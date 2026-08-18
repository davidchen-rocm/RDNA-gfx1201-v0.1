#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int ROWS = 12288;
constexpr int COLUMNS = 4096;
constexpr int WAVE = 32;
constexpr int WARPS = 8;
constexpr int THREADS = WAVE * WARPS;
constexpr int GROUP = 64;
constexpr int GROUPS_PER_ROW = COLUMNS / GROUP;
constexpr int Q4_RDNA_TILE_BYTES = 1088;
constexpr int Q4_K_BLOCK = 256;
constexpr int Q4_K_BLOCK_BYTES = 144;
constexpr int Q4_K_BLOCKS_PER_ROW = COLUMNS / Q4_K_BLOCK;
constexpr int WEIGHT_COPIES = 8;

void check(hipError_t status) {
    if (status != hipSuccess) {
        throw std::runtime_error(hipGetErrorString(status));
    }
}

std::vector<uint8_t> read_file_range(const char * path, uint64_t offset, uint64_t size) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(std::string("cannot open ") + path);
    }
    input.seekg(static_cast<std::streamoff>(offset));
    std::vector<uint8_t> data(size);
    input.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!input) {
        throw std::runtime_error(std::string("cannot read requested bytes from ") + path);
    }
    return data;
}

__device__ __forceinline__ int signed_int4_low(uint8_t value) {
    return int(static_cast<int8_t>(value << 4)) >> 4;
}

__device__ __forceinline__ int signed_int4_high(uint8_t value) {
    return int(static_cast<int8_t>(value)) >> 4;
}

__device__ __forceinline__ void get_scale_min(int index, const uint8_t * packed, uint8_t & scale, uint8_t & min) {
    if (index < 4) {
        scale = packed[index] & 63;
        min = packed[index + 4] & 63;
    } else {
        scale = (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4);
        min = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
    }
}

__global__ __launch_bounds__(THREADS) void q4_rdna_gemv(const uint8_t * weights, const float * activation, float * output) {
    __shared__ float activation_tile[WARPS][GROUP];
    const int lane = threadIdx.x & (WAVE - 1);
    const int warp = threadIdx.x / WAVE;
    const int row_tile = blockIdx.x * WARPS + warp;
    const int row = row_tile * WAVE + lane;
    float sum = 0.0f;

    for (int group = 0; group < GROUPS_PER_ROW; ++group) {
        activation_tile[warp][lane] = activation[group * GROUP + lane];
        activation_tile[warp][lane + WAVE] = activation[group * GROUP + lane + WAVE];
        __syncwarp();

        const uint8_t * tile = weights + (row_tile * GROUPS_PER_ROW + group) * Q4_RDNA_TILE_BYTES;
        const float scale = __half2float(reinterpret_cast<const __half *>(tile)[lane]);
        for (int i = 0; i < GROUP / 2; ++i) {
            const uint8_t packed = tile[64 + i * WAVE + lane];
            sum = fmaf(scale * float(signed_int4_low(packed)), activation_tile[warp][2 * i], sum);
            sum = fmaf(scale * float(signed_int4_high(packed)), activation_tile[warp][2 * i + 1], sum);
        }
        __syncwarp();
    }
    output[row] = sum;
}

__global__ __launch_bounds__(THREADS) void q4_k_gemv(const uint8_t * weights, const float * activation, float * output) {
    __shared__ float activation_tile[WARPS][Q4_K_BLOCK];
    const int lane = threadIdx.x & (WAVE - 1);
    const int warp = threadIdx.x / WAVE;
    const int row_tile = blockIdx.x * WARPS + warp;
    const int row = row_tile * WAVE + lane;
    const uint8_t * row_weights = weights + row * Q4_K_BLOCKS_PER_ROW * Q4_K_BLOCK_BYTES;
    float sum = 0.0f;

    for (int block_index = 0; block_index < Q4_K_BLOCKS_PER_ROW; ++block_index) {
        for (int i = lane; i < Q4_K_BLOCK; i += WAVE) {
            activation_tile[warp][i] = activation[block_index * Q4_K_BLOCK + i];
        }
        __syncwarp();

        const uint8_t * block = row_weights + block_index * Q4_K_BLOCK_BYTES;
        const float d = __half2float(reinterpret_cast<const __half *>(block)[0]);
        const float dmin = __half2float(reinterpret_cast<const __half *>(block)[1]);
        const uint8_t * scales = block + 4;
        const uint8_t * quants = block + 16;
        for (int chunk = 0; chunk < 4; ++chunk) {
            uint8_t scale0;
            uint8_t min0;
            uint8_t scale1;
            uint8_t min1;
            get_scale_min(2 * chunk, scales, scale0, min0);
            get_scale_min(2 * chunk + 1, scales, scale1, min1);
            const float ds0 = d * scale0;
            const float dm0 = dmin * min0;
            const float ds1 = d * scale1;
            const float dm1 = dmin * min1;
            for (int i = 0; i < 32; ++i) {
                const uint8_t packed = quants[chunk * 32 + i];
                const float weight0 = ds0 * float(packed & 15) - dm0;
                const float weight1 = ds1 * float(packed >> 4) - dm1;
                sum = fmaf(weight0, activation_tile[warp][chunk * 64 + i], sum);
                sum = fmaf(weight1, activation_tile[warp][chunk * 64 + i + 32], sum);
            }
        }
        __syncwarp();
    }
    output[row] = sum;
}

template <typename Kernel>
float benchmark(Kernel kernel, const uint8_t * weights, uint64_t weight_bytes, const float * activation, float * output) {
    constexpr int grid = ROWS / (WAVE * WARPS);
    constexpr int iterations = 1000;
    for (int i = 0; i < 100; ++i) {
        hipLaunchKernelGGL(kernel, dim3(grid), dim3(THREADS), 0, 0, weights + (i % WEIGHT_COPIES) * weight_bytes, activation, output);
    }
    check(hipDeviceSynchronize());

    hipEvent_t begin;
    hipEvent_t end;
    check(hipEventCreate(&begin));
    check(hipEventCreate(&end));
    check(hipEventRecord(begin));
    for (int i = 0; i < iterations; ++i) {
        hipLaunchKernelGGL(kernel, dim3(grid), dim3(THREADS), 0, 0, weights + (i % WEIGHT_COPIES) * weight_bytes, activation, output);
    }
    check(hipEventRecord(end));
    check(hipEventSynchronize(end));
    float milliseconds = 0.0f;
    check(hipEventElapsedTime(&milliseconds, begin, end));
    check(hipEventDestroy(begin));
    check(hipEventDestroy(end));
    return milliseconds * 1000.0f / iterations;
}

float half_to_float(uint16_t value) {
    const uint32_t sign = uint32_t(value & 0x8000) << 16;
    uint32_t exponent = (value >> 10) & 31;
    uint32_t mantissa = value & 1023;
    if (exponent == 0) {
        if (mantissa == 0) {
            float result;
            std::memcpy(&result, &sign, sizeof(result));
            return result;
        }
        int shift = 0;
        while ((mantissa & 1024) == 0) {
            mantissa <<= 1;
            ++shift;
        }
        mantissa &= 1023;
        exponent = uint32_t(127 - 15 - shift);
    } else {
        exponent += 127 - 15;
    }
    const uint32_t bits = sign | (exponent << 23) | (mantissa << 13);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::vector<float> reference_q4_rdna(const std::vector<uint8_t> & weights, const std::vector<float> & activation) {
    std::vector<float> output(ROWS);
    for (int row = 0; row < ROWS; ++row) {
        const int row_tile = row / WAVE;
        const int lane = row % WAVE;
        float sum = 0.0f;
        for (int group = 0; group < GROUPS_PER_ROW; ++group) {
            const uint8_t * tile = weights.data() + (row_tile * GROUPS_PER_ROW + group) * Q4_RDNA_TILE_BYTES;
            uint16_t scale_bits;
            std::memcpy(&scale_bits, tile + lane * 2, sizeof(scale_bits));
            const float scale = half_to_float(scale_bits);
            for (int i = 0; i < GROUP / 2; ++i) {
                const uint8_t packed = tile[64 + i * WAVE + lane];
                const int low = int(static_cast<int8_t>(packed << 4)) >> 4;
                const int high = int(static_cast<int8_t>(packed)) >> 4;
                sum = std::fma(scale * low, activation[group * GROUP + 2 * i], sum);
                sum = std::fma(scale * high, activation[group * GROUP + 2 * i + 1], sum);
            }
        }
        output[row] = sum;
    }
    return output;
}

} // namespace

int main(int argc, char ** argv) try {
    if (argc != 4) {
        std::cerr << "usage: q4rdna_gemv_bench Q4_RDNA_FILE Q4_K_GGUF Q4_K_OFFSET\n";
        return 2;
    }
    constexpr uint64_t q4rdna_size = uint64_t(ROWS) * COLUMNS / GROUP * 34;
    constexpr uint64_t q4k_size = uint64_t(ROWS) * COLUMNS / Q4_K_BLOCK * Q4_K_BLOCK_BYTES;
    const std::vector<uint8_t> q4rdna = read_file_range(argv[1], 0, q4rdna_size);
    const std::vector<uint8_t> q4k = read_file_range(argv[2], std::stoull(argv[3]), q4k_size);
    std::vector<float> activation(COLUMNS);
    for (int i = 0; i < COLUMNS; ++i) {
        activation[i] = std::sin(float(i) * 0.013f) * 0.25f + std::cos(float(i) * 0.007f) * 0.125f;
    }
    const std::vector<float> reference = reference_q4_rdna(q4rdna, activation);

    uint8_t * device_q4rdna = nullptr;
    uint8_t * device_q4k = nullptr;
    float * device_activation = nullptr;
    float * device_output = nullptr;
    check(hipSetDevice(0));
    check(hipMalloc(&device_q4rdna, q4rdna.size() * WEIGHT_COPIES));
    check(hipMalloc(&device_q4k, q4k.size() * WEIGHT_COPIES));
    check(hipMalloc(&device_activation, activation.size() * sizeof(float)));
    check(hipMalloc(&device_output, ROWS * sizeof(float)));
    for (int copy = 0; copy < WEIGHT_COPIES; ++copy) {
        check(hipMemcpy(device_q4rdna + copy * q4rdna.size(), q4rdna.data(), q4rdna.size(), hipMemcpyHostToDevice));
        check(hipMemcpy(device_q4k + copy * q4k.size(), q4k.data(), q4k.size(), hipMemcpyHostToDevice));
    }
    check(hipMemcpy(device_activation, activation.data(), activation.size() * sizeof(float), hipMemcpyHostToDevice));

    constexpr int grid = ROWS / (WAVE * WARPS);
    hipLaunchKernelGGL(q4_rdna_gemv, dim3(grid), dim3(THREADS), 0, 0, device_q4rdna, device_activation, device_output);
    check(hipDeviceSynchronize());
    std::vector<float> actual(ROWS);
    check(hipMemcpy(actual.data(), device_output, actual.size() * sizeof(float), hipMemcpyDeviceToHost));
    double max_error = 0.0;
    double sum_squared = 0.0;
    double reference_squared = 0.0;
    for (int i = 0; i < ROWS; ++i) {
        const double error = double(actual[i]) - reference[i];
        max_error = std::max(max_error, std::abs(error));
        sum_squared += error * error;
        reference_squared += double(reference[i]) * reference[i];
    }
    if (std::sqrt(sum_squared / reference_squared) > 1e-5) {
        throw std::runtime_error("Q4_RDNA GPU result failed the CPU reference check");
    }

    const float q4rdna_ms = benchmark(q4_rdna_gemv, device_q4rdna, q4rdna.size(), device_activation, device_output);
    const float q4k_ms = benchmark(q4_k_gemv, device_q4k, q4k.size(), device_activation, device_output);
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0));

    std::cout << properties.name << " (" << properties.gcnArchName << ")\n";
    std::cout << "matrix: " << ROWS << " outputs x " << COLUMNS << " inputs\n";
    std::cout << "weight rotation: " << WEIGHT_COPIES << " copies (cold-stream approximation)\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Q4_RDNA correctness max abs error: " << max_error << '\n';
    std::cout << "Q4_K naive wave32 GEMV:           " << q4k_ms << " us\n";
    std::cout << "Q4_RDNA wave32 GEMV:              " << q4rdna_ms << " us\n";
    std::cout << "Q4_RDNA / Q4_K time:              " << q4rdna_ms / q4k_ms << "x\n";
    std::cout << "Q4_K effective weight bandwidth:  " << double(q4k_size) / (q4k_ms * 1000.0) << " GB/s\n";
    std::cout << "Q4_RDNA effective weight bandwidth: " << double(q4rdna_size) / (q4rdna_ms * 1000.0) << " GB/s\n";

    check(hipFree(device_output));
    check(hipFree(device_activation));
    check(hipFree(device_q4k));
    check(hipFree(device_q4rdna));
    return 0;
} catch (const std::exception & error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}
