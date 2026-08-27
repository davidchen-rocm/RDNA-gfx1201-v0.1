#include "q4rdna.cuh"
#include "q4rdna_kernels.cuh"

#include "ggml.h"
#include "unary.cuh"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int Q4RDNA_GROUP = q4rdna_kernel::GROUP;
constexpr int Q4RDNA_TILE_ROWS = q4rdna_kernel::TILE_ROWS;
constexpr int Q4RDNA_TILE_BYTES = q4rdna_kernel::TILE_BYTES;
constexpr int Q4RDNA_WARPS = q4rdna_kernel::WAVES;
constexpr int Q4RDNA_THREADS = q4rdna_kernel::THREADS;

#pragma pack(push, 1)
struct q4rdna_file_header {
    char magic[8];
    uint32_t version;
    uint32_t count;
    uint32_t group;
    uint32_t tile_rows;
    uint32_t tile_bytes;
    uint32_t reserved;
    uint64_t data_offset;
    uint64_t data_bytes;
};

struct q4rdna_file_entry {
    char name[64];
    uint32_t rows;
    uint32_t columns;
    uint64_t offset;
    uint64_t size;
    uint64_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(q4rdna_file_header) == 48, "unexpected Q4_RDNA header size");
static_assert(sizeof(q4rdna_file_entry) == 96, "unexpected Q4_RDNA entry size");

struct q4rdna_device_entry {
    uint32_t rows;
    uint32_t columns;
    uint64_t relative_offset;
    uint64_t size;
    uint64_t hits = 0;
};

struct q4rdna_registry {
    std::once_flag init_once;
    bool enabled = false;
    uint8_t * data = nullptr;
    std::unordered_map<std::string, q4rdna_device_entry> entries;
};

q4rdna_registry & registry() {
    // The sidecar is an experiment-scoped, process-lifetime allocation.
    static q4rdna_registry * value = new q4rdna_registry;
    return *value;
}

std::atomic<uint64_t> q4rdna_launches{0};

void q4rdna_report_stats() {
    const uint64_t launches = q4rdna_launches.load();
    if (launches != 0) {
        std::fprintf(stderr, "Q4_RDNA: launched %llu decode GEMV kernels\n",
                     static_cast<unsigned long long>(launches));
    }
    q4rdna_registry & value = registry();
    uint64_t shape_12288_4096 = 0;
    uint64_t shape_4096_12288 = 0;
    uint64_t shape_4096_4096 = 0;
    uint64_t shape_1024_4096 = 0;
    uint64_t other = 0;
    uint32_t unique = 0;
    const bool trace = std::getenv("LLAMA_Q4_RDNA_TRACE") != nullptr;
    for (const auto & item : value.entries) {
        const q4rdna_device_entry & entry = item.second;
        if (entry.hits == 0) continue;
        ++unique;
        if (entry.rows == 12288 && entry.columns == 4096) shape_12288_4096 += entry.hits;
        else if (entry.rows == 4096 && entry.columns == 12288) shape_4096_12288 += entry.hits;
        else if (entry.rows == 4096 && entry.columns == 4096) shape_4096_4096 += entry.hits;
        else if (entry.rows == 1024 && entry.columns == 4096) shape_1024_4096 += entry.hits;
        else other += entry.hits;
        if (trace) {
            std::fprintf(stderr, "Q4_RDNA: hit %-31s %llu\n", item.first.c_str(),
                         static_cast<unsigned long long>(entry.hits));
        }
    }
    std::fprintf(stderr,
                 "Q4_RDNA: unique=%u, hits by rows x columns: 12288x4096=%llu, 4096x12288=%llu, "
                 "4096x4096=%llu, 1024x4096=%llu, other=%llu\n",
                 unique,
                 static_cast<unsigned long long>(shape_12288_4096),
                 static_cast<unsigned long long>(shape_4096_12288),
                 static_cast<unsigned long long>(shape_4096_4096),
                 static_cast<unsigned long long>(shape_1024_4096),
                 static_cast<unsigned long long>(other));
}

bool read_exact(std::ifstream & input, void * destination, size_t size) {
    input.read(static_cast<char *>(destination), static_cast<std::streamsize>(size));
    return input.good();
}

void q4rdna_initialize(q4rdna_registry & value, int device) {
    const char * path = std::getenv("LLAMA_Q4_RDNA_SIDECAR");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    if (device != 0) {
        std::fprintf(stderr, "Q4_RDNA: minimal integration only supports CUDA device 0\n");
        return;
    }

    std::ifstream input(path, std::ios::binary);
    q4rdna_file_header header{};
    if (!input || !read_exact(input, &header, sizeof(header)) ||
            std::memcmp(header.magic, "Q4RDNA1", 8) != 0 || header.version != 1 ||
            header.group != Q4RDNA_GROUP || header.tile_rows != Q4RDNA_TILE_ROWS ||
            header.tile_bytes != Q4RDNA_TILE_BYTES) {
        std::fprintf(stderr, "Q4_RDNA: invalid sidecar header: %s\n", path);
        return;
    }

    std::vector<q4rdna_file_entry> file_entries(header.count);
    if (!read_exact(input, file_entries.data(), file_entries.size() * sizeof(file_entries[0]))) {
        std::fprintf(stderr, "Q4_RDNA: truncated sidecar index: %s\n", path);
        return;
    }
    for (const q4rdna_file_entry & entry : file_entries) {
        if (std::memchr(entry.name, '\0', sizeof(entry.name)) == nullptr ||
                entry.offset < header.data_offset || entry.offset + entry.size < entry.offset ||
                entry.offset + entry.size > header.data_offset + header.data_bytes ||
                entry.rows % Q4RDNA_TILE_ROWS != 0 || entry.columns % Q4RDNA_GROUP != 0 ||
                entry.size != uint64_t(entry.rows) * entry.columns * 34 / Q4RDNA_GROUP) {
            std::fprintf(stderr, "Q4_RDNA: invalid sidecar entry\n");
            return;
        }
        value.entries.emplace(entry.name, q4rdna_device_entry{
            entry.rows, entry.columns, entry.offset - header.data_offset, entry.size, 0});
    }

    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaMalloc(&value.data, static_cast<size_t>(header.data_bytes)));
    input.seekg(static_cast<std::streamoff>(header.data_offset));
    constexpr size_t chunk_size = 64 * 1024 * 1024;
    std::vector<uint8_t> host(chunk_size);
    uint64_t copied = 0;
    while (copied < header.data_bytes) {
        const size_t count = static_cast<size_t>(std::min<uint64_t>(chunk_size, header.data_bytes - copied));
        if (!read_exact(input, host.data(), count)) {
            std::fprintf(stderr, "Q4_RDNA: truncated sidecar data: %s\n", path);
            CUDA_CHECK(cudaFree(value.data));
            value.data = nullptr;
            value.entries.clear();
            return;
        }
        CUDA_CHECK(cudaMemcpy(value.data + copied, host.data(), count, cudaMemcpyHostToDevice));
        copied += count;
    }
    value.enabled = true;
    std::atexit(q4rdna_report_stats);
    std::fprintf(stderr, "Q4_RDNA: loaded %u tensors, %.2f GiB on device 0 from %s\n",
                 header.count, double(header.data_bytes) / (1024.0 * 1024.0 * 1024.0), path);
}

__device__ __forceinline__ int q4rdna_low(uint8_t value) {
    return int(static_cast<int8_t>(value << 4)) >> 4;
}

__device__ __forceinline__ int q4rdna_high(uint8_t value) {
    return int(static_cast<int8_t>(value)) >> 4;
}

template <bool use_gate, bool use_add>
__global__ __launch_bounds__(Q4RDNA_THREADS) void q4rdna_gemv_pairwise(
        const uint8_t * weights, const uint8_t * gate_weights, const float * activation,
        const float * add, float * output, int columns) {
    const int lane = threadIdx.x & (Q4RDNA_TILE_ROWS - 1);
    const int warp = threadIdx.x / Q4RDNA_TILE_ROWS;
    const int pair_lane = lane & 1;
    const int row_lane = lane & ~1;
    const int row_tile = blockIdx.x * Q4RDNA_WARPS + warp;
    const int groups = columns / Q4RDNA_GROUP;
    float sum_0 = 0.0f;
    float sum_1 = 0.0f;
    float gate_sum_0 = 0.0f;
    float gate_sum_1 = 0.0f;

    for (int group = 0; group < groups; ++group) {
        const uint8_t * tile = weights + uint64_t(row_tile * groups + group) * Q4RDNA_TILE_BYTES;
        const uint8_t * gate_tile = nullptr;
        if constexpr (use_gate) {
            gate_tile = gate_weights + uint64_t(row_tile * groups + group) * Q4RDNA_TILE_BYTES;
        }
        float partial_0 = 0.0f;
        float partial_1 = 0.0f;
        float gate_partial_0 = 0.0f;
        float gate_partial_1 = 0.0f;
#pragma unroll
        for (int i = 0; i < Q4RDNA_GROUP / 2; ++i) {
            const float activation_low = activation[group * Q4RDNA_GROUP + 2 * i];
            const float activation_high = activation[group * Q4RDNA_GROUP + 2 * i + 1];
            const int packed_self = tile[64 + i * Q4RDNA_TILE_ROWS + lane];
            const int packed_peer = __shfl_xor(packed_self, 1, Q4RDNA_TILE_ROWS);
            int gate_packed_self = 0;
            int gate_packed_peer = 0;
            if constexpr (use_gate) {
                gate_packed_self = gate_tile[64 + i * Q4RDNA_TILE_ROWS + lane];
                gate_packed_peer = __shfl_xor(gate_packed_self, 1, Q4RDNA_TILE_ROWS);
            }
            if (pair_lane == (i & 1)) {
                const uint8_t packed_0 = pair_lane == 0 ? packed_self : packed_peer;
                const uint8_t packed_1 = pair_lane == 0 ? packed_peer : packed_self;
                partial_0 = fmaf(float(q4rdna_low(packed_0)), activation_low, partial_0);
                partial_0 = fmaf(float(q4rdna_high(packed_0)), activation_high, partial_0);
                partial_1 = fmaf(float(q4rdna_low(packed_1)), activation_low, partial_1);
                partial_1 = fmaf(float(q4rdna_high(packed_1)), activation_high, partial_1);
                if constexpr (use_gate) {
                    const uint8_t gate_packed_0 = pair_lane == 0 ? gate_packed_self : gate_packed_peer;
                    const uint8_t gate_packed_1 = pair_lane == 0 ? gate_packed_peer : gate_packed_self;
                    gate_partial_0 =
                        fmaf(float(q4rdna_low(gate_packed_0)), activation_low, gate_partial_0);
                    gate_partial_0 =
                        fmaf(float(q4rdna_high(gate_packed_0)), activation_high, gate_partial_0);
                    gate_partial_1 =
                        fmaf(float(q4rdna_low(gate_packed_1)), activation_low, gate_partial_1);
                    gate_partial_1 =
                        fmaf(float(q4rdna_high(gate_packed_1)), activation_high, gate_partial_1);
                }
            }
        }
        partial_0 += __shfl_xor(partial_0, 1, Q4RDNA_TILE_ROWS);
        partial_1 += __shfl_xor(partial_1, 1, Q4RDNA_TILE_ROWS);
        if constexpr (use_gate) {
            gate_partial_0 += __shfl_xor(gate_partial_0, 1, Q4RDNA_TILE_ROWS);
            gate_partial_1 += __shfl_xor(gate_partial_1, 1, Q4RDNA_TILE_ROWS);
        }
        if (pair_lane == 0) {
            const half * scales = reinterpret_cast<const half *>(tile);
            sum_0 = fmaf(__half2float(scales[row_lane]), partial_0, sum_0);
            sum_1 = fmaf(__half2float(scales[row_lane + 1]), partial_1, sum_1);
            if constexpr (use_gate) {
                const half * gate_scales = reinterpret_cast<const half *>(gate_tile);
                gate_sum_0 = fmaf(__half2float(gate_scales[row_lane]), gate_partial_0, gate_sum_0);
                gate_sum_1 = fmaf(__half2float(gate_scales[row_lane + 1]), gate_partial_1, gate_sum_1);
            }
        }
    }
    if (pair_lane == 0) {
        const int row_0 = row_tile * Q4RDNA_TILE_ROWS + row_lane;
        const int row_1 = row_0 + 1;
        if constexpr (use_add) {
            sum_0 += add[row_0];
            sum_1 += add[row_1];
        }
        if constexpr (use_gate) {
            sum_0 *= ggml_cuda_op_silu_single(gate_sum_0);
            sum_1 *= ggml_cuda_op_silu_single(gate_sum_1);
        }
        output[row_0] = sum_0;
        output[row_1] = sum_1;
    }
}

template <bool use_gate, bool use_add, int lanes_per_row>
__global__ __launch_bounds__(Q4RDNA_THREADS) void q4rdna_gemv_cooperative(
        const uint8_t * weights, const uint8_t * gate_weights, const float * activation,
        const float * add, float * output, int columns) {
    static_assert(lanes_per_row == 2 || lanes_per_row == 4 || lanes_per_row == 8,
                  "unsupported Q4_RDNA cooperative width");
    constexpr int quant_words = (Q4RDNA_TILE_BYTES - 64) / sizeof(uint32_t);
    __shared__ uint32_t quant_shared[Q4RDNA_WARPS][quant_words];
    __shared__ uint32_t gate_shared[use_gate ? Q4RDNA_WARPS : 1][quant_words];
    const int lane = threadIdx.x & (Q4RDNA_TILE_ROWS - 1);
    const int warp = threadIdx.x / Q4RDNA_TILE_ROWS;
    const int sub_lane = lane & (lanes_per_row - 1);
    const int row_base = lane - sub_lane;
    const int row_tile = blockIdx.x * Q4RDNA_WARPS + warp;
    const int groups = columns / Q4RDNA_GROUP;
    float sum[lanes_per_row] = {0.0f};
    float gate_sum[lanes_per_row] = {0.0f};

    for (int group = 0; group < groups; ++group) {
        const uint8_t * tile = weights + uint64_t(row_tile * groups + group) * Q4RDNA_TILE_BYTES;
        const uint8_t * gate_tile = nullptr;
        if constexpr (use_gate) {
            gate_tile = gate_weights + uint64_t(row_tile * groups + group) * Q4RDNA_TILE_BYTES;
        }
        const uint32_t * quant_words_global = reinterpret_cast<const uint32_t *>(tile + 64);
        const uint32_t * gate_words_global = nullptr;
        if constexpr (use_gate) {
            gate_words_global = reinterpret_cast<const uint32_t *>(gate_tile + 64);
        }
#pragma unroll
        for (int word = lane; word < quant_words; word += Q4RDNA_TILE_ROWS) {
            quant_shared[warp][word] = quant_words_global[word];
            if constexpr (use_gate) {
                gate_shared[warp][word] = gate_words_global[word];
            }
        }
        __syncwarp();

        const uint8_t * quant_bytes = reinterpret_cast<const uint8_t *>(quant_shared[warp]);
        const uint8_t * gate_bytes = nullptr;
        if constexpr (use_gate) {
            gate_bytes = reinterpret_cast<const uint8_t *>(gate_shared[warp]);
        }
        float partial[lanes_per_row] = {0.0f};
        float gate_partial[lanes_per_row] = {0.0f};
#pragma unroll
        for (int i = sub_lane; i < Q4RDNA_GROUP / 2; i += lanes_per_row) {
            float activation_low = 0.0f;
            float activation_high = 0.0f;
            if (row_base == 0) {
                activation_low = activation[group * Q4RDNA_GROUP + 2 * i];
                activation_high = activation[group * Q4RDNA_GROUP + 2 * i + 1];
            }
            activation_low = __shfl(activation_low, sub_lane, Q4RDNA_TILE_ROWS);
            activation_high = __shfl(activation_high, sub_lane, Q4RDNA_TILE_ROWS);
#pragma unroll
            for (int row = 0; row < lanes_per_row; ++row) {
                const uint8_t packed = quant_bytes[i * Q4RDNA_TILE_ROWS + row_base + row];
                partial[row] = fmaf(float(q4rdna_low(packed)), activation_low, partial[row]);
                partial[row] = fmaf(float(q4rdna_high(packed)), activation_high, partial[row]);
                if constexpr (use_gate) {
                    const uint8_t gate_packed = gate_bytes[i * Q4RDNA_TILE_ROWS + row_base + row];
                    gate_partial[row] =
                        fmaf(float(q4rdna_low(gate_packed)), activation_low, gate_partial[row]);
                    gate_partial[row] =
                        fmaf(float(q4rdna_high(gate_packed)), activation_high, gate_partial[row]);
                }
            }
        }

#pragma unroll
        for (int row = 0; row < lanes_per_row; ++row) {
#pragma unroll
            for (int offset = lanes_per_row / 2; offset > 0; offset /= 2) {
                partial[row] += __shfl_down(partial[row], offset, lanes_per_row);
                if constexpr (use_gate) {
                    gate_partial[row] += __shfl_down(gate_partial[row], offset, lanes_per_row);
                }
            }
            if (sub_lane == 0) {
                const int row_lane = row_base + row;
                const float scale = __half2float(reinterpret_cast<const half *>(tile)[row_lane]);
                sum[row] = fmaf(scale, partial[row], sum[row]);
                if constexpr (use_gate) {
                    const float gate_scale =
                        __half2float(reinterpret_cast<const half *>(gate_tile)[row_lane]);
                    gate_sum[row] = fmaf(gate_scale, gate_partial[row], gate_sum[row]);
                }
            }
        }
        __syncwarp();
    }
    if (sub_lane == 0) {
#pragma unroll
        for (int row = 0; row < lanes_per_row; ++row) {
            const int output_row = row_tile * Q4RDNA_TILE_ROWS + row_base + row;
            if constexpr (use_add) sum[row] += add[output_row];
            if constexpr (use_gate) sum[row] *= ggml_cuda_op_silu_single(gate_sum[row]);
            output[output_row] = sum[row];
        }
    }
}

int q4rdna_cooperative_width(int rows, int columns) {
    const char * override_width = std::getenv("LLAMA_Q4_RDNA_COOP");
    if (override_width != nullptr) {
        const int width = std::atoi(override_width);
        if (width == 2 || width == 4 || width == 8 ||
                ((width == 16 || width == 32) && rows == 1024 && columns == 4096)) return width;
    }
    if (columns == 4096 && rows == 12288) return 8;
    if (columns == 4096 && rows == 4096) return 8;
    if (columns == 4096 && rows == 1024) return 32;
    return 8;
}

int q4rdna_small_slice_rows() {
    const char * override_rows = std::getenv("LLAMA_Q4_RDNA_SMALL_ROWS");
    if (override_rows != nullptr) {
        const int rows = std::atoi(override_rows);
        if (rows == 8 || rows == 16 || rows == 32) return rows;
    }
    return 32;
}

template <bool use_gate, bool use_add, int split_waves, int slice_rows>
void q4rdna_launch_small(
        const uint8_t * weights, const uint8_t * gate_weights, const float * activation,
        const float * add, float * output, int rows, int columns, cudaStream_t stream) {
    GGML_ASSERT(rows % slice_rows == 0);
    q4rdna_kernel::gemv_small_split_k<use_gate, use_add, split_waves, slice_rows>
        <<<rows / slice_rows, split_waves * Q4RDNA_TILE_ROWS, 0, stream>>>(
            weights, gate_weights, activation, add, output, columns);
}

template <bool use_gate, bool use_add, int lanes_per_row>
void q4rdna_launch_width(
        const uint8_t * weights, const uint8_t * gate_weights, const float * activation,
        const float * add, float * output, int rows, int columns, cudaStream_t stream) {
    constexpr int tiles_per_block = Q4RDNA_WARPS / lanes_per_row;
    constexpr int rows_per_block = Q4RDNA_TILE_ROWS * tiles_per_block;
    GGML_ASSERT(rows % rows_per_block == 0);
    q4rdna_kernel::gemv_split_k<use_gate, use_add, lanes_per_row>
        <<<rows / rows_per_block, Q4RDNA_THREADS, 0, stream>>>(
            weights, gate_weights, activation, add, output, columns);
}

template <bool use_gate, bool use_add>
void q4rdna_launch(
        const uint8_t * weights, const uint8_t * gate_weights, const float * activation,
        const float * add, float * output, int rows, int columns, cudaStream_t stream) {
    const char * mapping = std::getenv("LLAMA_Q4_RDNA_MAPPING");
    if (mapping != nullptr && std::strcmp(mapping, "old") == 0) {
        constexpr int rows_per_block = Q4RDNA_TILE_ROWS * Q4RDNA_WARPS;
        GGML_ASSERT(rows % rows_per_block == 0);
        q4rdna_kernel::gemv_old<use_gate, use_add>
            <<<rows / rows_per_block, Q4RDNA_THREADS, 0, stream>>>(
                weights, gate_weights, activation, add, output, columns);
        return;
    }
    const int width = q4rdna_cooperative_width(rows, columns);
    if (rows == 1024 && columns == 4096) {
        const int slice_rows = q4rdna_small_slice_rows();
        if (width == 8) {
            switch (slice_rows) {
                case 8:
                    q4rdna_launch_small<use_gate, use_add, 8, 8>(
                        weights, gate_weights, activation, add, output, rows, columns, stream);
                    return;
                case 16:
                    q4rdna_launch_small<use_gate, use_add, 8, 16>(
                        weights, gate_weights, activation, add, output, rows, columns, stream);
                    return;
                default:
                    break;
            }
        } else if (width == 16) {
            if (slice_rows == 16) {
                q4rdna_launch_small<use_gate, use_add, 16, 16>(
                    weights, gate_weights, activation, add, output, rows, columns, stream);
            } else {
                q4rdna_launch_small<use_gate, use_add, 16, 32>(
                    weights, gate_weights, activation, add, output, rows, columns, stream);
            }
            return;
        } else if (width == 32) {
            q4rdna_launch_small<use_gate, use_add, 32, 32>(
                weights, gate_weights, activation, add, output, rows, columns, stream);
            return;
        }
    }
    switch (width) {
        case 2:
            q4rdna_launch_width<use_gate, use_add, 2>(
                weights, gate_weights, activation, add, output, rows, columns, stream);
            break;
        case 8:
            q4rdna_launch_width<use_gate, use_add, 8>(
                weights, gate_weights, activation, add, output, rows, columns, stream);
            break;
        default:
            q4rdna_launch_width<use_gate, use_add, 4>(
                weights, gate_weights, activation, add, output, rows, columns, stream);
            break;
    }
}

q4rdna_device_entry * q4rdna_find(ggml_backend_cuda_context & ctx, const ggml_tensor * weight) {
    const char * path = std::getenv("LLAMA_Q4_RDNA_SIDECAR");
    if (path == nullptr || path[0] == '\0' || ctx.device != 0) return nullptr;
    const char * scope = std::getenv("LLAMA_Q4_RDNA_SCOPE");
    if (scope != nullptr) {
        if (std::strcmp(scope, "ffn") == 0 && std::strstr(weight->name, ".ffn_") == nullptr) return nullptr;
        if (std::strcmp(scope, "qkv-fallback") == 0 &&
                (std::strstr(weight->name, ".attn_q.") != nullptr ||
                 std::strstr(weight->name, ".attn_k.") != nullptr ||
                 std::strstr(weight->name, ".attn_v.") != nullptr)) return nullptr;
        if (std::strcmp(scope, "hotspot") == 0 &&
                std::strstr(weight->name, ".ffn_gate.") == nullptr &&
                std::strstr(weight->name, ".ffn_up.") == nullptr) return nullptr;
        if (std::strcmp(scope, "attn") == 0 && std::strstr(weight->name, ".attn_") == nullptr) return nullptr;
    }
    q4rdna_registry & value = registry();
    std::call_once(value.init_once, [&value, &ctx]() { q4rdna_initialize(value, ctx.device); });
    if (!value.enabled) return nullptr;
    const auto iterator = value.entries.find(weight->name);
    if (iterator == value.entries.end()) return nullptr;
    q4rdna_device_entry & entry = iterator->second;
    if (weight->ne[0] != entry.columns || weight->ne[1] != entry.rows) return nullptr;
    return &entry;
}

void q4rdna_count(q4rdna_device_entry & entry) {
    ++entry.hits;
}

} // namespace

bool ggml_cuda_q4rdna_mul_mat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    const char * path = std::getenv("LLAMA_Q4_RDNA_SIDECAR");
    if (path == nullptr || path[0] == '\0' || ctx.device != 0 ||
            src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            src0->ne[2] != 1 || src0->ne[3] != 1 ||
            src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            dst->ne[1] != 1 || dst->ne[2] != 1 || dst->ne[3] != 1 ||
            !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }

    q4rdna_device_entry * entry = q4rdna_find(ctx, src0);
    if (entry == nullptr) return false;
    if (src1->ne[0] != entry->columns || dst->ne[0] != entry->rows) {
        std::fprintf(stderr, "Q4_RDNA: shape mismatch for %s\n", src0->name);
        return false;
    }

    q4rdna_registry & value = registry();
    q4rdna_launch<false, false>(
        value.data + entry->relative_offset, nullptr,
        static_cast<const float *>(src1->data),
        nullptr,
        static_cast<float *>(dst->data),
        entry->rows, entry->columns, ctx.stream());
    q4rdna_launches.fetch_add(1, std::memory_order_relaxed);
    q4rdna_count(*entry);
    return true;
}

bool ggml_cuda_q4rdna_mul_mat_add(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * add,
        ggml_tensor * dst) {
    if (src1->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
            !ggml_is_contiguous(src1) || !ggml_is_contiguous(add) || !ggml_is_contiguous(dst)) return false;
    q4rdna_device_entry * entry = q4rdna_find(ctx, src0);
    if (entry == nullptr || src1->ne[0] != entry->columns || dst->ne[0] != entry->rows ||
            ggml_nelements(add) != entry->rows || ggml_nelements(dst) != entry->rows) return false;
    q4rdna_registry & value = registry();
    q4rdna_launch<false, true>(
        value.data + entry->relative_offset, nullptr,
        static_cast<const float *>(src1->data), static_cast<const float *>(add->data),
        static_cast<float *>(dst->data), entry->rows, entry->columns, ctx.stream());
    q4rdna_launches.fetch_add(1, std::memory_order_relaxed);
    q4rdna_count(*entry);
    return true;
}

bool ggml_cuda_q4rdna_mul_mat_glu(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * up,
        const ggml_tensor * gate,
        const ggml_tensor * activation,
        enum ggml_glu_op glu_op,
        ggml_tensor * dst) {
    if (glu_op != GGML_GLU_OP_SWIGLU || activation->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
            activation->ne[1] != 1 || activation->ne[2] != 1 || activation->ne[3] != 1 ||
            !ggml_is_contiguous(activation) || !ggml_is_contiguous(dst)) return false;
    q4rdna_device_entry * up_entry = q4rdna_find(ctx, up);
    q4rdna_device_entry * gate_entry = q4rdna_find(ctx, gate);
    if (up_entry == nullptr || gate_entry == nullptr ||
            up_entry->rows != gate_entry->rows || up_entry->columns != gate_entry->columns ||
            activation->ne[0] != up_entry->columns || ggml_nelements(dst) != up_entry->rows) return false;
    q4rdna_registry & value = registry();
    q4rdna_launch<true, false>(
        value.data + up_entry->relative_offset, value.data + gate_entry->relative_offset,
        static_cast<const float *>(activation->data), nullptr,
        static_cast<float *>(dst->data), up_entry->rows, up_entry->columns, ctx.stream());
    q4rdna_launches.fetch_add(1, std::memory_order_relaxed);
    q4rdna_count(*up_entry);
    q4rdna_count(*gate_entry);
    return true;
}
