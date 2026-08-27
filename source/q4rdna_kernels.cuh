#pragma once

#include <cstdint>

namespace q4rdna_kernel {

constexpr int GROUP = 64;
constexpr int TILE_ROWS = 32;
constexpr int TILE_BYTES = 1088;
constexpr int WAVES = 8;
constexpr int THREADS = TILE_ROWS * WAVES;

__device__ __forceinline__ int low(uint8_t value) {
    return int(static_cast<int8_t>(value << 4)) >> 4;
}

__device__ __forceinline__ int high(uint8_t value) {
    return int(static_cast<int8_t>(value)) >> 4;
}

__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

template <bool use_gate, bool use_add>
__global__ __launch_bounds__(THREADS) void gemv_old(
        const uint8_t * weights, const uint8_t * gate_weights, const float * activation,
        const float * add, float * output, int columns) {
    __shared__ float activation_tile[WAVES][GROUP];
    const int lane = threadIdx.x & (TILE_ROWS - 1);
    const int wave = threadIdx.x / TILE_ROWS;
    const int row_tile = blockIdx.x * WAVES + wave;
    const int row = row_tile * TILE_ROWS + lane;
    const int groups = columns / GROUP;
    float sum = 0.0f;
    float gate_sum = 0.0f;

    for (int group = 0; group < groups; ++group) {
        activation_tile[wave][lane] = activation[group * GROUP + lane];
        activation_tile[wave][lane + TILE_ROWS] = activation[group * GROUP + lane + TILE_ROWS];
        __syncwarp();

        const uint8_t * tile = weights + uint64_t(row_tile * groups + group) * TILE_BYTES;
        const float scale = __half2float(reinterpret_cast<const __half *>(tile)[lane]);
        const uint8_t * gate_tile = nullptr;
        float gate_scale = 0.0f;
        if constexpr (use_gate) {
            gate_tile = gate_weights + uint64_t(row_tile * groups + group) * TILE_BYTES;
            gate_scale = __half2float(reinterpret_cast<const __half *>(gate_tile)[lane]);
        }
#pragma unroll
        for (int i = 0; i < GROUP / 2; ++i) {
            const uint8_t packed = tile[64 + i * TILE_ROWS + lane];
            sum = fmaf(scale * float(low(packed)), activation_tile[wave][2 * i], sum);
            sum = fmaf(scale * float(high(packed)), activation_tile[wave][2 * i + 1], sum);
            if constexpr (use_gate) {
                const uint8_t gate_packed = gate_tile[64 + i * TILE_ROWS + lane];
                gate_sum = fmaf(
                    gate_scale * float(low(gate_packed)), activation_tile[wave][2 * i], gate_sum);
                gate_sum = fmaf(
                    gate_scale * float(high(gate_packed)), activation_tile[wave][2 * i + 1], gate_sum);
            }
        }
        __syncwarp();
    }
    if constexpr (use_add) sum += add[row];
    if constexpr (use_gate) sum *= silu(gate_sum);
    output[row] = sum;
}

template <bool use_gate, bool use_add, int split_waves>
__global__ __launch_bounds__(THREADS) void gemv_split_k(
        const uint8_t * weights, const uint8_t * gate_weights, const float * activation,
        const float * add, float * output, int columns) {
    static_assert(split_waves == 2 || split_waves == 4 || split_waves == 8,
                  "unsupported Q4_RDNA split-K width");
    constexpr int tiles_per_block = WAVES / split_waves;
    __shared__ float partial_shared[WAVES][TILE_ROWS];
    __shared__ float gate_partial_shared[use_gate ? WAVES : 1][TILE_ROWS];
    const int lane = threadIdx.x & (TILE_ROWS - 1);
    const int wave = threadIdx.x / TILE_ROWS;
    const int split = wave & (split_waves - 1);
    const int tile_in_block = wave / split_waves;
    const int row_tile = blockIdx.x * tiles_per_block + tile_in_block;
    const int groups = columns / GROUP;
    float sum = 0.0f;
    float gate_sum = 0.0f;

    for (int group = split; group < groups; group += split_waves) {
        const uint8_t * tile = weights + uint64_t(row_tile * groups + group) * TILE_BYTES;
        const uint8_t * gate_tile = nullptr;
        if constexpr (use_gate) {
            gate_tile = gate_weights + uint64_t(row_tile * groups + group) * TILE_BYTES;
        }
        float partial_low = 0.0f;
        float partial_high = 0.0f;
        float gate_partial_low = 0.0f;
        float gate_partial_high = 0.0f;
#pragma unroll 4
        for (int i = 0; i < GROUP / 2; ++i) {
            const float activation_low = activation[group * GROUP + 2 * i];
            const float activation_high = activation[group * GROUP + 2 * i + 1];
            const uint8_t packed = tile[64 + i * TILE_ROWS + lane];
            partial_low = fmaf(float(low(packed)), activation_low, partial_low);
            partial_high = fmaf(float(high(packed)), activation_high, partial_high);
            if constexpr (use_gate) {
                const uint8_t gate_packed = gate_tile[64 + i * TILE_ROWS + lane];
                gate_partial_low = fmaf(float(low(gate_packed)), activation_low, gate_partial_low);
                gate_partial_high = fmaf(float(high(gate_packed)), activation_high, gate_partial_high);
            }
        }
        const float scale = __half2float(reinterpret_cast<const __half *>(tile)[lane]);
        sum = fmaf(scale, partial_low + partial_high, sum);
        if constexpr (use_gate) {
            const float gate_scale = __half2float(reinterpret_cast<const __half *>(gate_tile)[lane]);
            gate_sum = fmaf(gate_scale, gate_partial_low + gate_partial_high, gate_sum);
        }
    }

    partial_shared[wave][lane] = sum;
    if constexpr (use_gate) {
        gate_partial_shared[wave][lane] = gate_sum;
    }
    __syncthreads();

    if (split == 0) {
#pragma unroll
        for (int other = 1; other < split_waves; ++other) {
            sum += partial_shared[wave + other][lane];
            if constexpr (use_gate) {
                gate_sum += gate_partial_shared[wave + other][lane];
            }
        }
        const int row = row_tile * TILE_ROWS + lane;
        if constexpr (use_add) sum += add[row];
        if constexpr (use_gate) sum *= silu(gate_sum);
        output[row] = sum;
    }
}

template <bool use_gate, bool use_add, int split_waves, int slice_rows>
__global__ __launch_bounds__(split_waves * TILE_ROWS) void gemv_small_split_k(
        const uint8_t * weights, const uint8_t * gate_weights, const float * activation,
        const float * add, float * output, int columns) {
    static_assert(split_waves == 8 || split_waves == 16 || split_waves == 32,
                  "unsupported Q4_RDNA small split-K width");
    static_assert(slice_rows == 8 || slice_rows == 16 || slice_rows == 32,
                  "unsupported Q4_RDNA small-row slice");
    constexpr int slices_per_tile = TILE_ROWS / slice_rows;
    __shared__ float partial_shared[split_waves][TILE_ROWS];
    __shared__ float gate_partial_shared[use_gate ? split_waves : 1][TILE_ROWS];
    const int lane = threadIdx.x & (TILE_ROWS - 1);
    const int wave = threadIdx.x / TILE_ROWS;
    const int row_slice = blockIdx.x;
    const int row_tile = row_slice / slices_per_tile;
    const int row_lane = (row_slice % slices_per_tile) * slice_rows + lane;
    const int groups = columns / GROUP;
    float sum = 0.0f;
    float gate_sum = 0.0f;

    if (lane < slice_rows) {
        for (int group = wave; group < groups; group += split_waves) {
            const uint8_t * tile = weights + uint64_t(row_tile * groups + group) * TILE_BYTES;
            const uint8_t * gate_tile = nullptr;
            if constexpr (use_gate) {
                gate_tile = gate_weights + uint64_t(row_tile * groups + group) * TILE_BYTES;
            }
            float partial_low = 0.0f;
            float partial_high = 0.0f;
            float gate_partial_low = 0.0f;
            float gate_partial_high = 0.0f;
#pragma unroll 4
            for (int i = 0; i < GROUP / 2; ++i) {
                const float activation_low = activation[group * GROUP + 2 * i];
                const float activation_high = activation[group * GROUP + 2 * i + 1];
                const uint8_t packed = tile[64 + i * TILE_ROWS + row_lane];
                partial_low = fmaf(float(low(packed)), activation_low, partial_low);
                partial_high = fmaf(float(high(packed)), activation_high, partial_high);
                if constexpr (use_gate) {
                    const uint8_t gate_packed = gate_tile[64 + i * TILE_ROWS + row_lane];
                    gate_partial_low = fmaf(float(low(gate_packed)), activation_low, gate_partial_low);
                    gate_partial_high = fmaf(float(high(gate_packed)), activation_high, gate_partial_high);
                }
            }
            const float scale = __half2float(reinterpret_cast<const __half *>(tile)[row_lane]);
            sum = fmaf(scale, partial_low + partial_high, sum);
            if constexpr (use_gate) {
                const float gate_scale = __half2float(reinterpret_cast<const __half *>(gate_tile)[row_lane]);
                gate_sum = fmaf(gate_scale, gate_partial_low + gate_partial_high, gate_sum);
            }
        }
    }

    partial_shared[wave][lane] = sum;
    if constexpr (use_gate) {
        gate_partial_shared[wave][lane] = gate_sum;
    }
    __syncthreads();

    if (wave == 0 && lane < slice_rows) {
#pragma unroll
        for (int other = 1; other < split_waves; ++other) {
            sum += partial_shared[other][lane];
            if constexpr (use_gate) {
                gate_sum += gate_partial_shared[other][lane];
            }
        }
        const int row = row_tile * TILE_ROWS + row_lane;
        if constexpr (use_add) sum += add[row];
        if constexpr (use_gate) sum *= silu(gate_sum);
        output[row] = sum;
    }
}

} // namespace q4rdna_kernel
