#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint64_t QK_K = 256;
constexpr uint64_t Q4_K_BLOCK_BYTES = 144;
constexpr uint64_t Q4_RDNA_GROUP = 64;
constexpr uint64_t Q4_RDNA_GROUP_BYTES = 34;
constexpr uint64_t Q4_RDNA_TILE_ROWS = 32;
constexpr uint64_t Q4_RDNA_TILE_BYTES = Q4_RDNA_TILE_ROWS * Q4_RDNA_GROUP_BYTES;

enum class ScaleAlgorithm {
    Absmax,
    Range,
    Mse,
};

float bits_to_float(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float bf16_to_float(uint16_t value) {
    return bits_to_float(uint32_t(value) << 16);
}

float fp16_to_float(uint16_t value) {
    const uint32_t sign = uint32_t(value & 0x8000) << 16;
    uint32_t exponent = (value >> 10) & 0x1f;
    uint32_t mantissa = value & 0x03ff;
    if (exponent == 0) {
        if (mantissa == 0) {
            return bits_to_float(sign);
        }
        int shift = 0;
        while ((mantissa & 0x0400) == 0) {
            mantissa <<= 1;
            ++shift;
        }
        mantissa &= 0x03ff;
        exponent = uint32_t(127 - 15 - shift);
    } else if (exponent == 31) {
        exponent = 255;
    } else {
        exponent += 127 - 15;
    }
    return bits_to_float(sign | (exponent << 23) | (mantissa << 13));
}

uint16_t float_to_fp16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = uint16_t((bits >> 16) & 0x8000);
    const uint32_t mantissa = bits & 0x007fffff;
    const int exponent = int((bits >> 23) & 0xff) - 127 + 15;

    if (exponent <= 0) {
        if (exponent < -10) {
            return sign;
        }
        uint32_t normalized = mantissa | 0x00800000;
        const int shift = 14 - exponent;
        uint32_t rounded = normalized >> shift;
        const uint32_t remainder = normalized & ((uint32_t(1) << shift) - 1);
        const uint32_t halfway = uint32_t(1) << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1))) {
            ++rounded;
        }
        return sign | uint16_t(rounded);
    }
    if (exponent >= 31) {
        return sign | 0x7c00;
    }

    uint32_t rounded = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fff;
    if (remainder > 0x1000 || (remainder == 0x1000 && (rounded & 1))) {
        ++rounded;
        if (rounded == 0x400) {
            rounded = 0;
            if (exponent + 1 >= 31) {
                return sign | 0x7c00;
            }
            return sign | uint16_t((exponent + 1) << 10);
        }
    }
    return sign | uint16_t(exponent << 10) | uint16_t(rounded);
}

void get_scale_min(int index, const uint8_t * packed, uint8_t & scale, uint8_t & min) {
    if (index < 4) {
        scale = packed[index] & 63;
        min = packed[index + 4] & 63;
    } else {
        scale = (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4);
        min = (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4);
    }
}

void dequantize_q4_k(const std::array<uint8_t, Q4_K_BLOCK_BYTES> & block, std::array<float, QK_K> & output) {
    uint16_t d_bits;
    uint16_t dmin_bits;
    std::memcpy(&d_bits, block.data(), sizeof(d_bits));
    std::memcpy(&dmin_bits, block.data() + 2, sizeof(dmin_bits));
    const float d = fp16_to_float(d_bits);
    const float dmin = fp16_to_float(dmin_bits);
    const uint8_t * scales = block.data() + 4;
    const uint8_t * quants = block.data() + 16;
    for (int chunk = 0; chunk < 4; ++chunk) {
        uint8_t scale0;
        uint8_t min0;
        uint8_t scale1;
        uint8_t min1;
        get_scale_min(chunk * 2, scales, scale0, min0);
        get_scale_min(chunk * 2 + 1, scales, scale1, min1);
        for (int i = 0; i < 32; ++i) {
            const uint8_t packed = quants[chunk * 32 + i];
            output[chunk * 64 + i] = d * scale0 * float(packed & 15) - dmin * min0;
            output[chunk * 64 + 32 + i] = d * scale1 * float(packed >> 4) - dmin * min1;
        }
    }
}

int quantize_value(float value, float scale) {
    if (scale == 0.0f) {
        return 0;
    }
    return std::clamp(int(std::round(value / scale)), -8, 7);
}

double scale_error(const float * input, uint16_t scale_bits) {
    const float scale = fp16_to_float(scale_bits);
    double error = 0.0;
    for (uint64_t i = 0; i < Q4_RDNA_GROUP; ++i) {
        const int quant = quantize_value(input[i], scale);
        const double delta = double(scale) * quant - input[i];
        error += delta * delta;
    }
    return error;
}

void consider_scale(const float * input, float scale, uint16_t & best_bits, double & best_error) {
    if (!std::isfinite(scale)) {
        return;
    }
    const uint16_t bits = float_to_fp16(scale);
    const double error = scale_error(input, bits);
    if (error < best_error) {
        best_bits = bits;
        best_error = error;
    }
}

void refine_mse_scale(const float * input, float initial_scale, uint16_t & best_bits, double & best_error) {
    uint16_t scale_bits = float_to_fp16(initial_scale);
    for (int iteration = 0; iteration < 12; ++iteration) {
        const float scale = fp16_to_float(scale_bits);
        consider_scale(input, scale, best_bits, best_error);
        double numerator = 0.0;
        double denominator = 0.0;
        for (uint64_t i = 0; i < Q4_RDNA_GROUP; ++i) {
            const int quant = quantize_value(input[i], scale);
            numerator += double(input[i]) * quant;
            denominator += double(quant) * quant;
        }
        if (denominator == 0.0) {
            break;
        }
        const uint16_t next_bits = float_to_fp16(float(numerator / denominator));
        if (next_bits == scale_bits) {
            break;
        }
        scale_bits = next_bits;
    }

    // The optimum above is continuous; check the adjacent FP16 scales too.
    if ((best_bits & 0x7fff) != 0 && (best_bits & 0x7fff) < 0x7bff) {
        const uint16_t magnitude = best_bits & 0x7fff;
        const uint16_t sign = best_bits & 0x8000;
        consider_scale(input, fp16_to_float(sign | uint16_t(magnitude - 1)), best_bits, best_error);
        consider_scale(input, fp16_to_float(sign | uint16_t(magnitude + 1)), best_bits, best_error);
    }
}

uint16_t select_scale(const float * input, ScaleAlgorithm algorithm) {
    float positive_max = 0.0f;
    float negative_max = 0.0f;
    float max_abs = 0.0f;
    for (uint64_t i = 0; i < Q4_RDNA_GROUP; ++i) {
        positive_max = std::max(positive_max, input[i]);
        negative_max = std::max(negative_max, -input[i]);
        max_abs = std::max(max_abs, std::abs(input[i]));
    }
    if (max_abs == 0.0f) {
        return 0;
    }
    if (algorithm == ScaleAlgorithm::Absmax) {
        return float_to_fp16(max_abs / 7.0f);
    }

    // Positive scale has representable range [-8s, +7s]. A negative scale
    // reverses that asymmetry to [-7|s|, +8|s|].
    const float positive_range_scale = std::max(positive_max / 7.0f, negative_max / 8.0f);
    const float negative_range_scale = -std::max(positive_max / 8.0f, negative_max / 7.0f);
    uint16_t best_bits = 0;
    double best_error = std::numeric_limits<double>::infinity();
    consider_scale(input, positive_range_scale, best_bits, best_error);
    consider_scale(input, negative_range_scale, best_bits, best_error);
    if (algorithm == ScaleAlgorithm::Range) {
        return best_bits;
    }

    // Multiple clipped starts avoid the local basin produced by a single
    // absmax initialization. Quant assignments and error use the stored FP16
    // scale, exactly as the unchanged decoder does.
    constexpr std::array<float, 12> START_FACTORS = {
        1.0f, 0.9f, 0.8f, 0.7f, 0.6f, 0.55f,
        0.5f, 0.45f, 0.4f, 0.35f, 0.3f, 0.25f,
    };
    for (float factor : START_FACTORS) {
        refine_mse_scale(input, positive_range_scale * factor, best_bits, best_error);
        refine_mse_scale(input, negative_range_scale * factor, best_bits, best_error);
    }
    return best_bits;
}

const char * algorithm_name(ScaleAlgorithm algorithm) {
    switch (algorithm) {
        case ScaleAlgorithm::Absmax: return "absmax";
        case ScaleAlgorithm::Range: return "range";
        case ScaleAlgorithm::Mse: return "mse";
    }
    return "unknown";
}

ScaleAlgorithm parse_algorithm(const std::string & name) {
    if (name == "absmax") {
        return ScaleAlgorithm::Absmax;
    }
    if (name == "range") {
        return ScaleAlgorithm::Range;
    }
    if (name == "mse") {
        return ScaleAlgorithm::Mse;
    }
    throw std::runtime_error("scale algorithm must be one of: absmax, range, mse");
}

void quantize_q4_rdna(const float * input, ScaleAlgorithm algorithm,
                      std::array<uint8_t, Q4_RDNA_GROUP_BYTES> & packed, float * output) {
    const uint16_t scale_bits = select_scale(input, algorithm);
    const float stored_scale = fp16_to_float(scale_bits);
    std::memcpy(packed.data(), &scale_bits, sizeof(scale_bits));
    for (uint64_t i = 0; i < Q4_RDNA_GROUP; i += 2) {
        const int low = quantize_value(input[i], stored_scale);
        const int high = quantize_value(input[i + 1], stored_scale);
        packed[2 + i / 2] = uint8_t(low & 15) | uint8_t((high & 15) << 4);
        output[i] = stored_scale * float(low);
        output[i + 1] = stored_scale * float(high);
    }
}

struct Metrics {
    long double squared_error = 0.0;
    long double absolute_error = 0.0;
    long double reference_squared = 0.0;
    long double candidate_squared = 0.0;
    long double dot = 0.0;
    double max_error = 0.0;
    uint64_t count = 0;

    void add(float reference, float candidate) {
        const double error = double(candidate) - double(reference);
        squared_error += error * error;
        absolute_error += std::abs(error);
        reference_squared += double(reference) * reference;
        candidate_squared += double(candidate) * candidate;
        dot += double(reference) * candidate;
        max_error = std::max(max_error, std::abs(error));
        ++count;
    }

    double mse() const {
        return double(squared_error / count);
    }

    double mae() const {
        return double(absolute_error / count);
    }

    double relative_l2() const {
        return std::sqrt(double(squared_error / reference_squared));
    }

    double cosine() const {
        return double(dot / std::sqrt(reference_squared * candidate_squared));
    }
};

void seek(std::ifstream & input, uint64_t offset) {
    input.seekg(static_cast<std::streamoff>(offset));
    if (!input) {
        throw std::runtime_error("seek failed");
    }
}

template <typename T, size_t N>
void read(std::ifstream & input, std::array<T, N> & buffer) {
    input.read(reinterpret_cast<char *>(buffer.data()), sizeof(T) * N);
    if (!input) {
        throw std::runtime_error("input ended before the matrix was complete");
    }
}

void print_metrics(const char * name, const Metrics & metrics, double mse_ratio) {
    std::cout << name << '\n';
    std::cout << "  MSE:               " << metrics.mse() << '\n';
    std::cout << "  MSE / Q4_K MSE:    " << mse_ratio << '\n';
    std::cout << "  relative L2 error: " << metrics.relative_l2() << '\n';
    std::cout << "  cosine similarity: " << metrics.cosine() << '\n';
    std::cout << "  mean abs error:    " << metrics.mae() << '\n';
    std::cout << "  max abs error:     " << metrics.max_error << '\n';
}

} // namespace

int main(int argc, char ** argv) try {
    if (argc != 9 && argc != 10) {
        std::cerr << "usage: q4rdna_cpu_experiment BF16_FILE BF16_OFFSET Q4K_GGUF Q4K_OFFSET ROWS COLS OUTPUT_Q4RDNA OUTPUT_JSON [absmax|range|mse]\n";
        return 2;
    }
    const uint64_t bf16_offset = std::stoull(argv[2]);
    const uint64_t q4k_offset = std::stoull(argv[4]);
    const uint64_t rows = std::stoull(argv[5]);
    const uint64_t columns = std::stoull(argv[6]);
    const ScaleAlgorithm algorithm = parse_algorithm(argc == 10 ? argv[9] : "absmax");
    const uint64_t elements = rows * columns;
    if (elements % QK_K != 0) {
        throw std::runtime_error("matrix element count must be divisible by 256");
    }

    std::ifstream bf16_input(argv[1], std::ios::binary);
    std::ifstream q4k_input(argv[3], std::ios::binary);
    if (!bf16_input || !q4k_input) {
        throw std::runtime_error("failed to open an input file");
    }
    seek(bf16_input, bf16_offset);
    seek(q4k_input, q4k_offset);

    Metrics q4k_metrics;
    Metrics q4rdna_metrics;
    std::array<uint16_t, QK_K> bf16_block{};
    std::array<float, QK_K> reference{};
    std::array<uint8_t, Q4_K_BLOCK_BYTES> q4k_block{};
    std::array<float, QK_K> q4k_values{};
    std::array<float, Q4_RDNA_GROUP> q4rdna_values{};
    std::array<uint8_t, Q4_RDNA_GROUP_BYTES> q4rdna_group{};
    const uint64_t groups_per_row = columns / Q4_RDNA_GROUP;
    const uint64_t q4rdna_bytes = elements / Q4_RDNA_GROUP * Q4_RDNA_GROUP_BYTES;
    if (columns % Q4_RDNA_GROUP != 0 || rows % Q4_RDNA_TILE_ROWS != 0) {
        throw std::runtime_error("Q4_RDNA layout requires columns divisible by 64 and rows divisible by 32");
    }
    std::vector<uint8_t> q4rdna_packed(q4rdna_bytes);

    for (uint64_t block = 0; block < elements / QK_K; ++block) {
        read(bf16_input, bf16_block);
        read(q4k_input, q4k_block);
        for (uint64_t i = 0; i < QK_K; ++i) {
            reference[i] = bf16_to_float(bf16_block[i]);
        }
        dequantize_q4_k(q4k_block, q4k_values);
        for (uint64_t i = 0; i < QK_K; ++i) {
            q4k_metrics.add(reference[i], q4k_values[i]);
        }
        for (uint64_t group = 0; group < QK_K / Q4_RDNA_GROUP; ++group) {
            const float * source = reference.data() + group * Q4_RDNA_GROUP;
            quantize_q4_rdna(source, algorithm, q4rdna_group, q4rdna_values.data());
            const uint64_t global_group = block * (QK_K / Q4_RDNA_GROUP) + group;
            const uint64_t row = global_group / groups_per_row;
            const uint64_t column_group = global_group % groups_per_row;
            const uint64_t tile = (row / Q4_RDNA_TILE_ROWS) * groups_per_row + column_group;
            const uint64_t lane = row % Q4_RDNA_TILE_ROWS;
            const uint64_t tile_offset = tile * Q4_RDNA_TILE_BYTES;
            std::memcpy(q4rdna_packed.data() + tile_offset + lane * 2, q4rdna_group.data(), 2);
            for (uint64_t i = 0; i < Q4_RDNA_GROUP / 2; ++i) {
                q4rdna_packed[tile_offset + 64 + i * Q4_RDNA_TILE_ROWS + lane] = q4rdna_group[2 + i];
            }
            for (uint64_t i = 0; i < Q4_RDNA_GROUP; ++i) {
                q4rdna_metrics.add(source[i], q4rdna_values[i]);
            }
        }
    }

    std::ofstream q4rdna_output(argv[7], std::ios::binary | std::ios::trunc);
    q4rdna_output.write(reinterpret_cast<const char *>(q4rdna_packed.data()), static_cast<std::streamsize>(q4rdna_packed.size()));
    if (!q4rdna_output) {
        throw std::runtime_error("failed while writing Q4_RDNA output");
    }

    const uint64_t bf16_bytes = elements * 2;
    const uint64_t q4k_bytes = elements / QK_K * Q4_K_BLOCK_BYTES;
    const double ratio = q4rdna_metrics.mse() / q4k_metrics.mse();

    std::cout << std::scientific << std::setprecision(9);
    std::cout << "matrix: " << rows << " x " << columns << " (" << elements << " weights)\n";
    std::cout << "BF16 bytes:    " << bf16_bytes << '\n';
    std::cout << "Q4_K bytes:    " << q4k_bytes << " (" << 8.0 * q4k_bytes / elements << " bits/weight)\n";
    std::cout << "Q4_RDNA bytes: " << q4rdna_bytes << " (" << 8.0 * q4rdna_bytes / elements << " bits/weight)\n";
    std::cout << "scale algorithm: " << algorithm_name(algorithm) << '\n';
    std::cout << "Q4_RDNA is " << 100.0 * (1.0 - double(q4rdna_bytes) / q4k_bytes) << "% smaller than Q4_K\n\n";
    print_metrics("Q4_K vs BF16", q4k_metrics, 1.0);
    std::cout << '\n';
    print_metrics("Q4_RDNA vs BF16", q4rdna_metrics, ratio);

    std::ofstream json(argv[8], std::ios::trunc);
    if (!json) {
        throw std::runtime_error("failed to open JSON output");
    }
    json << std::scientific << std::setprecision(17);
    json << "{\n";
    json << "  \"matrix\": \"model.layers.0.mlp.gate_proj.weight\",\n";
    json << "  \"shape\": [" << rows << ", " << columns << "],\n";
    json << "  \"q4_rdna_group_size\": " << Q4_RDNA_GROUP << ",\n";
    json << "  \"q4_rdna_tile_rows\": " << Q4_RDNA_TILE_ROWS << ",\n";
    json << "  \"q4_rdna_tile_bytes\": " << Q4_RDNA_TILE_BYTES << ",\n";
    json << "  \"q4_rdna_layout\": \"64-byte lane scales, then 32 x 32-byte lane-interleaved quant bytes\",\n";
    json << "  \"scale_algorithm\": \"" << algorithm_name(algorithm) << "\",\n";
    json << "  \"q4_rdna_bits_per_weight\": " << 8.0 * q4rdna_bytes / elements << ",\n";
    json << "  \"q4_rdna_smaller_than_q4_k_fraction\": " << 1.0 - double(q4rdna_bytes) / q4k_bytes << ",\n";
    json << "  \"q4_k\": {\"mse\": " << q4k_metrics.mse() << ", \"relative_l2\": " << q4k_metrics.relative_l2() << ", \"cosine\": " << q4k_metrics.cosine() << ", \"max_abs_error\": " << q4k_metrics.max_error << "},\n";
    json << "  \"q4_rdna\": {\"mse\": " << q4rdna_metrics.mse() << ", \"mse_over_q4_k\": " << ratio << ", \"relative_l2\": " << q4rdna_metrics.relative_l2() << ", \"cosine\": " << q4rdna_metrics.cosine() << ", \"max_abs_error\": " << q4rdna_metrics.max_error << "}\n";
    json << "}\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}
