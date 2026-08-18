#include <algorithm>
#include <array>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr uint64_t GROUP = 64;
constexpr uint64_t TILE_ROWS = 32;
constexpr uint64_t TILE_BYTES = 1088;

#pragma pack(push, 1)
struct Header {
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

struct FileEntry {
    char name[64];
    uint32_t rows;
    uint32_t columns;
    uint64_t offset;
    uint64_t size;
    uint64_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(Header) == 48);
static_assert(sizeof(FileEntry) == 96);

struct Tensor {
    std::string path;
    uint64_t source_offset;
    uint32_t rows;
    uint32_t columns;
    std::string name;
    uint64_t output_offset = 0;
    uint64_t output_size = 0;
};

uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

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
        if (exponent < -10) return sign;
        const uint32_t normalized = mantissa | 0x00800000;
        const int shift = 14 - exponent;
        uint32_t rounded = normalized >> shift;
        const uint32_t remainder = normalized & ((uint32_t(1) << shift) - 1);
        const uint32_t halfway = uint32_t(1) << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (rounded & 1))) ++rounded;
        return sign | uint16_t(rounded);
    }
    if (exponent >= 31) return sign | 0x7c00;
    uint32_t rounded = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fff;
    if (remainder > 0x1000 || (remainder == 0x1000 && (rounded & 1))) {
        ++rounded;
        if (rounded == 0x400) {
            rounded = 0;
            if (exponent + 1 >= 31) return sign | 0x7c00;
            return sign | uint16_t((exponent + 1) << 10);
        }
    }
    return sign | uint16_t(exponent << 10) | uint16_t(rounded);
}

int quantize_value(float value, float scale) {
    if (scale == 0.0f) return 0;
    return std::clamp(int(std::nearbyint(value / scale)), -8, 7);
}

float scale_error(const float * input, float scale) {
    float error = 0.0f;
    for (uint64_t i = 0; i < GROUP; ++i) {
        const float delta = scale * quantize_value(input[i], scale) - input[i];
        error += delta * delta;
    }
    return error;
}

uint16_t select_scale(const float * input) {
    float positive_max = 0.0f;
    float negative_max = 0.0f;
    for (uint64_t i = 0; i < GROUP; ++i) {
        positive_max = std::max(positive_max, input[i]);
        negative_max = std::max(negative_max, -input[i]);
    }
    if (positive_max == 0.0f && negative_max == 0.0f) return 0;
    const float positive = std::max(positive_max / 7.0f, negative_max / 8.0f);
    const float negative = -std::max(positive_max / 8.0f, negative_max / 7.0f);
    constexpr std::array<float, 5> factors = {1.0f, 0.9f, 0.8f, 0.7f, 0.6f};
    uint16_t best_bits = 0;
    float best_error = std::numeric_limits<float>::infinity();

    for (float factor : factors) {
        for (float initial : {positive * factor, negative * factor}) {
            uint16_t scale_bits = float_to_fp16(initial);
            for (int iteration = 0; iteration < 12; ++iteration) {
                const float scale = fp16_to_float(scale_bits);
                const float error = scale_error(input, scale);
                if (error < best_error) {
                    best_error = error;
                    best_bits = scale_bits;
                }
                float numerator = 0.0f;
                float denominator = 0.0f;
                for (uint64_t i = 0; i < GROUP; ++i) {
                    const int quant = quantize_value(input[i], scale);
                    numerator += input[i] * quant;
                    denominator += quant * quant;
                }
                if (denominator == 0.0f) break;
                const uint16_t next_bits = float_to_fp16(numerator / denominator);
                if (next_bits == scale_bits) break;
                scale_bits = next_bits;
            }
        }
    }
    return best_bits;
}

std::vector<Tensor> read_manifest(const char * path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open manifest");
    std::vector<Tensor> tensors;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        std::istringstream fields(line);
        Tensor tensor;
        if (!std::getline(fields, tensor.path, '\t') ||
                !(fields >> tensor.source_offset) || fields.get() != '\t' ||
                !(fields >> tensor.rows) || fields.get() != '\t' ||
                !(fields >> tensor.columns) || fields.get() != '\t' ||
                !std::getline(fields, tensor.name)) {
            throw std::runtime_error("invalid manifest line");
        }
        if (tensor.rows % TILE_ROWS || tensor.columns % GROUP || tensor.name.size() >= 64) {
            throw std::runtime_error("unsupported tensor in manifest");
        }
        tensors.push_back(std::move(tensor));
    }
    return tensors;
}

void write_zeros(std::ofstream & output, uint64_t count) {
    static const std::array<char, 4096> zeros{};
    while (count != 0) {
        const size_t chunk = std::min<uint64_t>(count, zeros.size());
        output.write(zeros.data(), chunk);
        count -= chunk;
    }
}

} // namespace

int main(int argc, char ** argv) try {
    if (argc != 3) {
        std::cerr << "usage: q4rdna_sidecar_pack MANIFEST OUTPUT\n";
        return 2;
    }
    std::fesetround(FE_TONEAREST);
    std::vector<Tensor> tensors = read_manifest(argv[1]);
    const uint64_t data_offset = align_up(sizeof(Header) + tensors.size() * sizeof(FileEntry), 4096);
    uint64_t offset = data_offset;
    for (Tensor & tensor : tensors) {
        tensor.output_offset = offset;
        tensor.output_size = uint64_t(tensor.rows) * tensor.columns * 34 / GROUP;
        offset = align_up(offset + tensor.output_size, 256);
    }

    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open output");
    Header header{{'Q','4','R','D','N','A','1','\0'}, 1, uint32_t(tensors.size()), uint32_t(GROUP),
                  uint32_t(TILE_ROWS), uint32_t(TILE_BYTES), 0, data_offset, offset - data_offset};
    output.write(reinterpret_cast<const char *>(&header), sizeof(header));
    for (const Tensor & tensor : tensors) {
        FileEntry entry{};
        std::memcpy(entry.name, tensor.name.c_str(), tensor.name.size());
        entry.rows = tensor.rows;
        entry.columns = tensor.columns;
        entry.offset = tensor.output_offset;
        entry.size = tensor.output_size;
        output.write(reinterpret_cast<const char *>(&entry), sizeof(entry));
    }
    write_zeros(output, data_offset - uint64_t(output.tellp()));

    const auto started = std::chrono::steady_clock::now();
    for (size_t number = 0; number < tensors.size(); ++number) {
        const Tensor & tensor = tensors[number];
        const auto tensor_started = std::chrono::steady_clock::now();
        write_zeros(output, tensor.output_offset - uint64_t(output.tellp()));
        const uint64_t elements = uint64_t(tensor.rows) * tensor.columns;
        std::vector<uint16_t> bf16(elements);
        std::ifstream input(tensor.path, std::ios::binary);
        input.seekg(tensor.source_offset);
        input.read(reinterpret_cast<char *>(bf16.data()), elements * sizeof(uint16_t));
        if (!input) throw std::runtime_error("failed to read " + tensor.name);
        std::vector<uint8_t> packed(tensor.output_size);
        const uint64_t groups_per_row = tensor.columns / GROUP;
        const uint64_t group_count = elements / GROUP;

#pragma omp parallel for schedule(static)
        for (uint64_t group = 0; group < group_count; ++group) {
            std::array<float, GROUP> values;
            for (uint64_t i = 0; i < GROUP; ++i) values[i] = bf16_to_float(bf16[group * GROUP + i]);
            const uint16_t scale_bits = select_scale(values.data());
            const float scale = fp16_to_float(scale_bits);
            const uint64_t row = group / groups_per_row;
            const uint64_t column_group = group % groups_per_row;
            const uint64_t tile = (row / TILE_ROWS) * groups_per_row + column_group;
            const uint64_t lane = row % TILE_ROWS;
            const uint64_t tile_offset = tile * TILE_BYTES;
            std::memcpy(packed.data() + tile_offset + lane * 2, &scale_bits, sizeof(scale_bits));
            for (uint64_t i = 0; i < GROUP / 2; ++i) {
                const int low = quantize_value(values[2 * i], scale);
                const int high = quantize_value(values[2 * i + 1], scale);
                packed[tile_offset + 64 + i * TILE_ROWS + lane] = uint8_t(low & 15) | uint8_t((high & 15) << 4);
            }
        }
        output.write(reinterpret_cast<const char *>(packed.data()), packed.size());
        if (!output) throw std::runtime_error("failed to write " + tensor.name);
        const double tensor_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - tensor_started).count();
        const double total_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::cout << '[' << std::setw(3) << number + 1 << '/' << tensors.size() << "] "
                  << std::left << std::setw(31) << tensor.name << std::right << ' '
                  << std::setw(5) << tensor.rows << 'x' << std::setw(5) << tensor.columns << ' '
                  << std::fixed << std::setprecision(1) << tensor_seconds << "s total " << total_seconds << "s\n" << std::flush;
    }
    write_zeros(output, offset - uint64_t(output.tellp()));
    std::cout << "wrote " << argv[2] << " (" << offset << " bytes)\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}
