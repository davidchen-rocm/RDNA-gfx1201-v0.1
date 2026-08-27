#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "q4rdna_kernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int GUARD_FLOATS = 32;
constexpr float GUARD_VALUE = 1234567.0f;
constexpr float OUTPUT_SENTINEL = -7654321.0f;

enum class Mode {
    Plain,
    Add,
    GateUp,
};

enum class Mapping {
    Old,
    Split2,
    Split4,
    Split8,
    Small8x8,
    Small8x16,
    Small8x32,
    Small16x16,
    Small16x32,
    Small32x32,
    Automatic,
};

enum class Pattern {
    Random,
    Zero,
    ZeroGroup,
    Alternating,
    Extreme,
    Group,
};

struct Options {
    int rows = 12288;
    int columns = 4096;
    Mode mode = Mode::Plain;
    Mapping mapping = Mapping::Split8;
    Pattern pattern = Pattern::Random;
    uint32_t seed = 20260827;
    int warmup = 100;
    int samples = 30;
    double target_sample_ms = 100.0;
    int weight_copies = 8;
    std::string cache = "rotating";
    std::string output;
};

void check(hipError_t status) {
    if (status != hipSuccess) {
        throw std::runtime_error(hipGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        check(hipMalloc(&data_, count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            [[maybe_unused]] const hipError_t status = hipFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer & operator=(const DeviceBuffer &) = delete;

    T * get() const {
        return data_;
    }

    size_t count() const {
        return count_;
    }

private:
    T * data_ = nullptr;
    size_t count_ = 0;
};

uint32_t next_random(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

float random_float(uint32_t & state, float low, float high) {
    const float unit = float(next_random(state) >> 8) * (1.0f / 16777216.0f);
    return low + (high - low) * unit;
}

uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bits_float(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint16_t float_to_half(float value) {
    const uint32_t bits = float_bits(value);
    const uint16_t sign = uint16_t((bits >> 16) & 0x8000);
    const uint32_t mantissa = bits & 0x007fffff;
    int exponent = int((bits >> 23) & 0xff) - 127 + 15;

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
            ++exponent;
            if (exponent >= 31) return sign | 0x7c00;
        }
    }
    return sign | uint16_t(exponent << 10) | uint16_t(rounded);
}

float half_to_float(uint16_t value) {
    const uint32_t sign = uint32_t(value & 0x8000) << 16;
    uint32_t exponent = (value >> 10) & 31;
    uint32_t mantissa = value & 1023;
    if (exponent == 0) {
        if (mantissa == 0) return bits_float(sign);
        int shift = 0;
        while ((mantissa & 1024) == 0) {
            mantissa <<= 1;
            ++shift;
        }
        mantissa &= 1023;
        exponent = uint32_t(127 - 15 - shift);
    } else if (exponent == 31) {
        exponent = 255;
    } else {
        exponent += 127 - 15;
    }
    return bits_float(sign | (exponent << 23) | (mantissa << 13));
}

int signed_low(uint8_t packed) {
    return int(static_cast<int8_t>(packed << 4)) >> 4;
}

int signed_high(uint8_t packed) {
    return int(static_cast<int8_t>(packed)) >> 4;
}

uint8_t pack_pair(int low, int high) {
    return uint8_t(low & 15) | uint8_t((high & 15) << 4);
}

const char * mode_name(Mode mode) {
    switch (mode) {
        case Mode::Plain: return "plain";
        case Mode::Add: return "add";
        case Mode::GateUp: return "gate-up";
    }
    return "unknown";
}

const char * mapping_name(Mapping mapping) {
    switch (mapping) {
        case Mapping::Old: return "old";
        case Mapping::Split2: return "split2";
        case Mapping::Split4: return "split4";
        case Mapping::Split8: return "split8";
        case Mapping::Small8x8: return "small8x8";
        case Mapping::Small8x16: return "small8x16";
        case Mapping::Small8x32: return "small8x32";
        case Mapping::Small16x16: return "small16x16";
        case Mapping::Small16x32: return "small16x32";
        case Mapping::Small32x32: return "small32x32";
        case Mapping::Automatic: return "auto";
    }
    return "unknown";
}

const char * pattern_name(Pattern pattern) {
    switch (pattern) {
        case Pattern::Random: return "random";
        case Pattern::Zero: return "zero";
        case Pattern::ZeroGroup: return "zero-group";
        case Pattern::Alternating: return "alternating";
        case Pattern::Extreme: return "extreme";
        case Pattern::Group: return "group";
    }
    return "unknown";
}

Mode parse_mode(const std::string & value) {
    if (value == "plain") return Mode::Plain;
    if (value == "add") return Mode::Add;
    if (value == "gate-up") return Mode::GateUp;
    throw std::invalid_argument("invalid mode: " + value);
}

Mapping parse_mapping(const std::string & value) {
    if (value == "old") return Mapping::Old;
    if (value == "split2") return Mapping::Split2;
    if (value == "split4") return Mapping::Split4;
    if (value == "split8") return Mapping::Split8;
    if (value == "small8x8") return Mapping::Small8x8;
    if (value == "small8x16") return Mapping::Small8x16;
    if (value == "small8x32") return Mapping::Small8x32;
    if (value == "small16x16") return Mapping::Small16x16;
    if (value == "small16x32") return Mapping::Small16x32;
    if (value == "small32x32") return Mapping::Small32x32;
    if (value == "auto") return Mapping::Automatic;
    throw std::invalid_argument("invalid mapping: " + value);
}

Pattern parse_pattern(const std::string & value) {
    if (value == "random") return Pattern::Random;
    if (value == "zero") return Pattern::Zero;
    if (value == "zero-group") return Pattern::ZeroGroup;
    if (value == "alternating") return Pattern::Alternating;
    if (value == "extreme") return Pattern::Extreme;
    if (value == "group") return Pattern::Group;
    throw std::invalid_argument("invalid pattern: " + value);
}

Mapping resolve_mapping(Mapping mapping, int rows, int columns) {
    if (mapping != Mapping::Automatic) return mapping;
    if (rows == 1024 && columns == 4096) return Mapping::Small32x32;
    return Mapping::Split8;
}

int mapping_row_multiple(Mapping mapping) {
    switch (mapping) {
        case Mapping::Old: return q4rdna_kernel::TILE_ROWS * q4rdna_kernel::WAVES;
        case Mapping::Split2: return 128;
        case Mapping::Split4: return 64;
        case Mapping::Split8: return 32;
        case Mapping::Small8x8: return 8;
        case Mapping::Small8x16: return 16;
        case Mapping::Small8x32: return 32;
        case Mapping::Small16x16: return 16;
        case Mapping::Small16x32: return 32;
        case Mapping::Small32x32: return 32;
        case Mapping::Automatic: break;
    }
    return 0;
}

void validate_options(Options & options) {
    if (options.rows <= 0 || options.rows % q4rdna_kernel::TILE_ROWS != 0) {
        throw std::invalid_argument("rows must be a positive multiple of 32");
    }
    if (options.columns <= 0 || options.columns % q4rdna_kernel::GROUP != 0) {
        throw std::invalid_argument("columns must be a positive multiple of 64");
    }
    if (options.warmup < 0 || options.samples < 0 || options.target_sample_ms <= 0.0) {
        throw std::invalid_argument("warmup and samples must be non-negative; target sample time must be positive");
    }
    if (options.cache != "hot" && options.cache != "rotating") {
        throw std::invalid_argument("cache must be hot or rotating");
    }
    if (options.cache == "hot") options.weight_copies = 1;
    if (options.weight_copies <= 0) throw std::invalid_argument("weight copies must be positive");
    const Mapping resolved = resolve_mapping(options.mapping, options.rows, options.columns);
    if (options.rows % mapping_row_multiple(resolved) != 0) {
        throw std::invalid_argument(
            std::string("rows are incompatible with mapping ") + mapping_name(resolved));
    }
}

Options parse_options(int argc, char ** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::invalid_argument("missing value for " + key);
            return argv[i];
        };
        if (key == "--rows") options.rows = std::stoi(value());
        else if (key == "--columns") options.columns = std::stoi(value());
        else if (key == "--mode") options.mode = parse_mode(value());
        else if (key == "--mapping") options.mapping = parse_mapping(value());
        else if (key == "--pattern") options.pattern = parse_pattern(value());
        else if (key == "--seed") options.seed = uint32_t(std::stoul(value()));
        else if (key == "--warmup") options.warmup = std::stoi(value());
        else if (key == "--samples") options.samples = std::stoi(value());
        else if (key == "--target-sample-ms") options.target_sample_ms = std::stod(value());
        else if (key == "--weight-copies") options.weight_copies = std::stoi(value());
        else if (key == "--cache") options.cache = value();
        else if (key == "--output") options.output = value();
        else if (key == "--help") {
            std::cout
                << "q4rdna_splitk_bench [options]\n"
                << "  --rows N --columns K\n"
                << "  --mode plain|add|gate-up\n"
                << "  --mapping old|split2|split4|split8|small8x8|small8x16|small8x32|small16x16|small16x32|small32x32|auto\n"
                << "  --pattern random|zero|zero-group|alternating|extreme|group\n"
                << "  --warmup N --samples N --target-sample-ms MS\n"
                << "  --cache hot|rotating --weight-copies N --seed N --output FILE\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + key);
        }
    }
    validate_options(options);
    return options;
}

struct HostInputs {
    std::vector<uint8_t> weights;
    std::vector<uint8_t> gate_weights;
    std::vector<float> activation;
    std::vector<float> add;
};

int pattern_quant(Pattern pattern, uint32_t & state, int row, int group, int element, bool gate) {
    switch (pattern) {
        case Pattern::Zero: return 0;
        case Pattern::ZeroGroup: return int(next_random(state) % 16) - 8;
        case Pattern::Alternating: return ((row + group + element + int(gate)) & 1) ? 7 : -8;
        case Pattern::Extreme: return (element & 1) ? 7 : -8;
        case Pattern::Group: return ((group + int(gate) * 3) % 15) - 7;
        case Pattern::Random: return int(next_random(state) % 16) - 8;
    }
    return 0;
}

float pattern_scale(Pattern pattern, int row, int group, bool gate) {
    if (pattern == Pattern::Zero) return 0.0f;
    if (pattern == Pattern::ZeroGroup && group % 3 == 1) return 0.0f;
    const float magnitude = 0.00390625f * float(1 + ((row * 3 + group * 5 + int(gate) * 7) % 15));
    if (pattern == Pattern::Alternating && ((row + group + int(gate)) & 1)) return -magnitude;
    return magnitude;
}

std::vector<uint8_t> make_weights(const Options & options, bool gate) {
    const int groups = options.columns / q4rdna_kernel::GROUP;
    const int row_tiles = options.rows / q4rdna_kernel::TILE_ROWS;
    std::vector<uint8_t> result(uint64_t(row_tiles) * groups * q4rdna_kernel::TILE_BYTES);
    uint32_t state = options.seed ^ (gate ? 0x9e3779b9u : 0u);
    for (int row_tile = 0; row_tile < row_tiles; ++row_tile) {
        for (int group = 0; group < groups; ++group) {
            uint8_t * tile = result.data() + uint64_t(row_tile * groups + group) * q4rdna_kernel::TILE_BYTES;
            for (int lane = 0; lane < q4rdna_kernel::TILE_ROWS; ++lane) {
                const int row = row_tile * q4rdna_kernel::TILE_ROWS + lane;
                const uint16_t scale = float_to_half(pattern_scale(options.pattern, row, group, gate));
                std::memcpy(tile + lane * 2, &scale, sizeof(scale));
                for (int pair = 0; pair < q4rdna_kernel::GROUP / 2; ++pair) {
                    const int low_value = pattern_quant(options.pattern, state, row, group, pair * 2, gate);
                    const int high_value = pattern_quant(options.pattern, state, row, group, pair * 2 + 1, gate);
                    tile[64 + pair * q4rdna_kernel::TILE_ROWS + lane] = pack_pair(low_value, high_value);
                }
            }
        }
    }
    return result;
}

HostInputs make_inputs(const Options & options) {
    HostInputs inputs;
    inputs.weights = make_weights(options, false);
    if (options.mode == Mode::GateUp) inputs.gate_weights = make_weights(options, true);
    inputs.activation.resize(options.columns);
    inputs.add.resize(options.rows);
    uint32_t state = options.seed ^ 0xa511e9b3u;
    for (int column = 0; column < options.columns; ++column) {
        switch (options.pattern) {
            case Pattern::Zero: inputs.activation[column] = 0.0f; break;
            case Pattern::ZeroGroup: inputs.activation[column] = random_float(state, -0.5f, 0.5f); break;
            case Pattern::Alternating: inputs.activation[column] = (column & 1) ? -0.25f : 0.25f; break;
            case Pattern::Extreme: inputs.activation[column] = (column & 1) ? 1.0f : -1.0f; break;
            case Pattern::Group: inputs.activation[column] = 1.0f + 0.001f * float(column % 64); break;
            case Pattern::Random: inputs.activation[column] = random_float(state, -0.5f, 0.5f); break;
        }
    }
    for (int row = 0; row < options.rows; ++row) {
        inputs.add[row] = options.pattern == Pattern::Zero ? 0.0f : 0.0005f * float((row % 31) - 15);
    }
    return inputs;
}

float read_scale(const uint8_t * tile, int lane) {
    uint16_t bits;
    std::memcpy(&bits, tile + lane * 2, sizeof(bits));
    return half_to_float(bits);
}

std::vector<float> cpu_reference(const Options & options, const HostInputs & inputs) {
    const int groups = options.columns / q4rdna_kernel::GROUP;
    std::vector<float> output(options.rows);
    for (int row = 0; row < options.rows; ++row) {
        const int row_tile = row / q4rdna_kernel::TILE_ROWS;
        const int lane = row % q4rdna_kernel::TILE_ROWS;
        float sum = 0.0f;
        float gate_sum = 0.0f;
        for (int group = 0; group < groups; ++group) {
            const uint8_t * tile = inputs.weights.data() + uint64_t(row_tile * groups + group) * q4rdna_kernel::TILE_BYTES;
            const uint8_t * gate_tile = options.mode == Mode::GateUp
                ? inputs.gate_weights.data() + uint64_t(row_tile * groups + group) * q4rdna_kernel::TILE_BYTES
                : nullptr;
            const float scale = read_scale(tile, lane);
            const float gate_scale = gate_tile == nullptr ? 0.0f : read_scale(gate_tile, lane);
            for (int pair = 0; pair < q4rdna_kernel::GROUP / 2; ++pair) {
                const uint8_t packed = tile[64 + pair * q4rdna_kernel::TILE_ROWS + lane];
                const float activation_low = inputs.activation[group * q4rdna_kernel::GROUP + pair * 2];
                const float activation_high = inputs.activation[group * q4rdna_kernel::GROUP + pair * 2 + 1];
                sum = std::fma(scale * float(signed_low(packed)), activation_low, sum);
                sum = std::fma(scale * float(signed_high(packed)), activation_high, sum);
                if (gate_tile != nullptr) {
                    const uint8_t gate_packed = gate_tile[64 + pair * q4rdna_kernel::TILE_ROWS + lane];
                    gate_sum = std::fma(gate_scale * float(signed_low(gate_packed)), activation_low, gate_sum);
                    gate_sum = std::fma(gate_scale * float(signed_high(gate_packed)), activation_high, gate_sum);
                }
            }
        }
        if (options.mode == Mode::Add) sum += inputs.add[row];
        if (options.mode == Mode::GateUp) sum *= gate_sum / (1.0f + std::exp(-gate_sum));
        output[row] = sum;
    }
    return output;
}

template <bool use_gate, bool use_add>
void launch_typed(
        Mapping mapping, const uint8_t * weights, const uint8_t * gate_weights,
        const float * activation, const float * add, float * output,
        int rows, int columns, hipStream_t stream = nullptr) {
    mapping = resolve_mapping(mapping, rows, columns);
    switch (mapping) {
        case Mapping::Old:
            q4rdna_kernel::gemv_old<use_gate, use_add>
                <<<rows / 256, q4rdna_kernel::THREADS, 0, stream>>>(
                    weights, gate_weights, activation, add, output, columns);
            return;
        case Mapping::Split2:
            q4rdna_kernel::gemv_split_k<use_gate, use_add, 2>
                <<<rows / 128, q4rdna_kernel::THREADS, 0, stream>>>(
                    weights, gate_weights, activation, add, output, columns);
            return;
        case Mapping::Split4:
            q4rdna_kernel::gemv_split_k<use_gate, use_add, 4>
                <<<rows / 64, q4rdna_kernel::THREADS, 0, stream>>>(
                    weights, gate_weights, activation, add, output, columns);
            return;
        case Mapping::Split8:
            q4rdna_kernel::gemv_split_k<use_gate, use_add, 8>
                <<<rows / 32, q4rdna_kernel::THREADS, 0, stream>>>(
                    weights, gate_weights, activation, add, output, columns);
            return;
        case Mapping::Small8x8:
            q4rdna_kernel::gemv_small_split_k<use_gate, use_add, 8, 8>
                <<<rows / 8, 8 * q4rdna_kernel::TILE_ROWS, 0, stream>>>(
                    weights, gate_weights, activation, add, output, columns);
            return;
        case Mapping::Small8x16:
            q4rdna_kernel::gemv_small_split_k<use_gate, use_add, 8, 16>
                <<<rows / 16, 8 * q4rdna_kernel::TILE_ROWS, 0, stream>>>(
                    weights, gate_weights, activation, add, output, columns);
            return;
        case Mapping::Small8x32:
            q4rdna_kernel::gemv_small_split_k<use_gate, use_add, 8, 32>
                <<<rows / 32, 8 * q4rdna_kernel::TILE_ROWS, 0, stream>>>(
                    weights, gate_weights, activation, add, output, columns);
            return;
        case Mapping::Small16x16:
            q4rdna_kernel::gemv_small_split_k<use_gate, use_add, 16, 16>
                <<<rows / 16, 16 * q4rdna_kernel::TILE_ROWS, 0, stream>>>(
                    weights, gate_weights, activation, add, output, columns);
            return;
        case Mapping::Small16x32:
            q4rdna_kernel::gemv_small_split_k<use_gate, use_add, 16, 32>
                <<<rows / 32, 16 * q4rdna_kernel::TILE_ROWS, 0, stream>>>(
                    weights, gate_weights, activation, add, output, columns);
            return;
        case Mapping::Small32x32:
            q4rdna_kernel::gemv_small_split_k<use_gate, use_add, 32, 32>
                <<<rows / 32, 32 * q4rdna_kernel::TILE_ROWS, 0, stream>>>(
                    weights, gate_weights, activation, add, output, columns);
            return;
        case Mapping::Automatic: break;
    }
    throw std::logic_error("unresolved mapping");
}

void launch(
        const Options & options, const uint8_t * weights, const uint8_t * gate_weights,
        const float * activation, const float * add, float * output,
        hipStream_t stream = nullptr) {
    if (options.mode == Mode::GateUp) {
        launch_typed<true, false>(
            options.mapping, weights, gate_weights, activation, add, output,
            options.rows, options.columns, stream);
    } else if (options.mode == Mode::Add) {
        launch_typed<false, true>(
            options.mapping, weights, gate_weights, activation, add, output,
            options.rows, options.columns, stream);
    } else {
        launch_typed<false, false>(
            options.mapping, weights, gate_weights, activation, add, output,
            options.rows, options.columns, stream);
    }
    check(hipGetLastError());
}

struct Correctness {
    double relative_l2 = 0.0;
    double relative_l2_limit = 0.0;
    double max_absolute = 0.0;
    double max_reference_absolute = 0.0;
    double expected_at_max_error = 0.0;
    double actual_at_max_error = 0.0;
    int max_error_row = -1;
    bool finite = true;
    bool all_written = true;
    bool canary_ok = true;
    bool passed = false;
};

Correctness check_correctness(
        const Options & options, const std::vector<float> & expected,
        const std::vector<float> & guarded_output) {
    Correctness result;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (int i = 0; i < GUARD_FLOATS; ++i) {
        if (guarded_output[i] != GUARD_VALUE || guarded_output[GUARD_FLOATS + options.rows + i] != GUARD_VALUE) {
            result.canary_ok = false;
        }
    }
    for (int row = 0; row < options.rows; ++row) {
        const float actual = guarded_output[GUARD_FLOATS + row];
        if (!std::isfinite(actual)) result.finite = false;
        if (actual == OUTPUT_SENTINEL) result.all_written = false;
        const double error = double(actual) - expected[row];
        result.max_reference_absolute = std::max(
            result.max_reference_absolute, std::abs(double(expected[row])));
        if (std::abs(error) > result.max_absolute) {
            result.max_absolute = std::abs(error);
            result.expected_at_max_error = expected[row];
            result.actual_at_max_error = actual;
            result.max_error_row = row;
        }
        squared_error += error * error;
        squared_reference += double(expected[row]) * expected[row];
    }
    result.relative_l2 = squared_reference == 0.0
        ? std::sqrt(squared_error)
        : std::sqrt(squared_error / squared_reference);
    // Split-K changes the FP32 reduction order.  The fused SiLU path can amplify
    // that rounding slightly, so keep a separately recorded bound for it.
    result.relative_l2_limit = options.mode == Mode::GateUp ? 2e-5 : 1e-5;
    result.passed = result.finite && result.all_written && result.canary_ok
        && result.relative_l2 <= result.relative_l2_limit;
    return result;
}

float timed_batch(
        const Options & options, int iterations, const uint8_t * weights,
        const uint8_t * gate_weights, uint64_t weight_bytes, const float * activation,
        const float * add, float * output) {
    hipEvent_t begin;
    hipEvent_t end;
    check(hipEventCreate(&begin));
    check(hipEventCreate(&end));
    check(hipEventRecord(begin));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int copy = iteration % options.weight_copies;
        launch(
            options,
            weights + uint64_t(copy) * weight_bytes,
            gate_weights == nullptr ? nullptr : gate_weights + uint64_t(copy) * weight_bytes,
            activation, add, output);
    }
    check(hipEventRecord(end));
    check(hipEventSynchronize(end));
    float milliseconds = 0.0f;
    check(hipEventElapsedTime(&milliseconds, begin, end));
    check(hipEventDestroy(begin));
    check(hipEventDestroy(end));
    return milliseconds * 1000.0f / float(iterations);
}

double quantile(const std::vector<double> & sorted, double q) {
    if (sorted.empty()) return 0.0;
    const double position = q * double(sorted.size() - 1);
    const size_t lower = size_t(position);
    const size_t upper = std::min(lower + 1, sorted.size() - 1);
    const double fraction = position - double(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

struct Benchmark {
    int iterations_per_sample = 0;
    std::vector<double> samples_us;
    double mean_us = 0.0;
    double sample_stddev_us = 0.0;
    double median_us = 0.0;
    double p10_us = 0.0;
    double p90_us = 0.0;
    double effective_gbs = 0.0;
};

Benchmark benchmark(
        const Options & options, const uint8_t * weights, const uint8_t * gate_weights,
        uint64_t weight_bytes, const float * activation, const float * add, float * output) {
    Benchmark result;
    if (options.samples == 0) return result;

    for (int i = 0; i < options.warmup; ++i) {
        const int copy = i % options.weight_copies;
        launch(
            options,
            weights + uint64_t(copy) * weight_bytes,
            gate_weights == nullptr ? nullptr : gate_weights + uint64_t(copy) * weight_bytes,
            activation, add, output);
    }
    check(hipDeviceSynchronize());

    const float calibration_us = timed_batch(
        options, 100, weights, gate_weights, weight_bytes, activation, add, output);
    result.iterations_per_sample = std::clamp(
        int(std::ceil(options.target_sample_ms * 1000.0 / std::max(0.01f, calibration_us))),
        10, 2000000);
    result.samples_us.reserve(options.samples);
    for (int sample = 0; sample < options.samples; ++sample) {
        result.samples_us.push_back(timed_batch(
            options, result.iterations_per_sample, weights, gate_weights, weight_bytes,
            activation, add, output));
    }
    result.mean_us = std::accumulate(result.samples_us.begin(), result.samples_us.end(), 0.0)
        / double(result.samples_us.size());
    if (result.samples_us.size() > 1) {
        double squared = 0.0;
        for (double sample : result.samples_us) {
            const double delta = sample - result.mean_us;
            squared += delta * delta;
        }
        result.sample_stddev_us = std::sqrt(squared / double(result.samples_us.size() - 1));
    }
    std::vector<double> sorted = result.samples_us;
    std::sort(sorted.begin(), sorted.end());
    result.median_us = quantile(sorted, 0.5);
    result.p10_us = quantile(sorted, 0.1);
    result.p90_us = quantile(sorted, 0.9);
    const double bytes_per_launch = double(weight_bytes) * (options.mode == Mode::GateUp ? 2.0 : 1.0);
    result.effective_gbs = bytes_per_launch / (result.median_us * 1000.0);
    return result;
}

std::string json_escape(const std::string & value) {
    std::string escaped;
    for (char character : value) {
        if (character == '\\' || character == '"') escaped += '\\';
        escaped += character;
    }
    return escaped;
}

void write_json(
        std::ostream & output, const Options & options, const hipDeviceProp_t & properties,
        int runtime_version, int driver_version, uint64_t weight_bytes,
        const Correctness & correctness, const Benchmark & benchmark_result) {
    output << std::setprecision(12);
    output << "{\n";
    output << "  \"schema\": \"q4rdna-splitk-bench-v1\",\n";
    output << "  \"environment\": {\n";
    output << "    \"gpu\": \"" << json_escape(properties.name) << "\",\n";
    output << "    \"gfx\": \"" << json_escape(properties.gcnArchName) << "\",\n";
    output << "    \"hip_multiprocessor_count\": " << properties.multiProcessorCount << ",\n";
    output << "    \"warp_size\": " << properties.warpSize << ",\n";
    output << "    \"runtime_version\": " << runtime_version << ",\n";
    output << "    \"driver_version\": " << driver_version << "\n";
    output << "  },\n";
    output << "  \"case\": {\n";
    output << "    \"rows\": " << options.rows << ",\n";
    output << "    \"columns\": " << options.columns << ",\n";
    output << "    \"groups\": " << options.columns / q4rdna_kernel::GROUP << ",\n";
    output << "    \"mode\": \"" << mode_name(options.mode) << "\",\n";
    output << "    \"requested_mapping\": \"" << mapping_name(options.mapping) << "\",\n";
    output << "    \"resolved_mapping\": \""
           << mapping_name(resolve_mapping(options.mapping, options.rows, options.columns)) << "\",\n";
    output << "    \"pattern\": \"" << pattern_name(options.pattern) << "\",\n";
    output << "    \"seed\": " << options.seed << ",\n";
    output << "    \"packed_weight_bytes\": " << weight_bytes << ",\n";
    output << "    \"cache_mode\": \"" << options.cache << "\",\n";
    output << "    \"weight_copies\": " << options.weight_copies << "\n";
    output << "  },\n";
    output << "  \"correctness\": {\n";
    output << "    \"relative_l2\": " << correctness.relative_l2 << ",\n";
    output << "    \"relative_l2_limit\": " << correctness.relative_l2_limit << ",\n";
    output << "    \"max_absolute\": " << correctness.max_absolute << ",\n";
    output << "    \"max_reference_absolute\": " << correctness.max_reference_absolute << ",\n";
    output << "    \"expected_at_max_error\": " << correctness.expected_at_max_error << ",\n";
    output << "    \"actual_at_max_error\": " << correctness.actual_at_max_error << ",\n";
    output << "    \"max_error_row\": " << correctness.max_error_row << ",\n";
    output << "    \"finite\": " << (correctness.finite ? "true" : "false") << ",\n";
    output << "    \"all_written\": " << (correctness.all_written ? "true" : "false") << ",\n";
    output << "    \"canary_ok\": " << (correctness.canary_ok ? "true" : "false") << ",\n";
    output << "    \"passed\": " << (correctness.passed ? "true" : "false") << "\n";
    output << "  },\n";
    output << "  \"benchmark\": {\n";
    output << "    \"warmup\": " << options.warmup << ",\n";
    output << "    \"target_sample_ms\": " << options.target_sample_ms << ",\n";
    output << "    \"iterations_per_sample\": " << benchmark_result.iterations_per_sample << ",\n";
    output << "    \"samples_us\": [";
    for (size_t i = 0; i < benchmark_result.samples_us.size(); ++i) {
        if (i != 0) output << ", ";
        output << benchmark_result.samples_us[i];
    }
    output << "],\n";
    output << "    \"mean_us\": " << benchmark_result.mean_us << ",\n";
    output << "    \"sample_stddev_us\": " << benchmark_result.sample_stddev_us << ",\n";
    output << "    \"median_us\": " << benchmark_result.median_us << ",\n";
    output << "    \"p10_us\": " << benchmark_result.p10_us << ",\n";
    output << "    \"p90_us\": " << benchmark_result.p90_us << ",\n";
    output << "    \"effective_weight_gbs\": " << benchmark_result.effective_gbs << "\n";
    output << "  }\n";
    output << "}\n";
}

} // namespace

int main(int argc, char ** argv) try {
    Options options = parse_options(argc, argv);
    check(hipSetDevice(0));

    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0));
    if (properties.warpSize != q4rdna_kernel::TILE_ROWS) {
        throw std::runtime_error("Q4_RDNA benchmark requires wave32");
    }
    int runtime_version = 0;
    int driver_version = 0;
    check(hipRuntimeGetVersion(&runtime_version));
    check(hipDriverGetVersion(&driver_version));

    HostInputs inputs = make_inputs(options);
    const std::vector<float> expected = cpu_reference(options, inputs);
    const uint64_t weight_bytes = inputs.weights.size();

    DeviceBuffer<uint8_t> device_weights(weight_bytes * uint64_t(options.weight_copies));
    for (int copy = 0; copy < options.weight_copies; ++copy) {
        check(hipMemcpy(
            device_weights.get() + uint64_t(copy) * weight_bytes,
            inputs.weights.data(), weight_bytes, hipMemcpyHostToDevice));
    }

    DeviceBuffer<uint8_t> device_gate(
        options.mode == Mode::GateUp ? weight_bytes * uint64_t(options.weight_copies) : 1);
    if (options.mode == Mode::GateUp) {
        for (int copy = 0; copy < options.weight_copies; ++copy) {
            check(hipMemcpy(
                device_gate.get() + uint64_t(copy) * weight_bytes,
                inputs.gate_weights.data(), weight_bytes, hipMemcpyHostToDevice));
        }
    }

    DeviceBuffer<float> device_activation(inputs.activation.size());
    DeviceBuffer<float> device_add(inputs.add.size());
    DeviceBuffer<float> device_output(options.rows + 2 * GUARD_FLOATS);
    check(hipMemcpy(
        device_activation.get(), inputs.activation.data(),
        inputs.activation.size() * sizeof(float), hipMemcpyHostToDevice));
    check(hipMemcpy(
        device_add.get(), inputs.add.data(),
        inputs.add.size() * sizeof(float), hipMemcpyHostToDevice));

    std::vector<float> guarded_output(options.rows + 2 * GUARD_FLOATS, GUARD_VALUE);
    std::fill(
        guarded_output.begin() + GUARD_FLOATS,
        guarded_output.begin() + GUARD_FLOATS + options.rows,
        OUTPUT_SENTINEL);
    check(hipMemcpy(
        device_output.get(), guarded_output.data(),
        guarded_output.size() * sizeof(float), hipMemcpyHostToDevice));

    float * output = device_output.get() + GUARD_FLOATS;
    const uint8_t * gate = options.mode == Mode::GateUp ? device_gate.get() : nullptr;
    launch(
        options, device_weights.get(), gate, device_activation.get(),
        device_add.get(), output);
    check(hipDeviceSynchronize());
    check(hipMemcpy(
        guarded_output.data(), device_output.get(),
        guarded_output.size() * sizeof(float), hipMemcpyDeviceToHost));
    const Correctness correctness = check_correctness(options, expected, guarded_output);

    const Benchmark benchmark_result = benchmark(
        options, device_weights.get(), gate, weight_bytes,
        device_activation.get(), device_add.get(), output);

    if (options.output.empty()) {
        write_json(
            std::cout, options, properties, runtime_version, driver_version,
            weight_bytes, correctness, benchmark_result);
    } else {
        const std::filesystem::path path(options.output);
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
        std::ofstream file(path);
        if (!file) throw std::runtime_error("cannot open output: " + options.output);
        write_json(
            file, options, properties, runtime_version, driver_version,
            weight_bytes, correctness, benchmark_result);
        std::cout << options.rows << "x" << options.columns
                  << " mode=" << mode_name(options.mode)
                  << " mapping=" << mapping_name(resolve_mapping(options.mapping, options.rows, options.columns))
                  << " rel_l2=" << correctness.relative_l2
                  << " median_us=" << benchmark_result.median_us
                  << " output=" << options.output << '\n';
    }
    return correctness.passed ? 0 : 1;
} catch (const std::invalid_argument & error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
} catch (const std::exception & error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
}
