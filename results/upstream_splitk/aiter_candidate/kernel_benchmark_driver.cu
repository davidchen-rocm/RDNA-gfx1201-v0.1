#include "csrc/kernels/q4_group64_gemv.cu"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define HIP_OK(call)                                                                      \
    do                                                                                    \
    {                                                                                     \
        const hipError_t error = (call);                                                   \
        if(error != hipSuccess)                                                            \
        {                                                                                 \
            throw std::runtime_error(std::string(#call) + ": " + hipGetErrorString(error)); \
        }                                                                                 \
    } while(false)

constexpr int kWarmup        = 100;
constexpr int kCalibration   = 100;
constexpr int kSamples       = 30;
constexpr int kWeightCopies  = 72;
constexpr double kTargetUs   = 100000.0;
constexpr int kMinIterations = 10;
constexpr int kMaxIterations = 2000000;

struct Shape
{
    int rows;
    int columns;
    Mapping selected;
    const char* selected_name;
};

constexpr std::array<Shape, 14> kShapes = {{{512, 3584, Mapping::Small32x32, "small32x32"},
                                             {1024, 3072, Mapping::Small32x32, "small32x32"},
                                             {1024, 4096, Mapping::Small32x32, "small32x32"},
                                             {3072, 3072, Mapping::Split8, "split8"},
                                             {3072, 8192, Mapping::Split8, "split8"},
                                             {3584, 3584, Mapping::Split8, "split8"},
                                             {3584, 18944, Mapping::Split8, "split8"},
                                             {4096, 4096, Mapping::Split8, "split8"},
                                             {4096, 12288, Mapping::Split8, "split8"},
                                             {4096, 14336, Mapping::Split8, "split8"},
                                             {8192, 3072, Mapping::Split4, "split4"},
                                             {12288, 4096, Mapping::Split8, "split8"},
                                             {14336, 4096, Mapping::Split8, "split8"},
                                             {18944, 3584, Mapping::Split8, "split8"}}};

struct DeviceBuffer
{
    void* pointer = nullptr;

    explicit DeviceBuffer(size_t bytes) { HIP_OK(hipMalloc(&pointer, bytes)); }

    ~DeviceBuffer()
    {
        if(pointer != nullptr)
        {
            (void)hipFree(pointer);
        }
    }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Events
{
    hipEvent_t start{};
    hipEvent_t end{};

    Events()
    {
        HIP_OK(hipEventCreate(&start));
        HIP_OK(hipEventCreate(&end));
    }

    ~Events()
    {
        (void)hipEventDestroy(end);
        (void)hipEventDestroy(start);
    }
};

void launch_direct(Mapping mapping,
                   const uint8_t* weights,
                   const float* x,
                   float* out,
                   int rows,
                   int columns,
                   hipStream_t stream)
{
    switch(mapping)
    {
    case Mapping::Old: {
        const int row_tiles = rows / kTileRows;
        const dim3 grid((row_tiles + kWaves - 1) / kWaves);
        const dim3 block(kThreads);
        if(row_tiles % kWaves == 0)
        {
            hipLaunchKernelGGL((q4_group64_gemv_old_kernel<false>),
                               grid,
                               block,
                               0,
                               stream,
                               weights,
                               x,
                               out,
                               row_tiles,
                               columns);
        }
        else
        {
            hipLaunchKernelGGL((q4_group64_gemv_old_kernel<true>),
                               grid,
                               block,
                               0,
                               stream,
                               weights,
                               x,
                               out,
                               row_tiles,
                               columns);
        }
        break;
    }
    case Mapping::Split2: launch_split<2>(weights, x, out, rows, columns, stream); break;
    case Mapping::Split4: launch_split<4>(weights, x, out, rows, columns, stream); break;
    case Mapping::Split8: launch_split<8>(weights, x, out, rows, columns, stream); break;
    case Mapping::Small8x8: launch_small<8, 8>(weights, x, out, rows, columns, stream); break;
    case Mapping::Small8x16: launch_small<8, 16>(weights, x, out, rows, columns, stream); break;
    case Mapping::Small8x32: launch_small<8, 32>(weights, x, out, rows, columns, stream); break;
    case Mapping::Small16x16: launch_small<16, 16>(weights, x, out, rows, columns, stream); break;
    case Mapping::Small16x32: launch_small<16, 32>(weights, x, out, rows, columns, stream); break;
    case Mapping::Small32x32: launch_small<32, 32>(weights, x, out, rows, columns, stream); break;
    case Mapping::Auto: throw std::runtime_error("auto must be resolved before launch");
    }
}

double elapsed_us(Events& events)
{
    HIP_OK(hipEventSynchronize(events.end));
    float milliseconds = 0.0f;
    HIP_OK(hipEventElapsedTime(&milliseconds, events.start, events.end));
    return static_cast<double>(milliseconds) * 1000.0;
}

double percentile(std::vector<double> values, double fraction)
{
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const size_t lower    = static_cast<size_t>(position);
    const size_t upper    = std::min(lower + 1, values.size() - 1);
    const double weight   = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

struct Measurement
{
    std::string label;
    Mapping mapping;
    int iterations{};
    double calibration_us{};
    std::vector<double> samples_us;

    double median_us() const { return percentile(samples_us, 0.5); }
    double p10_us() const { return percentile(samples_us, 0.1); }
    double p90_us() const { return percentile(samples_us, 0.9); }
};

void launch_batch(const Shape& shape,
                  Mapping mapping,
                  const uint8_t* weights,
                  size_t weight_bytes,
                  const float* x,
                  float* out,
                  int iterations,
                  int& rotation,
                  hipStream_t stream)
{
    for(int iteration = 0; iteration < iterations; ++iteration)
    {
        const auto* current_weights = weights + static_cast<size_t>(rotation) * weight_bytes;
        launch_direct(mapping, current_weights, x, out, shape.rows, shape.columns, stream);
        rotation = (rotation + 1) % kWeightCopies;
    }
}

int calibrate(const Shape& shape,
              Mapping mapping,
              const uint8_t* weights,
              size_t weight_bytes,
              const float* x,
              float* out,
              int& rotation,
              hipStream_t stream,
              Events& events,
              double& calibration_us)
{
    HIP_OK(hipEventRecord(events.start, stream));
    launch_batch(shape,
                 mapping,
                 weights,
                 weight_bytes,
                 x,
                 out,
                 kCalibration,
                 rotation,
                 stream);
    HIP_OK(hipEventRecord(events.end, stream));
    calibration_us = elapsed_us(events) / static_cast<double>(kCalibration);
    const int iterations = static_cast<int>(std::ceil(kTargetUs / calibration_us));
    return std::clamp(iterations, kMinIterations, kMaxIterations);
}

std::vector<Measurement> benchmark_shape(const Shape& shape)
{
    const size_t weight_bytes = static_cast<size_t>(shape.rows / kTileRows) *
                                static_cast<size_t>(shape.columns / kGroup) * kTileBytes;
    DeviceBuffer weights(weight_bytes * kWeightCopies);
    DeviceBuffer x(static_cast<size_t>(shape.columns) * sizeof(float));
    DeviceBuffer out(static_cast<size_t>(shape.rows) * sizeof(float));
    HIP_OK(hipMemset(weights.pointer, 0x55, weight_bytes * kWeightCopies));
    HIP_OK(hipMemset(x.pointer, 0, static_cast<size_t>(shape.columns) * sizeof(float)));
    HIP_OK(hipMemset(out.pointer, 0, static_cast<size_t>(shape.rows) * sizeof(float)));

    const auto* weight_pointer = static_cast<const uint8_t*>(weights.pointer);
    const auto* x_pointer      = static_cast<const float*>(x.pointer);
    auto* out_pointer          = static_cast<float*>(out.pointer);
    hipStream_t stream         = nullptr;
    Events events;
    int rotation = 0;

    std::vector<Measurement> measurements = {{"old", Mapping::Old},
                                              {"auto", shape.selected},
                                              {"selected", shape.selected}};
    for(auto& measurement : measurements)
    {
        launch_batch(shape,
                     measurement.mapping,
                     weight_pointer,
                     weight_bytes,
                     x_pointer,
                     out_pointer,
                     kWarmup,
                     rotation,
                     stream);
    }
    HIP_OK(hipDeviceSynchronize());
    for(auto& measurement : measurements)
    {
        measurement.iterations = calibrate(shape,
                                           measurement.mapping,
                                           weight_pointer,
                                           weight_bytes,
                                           x_pointer,
                                           out_pointer,
                                           rotation,
                                           stream,
                                           events,
                                           measurement.calibration_us);
    }

    for(int sample = 0; sample < kSamples; ++sample)
    {
        for(size_t offset = 0; offset < measurements.size(); ++offset)
        {
            auto& measurement = measurements[(static_cast<size_t>(sample) + offset) %
                                             measurements.size()];
            HIP_OK(hipEventRecord(events.start, stream));
            launch_batch(shape,
                         measurement.mapping,
                         weight_pointer,
                         weight_bytes,
                         x_pointer,
                         out_pointer,
                         measurement.iterations,
                         rotation,
                         stream);
            HIP_OK(hipEventRecord(events.end, stream));
            measurement.samples_us.push_back(
                elapsed_us(events) / static_cast<double>(measurement.iterations));
        }
    }
    HIP_OK(hipGetLastError());
    return measurements;
}

void write_json_string(std::ostream& output, const std::string& value)
{
    output << '"';
    for(const char character : value)
    {
        if(character == '"' || character == '\\')
        {
            output << '\\';
        }
        output << character;
    }
    output << '"';
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if(argc > 2)
        {
            throw std::runtime_error("usage: direct_kernel_benchmark [output-directory]");
        }
        hipDeviceProp_t properties{};
        HIP_OK(hipGetDeviceProperties(&properties, 0));
        if(std::string(properties.gcnArchName).rfind("gfx1201", 0) != 0)
        {
            throw std::runtime_error("benchmark requires gfx1201");
        }

        const std::filesystem::path result_dir = argc == 2 ? argv[1] : ".";
        std::filesystem::create_directories(result_dir);
        std::ofstream csv(result_dir / "kernel_benchmark_summary.csv");
        std::ofstream json(result_dir / "kernel_benchmark_raw.json");
        if(!csv || !json)
        {
            throw std::runtime_error("failed to open benchmark result files");
        }
        csv << "n,k,selected,mapping,median_us,p10_us,p90_us,calibration_us,batch_iterations,"
               "weight_copies,auto_over_old_speedup\n";
        json << std::setprecision(12);
        csv << std::setprecision(12);
        json << "{\n  \"schema\": 1,\n  \"device\": ";
        write_json_string(json, properties.name);
        json << ",\n  \"arch\": ";
        write_json_string(json, properties.gcnArchName);
        json << ",\n  \"timing_boundary\": "
                "\"HIP events around a calibrated batch of direct kernel launches on stream 0; "
                "elapsed device timeline divided by launches; excludes Python, pybind, tensor "
                "validation, device/arch guard, and public auto lookup\",\n"
                "  \"warmup_per_mapping\": "
             << kWarmup << ",\n  \"calibration_launches\": " << kCalibration
             << ",\n  \"target_sample_us\": " << kTargetUs << ",\n  \"samples\": " << kSamples
             << ",\n  \"weight_copies\": " << kWeightCopies << ",\n  \"shapes\": [\n";

        bool first_shape = true;
        for(const Shape& shape : kShapes)
        {
            const auto measurements = benchmark_shape(shape);
            const double old_median  = measurements[0].median_us();
            const double auto_median = measurements[1].median_us();
            const double speedup     = old_median / auto_median;
            std::cout << shape.rows << 'x' << shape.columns << " selected=" << shape.selected_name
                      << " old=" << old_median << "us auto=" << auto_median
                      << "us speedup=" << speedup << "x\n";

            if(!first_shape)
            {
                json << ",\n";
            }
            first_shape = false;
            json << "    {\"n\": " << shape.rows << ", \"k\": " << shape.columns
                 << ", \"selected\": ";
            write_json_string(json, shape.selected_name);
            json << ", \"auto_over_old_speedup\": " << speedup << ", \"measurements\": [";

            bool first_measurement = true;
            for(const auto& measurement : measurements)
            {
                if(!first_measurement)
                {
                    json << ',';
                }
                first_measurement = false;
                json << "{\"mapping\": ";
                write_json_string(json, measurement.label);
                json << ", \"median_us\": " << measurement.median_us()
                     << ", \"p10_us\": " << measurement.p10_us()
                     << ", \"p90_us\": " << measurement.p90_us()
                     << ", \"calibration_us\": " << measurement.calibration_us
                     << ", \"batch_iterations\": " << measurement.iterations
                     << ", \"raw_us\": [";
                for(size_t index = 0; index < measurement.samples_us.size(); ++index)
                {
                    if(index != 0)
                    {
                        json << ',';
                    }
                    json << measurement.samples_us[index];
                }
                json << "]}";
                csv << shape.rows << ',' << shape.columns << ',' << shape.selected_name << ','
                    << measurement.label << ',' << measurement.median_us() << ','
                    << measurement.p10_us() << ',' << measurement.p90_us() << ','
                    << measurement.calibration_us << ',' << measurement.iterations << ','
                    << kWeightCopies << ',' << speedup << '\n';
            }
            json << "]}";
            json.flush();
            csv.flush();
        }
        json << "\n  ]\n}\n";
        HIP_OK(hipDeviceSynchronize());
        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
