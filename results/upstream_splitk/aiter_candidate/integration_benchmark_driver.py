import csv
import ctypes
import datetime
import importlib
import importlib.util
import json
import os
import statistics
import sys
from array import array
from pathlib import Path

os.environ["AITER_AOT_IMPORT"] = "1"
sys.path.insert(0, "/tmp/aiter-test-tools")
sys.path.insert(0, "/home/david/Desktop/aiter-q4rdna")

import torch  # noqa: E402
import aiter  # noqa: E402,F401

CORE = importlib.import_module("aiter.jit.module_aiter_core")
SPEC = importlib.util.spec_from_file_location(
    "module_q4_group64_gemv",
    "/tmp/aiter-q4-build/module_q4_group64_gemv.benchmark.so",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

HIP = ctypes.CDLL("/opt/rocm/core-7.14/lib/libamdhip64.so.7")
HIP.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
HIP.hipMalloc.restype = ctypes.c_int
HIP.hipFree.argtypes = [ctypes.c_void_p]
HIP.hipFree.restype = ctypes.c_int
HIP.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
HIP.hipMemcpy.restype = ctypes.c_int
HIP.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
HIP.hipMemset.restype = ctypes.c_int
HIP.hipDeviceSynchronize.restype = ctypes.c_int
HIP.hipEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
HIP.hipEventCreate.restype = ctypes.c_int
HIP.hipEventDestroy.argtypes = [ctypes.c_void_p]
HIP.hipEventDestroy.restype = ctypes.c_int
HIP.hipEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
HIP.hipEventRecord.restype = ctypes.c_int
HIP.hipEventSynchronize.argtypes = [ctypes.c_void_p]
HIP.hipEventSynchronize.restype = ctypes.c_int
HIP.hipEventElapsedTime.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_void_p,
    ctypes.c_void_p,
]
HIP.hipEventElapsedTime.restype = ctypes.c_int
HIP.hipRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
HIP.hipRuntimeGetVersion.restype = ctypes.c_int
HIP.hipDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
HIP.hipDriverGetVersion.restype = ctypes.c_int

SHAPES = [
    (512, 3584, "small32x32"),
    (1024, 3072, "small32x32"),
    (1024, 4096, "small32x32"),
    (3072, 3072, "split8"),
    (3072, 8192, "split8"),
    (3584, 3584, "split8"),
    (3584, 18944, "split8"),
    (4096, 4096, "split8"),
    (4096, 12288, "split8"),
    (4096, 14336, "split8"),
    (8192, 3072, "split4"),
    (12288, 4096, "split8"),
    (14336, 4096, "split8"),
    (18944, 3584, "split8"),
]
MAPPING_IDS = {
    "auto": 0,
    "old": 1,
    "split4": 3,
    "split8": 4,
    "small32x32": 10,
}
WARMUP = 30
SAMPLES = 101
ROTATIONS = 16
RESULT_DIR = Path(
    "/home/david/Desktop/aiter-q4rdna/aiter_logs/q4_group64_gemv_gfx1201"
)


def check(status: int, operation: str) -> None:
    if status:
        raise RuntimeError(f"{operation} failed with HIP status {status}")


def allocate(size: int) -> ctypes.c_void_p:
    pointer = ctypes.c_void_p()
    check(HIP.hipMalloc(ctypes.byref(pointer), size), "hipMalloc")
    return pointer


def percentile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


class BufferSet:
    def __init__(self, n: int, k: int, seed: int):
        self.n = n
        self.k = k
        self.packed_bytes = (n // 32) * (k // 64) * 1088
        self.x_host = array(
            "f", [(float((i * 37 + seed * 11) % 101) - 50.0) / 200.0 for i in range(k)]
        )
        self.x_host_buffer = (
            ctypes.c_char * (len(self.x_host) * self.x_host.itemsize)
        ).from_buffer(self.x_host)
        self.allocations: list[tuple[ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]] = []
        self.tensors = []
        for rotation in range(ROTATIONS):
            x_pointer = allocate(k * 4)
            packed_pointer = allocate(self.packed_bytes)
            out_pointer = allocate(n * 4)
            check(
                HIP.hipMemcpy(x_pointer, self.x_host_buffer, k * 4, 1),
                "hipMemcpy x",
            )
            check(
                HIP.hipMemset(packed_pointer, 0x11 + rotation % 7, self.packed_bytes),
                "hipMemset packed",
            )
            self.allocations.append((x_pointer, packed_pointer, out_pointer))
            self.tensors.append(
                (
                    CORE.aiter_tensor_t(x_pointer.value, k, 1, [k], [1], 4, 0),
                    CORE.aiter_tensor_t(
                        packed_pointer.value,
                        self.packed_bytes,
                        3,
                        [n // 32, k // 64, 1088],
                        [(k // 64) * 1088, 1088, 1],
                        11,
                        0,
                    ),
                    CORE.aiter_tensor_t(out_pointer.value, n, 1, [n], [1], 4, 0),
                )
            )

    def close(self) -> None:
        check(HIP.hipDeviceSynchronize(), "hipDeviceSynchronize before free")
        for allocation in reversed(self.allocations):
            for pointer in reversed(allocation):
                check(HIP.hipFree(pointer), "hipFree")
        self.allocations.clear()
        self.tensors.clear()


def time_call(tensors, mapping_id: int, start, end) -> float:
    check(HIP.hipEventRecord(start, None), "hipEventRecord start")
    MODULE.q4_group64_gemv_out(*tensors, mapping_id)
    check(HIP.hipEventRecord(end, None), "hipEventRecord end")
    check(HIP.hipEventSynchronize(end), "hipEventSynchronize")
    milliseconds = ctypes.c_float()
    check(
        HIP.hipEventElapsedTime(ctypes.byref(milliseconds), start, end),
        "hipEventElapsedTime",
    )
    return float(milliseconds.value) * 1000.0


def summarize(samples: list[float], packed_bytes: int, n: int, k: int) -> dict:
    ordered = sorted(samples)
    median = statistics.median(ordered)
    traffic_bytes = packed_bytes + k * 4 + n * 4
    return {
        "count": len(samples),
        "raw_us": samples,
        "median_us": median,
        "p10_us": percentile(ordered, 0.10),
        "p90_us": percentile(ordered, 0.90),
        "mean_us": statistics.fmean(samples),
        "sample_stddev_us": statistics.stdev(samples),
        "effective_gbps": traffic_bytes / median / 1.0e3,
        "effective_tflops": 2.0 * n * k / median / 1.0e6,
    }


def benchmark_shape(n: int, k: int, selected: str, shape_index: int) -> dict:
    buffers = BufferSet(n, k, shape_index + 1)
    start = ctypes.c_void_p()
    end = ctypes.c_void_p()
    check(HIP.hipEventCreate(ctypes.byref(start)), "hipEventCreate start")
    check(HIP.hipEventCreate(ctypes.byref(end)), "hipEventCreate end")
    MODULE._set_current_hip_stream(0)
    requested = [("old", 1), ("auto", 0), ("selected", MAPPING_IDS[selected])]
    try:
        for iteration in range(WARMUP):
            for position, (_, mapping_id) in enumerate(requested):
                buffer_index = (iteration * len(requested) + position) % ROTATIONS
                MODULE.q4_group64_gemv_out(
                    *buffers.tensors[buffer_index], mapping_id
                )
        check(HIP.hipDeviceSynchronize(), "warmup synchronize")

        samples = {name: [] for name, _ in requested}
        for iteration in range(SAMPLES):
            order = requested[iteration % 3 :] + requested[: iteration % 3]
            for position, (name, mapping_id) in enumerate(order):
                buffer_index = (iteration * len(requested) + position) % ROTATIONS
                samples[name].append(
                    time_call(buffers.tensors[buffer_index], mapping_id, start, end)
                )

        summaries = {
            name: summarize(values, buffers.packed_bytes, n, k)
            for name, values in samples.items()
        }
        old_median = summaries["old"]["median_us"]
        auto_median = summaries["auto"]["median_us"]
        selected_median = summaries["selected"]["median_us"]
        result = {
            "n": n,
            "k": k,
            "selected_mapping": selected,
            "packed_bytes": buffers.packed_bytes,
            "mappings": summaries,
            "auto_speedup_vs_old": old_median / auto_median,
            "selected_speedup_vs_old": old_median / selected_median,
        }
        print(
            f"{n:7d}x{k:<7d} selected={selected:<11s} "
            f"old={old_median:8.3f}us auto={auto_median:8.3f}us "
            f"candidate={selected_median:8.3f}us speedup={old_median / auto_median:6.3f}x",
            flush=True,
        )
        return result
    finally:
        check(HIP.hipEventDestroy(end), "hipEventDestroy end")
        check(HIP.hipEventDestroy(start), "hipEventDestroy start")
        buffers.close()


def main() -> None:
    runtime_version = ctypes.c_int()
    driver_version = ctypes.c_int()
    check(HIP.hipRuntimeGetVersion(ctypes.byref(runtime_version)), "hipRuntimeGetVersion")
    check(HIP.hipDriverGetVersion(ctypes.byref(driver_version)), "hipDriverGetVersion")
    check(HIP.hipDeviceSynchronize(), "initial synchronize")
    started = datetime.datetime.now(datetime.timezone.utc)
    results = [
        benchmark_shape(n, k, selected, index)
        for index, (n, k, selected) in enumerate(SHAPES)
    ]
    finished = datetime.datetime.now(datetime.timezone.utc)
    document = {
        "schema": "aiter-q4-group64-pybind-benchmark-v1",
        "metadata": {
            "device": "AMD Radeon RX 9070 XT",
            "arch": "gfx1201",
            "aiter_commit": "48718fa7bb1b73d0800130144449fca3c625aba1",
            "branch": "perf/q4-group64-gemv",
            "torch_version": torch.__version__,
            "torch_hip_version": torch.version.hip,
            "hip_runtime_version": runtime_version.value,
            "hip_driver_version": driver_version.value,
            "timing": "HIP events around AITER pybind q4_group64_gemv_out",
            "frontend": "pybind aiter_tensor_t + direct HIP allocations; not ROCm PyTorch",
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
        },
        "configuration": {
            "processes": 1,
            "warmup_per_mapping": WARMUP,
            "samples_per_mapping": SAMPLES,
            "rotating_buffer_count": ROTATIONS,
            "mapping_order": "cyclically interleaved old/auto/selected",
            "packed_fill": "distinct physical allocations; deterministic finite payload bytes",
        },
        "results": results,
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULT_DIR / "benchmark_raw.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")

    csv_path = RESULT_DIR / "benchmark_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "n",
            "k",
            "requested",
            "resolved",
            "median_us",
            "p10_us",
            "p90_us",
            "mean_us",
            "sample_stddev_us",
            "effective_gbps",
            "effective_tflops",
            "speedup_vs_old",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            old_median = result["mappings"]["old"]["median_us"]
            for requested in ("old", "auto", "selected"):
                summary = result["mappings"][requested]
                writer.writerow(
                    {
                        "n": result["n"],
                        "k": result["k"],
                        "requested": requested,
                        "resolved": (
                            result["selected_mapping"]
                            if requested != "old"
                            else "old"
                        ),
                        "median_us": summary["median_us"],
                        "p10_us": summary["p10_us"],
                        "p90_us": summary["p90_us"],
                        "mean_us": summary["mean_us"],
                        "sample_stddev_us": summary["sample_stddev_us"],
                        "effective_gbps": summary["effective_gbps"],
                        "effective_tflops": summary["effective_tflops"],
                        "speedup_vs_old": old_median / summary["median_us"],
                    }
                )
    print(f"raw={json_path}", flush=True)
    print(f"summary={csv_path}", flush=True)


if __name__ == "__main__":
    main()
