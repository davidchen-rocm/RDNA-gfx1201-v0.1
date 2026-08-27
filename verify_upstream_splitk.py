#!/usr/bin/env python3
"""Independently verify the Q4_RDNA Split-K upstream evidence bundle.

This verifier intentionally uses only the Python standard library and does not
import any of the scripts that produced the evidence.  It follows references
back to raw artifacts, recomputes statistics and dispatch decisions, and exits
non-zero if any checked relationship is inconsistent.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import re
import statistics
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = REPOSITORY_ROOT / "results" / "upstream_splitk"

HARNESS_SCHEMA = "q4rdna-splitk-bench-v1"
SUITE_SCHEMA = "q4rdna-splitk-suite-v1"
DISPATCH_SCHEMA = "q4rdna-splitk-dispatch-v1"
END_TO_END_SCHEMA = "q4rdna-end-to-end-v1"
COMPLETION_SCHEMA = "q4rdna-completion-equivalence-v1"
GREEDY_COMPLETION_SCHEMA = "q4rdna-completion-equivalence-v2"
MODEL_ASSETS_SCHEMA = "q4rdna-model-assets-v1"
DYNAMIC_PROFILE_SCHEMA = "q4rdna-rocprof-summary-v1"
STATIC_PROFILE_SCHEMA = "q4rdna-static-profile-v1"

MODES = ("plain", "add", "gate-up")
EXPECTED_CORRECTNESS_CASES = 336
EXPECTED_INVALID_PROBES = 4
EXPECTED_MICROBENCH_MEASUREMENTS = 216
EXPECTED_BOOTSTRAP_RESAMPLES = 5000

Q4_ENV_PREFIX = "LLAMA_Q4_RDNA_"
LOAD_RE = re.compile(
    r"^Q4_RDNA: loaded (?P<tensors>\d+) tensors, (?P<gib>[0-9.]+) GiB "
    r"on device (?P<device>\d+) from (?P<path>.+)$",
    re.MULTILINE,
)
LAUNCH_RE = re.compile(
    r"^Q4_RDNA: launched (?P<count>\d+) decode GEMV kernels$", re.MULTILINE
)
UNIQUE_RE = re.compile(r"^Q4_RDNA: unique=(?P<count>\d+),", re.MULTILINE)
GREEDY_UNIQUE_RE = re.compile(
    r"^Q4_RDNA: unique=(?P<count>\d+)(?:, (?P<hits>.*))?$", re.MULTILINE
)
HIT_RE = re.compile(r"(?P<shape>\d+x\d+|other)=(?P<count>\d+)")
MAPPING_RE = re.compile(
    r"^Q4_RDNA: (?:selected )?mapping(?:=|:| )+(?P<mapping>[A-Za-z0-9_-]+)$",
    re.MULTILINE,
)
SAMPLER_CHAIN_RE = re.compile(
    r"^[^\r\n]*\bI sampler chain:\s*(?P<chain>.+?)\s*$", re.MULTILINE
)
SAMPLER_TEMP_RE = re.compile(r"(?:^|\s)temp = (?P<temperature>-?[0-9.]+)")
SAMPLER_SEED_RE = re.compile(r"\bI sampler seed:\s*(?P<seed>\d+)\s*$", re.MULTILINE)

TARGET_TENSOR_SUFFIXES = (
    ("self_attn.q_proj.weight", "attn_q.weight"),
    ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.v_proj.weight", "attn_v.weight"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("mlp.gate_proj.weight", "ffn_gate.weight"),
    ("mlp.up_proj.weight", "ffn_up.weight"),
    ("mlp.down_proj.weight", "ffn_down.weight"),
)

AITER_PYTEST_CASES_LEGACY = {
    "test_import_does_not_require_gfx1201",
    "test_arch_cache_is_keyed_by_device_and_does_not_cache_errors",
    "test_auto_dispatch_has_all_measured_plain_shapes",
    "test_python_and_cpp_auto_dispatch_tables_do_not_drift",
    "test_cpu_call_is_rejected_before_jit",
    "test_test_packer_rejects_non_tile_rows_and_non_group_columns",
    "test_packed_tile_round_trip_extremes_scales_and_byte_order",
    "test_all_explicit_mappings[old]",
    "test_all_explicit_mappings[split2]",
    "test_all_explicit_mappings[split4]",
    "test_all_explicit_mappings[split8]",
    "test_all_explicit_mappings[small8x8]",
    "test_all_explicit_mappings[small8x16]",
    "test_all_explicit_mappings[small8x32]",
    "test_all_explicit_mappings[small16x16]",
    "test_all_explicit_mappings[small16x32]",
    "test_all_explicit_mappings[small32x32]",
    "test_public_auto_known_shape_matches_selected_mapping",
    "test_unseen_auto_falls_back_to_boundary_safe_old",
    "test_extreme_int4_and_zero_scales",
    "test_non_default_stream_and_preallocated_private_output",
    "test_invalid_shapes_dtypes_layout_and_mapping",
    "test_invalid_explicit_split_shape_is_rejected",
}

AITER_PYTEST_CASES = (
    AITER_PYTEST_CASES_LEGACY
    - {"test_non_default_stream_and_preallocated_private_output"}
) | {
    "test_python_experimental_gate_rejects_unset_and_disabled",
    "test_cpp_auto_dispatch_has_exact_rx_9070_xt_guard",
    "test_benchmark_cli_rejects_when_no_requested_mapping_is_legal",
    "test_benchmark_auto_uses_public_call_and_controls_allocate_equally",
    "test_benchmark_pci_chip_id_uses_valid_aiter_helper_value",
    "test_benchmark_pci_chip_id_falls_back_from_rocm72_helper_value",
    "test_benchmark_pci_chip_id_fallback_fails_closed",
    "test_benchmark_requires_exact_rx_9070_xt_identity",
    "test_public_non_default_stream_and_preallocated_private_output",
    "test_runtime_gate_disables_loaded_public_and_direct_cpp_entries",
}


class SectionAbort(RuntimeError):
    """Stop one verification section after a missing structural prerequisite."""


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object, base: int = 20260827) -> int:
    payload = "\0".join(map(str, parts)).encode("utf-8")
    return (
        base ^ int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")
    ) & 0xFFFFFFFF


def linear_quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires a non-empty sample")
    ordered = sorted(float(value) for value in values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timing_statistics(samples: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in samples]
    finite = [value for value in values if math.isfinite(value)]
    if not finite or len(finite) != len(values):
        return {
            "count": len(values),
            "finite": len(finite) == len(values),
            "mean_us": None,
            "sample_stddev_us": None,
            "median_us": None,
            "p10_us": None,
            "p90_us": None,
        }
    return {
        "count": len(finite),
        "finite": True,
        "mean_us": statistics.fmean(finite),
        "sample_stddev_us": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "median_us": statistics.median(finite),
        "p10_us": linear_quantile(finite, 0.10),
        "p90_us": linear_quantile(finite, 0.90),
    }


def sample_summary(values: Sequence[float], expected_count: int) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "samples": [],
            "count": 0,
            "expected_count": expected_count,
            "complete": False,
            "mean": None,
            "sample_stddev": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "samples": samples,
        "count": len(samples),
        "expected_count": expected_count,
        "complete": len(samples) == expected_count,
        "mean": statistics.fmean(samples),
        "sample_stddev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "median": statistics.median(samples),
        "minimum": min(samples),
        "maximum": max(samples),
    }


def first_byte_difference(old: bytes, split: bytes) -> dict[str, Any] | None:
    common_length = min(len(old), len(split))
    for offset in range(common_length):
        if old[offset] != split[offset]:
            return {
                "offset": offset,
                "old_byte": old[offset],
                "split_byte": split[offset],
                "old_length": len(old),
                "split_length": len(split),
            }
    if len(old) != len(split):
        return {
            "offset": common_length,
            "old_byte": old[common_length] if common_length < len(old) else None,
            "split_byte": split[common_length] if common_length < len(split) else None,
            "old_length": len(old),
            "split_length": len(split),
        }
    return None


def parse_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as stream:
        encoded_size = stream.read(8)
        if len(encoded_size) != 8:
            raise ValueError(f"truncated safetensors header length: {path}")
        header_size = struct.unpack("<Q", encoded_size)[0]
        if header_size == 0 or header_size > 256 * 1024 * 1024:
            raise ValueError(
                f"implausible safetensors header length {header_size}: {path}"
            )
        encoded_header = stream.read(header_size)
        if len(encoded_header) != header_size:
            raise ValueError(f"truncated safetensors header: {path}")
    header = json.loads(encoded_header)
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    return 8 + header_size, header


def read_exact_binary(stream: Any, size: int, label: str) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise ValueError(f"truncated binary field {label}")
    return value


def read_u32(stream: Any, label: str) -> int:
    return struct.unpack("<I", read_exact_binary(stream, 4, label))[0]


def read_u64(stream: Any, label: str) -> int:
    return struct.unpack("<Q", read_exact_binary(stream, 8, label))[0]


def read_gguf_string(stream: Any, label: str) -> str:
    length = read_u64(stream, f"{label}.length")
    if length > 1024 * 1024 * 1024:
        raise ValueError(f"implausible GGUF string length {length} for {label}")
    return read_exact_binary(stream, length, label).decode("utf-8")


GGUF_SCALAR_SIZES = {
    0: 1,  # UINT8
    1: 1,  # INT8
    2: 2,  # UINT16
    3: 2,  # INT16
    4: 4,  # UINT32
    5: 4,  # INT32
    6: 4,  # FLOAT32
    7: 1,  # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}


def skip_gguf_value(stream: Any, value_type: int, label: str) -> None:
    if value_type in GGUF_SCALAR_SIZES:
        read_exact_binary(stream, GGUF_SCALAR_SIZES[value_type], label)
        return
    if value_type == 8:  # STRING
        read_gguf_string(stream, label)
        return
    if value_type == 9:  # ARRAY
        element_type = read_u32(stream, f"{label}.element_type")
        count = read_u64(stream, f"{label}.count")
        if count > 100_000_000:
            raise ValueError(f"implausible GGUF array length {count} for {label}")
        if element_type in GGUF_SCALAR_SIZES:
            read_exact_binary(stream, GGUF_SCALAR_SIZES[element_type] * count, label)
        else:
            for index in range(count):
                skip_gguf_value(stream, element_type, f"{label}[{index}]")
        return
    raise ValueError(f"unsupported GGUF metadata value type {value_type} for {label}")


def parse_gguf_tensor_directory(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        if read_exact_binary(stream, 4, "gguf.magic") != b"GGUF":
            raise ValueError(f"invalid GGUF magic: {path}")
        version = read_u32(stream, "gguf.version")
        tensor_count = read_u64(stream, "gguf.tensor_count")
        metadata_count = read_u64(stream, "gguf.metadata_count")
        if tensor_count > 10_000_000 or metadata_count > 10_000_000:
            raise ValueError(f"implausible GGUF directory counts: {path}")
        for index in range(metadata_count):
            key = read_gguf_string(stream, f"metadata[{index}].key")
            value_type = read_u32(stream, f"metadata[{index}].type")
            skip_gguf_value(stream, value_type, f"metadata[{index}] {key}")
        tensors: list[dict[str, Any]] = []
        for index in range(tensor_count):
            name = read_gguf_string(stream, f"tensor[{index}].name")
            dimensions = read_u32(stream, f"tensor[{index}].n_dimensions")
            if dimensions > 16:
                raise ValueError(f"implausible GGUF tensor rank {dimensions}: {name}")
            shape = [
                read_u64(stream, f"tensor[{index}].dimension")
                for _ in range(dimensions)
            ]
            tensor_type = read_u32(stream, f"tensor[{index}].type")
            offset = read_u64(stream, f"tensor[{index}].offset")
            tensors.append(
                {"name": name, "shape": shape, "type": tensor_type, "offset": offset}
            )
    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "tensors": tensors,
    }


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def bootstrap_speedup_ci(
    baseline_samples: Sequence[float],
    candidate_samples: Sequence[float],
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any] | None:
    baseline = [
        float(value)
        for value in baseline_samples
        if math.isfinite(float(value)) and float(value) > 0
    ]
    candidate = [
        float(value)
        for value in candidate_samples
        if math.isfinite(float(value)) and float(value) > 0
    ]
    if not baseline or not candidate:
        return None
    generator = random.Random(seed)
    ratios: list[float] = []
    for _ in range(resamples):
        baseline_median = statistics.median(
            generator.choices(baseline, k=len(baseline))
        )
        candidate_median = statistics.median(
            generator.choices(candidate, k=len(candidate))
        )
        if candidate_median > 0:
            ratios.append(baseline_median / candidate_median)
    if not ratios:
        return None
    tail = (1.0 - confidence) / 2.0
    return {
        "confidence": confidence,
        "method": "independent nonparametric bootstrap of median ratio",
        "resamples": resamples,
        "lower": linear_quantile(ratios, tail),
        "upper": linear_quantile(ratios, 1.0 - tail),
    }


def first_json_difference(actual: Any, expected: Any, path: str = "$") -> str | None:
    """Return the first structural/value difference, with tolerant float comparison."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        if type(actual) is not type(expected) or actual != expected:
            return f"{path}: actual={actual!r}, expected={expected!r}"
        return None
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        left = float(actual)
        right = float(expected)
        if math.isnan(left) or math.isnan(right):
            if not (math.isnan(left) and math.isnan(right)):
                return f"{path}: actual={actual!r}, expected={expected!r}"
        elif not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-9):
            return f"{path}: actual={actual!r}, expected={expected!r}"
        return None
    if type(actual) is not type(expected):
        return (
            f"{path}: type {type(actual).__name__}, expected {type(expected).__name__}"
        )
    if isinstance(actual, dict):
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            return f"{path}: missing keys={missing}, extra keys={extra}"
        for key in expected:
            difference = first_json_difference(
                actual[key], expected[key], f"{path}.{key}"
            )
            if difference is not None:
                return difference
        return None
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return f"{path}: length {len(actual)}, expected {len(expected)}"
        for index, (left, right) in enumerate(zip(actual, expected)):
            difference = first_json_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if actual != expected:
        return f"{path}: actual={actual!r}, expected={expected!r}"
    return None


class Verifier:
    def __init__(
        self,
        results_root: Path,
        binary_override: Path | None = None,
        aiter_source: Path | None = None,
    ) -> None:
        self.results_root = results_root.resolve()
        self.repository_root = REPOSITORY_ROOT.resolve()
        self.binary_override = binary_override.resolve() if binary_override else None
        self.aiter_source = aiter_source.resolve() if aiter_source else None
        self.checks = 0
        self.errors: list[str] = []
        self.notes: list[str] = []
        self.section_results: list[tuple[str, bool, str]] = []
        self.correctness: dict[str, Any] | None = None
        self.microbench: dict[str, Any] | None = None
        self.dispatch: dict[str, Any] | None = None
        self.sha256_cache: dict[Path, str] = {}
        self.sha256_cache_hits = 0
        self.aiter_patch_postimages: dict[str, bytes] = {}

    def expect(self, condition: bool, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.errors.append(message)
            return False
        return True

    def require(self, condition: bool, message: str) -> None:
        if not self.expect(condition, message):
            raise SectionAbort(message)

    def expect_equal(self, actual: Any, expected: Any, message: str) -> None:
        difference = first_json_difference(actual, expected)
        self.expect(
            difference is None, f"{message}: {difference}" if difference else message
        )

    def expect_exact(self, actual: Any, expected: Any, message: str) -> None:
        self.expect(
            actual == expected, f"{message}: actual={actual!r}, expected={expected!r}"
        )

    def cached_sha256(self, path: Path) -> str:
        resolved = path.expanduser().resolve()
        cached = self.sha256_cache.get(resolved)
        if cached is not None:
            self.sha256_cache_hits += 1
            return cached
        digest = sha256_file(resolved)
        self.sha256_cache[resolved] = digest
        return digest

    def verify_recorded_file(
        self,
        *,
        path_value: Any,
        size_value: Any,
        sha256_value: Any,
        label: str,
        missing_is_note: bool,
    ) -> Path | None:
        self.require(
            isinstance(path_value, str) and bool(path_value.strip()),
            f"{label}: path must be a non-empty string",
        )
        self.expect(
            isinstance(size_value, int)
            and not isinstance(size_value, bool)
            and size_value >= 0,
            f"{label}: size_bytes must be a non-negative integer",
        )
        self.expect(
            isinstance(sha256_value, str)
            and re.fullmatch(r"[0-9a-f]{64}", sha256_value) is not None,
            f"{label}: SHA-256 must be 64 lowercase hexadecimal characters",
        )
        path = Path(path_value).expanduser()
        if not path.exists():
            message = f"{label}: recorded file is not present locally: {path}"
            if missing_is_note:
                self.notes.append(message + "; size/SHA-256 check skipped")
                return None
            self.require(False, message)
        self.require(path.is_file(), f"{label}: recorded path is not a file: {path}")
        self.expect_exact(path.stat().st_size, size_value, f"{label}: size")
        self.expect_exact(self.cached_sha256(path), sha256_value, f"{label}: SHA-256")
        return path.resolve()

    def verify_csv_projection(
        self,
        json_rows: Sequence[dict[str, Any]],
        csv_path: Path,
        excluded_fields: set[str],
        label: str,
    ) -> None:
        self.require(bool(json_rows), f"{label}: JSON rows must be non-empty")
        self.require(csv_path.is_file(), f"{label}: missing CSV {csv_path}")
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            csv_rows = list(reader)
            fieldnames = reader.fieldnames
        expected_fields = [key for key in json_rows[0] if key not in excluded_fields]
        self.expect_exact(fieldnames, expected_fields, f"{label}: CSV columns")
        expected_rows = [
            {
                key: "" if row.get(key) is None else str(row.get(key))
                for key in expected_fields
            }
            for row in json_rows
        ]
        self.expect_exact(
            csv_rows, expected_rows, f"{label}: CSV rows vs JSON projection"
        )

    def safe_relative_path(self, base: Path, value: str, label: str) -> Path:
        candidate = Path(value)
        self.require(
            not candidate.is_absolute(),
            f"{label}: expected a relative path, got {value!r}",
        )
        resolved = (base / candidate).resolve()
        self.require(
            resolved.is_relative_to(base.resolve()),
            f"{label}: path escapes artifact root: {value!r}",
        )
        self.require(resolved.is_file(), f"{label}: missing file {resolved}")
        return resolved

    def run_section(self, name: str, function: Callable[[], str]) -> None:
        before = len(self.errors)
        detail = ""
        try:
            detail = function()
        except SectionAbort:
            detail = "structural prerequisite missing"
        except Exception as error:  # keep independent sections running
            self.errors.append(f"{name}: unexpected {type(error).__name__}: {error}")
            detail = "unexpected exception"
        passed = len(self.errors) == before
        self.section_results.append((name, passed, detail))

    @staticmethod
    def logical_correctness(result: dict[str, Any]) -> bool:
        correctness = result["correctness"]
        relative_l2 = float(correctness["relative_l2"])
        limit = float(correctness["relative_l2_limit"])
        return bool(
            correctness["finite"]
            and correctness["all_written"]
            and correctness["canary_ok"]
            and math.isfinite(relative_l2)
            and relative_l2 <= limit
        )

    @staticmethod
    def request_matches_raw(record: dict[str, Any]) -> bool:
        request = record["request"]
        case = record["result"]["case"]
        return all(
            (
                request["rows"] == case["rows"],
                request["columns"] == case["columns"],
                request["mode"] == case["mode"],
                request["mapping"] == case["requested_mapping"],
                request["pattern"] == case["pattern"],
                request["seed"] == case["seed"],
                request["cache"] == case["cache_mode"],
            )
        )

    @staticmethod
    def record_passed(record: dict[str, Any]) -> bool:
        result = record.get("result")
        return bool(
            isinstance(result, dict)
            and record.get("execution", {}).get("returncode") == 0
            and not record.get("execution", {}).get("timed_out", False)
            and Verifier.request_matches_raw(record)
            and Verifier.logical_correctness(result)
        )

    @staticmethod
    def relative_l2_policy(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
        policy: dict[str, Any] = {}
        for mode in MODES:
            results = [
                record["result"]["correctness"]
                for record in records
                if record["request"]["mode"] == mode
                and isinstance(record.get("result"), dict)
            ]
            limits = sorted({float(result["relative_l2_limit"]) for result in results})
            errors = [
                float(result["relative_l2"])
                for result in results
                if math.isfinite(float(result["relative_l2"]))
            ]
            policy[mode] = {
                "relative_l2_limits_reported_by_harness": limits,
                "maximum_relative_l2_observed": max(errors, default=None),
            }
        return policy

    def verify_correctness(self) -> str:
        path = self.results_root / "correctness.json"
        self.require(path.is_file(), f"correctness: missing {path}")
        document = read_json(path)
        self.correctness = document
        self.expect_exact(document.get("schema"), SUITE_SCHEMA, "correctness: schema")
        self.expect_exact(document.get("kind"), "correctness", "correctness: kind")
        cases = document.get("cases")
        probes = document.get("invalid_option_probes")
        self.require(isinstance(cases, list), "correctness: cases must be a list")
        self.require(
            isinstance(probes, list),
            "correctness: invalid_option_probes must be a list",
        )
        self.expect_exact(
            len(cases), EXPECTED_CORRECTNESS_CASES, "correctness: positive case count"
        )
        self.expect_exact(
            len(probes), EXPECTED_INVALID_PROBES, "correctness: invalid probe count"
        )

        referenced: set[Path] = set()
        environments: list[dict[str, Any]] = []
        for record in cases:
            raw_path = self.safe_relative_path(
                self.results_root,
                record["raw_path"],
                f"correctness {record.get('id')} raw",
            )
            referenced.add(raw_path)
            raw = read_json(raw_path)
            self.expect_exact(
                raw.get("schema"), HARNESS_SCHEMA, f"{record['id']}: harness schema"
            )
            self.expect(
                raw == record.get("result"),
                f"{record['id']}: embedded result differs from raw JSON",
            )
            self.expect(
                self.request_matches_raw(record),
                f"{record['id']}: request/raw case mismatch",
            )
            expected_correct = self.logical_correctness(raw)
            self.expect_exact(
                raw["correctness"]["passed"],
                expected_correct,
                f"{record['id']}: correctness predicate",
            )
            expected_pass = self.record_passed(record)
            self.expect_exact(
                record.get("passed"), expected_pass, f"{record['id']}: record pass flag"
            )
            self.expect(
                not raw["benchmark"]["samples_us"],
                f"{record['id']}: correctness case contains timing samples",
            )
            environments.append(raw["environment"])

        raw_files = {
            path.resolve()
            for path in (self.results_root / "raw" / "correctness").glob("*.json")
        }
        self.expect_exact(raw_files, referenced, "correctness: referenced/raw file set")
        if environments:
            self.expect(
                all(environment == environments[0] for environment in environments),
                "correctness: harness environment changes across raw cases",
            )
            self.expect_equal(
                document.get("environment"), environments[0], "correctness: environment"
            )

        recomputed_probe_passes = 0
        names: set[str] = set()
        for probe in probes:
            name = str(probe.get("name"))
            self.expect(
                name not in names, f"correctness invalid probe: duplicate name {name}"
            )
            names.add(name)
            expected_pass = bool(
                not probe.get("timed_out")
                and probe.get("returncode") == probe.get("expected_returncode") == 2
                and str(probe.get("expected_error", "")) in str(probe.get("stderr", ""))
            )
            recomputed_probe_passes += expected_pass
            self.expect_exact(
                probe.get("passed"), expected_pass, f"invalid probe {name}: pass flag"
            )

        passed = sum(self.record_passed(record) for record in cases)
        relative_l2 = [
            float(record["result"]["correctness"]["relative_l2"]) for record in cases
        ]
        maximum_absolute = [
            float(record["result"]["correctness"]["max_absolute"]) for record in cases
        ]
        expected_summary = {
            "positive_cases": len(cases),
            "positive_passed": passed,
            "positive_failed": len(cases) - passed,
            "invalid_cases": len(probes),
            "invalid_passed": recomputed_probe_passes,
            "invalid_failed": len(probes) - recomputed_probe_passes,
            "maximum_relative_l2": max(relative_l2),
            "maximum_absolute_error": max(maximum_absolute),
            "relative_l2_policy_by_mode": self.relative_l2_policy(cases),
        }
        expected_coverage = {
            "mappings": sorted({record["request"]["mapping"] for record in cases}),
            "modes": sorted({record["request"]["mode"] for record in cases}),
            "patterns": sorted({record["request"]["pattern"] for record in cases}),
            "rows": sorted({record["request"]["rows"] for record in cases}),
            "columns": sorted({record["request"]["columns"] for record in cases}),
        }
        self.expect_equal(
            document.get("summary"), expected_summary, "correctness: recomputed summary"
        )
        self.expect_equal(
            document.get("coverage"),
            expected_coverage,
            "correctness: recomputed coverage",
        )
        expected_status = (
            "passed"
            if passed == len(cases) and recomputed_probe_passes == len(probes)
            else "failed"
        )
        self.expect_exact(
            document.get("status"), expected_status, "correctness: status"
        )
        return f"{passed}/{len(cases)} positive, {recomputed_probe_passes}/{len(probes)} invalid probes"

    @staticmethod
    def comparison_rejection_reasons(
        correct: bool,
        speedup: float | None,
        interval: dict[str, Any] | None,
        minimum_speedup: float,
        minimum_lower_bound: float,
    ) -> list[str]:
        reasons: list[str] = []
        if not correct:
            reasons.append("correctness-failed")
        if speedup is None:
            reasons.append("missing-or-invalid-timing-samples")
        elif speedup < minimum_speedup:
            reasons.append(f"median-speedup-below-{minimum_speedup:.3f}")
        if interval is None:
            reasons.append("bootstrap-ci-unavailable")
        elif float(interval["lower"]) <= minimum_lower_bound:
            reasons.append(f"bootstrap-ci-lower-not-above-{minimum_lower_bound:.1f}")
        return reasons

    def verify_microbench_and_dispatch(self) -> str:
        micro_path = self.results_root / "microbench.json"
        dispatch_path = self.results_root / "dispatch.json"
        self.require(micro_path.is_file(), f"microbench: missing {micro_path}")
        self.require(dispatch_path.is_file(), f"dispatch: missing {dispatch_path}")
        microbench = read_json(micro_path)
        dispatch = read_json(dispatch_path)
        self.microbench = microbench
        self.dispatch = dispatch
        self.expect_exact(microbench.get("schema"), SUITE_SCHEMA, "microbench: schema")
        self.expect_exact(microbench.get("kind"), "microbench", "microbench: kind")
        self.expect_exact(dispatch.get("schema"), DISPATCH_SCHEMA, "dispatch: schema")
        measurements = microbench.get("measurements")
        self.require(
            isinstance(measurements, list), "microbench: measurements must be a list"
        )
        self.expect_exact(
            len(measurements),
            EXPECTED_MICROBENCH_MEASUREMENTS,
            "microbench: measurement count",
        )

        referenced: set[Path] = set()
        environments: list[dict[str, Any]] = []
        recomputed_pass: dict[str, bool] = {}
        for record in measurements:
            record_id = record["id"]
            raw_path = self.safe_relative_path(
                self.results_root, record["raw_path"], f"microbench {record_id} raw"
            )
            referenced.add(raw_path)
            raw = read_json(raw_path)
            self.expect_exact(
                raw.get("schema"), HARNESS_SCHEMA, f"{record_id}: harness schema"
            )
            self.expect(
                raw == record.get("result"),
                f"{record_id}: embedded result differs from raw JSON",
            )
            self.expect(
                self.request_matches_raw(record),
                f"{record_id}: request/raw case mismatch",
            )
            expected_correct = self.logical_correctness(raw)
            self.expect_exact(
                raw["correctness"]["passed"],
                expected_correct,
                f"{record_id}: correctness predicate",
            )
            expected_pass = self.record_passed(record)
            recomputed_pass[record_id] = expected_pass
            self.expect_exact(
                record.get("passed"), expected_pass, f"{record_id}: record pass flag"
            )

            samples = [float(value) for value in raw["benchmark"]["samples_us"]]
            self.expect_exact(len(samples), 30, f"{record_id}: sample count")
            statistics_record = timing_statistics(samples)
            self.expect(
                statistics_record["finite"], f"{record_id}: non-finite timing sample"
            )
            for key in ("mean_us", "sample_stddev_us", "median_us", "p10_us", "p90_us"):
                self.expect_equal(
                    raw["benchmark"].get(key),
                    statistics_record[key],
                    f"{record_id}: raw {key}",
                )
            median = statistics_record["median_us"]
            mode_multiplier = 2.0 if raw["case"]["mode"] == "gate-up" else 1.0
            expected_gbs = (
                float(raw["case"]["packed_weight_bytes"])
                * mode_multiplier
                / (float(median) * 1000.0)
                if median not in (None, 0)
                else 0.0
            )
            self.expect_equal(
                raw["benchmark"].get("effective_weight_gbs"),
                expected_gbs,
                f"{record_id}: effective weight GB/s",
            )
            environments.append(raw["environment"])

        raw_files = {
            path.resolve()
            for path in (self.results_root / "raw" / "microbench").glob("*.json")
        }
        self.expect_exact(raw_files, referenced, "microbench: referenced/raw file set")
        if environments:
            self.expect(
                all(environment == environments[0] for environment in environments),
                "microbench: harness environment changes across raw measurements",
            )
            self.expect_equal(
                microbench.get("environment"),
                environments[0],
                "microbench: environment",
            )

        policy = microbench.get("selection_policy")
        self.require(
            isinstance(policy, dict), "microbench: selection_policy must be an object"
        )
        minimum_speedup = float(policy["minimum_median_speedup"])
        confidence = float(policy["bootstrap_confidence"])
        lower_bound = float(policy["minimum_ci_lower_bound_exclusive"])
        self.expect_equal(
            minimum_speedup, 1.03, "microbench: minimum median speedup policy"
        )
        self.expect_equal(confidence, 0.95, "microbench: bootstrap confidence policy")
        self.expect_equal(lower_bound, 1.0, "microbench: CI lower-bound policy")
        self.expect_exact(policy.get("fallback"), "old", "microbench: fallback policy")
        self.expect_exact(
            policy.get("unseen_shape_fallback"), "old", "microbench: unseen fallback"
        )

        groups: dict[tuple[int, int, str, str], dict[str, dict[str, Any]]] = {}
        for record in measurements:
            request = record["request"]
            key = (
                request["rows"],
                request["columns"],
                request["mode"],
                request["cache"],
            )
            mapping_records = groups.setdefault(key, {})
            self.expect(
                request["mapping"] not in mapping_records,
                f"microbench: duplicate mapping {request['mapping']} for {key}",
            )
            mapping_records[request["mapping"]] = record
            if request["mapping"].startswith("small"):
                self.expect(
                    int(request["rows"]) <= 1024,
                    f"microbench: small mapping outside declared row scope for {record['id']}",
                )

        embedded_comparisons = microbench.get("comparisons")
        self.require(
            isinstance(embedded_comparisons, list),
            "microbench: comparisons must be a list",
        )
        resample_counts = {
            int(candidate["bootstrap_speedup_ci"]["resamples"])
            for comparison in embedded_comparisons
            for candidate in comparison.get("candidates", [])
            if candidate.get("bootstrap_speedup_ci") is not None
        }
        self.expect_exact(
            resample_counts,
            {EXPECTED_BOOTSTRAP_RESAMPLES},
            "microbench: bootstrap resample count",
        )
        bootstrap_resamples = EXPECTED_BOOTSTRAP_RESAMPLES

        expected_comparisons: list[dict[str, Any]] = []
        for (rows, columns, mode, cache), mapping_records in sorted(groups.items()):
            baseline = mapping_records.get("old")
            self.require(
                baseline is not None,
                f"microbench: {rows}x{columns}/{mode}/{cache} lacks old",
            )
            baseline_samples = [
                float(value) for value in baseline["result"]["benchmark"]["samples_us"]
            ]
            baseline_stats = timing_statistics(baseline_samples)
            baseline_median = baseline_stats["median_us"]
            baseline_correct = recomputed_pass[baseline["id"]]
            candidates: list[dict[str, Any]] = []
            for mapping, record in sorted(mapping_records.items()):
                if mapping == "old":
                    continue
                samples = [
                    float(value)
                    for value in record["result"]["benchmark"]["samples_us"]
                ]
                stats = timing_statistics(samples)
                candidate_median = stats["median_us"]
                speedup = (
                    float(baseline_median) / float(candidate_median)
                    if baseline_median is not None and candidate_median not in (None, 0)
                    else None
                )
                interval = bootstrap_speedup_ci(
                    baseline_samples,
                    samples,
                    bootstrap_resamples,
                    confidence,
                    stable_seed("bootstrap", rows, columns, mode, cache, mapping),
                )
                correct = baseline_correct and recomputed_pass[record["id"]]
                selection_candidate = mapping != "auto"
                reasons = self.comparison_rejection_reasons(
                    correct, speedup, interval, minimum_speedup, lower_bound
                )
                qualifies = selection_candidate and not reasons
                if not selection_candidate:
                    reasons = ["auto-is-observed-but-not-a-tuning-candidate", *reasons]
                candidates.append(
                    {
                        "mapping": mapping,
                        "resolved_mapping": record["result"]["case"][
                            "resolved_mapping"
                        ],
                        "raw_path": record["raw_path"],
                        "correctness_passed": recomputed_pass[record["id"]],
                        "statistics": stats,
                        "median_speedup_vs_old": speedup,
                        "bootstrap_speedup_ci": interval,
                        "selection_candidate": selection_candidate,
                        "qualifies": qualifies,
                        "selected": False,
                        "rejection_reasons": reasons,
                    }
                )

            qualifying = [
                candidate for candidate in candidates if candidate["qualifies"]
            ]
            selected = max(
                qualifying,
                key=lambda candidate: float(candidate["median_speedup_vs_old"]),
                default=None,
            )
            if selected is None:
                chosen_mapping = "old"
                chosen_median = baseline_median
                chosen_speedup = 1.0 if baseline_median is not None else None
                chosen_interval = {
                    "confidence": confidence,
                    "method": "identity fallback",
                    "resamples": 0,
                    "lower": 1.0,
                    "upper": 1.0,
                }
                reason = "old-fallback"
            else:
                selected["selected"] = True
                selected["rejection_reasons"] = []
                for candidate in qualifying:
                    if candidate is not selected:
                        candidate["rejection_reasons"] = [
                            "slower-than-selected-qualified-candidate"
                        ]
                chosen_mapping = selected["mapping"]
                chosen_median = selected["statistics"]["median_us"]
                chosen_speedup = selected["median_speedup_vs_old"]
                chosen_interval = selected["bootstrap_speedup_ci"]
                reason = "split-k-selected"
            expected_comparisons.append(
                {
                    "rows": rows,
                    "columns": columns,
                    "mode": mode,
                    "cache": cache,
                    "usages": baseline["request"]["usages"],
                    "baseline": {
                        "mapping": "old",
                        "raw_path": baseline["raw_path"],
                        "correctness_passed": baseline_correct,
                        "statistics": baseline_stats,
                    },
                    "candidates": candidates,
                    "decision": {
                        "mapping": chosen_mapping,
                        "median_us": chosen_median,
                        "median_speedup_vs_old": chosen_speedup,
                        "bootstrap_speedup_ci": chosen_interval,
                        "reason": reason,
                    },
                }
            )

        self.expect_equal(
            embedded_comparisons,
            expected_comparisons,
            "microbench: independently rebuilt comparisons",
        )
        primary_cache = microbench.get("summary", {}).get("primary_cache")
        self.expect_exact(primary_cache, "rotating", "microbench: primary cache")
        primary = [
            comparison
            for comparison in expected_comparisons
            if comparison["cache"] == primary_cache
        ]
        primary_speedups = [
            float(comparison["decision"]["median_speedup_vs_old"])
            for comparison in primary
            if comparison["decision"]["median_speedup_vs_old"] is not None
        ]
        weighted_baseline = 0.0
        weighted_selected = 0.0
        total_weight = 0
        for comparison in primary:
            baseline_median = comparison["baseline"]["statistics"]["median_us"]
            selected_median = comparison["decision"]["median_us"]
            if baseline_median is None or selected_median is None:
                continue
            weight = max(1, len(comparison["usages"]))
            weighted_baseline += float(baseline_median) * weight
            weighted_selected += float(selected_median) * weight
            total_weight += weight
        aggregate_statistics = {
            "primary_cache": primary_cache,
            "shape_mode_count": len(primary),
            "selected_split_k_count": sum(
                comparison["decision"]["mapping"] != "old" for comparison in primary
            ),
            "old_fallback_count": sum(
                comparison["decision"]["mapping"] == "old" for comparison in primary
            ),
            "median_dispatch_speedup": (
                statistics.median(primary_speedups) if primary_speedups else None
            ),
            "worst_dispatch_speedup": min(primary_speedups, default=None),
            "best_dispatch_speedup": max(primary_speedups, default=None),
            "baseline_latency_weighted_speedup": (
                weighted_baseline / weighted_selected if weighted_selected > 0 else None
            ),
            "weight_definition": "number of model-role references for each deduplicated shape and mode",
            "total_weight": total_weight,
            "worst_no_regression_threshold": 0.99,
            "worst_no_regression_passed": bool(primary_speedups)
            and min(primary_speedups) >= 0.99,
        }
        passed_count = sum(recomputed_pass.values())
        expected_summary = {
            "measurements": len(measurements),
            "measurements_passed": passed_count,
            "measurements_failed": len(measurements) - passed_count,
            "relative_l2_policy_by_mode": self.relative_l2_policy(measurements),
            **aggregate_statistics,
        }
        self.expect_equal(
            microbench.get("summary"),
            expected_summary,
            "microbench: recomputed summary",
        )
        self.expect_exact(
            microbench.get("status"),
            "passed" if passed_count == len(measurements) else "failed",
            "microbench: status",
        )

        expected_entries = [
            {
                "key": f"{comparison['rows']}x{comparison['columns']}:{comparison['mode']}",
                "rows": comparison["rows"],
                "columns": comparison["columns"],
                "mode": comparison["mode"],
                "mapping": comparison["decision"]["mapping"],
                "median_speedup_vs_old": comparison["decision"][
                    "median_speedup_vs_old"
                ],
                "bootstrap_speedup_ci": comparison["decision"]["bootstrap_speedup_ci"],
                "reason": comparison["decision"]["reason"],
                "usages": comparison["usages"],
                "candidates": comparison["candidates"],
            }
            for comparison in primary
        ]
        self.expect_equal(
            dispatch.get("entries"),
            expected_entries,
            "dispatch: independently rebuilt entries",
        )
        self.expect_equal(
            dispatch.get("summary"), aggregate_statistics, "dispatch: summary"
        )
        self.expect_equal(
            dispatch.get("selection_policy"), policy, "dispatch: selection policy"
        )
        self.expect_exact(
            dispatch.get("cache_basis"), primary_cache, "dispatch: cache basis"
        )
        self.expect_exact(
            dispatch.get("default_for_unseen_legal_shapes"),
            "old",
            "dispatch: unseen fallback",
        )
        if environments:
            self.expect_exact(
                dispatch.get("architecture"),
                environments[0].get("gfx"),
                "dispatch: architecture",
            )
        return (
            f"{passed_count}/{len(measurements)} measurements, "
            f"{len(expected_comparisons)} comparisons, {len(primary)} dispatch entries"
        )

    def verify_aggregate_and_environment(self) -> str:
        aggregate_path = self.results_root / "aggregate.json"
        environment_path = self.results_root / "environment.json"
        self.require(aggregate_path.is_file(), f"aggregate: missing {aggregate_path}")
        self.require(
            environment_path.is_file(), f"environment: missing {environment_path}"
        )
        self.require(
            self.correctness is not None, "aggregate: correctness document unavailable"
        )
        self.require(
            self.microbench is not None, "aggregate: microbench document unavailable"
        )
        self.require(
            self.dispatch is not None, "aggregate: dispatch document unavailable"
        )
        aggregate = read_json(aggregate_path)
        environment = read_json(environment_path)
        self.expect_exact(aggregate.get("schema"), SUITE_SCHEMA, "aggregate: schema")
        self.expect_exact(aggregate.get("kind"), "aggregate", "aggregate: kind")
        self.expect_equal(
            aggregate.get("correctness_summary"),
            self.correctness["summary"],
            "aggregate: correctness summary",
        )
        self.expect_equal(
            aggregate.get("microbench_summary"),
            self.microbench["summary"],
            "aggregate: microbench summary",
        )
        self.expect_equal(
            aggregate.get("dispatch_summary"),
            self.dispatch["summary"],
            "aggregate: dispatch summary",
        )
        expected_status = (
            "passed"
            if self.correctness.get("status") == "passed"
            and self.microbench.get("status") == "passed"
            else "failed"
        )
        self.expect_exact(aggregate.get("status"), expected_status, "aggregate: status")
        artifacts = aggregate.get("artifacts", {})
        for name, value in artifacts.items():
            if value is None:
                continue
            artifact_path = (self.results_root / str(value)).resolve()
            self.expect(
                artifact_path.is_relative_to(self.results_root)
                and artifact_path.exists(),
                f"aggregate artifact {name}: missing or outside results root: {value}",
            )

        expected_environment = self.microbench.get(
            "environment"
        ) or self.correctness.get("environment")
        self.expect_equal(
            environment.get("harness_environment"),
            expected_environment,
            "environment: harness metadata",
        )
        self.expect_equal(
            self.correctness.get("environment"),
            self.microbench.get("environment"),
            "cross-document environment",
        )
        hashes = {
            self.correctness.get("runtime_setup", {}).get("binary_sha256"),
            self.microbench.get("runtime_setup", {}).get("binary_sha256"),
            environment.get("runtime_setup", {}).get("binary_sha256"),
        }
        hashes.discard(None)
        self.expect_exact(len(hashes), 1, "cross-document harness binary SHA-256")
        return f"status={expected_status}, shared harness hash={next(iter(hashes), 'unavailable')}"

    @staticmethod
    def result_for_generation(
        record: dict[str, Any], generation: int
    ) -> dict[str, Any] | None:
        for result in record.get("results", []):
            if result.get("n_gen") == generation:
                return result
        return None

    def verify_end_to_end_provenance(self, document: dict[str, Any], label: str) -> int:
        provenance = document.get("provenance")
        self.require(
            isinstance(provenance, dict), f"{label}: provenance must be an object"
        )
        local_files_checked = 0
        for artifact_name in ("binary", "model", "sidecar"):
            artifact = provenance.get(artifact_name)
            self.require(
                isinstance(artifact, dict),
                f"{label}: provenance.{artifact_name} must be an object",
            )
            recorded_path = artifact.get("path")
            recorded_hash = artifact.get("sha256")
            recorded_size = artifact.get("size_bytes")
            self.require(
                isinstance(recorded_path, str) and bool(recorded_path.strip()),
                f"{label}: provenance.{artifact_name}.path must be non-empty",
            )
            self.expect(
                isinstance(recorded_size, int) and recorded_size >= 0,
                f"{label}: provenance.{artifact_name}.size_bytes is invalid",
            )
            self.expect(
                isinstance(recorded_hash, str)
                and re.fullmatch(r"[0-9a-f]{64}", recorded_hash) is not None,
                f"{label}: provenance.{artifact_name}.sha256 is invalid",
            )
            local_path = Path(recorded_path).expanduser()
            if not local_path.exists():
                self.notes.append(
                    f"{label}: external provenance {artifact_name} is not present locally; "
                    f"size/SHA-256 check skipped ({recorded_path})"
                )
                continue
            self.require(
                local_path.is_file(),
                f"{label}: provenance.{artifact_name} path exists but is not a file: {local_path}",
            )
            local_files_checked += 1
            self.expect_exact(
                local_path.stat().st_size,
                recorded_size,
                f"{label}: provenance {artifact_name} size",
            )
            self.expect_exact(
                self.cached_sha256(local_path),
                recorded_hash,
                f"{label}: provenance {artifact_name} SHA-256",
            )
        return local_files_checked

    def recompute_runtime_evidence(
        self,
        record: dict[str, Any],
        stderr: str,
        sidecar_path: str,
    ) -> dict[str, Any]:
        expected_mapping = str(record["expected_mapping"])
        controlled_environment = record["environment_summary"]["controlled_environment"]
        observed_q4_environment = {
            key: value
            for key, value in controlled_environment.items()
            if key.startswith(Q4_ENV_PREFIX)
        }
        if expected_mapping == "production":
            expected_q4_environment: dict[str, str] = {}
            uses_sidecar = False
        elif expected_mapping == "auto":
            expected_q4_environment = {"LLAMA_Q4_RDNA_SIDECAR": sidecar_path}
            uses_sidecar = True
        else:
            expected_q4_environment = {
                "LLAMA_Q4_RDNA_MAPPING": expected_mapping,
                "LLAMA_Q4_RDNA_SIDECAR": sidecar_path,
            }
            uses_sidecar = True
        environment_matches = observed_q4_environment == expected_q4_environment

        load_matches = list(LOAD_RE.finditer(stderr))
        loaded = [
            {
                "tensors": int(match.group("tensors")),
                "gib": float(match.group("gib")),
                "device": int(match.group("device")),
                "path": match.group("path").strip(),
            }
            for match in load_matches
        ]
        launch_counts = [
            int(match.group("count")) for match in LAUNCH_RE.finditer(stderr)
        ]
        unique_counts = [
            int(match.group("count")) for match in UNIQUE_RE.finditer(stderr)
        ]
        mapping_matches = [
            match.group("mapping") for match in MAPPING_RE.finditer(stderr)
        ]
        error_lines = [
            line
            for line in stderr.splitlines()
            if line.startswith("Q4_RDNA:")
            and any(
                word in line.lower()
                for word in ("invalid", "truncated", "mismatch", "only supports")
            )
        ]

        errors: list[str] = []
        if not environment_matches:
            errors.append(
                f"Q4_RDNA environment is {observed_q4_environment!r}, "
                f"expected {expected_q4_environment!r}"
            )
        if error_lines:
            errors.append("Q4_RDNA emitted an error diagnostic")
        if uses_sidecar:
            if len(loaded) != 1:
                errors.append(
                    f"expected exactly one sidecar load log, observed {len(loaded)}"
                )
            elif loaded[0]["path"] != sidecar_path:
                errors.append(
                    f"loaded sidecar path is {loaded[0]['path']!r}, expected {sidecar_path!r}"
                )
            elif loaded[0]["tensors"] <= 0:
                errors.append("sidecar load log reports no tensors")
            if not launch_counts or max(launch_counts) <= 0:
                errors.append("no non-zero Q4_RDNA decode launch count was logged")
            if not unique_counts or max(unique_counts) <= 0:
                errors.append("no non-zero Q4_RDNA unique tensor hit count was logged")
        elif loaded or launch_counts or unique_counts or "Q4_RDNA:" in stderr:
            errors.append(
                "production baseline unexpectedly emitted Q4_RDNA runtime logs"
            )

        if mapping_matches:
            accepted = {expected_mapping}
            if expected_mapping == "auto":
                accepted.update(("split", "split-k", "default"))
            if any(mapping not in accepted for mapping in mapping_matches):
                errors.append(
                    f"runtime mapping marker is {mapping_matches!r}, "
                    f"expected one of {sorted(accepted)!r}"
                )
        return {
            "passed": not errors,
            "expected_mapping": expected_mapping,
            "mapping_contract": (
                "old is selected by LLAMA_Q4_RDNA_MAPPING=old; auto/default Split-K is selected "
                "by explicitly leaving LLAMA_Q4_RDNA_MAPPING unset"
            ),
            "mapping_marker_emitted": bool(mapping_matches),
            "mapping_markers": mapping_matches,
            "expected_q4rdna_environment": expected_q4_environment,
            "observed_q4rdna_environment": observed_q4_environment,
            "environment_matches": environment_matches,
            "sidecar_loads": loaded,
            "decode_launch_counts": launch_counts,
            "unique_tensor_counts": unique_counts,
            "q4rdna_error_lines": error_lines,
            "errors": errors,
        }

    def verify_end_to_end_document(self, path: Path) -> str:
        directory = path.parent
        document = read_json(path)
        label = directory.name
        self.expect_exact(document.get("schema"), END_TO_END_SCHEMA, f"{label}: schema")
        protocol = document.get("protocol")
        self.require(isinstance(protocol, dict), f"{label}: protocol missing")
        for optional_label in ("expected_model_family", "model_label"):
            if optional_label in protocol:
                value = protocol[optional_label]
                self.expect(
                    isinstance(value, str) and bool(value.strip()),
                    f"{label}: protocol.{optional_label} must be non-empty when present",
                )
        local_provenance_files = self.verify_end_to_end_provenance(document, label)
        sidecar_path = str(document["provenance"]["sidecar"]["path"])
        rounds = int(protocol["rounds"])
        generations = [int(value) for value in protocol["generations"]]
        warmups = document.get("warmups")
        measurements = document.get("measurements")
        self.require(isinstance(warmups, list), f"{label}: warmups must be a list")
        self.require(
            isinstance(measurements, list), f"{label}: measurements must be a list"
        )

        referenced_meta: set[Path] = set()
        referenced_stdout: set[Path] = set()
        referenced_stderr: set[Path] = set()
        for record in [*warmups, *measurements]:
            artifacts = record["artifacts"]
            meta_path = self.safe_relative_path(
                directory, artifacts["metadata"], f"{label} {record['id']} meta"
            )
            stdout_path = self.safe_relative_path(
                directory, artifacts["stdout_json"], f"{label} {record['id']} stdout"
            )
            stderr_path = self.safe_relative_path(
                directory, artifacts["stderr"], f"{label} {record['id']} stderr"
            )
            referenced_meta.add(meta_path)
            referenced_stdout.add(stdout_path)
            referenced_stderr.add(stderr_path)
            self.expect(
                read_json(meta_path) == record,
                f"{label} {record['id']}: embedded record differs from raw metadata",
            )
            self.expect(
                read_json(stdout_path) == record.get("results"),
                f"{label} {record['id']}: embedded results differ from raw stdout JSON",
            )
            stderr_lines = stderr_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            self.expect_exact(
                record.get("stderr_tail"),
                stderr_lines[-20:],
                f"{label} {record['id']}: stderr tail",
            )
            stderr_text = "\n".join(stderr_lines)
            recomputed_runtime = self.recompute_runtime_evidence(
                record, stderr_text, sidecar_path
            )
            self.expect_equal(
                record.get("runtime_evidence"),
                recomputed_runtime,
                f"{label} {record['id']}: directly recomputed runtime markers",
            )
            basic_pass = bool(
                record.get("returncode") == 0
                and not record.get("timed_out")
                and not record.get("errors")
                and record.get("runtime_evidence", {}).get("passed")
                and len(record.get("results", [])) == len(generations)
                and {row.get("n_gen") for row in record.get("results", [])}
                == set(generations)
            )
            self.expect_exact(
                record.get("passed"),
                basic_pass,
                f"{label} {record['id']}: basic pass predicate",
            )

        raw_root = directory / "raw"
        self.expect_exact(
            {item.resolve() for item in raw_root.glob("**/*.meta.json")},
            referenced_meta,
            f"{label}: raw metadata file set",
        )
        self.expect_exact(
            {item.resolve() for item in raw_root.glob("**/*.stdout.json")},
            referenced_stdout,
            f"{label}: raw stdout file set",
        )
        self.expect_exact(
            {item.resolve() for item in raw_root.glob("**/*.stderr.log")},
            referenced_stderr,
            f"{label}: raw stderr file set",
        )

        route_order: list[str] = []
        for record in measurements:
            route = str(record["route"])
            if route not in route_order:
                route_order.append(route)
        series = document.get("series")
        self.require(isinstance(series, dict), f"{label}: series must be an object")
        self.expect_exact(set(route_order), set(series), f"{label}: route set")
        self.expect_exact(
            len(measurements), rounds * len(route_order), f"{label}: measurement count"
        )
        self.expect_exact(len(warmups), len(route_order), f"{label}: warmup count")

        schedule = document.get("schedule")
        self.require(
            isinstance(schedule, list) and schedule, f"{label}: schedule missing"
        )
        self.expect_exact(len(schedule), rounds, f"{label}: schedule round count")
        base_order = list(schedule[0]["order"])
        self.expect_exact(
            set(base_order), set(route_order), f"{label}: schedule route set"
        )
        for round_number in range(1, rounds + 1):
            offset = (round_number - 1) % len(base_order)
            expected_order = base_order[offset:] + base_order[:offset]
            schedule_record = schedule[round_number - 1]
            self.expect_exact(
                schedule_record.get("round"),
                round_number,
                f"{label}: schedule round number",
            )
            self.expect_exact(
                schedule_record.get("order"),
                expected_order,
                f"{label}: Latin rotation round {round_number}",
            )
            actual_records = sorted(
                (
                    record
                    for record in measurements
                    if record.get("round") == round_number
                ),
                key=lambda record: record["order_position"],
            )
            self.expect_exact(
                [record["route"] for record in actual_records],
                expected_order,
                f"{label}: execution order round {round_number}",
            )

        expected_series: dict[str, Any] = {}
        for route in route_order:
            route_records = [
                record
                for record in measurements
                if record["route"] == route and record["passed"]
            ]
            generation_results: dict[str, Any] = {}
            for generation in generations:
                rows = [
                    self.result_for_generation(record, generation)
                    for record in route_records
                ]
                valid_rows = [row for row in rows if row is not None]
                generation_results[str(generation)] = {
                    "throughput_tokens_per_second": sample_summary(
                        [float(row["avg_ts"]) for row in valid_rows], rounds
                    ),
                    "elapsed_nanoseconds": sample_summary(
                        [float(row["avg_ns"]) for row in valid_rows], rounds
                    ),
                }
            expected_series[route] = {
                "label": series[route]["label"],
                "expected_mapping": series[route]["expected_mapping"],
                "generations": generation_results,
            }
        self.expect_equal(series, expected_series, f"{label}: recomputed series")

        expected_gains: dict[str, Any] = {}
        gains = document.get("gains")
        self.require(isinstance(gains, dict), f"{label}: gains must be an object")
        for gain_label, embedded_gain in gains.items():
            numerator_route = embedded_gain["numerator"]
            denominator_route = embedded_gain["denominator"]
            by_generation: dict[str, Any] = {}
            for generation in generations:
                numerator = expected_series[numerator_route]["generations"][
                    str(generation)
                ]
                denominator = expected_series[denominator_route]["generations"][
                    str(generation)
                ]
                numerator_ts = numerator["throughput_tokens_per_second"]["mean"]
                denominator_ts = denominator["throughput_tokens_per_second"]["mean"]
                numerator_ns = numerator["elapsed_nanoseconds"]["mean"]
                denominator_ns = denominator["elapsed_nanoseconds"]["mean"]
                paired_ts: list[float] = []
                paired_ns: list[float] = []
                for round_number in range(1, rounds + 1):
                    by_route = {
                        record["route"]: self.result_for_generation(record, generation)
                        for record in measurements
                        if record.get("round") == round_number and record.get("passed")
                    }
                    numerator_row = by_route.get(numerator_route)
                    denominator_row = by_route.get(denominator_route)
                    if numerator_row is None or denominator_row is None:
                        continue
                    paired_ts.append(
                        (
                            float(numerator_row["avg_ts"])
                            / float(denominator_row["avg_ts"])
                            - 1.0
                        )
                        * 100.0
                    )
                    paired_ns.append(
                        (
                            1.0
                            - float(numerator_row["avg_ns"])
                            / float(denominator_row["avg_ns"])
                        )
                        * 100.0
                    )
                by_generation[str(generation)] = {
                    "gain_from_route_means_percent": (
                        (float(numerator_ts) / float(denominator_ts) - 1.0) * 100.0
                        if numerator_ts is not None and denominator_ts not in (None, 0)
                        else None
                    ),
                    "elapsed_reduction_from_route_means_percent": (
                        (1.0 - float(numerator_ns) / float(denominator_ns)) * 100.0
                        if numerator_ns is not None and denominator_ns not in (None, 0)
                        else None
                    ),
                    "paired_round_throughput_gain_percent": sample_summary(
                        paired_ts, rounds
                    ),
                    "paired_round_elapsed_reduction_percent": sample_summary(
                        paired_ns, rounds
                    ),
                }
            expected_gains[gain_label] = {
                "numerator": numerator_route,
                "denominator": denominator_route,
                "generations": by_generation,
            }
        self.expect_equal(gains, expected_gains, f"{label}: recomputed gains")

        all_records = [*warmups, *measurements]
        commits = sorted(
            {
                str(row["build_commit"])
                for record in all_records
                for row in record.get("results", [])
                if row.get("build_commit")
            }
        )
        self.expect_exact(
            document.get("provenance", {}).get("llama_build_commits"),
            commits,
            f"{label}: build commits",
        )
        all_processes_passed = bool(
            len(warmups) == len(route_order)
            and len(measurements) == rounds * len(route_order)
            and all(record["passed"] for record in all_records)
        )
        series_complete = all(
            expected_series[route]["generations"][str(generation)][metric]["complete"]
            for route in route_order
            for generation in generations
            for metric in ("throughput_tokens_per_second", "elapsed_nanoseconds")
        )
        expected_validation = {
            "warmups_passed": sum(bool(record["passed"]) for record in warmups),
            "warmups_expected": len(route_order),
            "measurements_passed": sum(
                bool(record["passed"]) for record in measurements
            ),
            "measurements_expected": rounds * len(route_order),
            "series_complete": series_complete,
            "single_build_commit": len(commits) == 1,
        }
        self.expect_equal(
            document.get("validation_summary"),
            expected_validation,
            f"{label}: validation summary",
        )
        expected_status = (
            "passed"
            if all_processes_passed and series_complete and len(commits) == 1
            else "failed"
        )
        self.expect_exact(document.get("status"), expected_status, f"{label}: status")
        return (
            f"{rounds} rounds x {len(route_order)} routes, generations={generations}, "
            f"local provenance files checked={local_provenance_files}"
        )

    def verify_all_end_to_end(self) -> str:
        documents = sorted(self.results_root.glob("end_to_end_*/end_to_end.json"))
        self.require(
            bool(documents),
            "end-to-end: no end_to_end_*/end_to_end.json documents found",
        )
        details: list[str] = []
        for document in documents:
            details.append(
                f"{document.parent.name} ({self.verify_end_to_end_document(document)})"
            )
        return "; ".join(details)

    def verify_mistral_assets(self) -> str:
        path = self.results_root / "mistral_assets.json"
        self.require(path.is_file(), f"Mistral assets: missing {path}")
        document = read_json(path)
        self.expect_exact(
            document.get("schema"), MODEL_ASSETS_SCHEMA, "Mistral assets: schema"
        )
        model = document.get("model")
        production = document.get("production_gguf")
        q4rdna = document.get("q4_rdna_sidecar")
        self.require(isinstance(model, dict), "Mistral assets: model must be an object")
        self.require(
            isinstance(production, dict),
            "Mistral assets: production_gguf must be an object",
        )
        self.require(
            isinstance(q4rdna, dict),
            "Mistral assets: q4_rdna_sidecar must be an object",
        )
        self.expect(
            isinstance(model.get("label"), str) and "mistral" in model["label"].lower(),
            "Mistral assets: model label must identify Mistral",
        )
        for owner, prefix in ((model, "model"), (production, "production_gguf")):
            self.expect(
                isinstance(owner.get("hugging_face_repository"), str)
                and bool(owner["hugging_face_repository"].strip()),
                f"Mistral assets: {prefix} repository must be non-empty",
            )
            self.expect(
                isinstance(owner.get("revision"), str)
                and re.fullmatch(r"[0-9a-f]{40}", owner["revision"]) is not None,
                f"Mistral assets: {prefix} revision must be a full Git commit",
            )

        configuration = model.get("configuration")
        files = model.get("files")
        validation = model.get("safetensors_validation")
        self.require(
            isinstance(configuration, dict), "Mistral assets: configuration missing"
        )
        self.require(
            isinstance(files, list), "Mistral assets: model.files must be a list"
        )
        self.require(
            isinstance(validation, dict),
            "Mistral assets: safetensors_validation missing",
        )
        recorded_files: dict[str, Path] = {}
        for index, record in enumerate(files):
            self.require(
                isinstance(record, dict),
                f"Mistral assets: model.files[{index}] is not an object",
            )
            verified = self.verify_recorded_file(
                path_value=record.get("path"),
                size_value=record.get("size_bytes"),
                sha256_value=record.get("sha256"),
                label=f"Mistral assets model file {index}",
                missing_is_note=False,
            )
            self.require(
                verified is not None, f"Mistral assets: model file {index} unavailable"
            )
            self.expect(
                verified.name not in recorded_files,
                f"Mistral assets: duplicate model filename {verified.name}",
            )
            recorded_files[verified.name] = verified
        self.require(
            "config.json" in recorded_files,
            "Mistral assets: config.json is not recorded",
        )
        self.require(
            "model.safetensors.index.json" in recorded_files,
            "Mistral assets: safetensors index is not recorded",
        )

        config = read_json(recorded_files["config.json"])
        expected_configuration = {
            "model_type": config.get("model_type"),
            "layers": config.get("num_hidden_layers"),
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "attention_heads": config.get("num_attention_heads"),
            "key_value_heads": config.get("num_key_value_heads"),
            "source_dtype": (
                "BF16"
                if config.get("torch_dtype") == "bfloat16"
                else config.get("torch_dtype")
            ),
        }
        self.expect_equal(
            configuration,
            expected_configuration,
            "Mistral assets: configuration vs config.json",
        )
        layers = int(configuration["layers"])
        self.expect_exact(layers, 32, "Mistral assets: layer count")
        expected_target_count = layers * len(TARGET_TENSOR_SUFFIXES)
        self.expect_exact(
            expected_target_count, 224, "Mistral assets: derived target tensor count"
        )

        index_document = read_json(recorded_files["model.safetensors.index.json"])
        weight_map = index_document.get("weight_map")
        self.require(
            isinstance(weight_map, dict),
            "Mistral assets: safetensors weight_map missing",
        )
        shard_names = sorted(set(weight_map.values()))
        self.expect_exact(
            set(recorded_files),
            {"config.json", "model.safetensors.index.json", *shard_names},
            "Mistral assets: recorded file set vs index shards",
        )
        tensor_headers: dict[str, dict[str, Any]] = {}
        tensor_locations: dict[str, tuple[Path, int]] = {}
        parsed_payload_bytes = 0
        for shard_name in shard_names:
            self.require(
                shard_name in recorded_files,
                f"Mistral assets: unrecorded shard {shard_name}",
            )
            shard_path = recorded_files[shard_name]
            data_start, header = parse_safetensors_header(shard_path)
            tensors = {
                name: value for name, value in header.items() if name != "__metadata__"
            }
            payload_bytes = shard_path.stat().st_size - data_start
            parsed_payload_bytes += payload_bytes
            maximum_end = 0
            intervals: list[tuple[int, int, str]] = []
            for name, metadata in tensors.items():
                self.require(
                    isinstance(metadata, dict),
                    f"Mistral assets: invalid metadata for {name}",
                )
                offsets = metadata.get("data_offsets")
                self.require(
                    isinstance(offsets, list) and len(offsets) == 2,
                    f"Mistral assets: invalid offsets for {name}",
                )
                begin, end = map(int, offsets)
                self.expect(
                    0 <= begin <= end <= payload_bytes,
                    f"Mistral assets: out-of-range offsets for {name}",
                )
                maximum_end = max(maximum_end, end)
                intervals.append((begin, end, name))
                self.expect(
                    name not in tensor_headers,
                    f"Mistral assets: duplicate shard tensor {name}",
                )
                tensor_headers[name] = metadata
                tensor_locations[name] = (shard_path, data_start)
            self.expect_exact(
                maximum_end,
                payload_bytes,
                f"Mistral assets: {shard_name} payload extent",
            )
            for (begin, end, name), (next_begin, _, next_name) in zip(
                sorted(intervals), sorted(intervals)[1:]
            ):
                self.expect(
                    end <= next_begin,
                    f"Mistral assets: overlapping shard tensors {name} and {next_name}",
                )

        index_keys = set(weight_map)
        header_keys = set(tensor_headers)
        for name, shard_name in weight_map.items():
            location = tensor_locations.get(name)
            self.expect(
                location is not None and location[0].name == shard_name,
                f"Mistral assets: index/header shard mismatch for {name}",
            )
        target_names = {
            f"model.layers.{layer}.{hf_suffix}"
            for layer in range(layers)
            for hf_suffix, _ in TARGET_TENSOR_SUFFIXES
        }
        target_found = target_names & index_keys & header_keys
        target_bf16 = {
            name for name in target_found if tensor_headers[name].get("dtype") == "BF16"
        }
        rows_aligned: set[str] = set()
        columns_aligned: set[str] = set()
        for name in target_found:
            metadata = tensor_headers[name]
            shape = metadata.get("shape")
            self.require(
                isinstance(shape, list) and len(shape) == 2,
                f"Mistral assets: target tensor is not a matrix: {name}",
            )
            rows, columns = map(int, shape)
            if rows % 32 == 0:
                rows_aligned.add(name)
            if columns % 64 == 0:
                columns_aligned.add(name)
            begin, end = map(int, metadata["data_offsets"])
            self.expect_exact(
                end - begin,
                rows * columns * 2,
                f"Mistral assets: BF16 payload byte count for {name}",
            )
        index_payload_bytes = int(
            index_document.get("metadata", {}).get("total_size", -1)
        )
        expected_validation = {
            "index_tensor_payload_bytes": index_payload_bytes,
            "parsed_shard_payload_bytes": parsed_payload_bytes,
            "index_keys": len(index_keys),
            "header_keys": len(header_keys),
            "missing_or_extra_keys": len(index_keys ^ header_keys),
            "target_tensors_expected": len(target_names),
            "target_tensors_found": len(target_found),
            "target_tensors_bf16": len(target_bf16),
            "rows_divisible_by_32": len(rows_aligned),
            "columns_divisible_by_64": len(columns_aligned),
        }
        self.expect_equal(
            validation, expected_validation, "Mistral assets: safetensors validation"
        )
        self.expect_exact(
            len(target_found),
            expected_target_count,
            "Mistral assets: all target tensors found",
        )

        production_path = self.verify_recorded_file(
            path_value=production.get("path"),
            size_value=production.get("size_bytes"),
            sha256_value=production.get("sha256"),
            label="Mistral assets production GGUF",
            missing_is_note=False,
        )
        self.require(
            production_path is not None, "Mistral assets: production GGUF unavailable"
        )
        gguf = parse_gguf_tensor_directory(production_path)
        self.expect_exact(
            production.get("gguf_version"),
            gguf["version"],
            "Mistral assets: GGUF version",
        )
        self.expect_exact(
            production.get("tensor_count"),
            gguf["tensor_count"],
            "Mistral assets: GGUF tensor count",
        )
        gguf_targets = {
            f"blk.{layer}.{gguf_suffix}"
            for layer in range(layers)
            for _, gguf_suffix in TARGET_TENSOR_SUFFIXES
        }
        target_tensor_rows = [
            tensor for tensor in gguf["tensors"] if tensor["name"] in gguf_targets
        ]
        self.expect_exact(
            {tensor["name"] for tensor in target_tensor_rows},
            gguf_targets,
            "Mistral assets: GGUF target tensor names",
        )
        gguf_type_names = {12: "Q4_K", 14: "Q6_K"}
        target_type_counts = Counter(
            gguf_type_names.get(int(tensor["type"]), f"TYPE_{tensor['type']}")
            for tensor in target_tensor_rows
        )
        self.expect_exact(
            production.get("target_tensors_found"),
            len(target_tensor_rows),
            "Mistral assets: GGUF target count",
        )
        self.expect_equal(
            production.get("target_quantization_types"),
            dict(sorted(target_type_counts.items())),
            "Mistral assets: GGUF target quantization types",
        )

        manifest_record = q4rdna.get("manifest")
        sidecar_record = q4rdna.get("sidecar")
        packer = q4rdna.get("packer")
        self.require(
            isinstance(manifest_record, dict), "Mistral assets: manifest record missing"
        )
        self.require(
            isinstance(sidecar_record, dict), "Mistral assets: sidecar record missing"
        )
        self.require(isinstance(packer, dict), "Mistral assets: packer record missing")
        manifest_path = self.verify_recorded_file(
            path_value=manifest_record.get("path"),
            size_value=manifest_record.get("size_bytes"),
            sha256_value=manifest_record.get("sha256"),
            label="Mistral assets Q4_RDNA manifest",
            missing_is_note=False,
        )
        sidecar_path = self.verify_recorded_file(
            path_value=sidecar_record.get("path"),
            size_value=sidecar_record.get("size_bytes"),
            sha256_value=sidecar_record.get("sha256"),
            label="Mistral assets Q4_RDNA sidecar",
            missing_is_note=False,
        )
        self.require(manifest_path is not None, "Mistral assets: manifest unavailable")
        self.require(sidecar_path is not None, "Mistral assets: sidecar unavailable")

        expected_manifest: list[tuple[str, int, int, int, str]] = []
        for layer in range(layers):
            for hf_suffix, gguf_suffix in TARGET_TENSOR_SUFFIXES:
                hf_name = f"model.layers.{layer}.{hf_suffix}"
                metadata = tensor_headers[hf_name]
                shard_path, data_start = tensor_locations[hf_name]
                begin = int(metadata["data_offsets"][0])
                rows, columns = map(int, metadata["shape"])
                expected_manifest.append(
                    (
                        str(shard_path),
                        data_start + begin,
                        rows,
                        columns,
                        f"blk.{layer}.{gguf_suffix}",
                    )
                )
        actual_manifest: list[tuple[str, int, int, int, str]] = []
        for line_number, line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = line.split("\t")
            self.require(
                len(fields) == 5, f"Mistral assets: invalid manifest line {line_number}"
            )
            actual_manifest.append(
                (fields[0], int(fields[1]), int(fields[2]), int(fields[3]), fields[4])
            )
        self.expect_exact(
            actual_manifest,
            expected_manifest,
            "Mistral assets: rebuilt 224-entry manifest",
        )

        with sidecar_path.open("rb") as stream:
            encoded_header = read_exact_binary(stream, 48, "Q4_RDNA header")
            (
                magic,
                version,
                entry_count,
                group,
                tile_rows,
                tile_bytes,
                reserved,
                data_offset,
                data_bytes,
            ) = struct.unpack("<8sIIIIIIQQ", encoded_header)
            sidecar_entries: list[dict[str, Any]] = []
            structurally_valid = 0
            for index in range(entry_count):
                encoded_entry = read_exact_binary(stream, 96, f"Q4_RDNA entry {index}")
                name_bytes, rows, columns, offset, size, entry_reserved = struct.unpack(
                    "<64sIIQQQ", encoded_entry
                )
                nul = name_bytes.find(b"\0")
                name = name_bytes[:nul].decode("utf-8") if nul >= 0 else ""
                valid = bool(
                    nul >= 0
                    and name
                    and rows % 32 == 0
                    and columns % 64 == 0
                    and size == rows * columns * 34 // 64
                    and offset >= data_offset
                    and offset + size >= offset
                    and offset + size <= data_offset + data_bytes
                    and entry_reserved == 0
                )
                structurally_valid += valid
                sidecar_entries.append(
                    {
                        "name": name,
                        "rows": rows,
                        "columns": columns,
                        "offset": offset,
                        "size": size,
                    }
                )
        self.expect_exact(magic, b"Q4RDNA1\0", "Mistral assets: sidecar magic")
        self.expect_exact(version, 1, "Mistral assets: sidecar version")
        self.expect_exact(group, 64, "Mistral assets: sidecar group")
        self.expect_exact(tile_rows, 32, "Mistral assets: sidecar tile rows")
        self.expect_exact(tile_bytes, 1088, "Mistral assets: sidecar tile bytes")
        self.expect_exact(reserved, 0, "Mistral assets: sidecar reserved field")
        self.expect_exact(
            entry_count,
            len(actual_manifest),
            "Mistral assets: sidecar/manifest entry count",
        )
        self.expect_exact(
            sidecar_record.get("entries"),
            entry_count,
            "Mistral assets: recorded sidecar entries",
        )
        self.expect_exact(
            sidecar_record.get("structurally_valid_entries"),
            structurally_valid,
            "Mistral assets: structurally valid sidecar entries",
        )
        expected_data_offset = align_up(48 + entry_count * 96, 4096)
        self.expect_exact(
            data_offset, expected_data_offset, "Mistral assets: sidecar data offset"
        )
        next_offset = data_offset
        expected_sidecar_entries: list[dict[str, Any]] = []
        for _, _, rows, columns, name in actual_manifest:
            size = rows * columns * 34 // 64
            expected_sidecar_entries.append(
                {
                    "name": name,
                    "rows": rows,
                    "columns": columns,
                    "offset": next_offset,
                    "size": size,
                }
            )
            next_offset = align_up(next_offset + size, 256)
        self.expect_exact(
            sidecar_entries,
            expected_sidecar_entries,
            "Mistral assets: sidecar index vs manifest",
        )
        self.expect_exact(
            data_bytes,
            next_offset - data_offset,
            "Mistral assets: sidecar data byte extent",
        )
        self.expect_exact(
            sidecar_path.stat().st_size,
            data_offset + data_bytes,
            "Mistral assets: sidecar file extent",
        )

        for prefix, path_key, size_key, hash_key in (
            ("packer source", "source_path", "source_size_bytes", "source_sha256"),
            (
                "manifest generator",
                "manifest_generator_path",
                "manifest_generator_size_bytes",
                "manifest_generator_sha256",
            ),
        ):
            self.verify_recorded_file(
                path_value=packer.get(path_key),
                size_value=packer.get(size_key),
                sha256_value=packer.get(hash_key),
                label=f"Mistral assets {prefix}",
                missing_is_note=False,
            )
        self.verify_recorded_file(
            path_value=packer.get("temporary_binary_path"),
            size_value=packer.get("temporary_binary_size_bytes"),
            sha256_value=packer.get("temporary_binary_sha256"),
            label="Mistral assets temporary packer binary",
            missing_is_note=True,
        )
        self.expect(
            isinstance(q4rdna.get("layout"), str) and "1088" in q4rdna["layout"],
            "Mistral assets: sidecar layout description",
        )
        cross_check = q4rdna.get("cross_model_packer_check")
        self.require(
            isinstance(cross_check, dict),
            "Mistral assets: cross-model packer check missing",
        )
        self.expect_exact(
            cross_check.get("native_packer_matches_published_sidecar_byte_for_byte"),
            True,
            "Mistral assets: cross-model packer byte check",
        )
        counts = {
            expected_target_count,
            len(target_found),
            len(actual_manifest),
            entry_count,
            structurally_valid,
            len(target_tensor_rows),
            int(production.get("target_tensors_found", -1)),
            int(sidecar_record.get("entries", -1)),
        }
        self.expect_exact(
            counts, {224}, "Mistral assets: all target/manifest/sidecar counts agree"
        )
        return (
            f"{len(index_keys)} indexed tensors, 224 targets, 224 manifest/sidecar entries, "
            f"GGUF types={dict(sorted(target_type_counts.items()))}"
        )

    @staticmethod
    def parse_greedy_runtime_evidence(
        stderr: bytes, sidecar_path: str
    ) -> dict[str, Any]:
        text = stderr.decode("utf-8", errors="replace")
        loads = [
            {
                "tensors": int(match.group("tensors")),
                "device_gib": float(match.group("gib")),
                "device": int(match.group("device")),
                "path": match.group("path").strip(),
            }
            for match in LOAD_RE.finditer(text)
        ]
        launches = [int(match.group("count")) for match in LAUNCH_RE.finditer(text)]
        unique_matches = list(GREEDY_UNIQUE_RE.finditer(text))
        unique_counts = [int(match.group("count")) for match in unique_matches]
        hit_summaries = [
            {
                match.group("shape"): int(match.group("count"))
                for match in HIT_RE.finditer(unique_match.group("hits") or "")
            }
            for unique_match in unique_matches
        ]
        q4_error_lines = [
            line
            for line in text.splitlines()
            if line.startswith("Q4_RDNA:")
            and any(
                word in line.lower()
                for word in (
                    "error",
                    "failed",
                    "invalid",
                    "mismatch",
                    "truncated",
                    "only supports",
                )
            )
        ]
        sampler_name_errors = [
            line
            for line in text.splitlines()
            if "unable to match sampler" in line.lower()
        ]
        sampler_chains = [
            match.group("chain").strip() for match in SAMPLER_CHAIN_RE.finditer(text)
        ]
        sampler_temperatures = [
            float(match.group("temperature"))
            for match in SAMPLER_TEMP_RE.finditer(text)
        ]
        sampler_seeds = [
            int(match.group("seed")) for match in SAMPLER_SEED_RE.finditer(text)
        ]
        exact_load = bool(
            len(loads) == 1
            and loads[0]["path"] == sidecar_path
            and loads[0]["tensors"] > 0
            and loads[0]["device_gib"] > 0
        )
        return {
            "sidecar_loads": loads,
            "exact_sidecar_loaded": exact_load,
            "launch_counts": launches,
            "launch_total": sum(launches),
            "unique_tensor_counts": unique_counts,
            "unique_tensor_maximum": max(unique_counts, default=0),
            "hit_summaries": hit_summaries,
            "q4rdna_error_lines": q4_error_lines,
            "sampler_name_errors": sampler_name_errors,
            "sampler_chains": sampler_chains,
            "sampler_temperatures": sampler_temperatures,
            "sampler_seeds": sampler_seeds,
        }

    @staticmethod
    def greedy_runtime_errors(
        evidence: dict[str, Any],
        expected_environment: dict[str, str],
        observed_environment: dict[str, str],
        seed: int,
    ) -> list[str]:
        errors: list[str] = []
        if observed_environment != expected_environment:
            errors.append(
                f"Q4_RDNA environment is {observed_environment!r}, expected {expected_environment!r}"
            )
        if not evidence["exact_sidecar_loaded"]:
            errors.append(
                "did not log exactly one positive sidecar load from the requested path"
            )
        if evidence["launch_total"] <= 0:
            errors.append("did not log a non-zero Q4_RDNA decode launch count")
        if evidence["unique_tensor_maximum"] <= 0:
            errors.append("did not log a non-zero Q4_RDNA unique tensor count")
        if evidence["q4rdna_error_lines"]:
            errors.append("Q4_RDNA emitted an error diagnostic")
        if evidence["sampler_name_errors"]:
            errors.append("llama-completion did not recognize the requested sampler")
        if not evidence["sampler_chains"]:
            errors.append("llama-completion did not log the effective sampler chain")
        elif any("temp" not in chain.lower() for chain in evidence["sampler_chains"]):
            errors.append(
                f"effective sampler chain does not contain temperature: {evidence['sampler_chains']!r}"
            )
        if not evidence["sampler_temperatures"]:
            errors.append("llama-completion did not log the effective temperature")
        elif any(
            temperature != 0.0 for temperature in evidence["sampler_temperatures"]
        ):
            errors.append(
                f"effective sampler temperature is not zero: {evidence['sampler_temperatures']!r}"
            )
        if evidence["sampler_seeds"] != [seed]:
            errors.append(
                f"effective sampler seed is {evidence['sampler_seeds']!r}, expected [{seed}]"
            )
        return errors

    def verify_greedy_completion_document(self, path: Path) -> dict[str, Any]:
        directory = path.parent
        label = directory.name
        document = read_json(path)
        self.expect_exact(
            document.get("schema"), GREEDY_COMPLETION_SCHEMA, f"{label}: schema"
        )
        self.expect_exact(
            document.get("derived_from_schema"),
            COMPLETION_SCHEMA,
            f"{label}: derived schema",
        )
        runtime = document.get("runtime")
        generation = document.get("generation")
        runs = document.get("runs")
        self.require(isinstance(runtime, dict), f"{label}: runtime must be an object")
        self.require(
            isinstance(generation, dict), f"{label}: generation must be an object"
        )
        self.require(
            isinstance(runs, list) and runs, f"{label}: runs must be non-empty"
        )

        local_provenance = 0
        for artifact, path_key, size_key, hash_key in (
            ("binary", "binary", "binary_size_bytes", "binary_sha256"),
            ("ggml-hip", "ggml_hip", "ggml_hip_size_bytes", "ggml_hip_sha256"),
            ("model", "model", "model_size_bytes", "model_sha256"),
            ("sidecar", "sidecar", "sidecar_size_bytes", "sidecar_sha256"),
        ):
            verified = self.verify_recorded_file(
                path_value=runtime.get(path_key),
                size_value=runtime.get(size_key),
                sha256_value=runtime.get(hash_key),
                label=f"{label} provenance {artifact}",
                missing_is_note=True,
            )
            local_provenance += verified is not None

        model_label = generation.get("model_label")
        self.require(
            isinstance(model_label, str) and bool(model_label.strip()),
            f"{label}: generation.model_label must be non-empty",
        )
        prompts = generation.get("prompts")
        self.require(
            isinstance(prompts, list)
            and prompts
            and all(isinstance(prompt, str) and prompt for prompt in prompts),
            f"{label}: generation.prompts must be non-empty strings",
        )
        self.expect_exact(len(runs), len(prompts), f"{label}: run/prompt count")
        self.expect_exact(
            generation.get("samplers"), "greedy", f"{label}: sampler classification"
        )
        self.expect(
            isinstance(generation.get("sampler_implementation"), str)
            and "temperature 0" in generation["sampler_implementation"].lower()
            and "argmax" in generation["sampler_implementation"].lower(),
            f"{label}: sampler implementation must state temperature-0 argmax",
        )
        temperature = generation.get("temperature")
        self.expect(
            isinstance(temperature, (int, float))
            and not isinstance(temperature, bool)
            and float(temperature) == 0.0,
            f"{label}: generation temperature must be exactly zero",
        )
        seed = generation.get("seed")
        n_predict = generation.get("n_predict")
        threads = generation.get("threads")
        gpu_layers = generation.get("gpu_layers")
        self.require(
            isinstance(seed, int) and not isinstance(seed, bool),
            f"{label}: seed must be an integer",
        )
        self.require(
            isinstance(n_predict, int)
            and not isinstance(n_predict, bool)
            and n_predict > 0,
            f"{label}: n_predict must be positive",
        )
        self.require(
            isinstance(threads, int) and not isinstance(threads, bool) and threads > 0,
            f"{label}: threads must be positive",
        )
        self.require(bool(str(gpu_layers)), f"{label}: gpu_layers must be non-empty")
        common_arguments = [
            "--samplers",
            "temperature",
            "--temp",
            "0",
            "--seed",
            str(seed),
            "-n",
            str(n_predict),
            "--no-display-prompt",
            "--no-conversation",
            "--simple-io",
        ]
        self.expect_exact(
            generation.get("common_arguments"),
            common_arguments,
            f"{label}: common command arguments",
        )
        expected_old_environment = {
            "LLAMA_Q4_RDNA_MAPPING": "old",
            "LLAMA_Q4_RDNA_SCOPE": None,
            "LLAMA_Q4_RDNA_COOP": None,
            "LLAMA_Q4_RDNA_SMALL_ROWS": None,
        }
        expected_split_environment = {
            "LLAMA_Q4_RDNA_MAPPING": None,
            "LLAMA_Q4_RDNA_SCOPE": None,
            "LLAMA_Q4_RDNA_COOP": None,
            "LLAMA_Q4_RDNA_SMALL_ROWS": None,
        }
        expected_shared_environment = {
            "LLAMA_Q4_RDNA_TRACE": "1",
            "LLAMA_Q4_RDNA_SIDECAR": runtime["sidecar"],
        }
        self.expect_equal(
            generation.get("old_environment"),
            expected_old_environment,
            f"{label}: old environment contract",
        )
        self.expect_equal(
            generation.get("split_environment"),
            expected_split_environment,
            f"{label}: split environment contract",
        )
        self.expect_equal(
            generation.get("shared_environment"),
            expected_shared_environment,
            f"{label}: shared environment contract",
        )
        self.expect_exact(
            runtime.get("hip_visible_devices"),
            runtime.get("rocr_visible_devices"),
            f"{label}: visible devices",
        )

        stdout_paths: set[Path] = set()
        stderr_paths: set[Path] = set()
        recomputed_runs: list[dict[str, Any]] = []
        runtime_routes_valid = 0
        for index, run in enumerate(runs, start=1):
            run_id = f"prompt{index}"
            prompt = prompts[index - 1]
            self.expect_exact(run.get("id"), run_id, f"{label}: run id {index}")
            self.expect_exact(run.get("prompt"), prompt, f"{label}: run prompt {index}")
            route_data: dict[str, dict[str, Any]] = {}
            expected_command = [
                runtime["binary"],
                "-m",
                runtime["model"],
                "-ngl",
                str(gpu_layers),
                "-t",
                str(threads),
                *common_arguments,
                "-p",
                prompt,
            ]
            for route in ("old", "split"):
                stdout_path = self.safe_relative_path(
                    directory,
                    run[f"{route}_stdout"],
                    f"{label} {run_id} {route} stdout",
                )
                stderr_path = self.safe_relative_path(
                    directory,
                    run[f"{route}_stderr"],
                    f"{label} {run_id} {route} stderr",
                )
                stdout_paths.add(stdout_path)
                stderr_paths.add(stderr_path)
                stdout = stdout_path.read_bytes()
                stderr = stderr_path.read_bytes()
                evidence = self.parse_greedy_runtime_evidence(
                    stderr, runtime["sidecar"]
                )
                expected_environment = dict(expected_shared_environment)
                if route == "old":
                    expected_environment["LLAMA_Q4_RDNA_MAPPING"] = "old"
                validation = run[f"{route}_runtime_validation"]
                self.require(
                    isinstance(validation, dict),
                    f"{label} {run_id} {route}: validation must be an object",
                )
                self.expect_equal(
                    validation.get("expected_q4rdna_environment"),
                    expected_environment,
                    f"{label} {run_id} {route}: expected controlled environment",
                )
                self.expect_equal(
                    validation.get("observed_q4rdna_environment"),
                    expected_environment,
                    f"{label} {run_id} {route}: observed controlled environment",
                )
                self.expect_equal(
                    validation.get("runtime_evidence"),
                    evidence,
                    f"{label} {run_id} {route}: recomputed stderr evidence",
                )
                runtime_errors = self.greedy_runtime_errors(
                    evidence,
                    expected_environment,
                    validation.get("observed_q4rdna_environment", {}),
                    seed,
                )
                self.expect(
                    not runtime_errors,
                    f"{label} {run_id} {route}: runtime errors {runtime_errors}",
                )
                self.expect_exact(
                    validation.get("command"),
                    expected_command,
                    f"{label} {run_id} {route}: exact command",
                )
                self.expect_exact(
                    validation.get("timed_out"),
                    False,
                    f"{label} {run_id} {route}: timeout",
                )
                self.expect_exact(
                    run.get(f"{route}_exit_code"),
                    0,
                    f"{label} {run_id} {route}: exit code",
                )
                expected_route_pass = bool(
                    not runtime_errors
                    and run.get(f"{route}_exit_code") == 0
                    and not validation.get("timed_out")
                    and stdout
                )
                runtime_routes_valid += expected_route_pass
                self.expect_exact(
                    validation.get("passed"),
                    expected_route_pass,
                    f"{label} {run_id} {route}: route pass",
                )
                self.expect_exact(
                    validation.get("errors"),
                    [],
                    f"{label} {run_id} {route}: route validation errors",
                )
                self.expect(
                    isinstance(validation.get("started_at"), str)
                    and bool(validation["started_at"]),
                    f"{label} {run_id} {route}: started_at missing",
                )
                self.expect(
                    isinstance(validation.get("duration_seconds"), (int, float))
                    and float(validation["duration_seconds"]) >= 0,
                    f"{label} {run_id} {route}: invalid duration",
                )
                route_data[route] = {
                    "stdout": stdout,
                    "stderr": stderr,
                    "evidence": evidence,
                    "passed": expected_route_pass,
                }

            old_stdout = route_data["old"]["stdout"]
            split_stdout = route_data["split"]["stdout"]
            difference = first_byte_difference(old_stdout, split_stdout)
            byte_identical = difference is None
            runtime_valid = (
                route_data["old"]["passed"] and route_data["split"]["passed"]
            )
            pair_errors = (
                []
                if difference is None
                else [f"stdout differs at byte offset {difference['offset']}"]
            )
            old_evidence = route_data["old"]["evidence"]
            split_evidence = route_data["split"]["evidence"]
            expected_pair_fields = {
                "byte_identical": byte_identical,
                "first_difference": difference,
                "runtime_valid": runtime_valid,
                "passed": runtime_valid and byte_identical,
                "errors": pair_errors,
                "stdout_bytes": len(old_stdout),
                "split_stdout_bytes": len(split_stdout),
                "stdout_sha256": hashlib.sha256(old_stdout).hexdigest(),
                "old_stdout_sha256": hashlib.sha256(old_stdout).hexdigest(),
                "split_stdout_sha256": hashlib.sha256(split_stdout).hexdigest(),
                "old_stderr_sha256": hashlib.sha256(
                    route_data["old"]["stderr"]
                ).hexdigest(),
                "split_stderr_sha256": hashlib.sha256(
                    route_data["split"]["stderr"]
                ).hexdigest(),
                "old_q4rdna_launches": old_evidence["launch_total"],
                "split_q4rdna_launches": split_evidence["launch_total"],
                "old_unique_tensors": old_evidence["unique_tensor_maximum"],
                "split_unique_tensors": split_evidence["unique_tensor_maximum"],
                "old_sidecar_loaded": old_evidence["exact_sidecar_loaded"],
                "split_sidecar_loaded": split_evidence["exact_sidecar_loaded"],
            }
            for field, expected in expected_pair_fields.items():
                self.expect_equal(
                    run.get(field), expected, f"{label} {run_id}: recomputed {field}"
                )
            recomputed_runs.append(
                {
                    "id": run_id,
                    "byte_identical": byte_identical,
                    "difference": difference,
                    "runtime_valid": runtime_valid,
                    "old_evidence": old_evidence,
                    "split_evidence": split_evidence,
                }
            )

        self.expect_exact(
            {item.resolve() for item in (directory / "raw").glob("*.stdout")},
            stdout_paths,
            f"{label}: raw stdout file set",
        )
        self.expect_exact(
            {item.resolve() for item in (directory / "raw").glob("*.stderr")},
            stderr_paths,
            f"{label}: raw stderr file set",
        )
        all_identical = all(run["byte_identical"] for run in recomputed_runs)
        all_runtime_valid = all(run["runtime_valid"] for run in recomputed_runs)
        first_difference = next(
            (
                {"run": run["id"], **run["difference"]}
                for run in recomputed_runs
                if run["difference"] is not None
            ),
            None,
        )
        self.expect_exact(
            document.get("all_byte_identical"),
            all_identical,
            f"{label}: all byte identical",
        )
        self.expect_exact(
            document.get("all_runtime_valid"),
            all_runtime_valid,
            f"{label}: all runtime valid",
        )
        self.expect_equal(
            document.get("first_difference"),
            first_difference,
            f"{label}: first difference",
        )
        expected_verdict = "pass" if all_identical and all_runtime_valid else "fail"
        self.expect_exact(
            document.get("verdict"), expected_verdict, f"{label}: recomputed verdict"
        )

        observations: list[dict[str, Any]] = []
        for run in recomputed_runs:
            for route in ("old", "split"):
                evidence = run[f"{route}_evidence"]
                load = (
                    evidence["sidecar_loads"][0]
                    if len(evidence["sidecar_loads"]) == 1
                    else None
                )
                hits = (
                    evidence["hit_summaries"][0]
                    if len(evidence["hit_summaries"]) == 1
                    else None
                )
                observations.append(
                    {
                        "run": run["id"],
                        "route": route,
                        "loaded_tensors": load["tensors"] if load else None,
                        "loaded_device_gib": load["device_gib"] if load else None,
                        "unique_tensors_hit": evidence["unique_tensor_maximum"],
                        "hits_by_shape": hits,
                    }
                )
        signatures = {
            json.dumps(
                {
                    "loaded_tensors": observation["loaded_tensors"],
                    "loaded_device_gib": observation["loaded_device_gib"],
                    "unique_tensors_hit": observation["unique_tensors_hit"],
                    "hits_by_shape": observation["hits_by_shape"],
                },
                sort_keys=True,
            )
            for observation in observations
        }
        first = observations[0] if observations else {}
        expected_launch_summary = {
            "loaded_tensors": first.get("loaded_tensors"),
            "loaded_device_gib": first.get("loaded_device_gib"),
            "unique_tensors_hit": first.get("unique_tensors_hit"),
            "hits_by_shape": first.get("hits_by_shape"),
            "consistent_across_runs": len(signatures) == 1,
            "observations": observations,
        }
        self.expect_equal(
            document.get("launch_summary_per_run"),
            expected_launch_summary,
            f"{label}: recomputed launch summary",
        )
        return {
            "model_label": model_label,
            "prompts": len(recomputed_runs),
            "exact_prompts": sum(run["byte_identical"] for run in recomputed_runs),
            "model_exact": all_identical,
            "runtime_routes": len(recomputed_runs) * 2,
            "runtime_routes_valid": runtime_routes_valid,
            "local_provenance": local_provenance,
        }

    def verify_all_greedy_completions(self) -> str:
        documents = sorted(
            self.results_root.glob(
                "completion_equivalence_*_greedy/completion_equivalence.json"
            )
        )
        self.require(
            len(documents) >= 2,
            "greedy completion v2: expected at least Qwen3 and Mistral documents",
        )
        results = [self.verify_greedy_completion_document(path) for path in documents]
        labels = [str(result["model_label"]) for result in results]
        self.expect_exact(
            len(set(labels)), len(labels), "greedy completion v2: unique model labels"
        )
        normalized = [re.sub(r"[^a-z0-9]", "", label.lower()) for label in labels]
        self.expect(
            any("qwen3" in label for label in normalized),
            "greedy completion v2: missing Qwen3 model label",
        )
        self.expect(
            any("mistral" in label for label in normalized),
            "greedy completion v2: missing Mistral model label",
        )
        prompts = sum(int(result["prompts"]) for result in results)
        exact_prompts = sum(int(result["exact_prompts"]) for result in results)
        exact_models = sum(bool(result["model_exact"]) for result in results)
        runtime_routes = sum(int(result["runtime_routes"]) for result in results)
        runtime_valid = sum(int(result["runtime_routes_valid"]) for result in results)
        local_provenance = sum(int(result["local_provenance"]) for result in results)
        self.expect_exact(
            runtime_valid,
            runtime_routes,
            "greedy completion v2: all runtime routes valid",
        )
        return (
            f"evidence-integrity verified; greedy exact={exact_prompts}/{prompts} prompts, "
            f"{exact_models}/{len(results)} models; runtime={runtime_valid}/{runtime_routes} routes; "
            f"local provenance entries={local_provenance}"
        )

    def verify_completion_equivalence(self) -> str:
        directory = self.results_root / "completion_equivalence"
        path = directory / "completion_equivalence.json"
        self.require(path.is_file(), f"completion: missing {path}")
        document = read_json(path)
        self.expect_exact(
            document.get("schema"), COMPLETION_SCHEMA, "completion: schema"
        )
        runs = document.get("runs")
        self.require(
            isinstance(runs, list) and runs, "completion: runs must be non-empty"
        )
        identities: list[bool] = []
        stdout_paths: set[Path] = set()
        stderr_paths: set[Path] = set()
        sampler_name_warnings: list[str] = []
        sampler_chains: list[str] = []
        sampler_temperatures: list[float] = []
        sampler_seeds: list[int] = []
        for run in runs:
            run_id = run["id"]
            old_stdout = self.safe_relative_path(
                directory, run["old_stdout"], f"completion {run_id} old stdout"
            )
            split_stdout = self.safe_relative_path(
                directory, run["split_stdout"], f"completion {run_id} split stdout"
            )
            old_stderr = self.safe_relative_path(
                directory, run["old_stderr"], f"completion {run_id} old stderr"
            )
            split_stderr = self.safe_relative_path(
                directory, run["split_stderr"], f"completion {run_id} split stderr"
            )
            stdout_paths.update((old_stdout, split_stdout))
            stderr_paths.update((old_stderr, split_stderr))
            old_bytes = old_stdout.read_bytes()
            split_bytes = split_stdout.read_bytes()
            identical = old_bytes == split_bytes
            identities.append(identical)
            self.expect_exact(
                run.get("byte_identical"),
                identical,
                f"completion {run_id}: byte equality",
            )
            self.expect_exact(
                run.get("stdout_bytes"),
                len(old_bytes),
                f"completion {run_id}: stdout bytes",
            )
            self.expect_exact(
                run.get("stdout_sha256"),
                hashlib.sha256(old_bytes).hexdigest(),
                f"completion {run_id}: stdout SHA-256",
            )
            self.expect_exact(
                run.get("old_stderr_sha256"),
                sha256_file(old_stderr),
                f"completion {run_id}: old stderr SHA-256",
            )
            self.expect_exact(
                run.get("split_stderr_sha256"),
                sha256_file(split_stderr),
                f"completion {run_id}: split stderr SHA-256",
            )
            old_stderr_text = old_stderr.read_text(encoding="utf-8", errors="replace")
            split_stderr_text = split_stderr.read_text(
                encoding="utf-8", errors="replace"
            )
            for stderr_text in (old_stderr_text, split_stderr_text):
                warnings = [
                    line
                    for line in stderr_text.splitlines()
                    if "unable to match sampler" in line.lower()
                ]
                chains = [
                    match.group("chain").strip()
                    for match in SAMPLER_CHAIN_RE.finditer(stderr_text)
                ]
                temperatures = [
                    float(match.group("temperature"))
                    for match in SAMPLER_TEMP_RE.finditer(stderr_text)
                ]
                seeds = [
                    int(match.group("seed"))
                    for match in SAMPLER_SEED_RE.finditer(stderr_text)
                ]
                self.expect(
                    bool(warnings),
                    f"completion {run_id}: legacy sampler-name warning missing",
                )
                self.expect_exact(
                    chains,
                    ["logits -> dist"],
                    f"completion {run_id}: legacy effective sampler chain",
                )
                self.expect(
                    bool(temperatures) and all(value != 0.0 for value in temperatures),
                    f"completion {run_id}: legacy distribution temperature was not non-zero",
                )
                sampler_name_warnings.extend(warnings)
                sampler_chains.extend(chains)
                sampler_temperatures.extend(temperatures)
                sampler_seeds.extend(seeds)
            old_launches = sum(
                int(value) for value in LAUNCH_RE.findall(old_stderr_text)
            )
            split_launches = sum(
                int(value) for value in LAUNCH_RE.findall(split_stderr_text)
            )
            self.expect_exact(
                run.get("old_q4rdna_launches"),
                old_launches,
                f"completion {run_id}: old launches",
            )
            self.expect_exact(
                run.get("split_q4rdna_launches"),
                split_launches,
                f"completion {run_id}: split launches",
            )
            self.expect_exact(
                run.get("old_sidecar_loaded"),
                bool(LOAD_RE.search(old_stderr_text)),
                f"completion {run_id}: old load marker",
            )
            self.expect_exact(
                run.get("split_sidecar_loaded"),
                bool(LOAD_RE.search(split_stderr_text)),
                f"completion {run_id}: split load marker",
            )
            self.expect_exact(
                run.get("old_exit_code"), 0, f"completion {run_id}: old exit code"
            )
            self.expect_exact(
                run.get("split_exit_code"), 0, f"completion {run_id}: split exit code"
            )

        self.expect_exact(
            {path.resolve() for path in (directory / "raw").glob("*.stdout")},
            stdout_paths,
            "completion: stdout file set",
        )
        self.expect_exact(
            {path.resolve() for path in (directory / "raw").glob("*.stderr")},
            stderr_paths,
            "completion: stderr file set",
        )
        all_identical = all(identities)
        self.expect_exact(
            document.get("all_byte_identical"),
            all_identical,
            "completion: aggregate equality",
        )
        self.expect_exact(
            document.get("first_difference"), None, "completion: first difference"
        )
        expected_verdict = (
            "pass"
            if all_identical
            and all(
                run.get("old_exit_code") == run.get("split_exit_code") == 0
                for run in runs
            )
            else "fail"
        )
        self.expect_exact(
            document.get("verdict"), expected_verdict, "completion: verdict"
        )
        generation = document.get("generation")
        self.require(
            isinstance(generation, dict),
            "completion legacy: generation must be an object",
        )
        seed = generation.get("seed")
        self.expect_exact(
            generation.get("samplers"),
            "seeded distribution",
            "completion legacy: classification",
        )
        self.expect_exact(
            generation.get("requested_sampler"),
            "greedy",
            "completion legacy: requested sampler",
        )
        self.expect_exact(
            generation.get("requested_sampler_recognized"),
            False,
            "completion legacy: sampler recognition",
        )
        self.expect_exact(
            generation.get("effective_sampler_chain"),
            "logits -> dist",
            "completion legacy: declared sampler chain",
        )
        self.expect_exact(
            generation.get("is_greedy"),
            False,
            "completion legacy: greedy classification",
        )
        self.expect(
            bool(sampler_name_warnings),
            "completion legacy: no raw sampler-name warnings were observed",
        )
        self.expect_exact(
            set(sampler_chains),
            {"logits -> dist"},
            "completion legacy: raw sampler chains",
        )
        self.expect(
            bool(sampler_temperatures)
            and all(value != 0.0 for value in sampler_temperatures),
            "completion legacy: raw temperatures do not prove seeded distribution",
        )
        self.expect(
            isinstance(seed, int) and sampler_seeds == [seed] * (2 * len(runs)),
            "completion legacy: raw sampler seeds do not match the declared seed",
        )
        self.expect_exact(
            document.get("protocol_status"),
            "legacy-seeded-distribution-superseded-by-v2-greedy",
            "completion legacy: protocol status",
        )
        protocol_note = document.get("protocol_note")
        self.expect(
            isinstance(protocol_note, str)
            and "not greedy evidence" in protocol_note.lower()
            and "logits -> dist" in protocol_note,
            "completion legacy: protocol note must explain why this is not greedy evidence",
        )
        superseded_by = document.get("superseded_by")
        self.require(
            isinstance(superseded_by, str) and bool(superseded_by),
            "completion legacy: superseded_by must be non-empty",
        )
        superseding_path = (directory / superseded_by).resolve()
        self.require(
            superseding_path.is_relative_to(self.results_root)
            and superseding_path.is_file(),
            f"completion legacy: missing/out-of-root superseding artifact {superseding_path}",
        )
        superseding = read_json(superseding_path)
        self.expect_exact(
            superseding.get("schema"),
            GREEDY_COMPLETION_SCHEMA,
            "completion legacy: superseding schema",
        )
        self.expect(
            "qwen3"
            in re.sub(
                r"[^a-z0-9]",
                "",
                str(superseding.get("generation", {}).get("model_label", "")).lower(),
            ),
            "completion legacy: superseding artifact is not the Qwen3 greedy result",
        )
        self.notes.append(
            "legacy completion_equivalence v1 is byte-equal under a seeded distribution, "
            "but raw sampler-name warnings and 'logits -> dist' exclude it from greedy counts"
        )
        return (
            f"{sum(identities)}/{len(runs)} stdout pairs byte-identical; "
            "classification=legacy seeded-distribution; greedy credit=0"
        )

    def verify_aiter_source_manifest(
        self, source: Any, label: str
    ) -> list[tuple[str, str]]:
        self.require(
            isinstance(source, dict), f"{label}: source manifest must be an object"
        )
        files = source.get("files")
        self.require(
            isinstance(files, list) and bool(files), f"{label}: files must be non-empty"
        )
        repository_value = source.get("repository")
        self.require(
            isinstance(repository_value, str) and bool(repository_value),
            f"{label}: repository must be non-empty",
        )
        repository = Path(repository_value)
        canonical_entries: list[tuple[str, str]] = []
        for index, record in enumerate(files):
            item_label = f"{label}: file {index}"
            self.require(
                isinstance(record, dict), f"{item_label}: record must be an object"
            )
            path_value = record.get("path")
            digest = record.get("sha256")
            self.require(
                isinstance(path_value, str) and bool(path_value), f"{item_label}: path"
            )
            self.require(
                isinstance(digest, str)
                and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                f"{item_label}: SHA-256",
            )
            candidate = Path(path_value)
            if candidate.is_absolute():
                self.require(
                    repository.is_absolute(),
                    f"{item_label}: absolute path with relative repository",
                )
                try:
                    relative = candidate.relative_to(repository)
                except ValueError:
                    self.require(
                        False, f"{item_label}: path is outside source repository"
                    )
                    raise AssertionError("unreachable")
            else:
                relative = candidate
            self.require(
                relative != Path(".") and ".." not in relative.parts,
                f"{item_label}: invalid relative path {relative}",
            )
            canonical_entries.append((relative.as_posix(), digest))
        self.expect_exact(
            len({path for path, _digest in canonical_entries}),
            len(canonical_entries),
            f"{label}: unique source paths",
        )
        payload = "".join(
            f"{digest}  {relative_path}\n"
            for relative_path, digest in canonical_entries
        ).encode("utf-8")
        aggregate = hashlib.sha256(payload).hexdigest()
        self.expect_exact(
            source.get("aggregate_candidate_files_sha256"),
            aggregate,
            f"{label}: canonical candidate-file manifest aggregate",
        )
        method = source.get("aggregate_hash_method")
        self.expect(
            isinstance(method, str)
            and "UTF-8 concatenation" in method
            and "<sha256>  <relative_path>\\n" in method
            and "listed order" in method,
            f"{label}: canonical aggregate algorithm declaration",
        )
        return canonical_entries

    def verify_aiter_patch(
        self,
        directory: Path,
        metadata_source: dict[str, Any],
        validation_source: dict[str, Any],
        manifest: Sequence[tuple[str, str]],
    ) -> str:
        metadata_patch = metadata_source.get("patch")
        validation_patch = validation_source.get("patch")
        self.require(
            isinstance(metadata_patch, dict),
            "AITER patch: metadata source.patch missing",
        )
        self.require(
            isinstance(validation_patch, dict),
            "AITER patch: container source.patch missing",
        )
        self.expect_equal(
            metadata_patch, validation_patch, "AITER patch: provenance records"
        )
        self.expect_exact(
            metadata_patch.get("applies_to_upstream_base_commit"),
            metadata_source.get("upstream_base_commit"),
            "AITER patch: metadata upstream-base binding",
        )
        self.expect_exact(
            validation_patch.get("applies_to_upstream_base_commit"),
            validation_source.get("upstream_base_commit"),
            "AITER patch: container upstream-base binding",
        )
        self.expect_exact(
            metadata_patch.get("candidate_commit"),
            metadata_source.get("candidate_commit"),
            "AITER patch: metadata candidate-commit binding",
        )
        self.expect_exact(
            validation_patch.get("candidate_commit"),
            validation_source.get("candidate_commit"),
            "AITER patch: container candidate-commit binding",
        )
        self.expect_exact(
            metadata_patch.get("format"), "git-diff-binary", "AITER patch: format"
        )
        patch_value = metadata_patch.get("path")
        self.require(
            isinstance(patch_value, str) and bool(patch_value),
            "AITER patch: path must be non-empty",
        )
        patch_path = self.safe_relative_path(directory, patch_value, "AITER patch")
        self.expect_exact(
            patch_path.stat().st_size,
            metadata_patch.get("size_bytes"),
            "AITER patch: size",
        )
        patch_sha256 = self.cached_sha256(patch_path)
        self.expect_exact(
            patch_sha256, metadata_patch.get("sha256"), "AITER patch: SHA-256"
        )
        patch_bytes = patch_path.read_bytes()
        self.expect(
            b"\x00" not in patch_bytes, "AITER patch: unexpected NUL/binary payload"
        )
        try:
            patch_text = patch_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            self.require(False, f"AITER patch: not UTF-8: {error}")
            raise AssertionError("unreachable")
        self.expect(
            "/home/" not in patch_text and "C:\\" not in patch_text,
            "AITER patch: contains an absolute developer path",
        )
        self.expect(patch_text.endswith("\n"), "AITER patch: missing final newline")
        headers = re.findall(r"^diff --git a/(.+) b/(.+)$", patch_text, re.MULTILINE)
        self.require(bool(headers), "AITER patch: no git diff headers")
        self.expect(
            all(left == right for left, right in headers),
            "AITER patch: rename/cross-path entries are not expected",
        )
        patch_paths = [left for left, right in headers if left == right]
        manifest_paths = [path for path, _digest in manifest]
        manifest_hashes = dict(manifest)
        self.expect_exact(
            patch_paths, manifest_paths, "AITER patch: ordered path coverage"
        )
        self.expect_exact(
            len(set(patch_paths)), len(patch_paths), "AITER patch: unique paths"
        )
        for marker in (
            "from .ops.q4_group64_gemv import *",
            '"module_q4_group64_gemv"',
            "csrc/pybind/q4_group64_gemv_pybind.cu",
            "csrc/kernels/q4_group64_gemv.cu",
        ):
            self.expect(
                marker in patch_text, f"AITER patch: public/JIT marker {marker}"
            )
        for relative_path in patch_paths:
            block_pattern = re.compile(
                rf"^diff --git a/{re.escape(relative_path)} b/{re.escape(relative_path)}$",
                re.MULTILINE,
            )
            self.expect_exact(
                len(block_pattern.findall(patch_text)),
                1,
                f"AITER patch {relative_path}: diff block",
            )
        blocks = re.split(r"(?=^diff --git a/)", patch_text, flags=re.MULTILINE)
        self.aiter_patch_postimages = {}
        for block in blocks:
            header = re.match(r"^diff --git a/(.+) b/(.+)$", block, re.MULTILINE)
            if header is None:
                continue
            self.expect(
                re.search(
                    r"^index [0-9a-f]{40}\.\.[0-9a-f]{40}(?: [0-7]{6})?$",
                    block,
                    re.MULTILINE,
                )
                is not None,
                f"AITER patch {header.group(1)}: full Git blob IDs",
            )
            if "\nnew file mode " not in block:
                continue
            relative_path = header.group(1)
            in_hunk = False
            reconstructed: list[bytes] = []
            for line in block.splitlines(keepends=True):
                if line.startswith("@@ "):
                    in_hunk = True
                    continue
                if not in_hunk:
                    continue
                if line.startswith("+") and not line.startswith("+++"):
                    reconstructed.append(line[1:].encode("utf-8"))
                elif line.startswith("\\ No newline at end of file") and reconstructed:
                    reconstructed[-1] = reconstructed[-1].rstrip(b"\r\n")
                elif line.startswith((" ", "-")):
                    self.expect(
                        False,
                        f"AITER patch {relative_path}: non-addition in new-file hunk",
                    )
            postimage = b"".join(reconstructed)
            self.aiter_patch_postimages[relative_path] = postimage
            self.expect_exact(
                hashlib.sha256(postimage).hexdigest(),
                manifest_hashes.get(relative_path),
                f"AITER patch {relative_path}: reconstructed post-image SHA-256",
            )
        self.expect_exact(
            metadata_patch.get("post_image_manifest_sha256"),
            metadata_source.get("aggregate_candidate_files_sha256"),
            "AITER patch: post-image manifest binding",
        )

        metadata_format_patch = metadata_source.get("dco_format_patch")
        validation_format_patch = validation_source.get("dco_format_patch")
        self.require(
            isinstance(metadata_format_patch, dict),
            "AITER DCO format-patch: metadata record missing",
        )
        self.require(
            isinstance(validation_format_patch, dict),
            "AITER DCO format-patch: container record missing",
        )
        self.expect_equal(
            metadata_format_patch,
            validation_format_patch,
            "AITER DCO format-patch: provenance records",
        )
        format_patch_value = metadata_format_patch.get("path")
        self.require(
            isinstance(format_patch_value, str) and bool(format_patch_value),
            "AITER DCO format-patch: path",
        )
        format_patch_path = self.safe_relative_path(
            directory, format_patch_value, "AITER DCO format-patch"
        )
        self.expect_exact(
            format_patch_path.stat().st_size,
            metadata_format_patch.get("size_bytes"),
            "AITER DCO format-patch: size",
        )
        self.expect_exact(
            self.cached_sha256(format_patch_path),
            metadata_format_patch.get("sha256"),
            "AITER DCO format-patch: SHA-256",
        )
        format_patch_text = format_patch_path.read_text(encoding="utf-8")
        candidate_commit = metadata_source.get("candidate_commit")
        signed_off_by = metadata_format_patch.get("signed_off_by")
        self.expect_exact(
            metadata_format_patch.get("candidate_commit"),
            candidate_commit,
            "AITER DCO format-patch: candidate commit",
        )
        self.expect(
            isinstance(candidate_commit, str)
            and format_patch_text.startswith(f"From {candidate_commit} "),
            "AITER DCO format-patch: From commit",
        )
        message_separator = format_patch_text.find("\n---\n")
        self.require(
            message_separator > 0,
            "AITER DCO format-patch: missing commit-message/diffstat separator",
        )
        commit_message_region = format_patch_text[:message_separator]
        self.expect(
            isinstance(signed_off_by, str)
            and f"Signed-off-by: {signed_off_by}" in commit_message_region,
            "AITER DCO format-patch: Signed-off-by trailer",
        )
        candidate_subject = metadata_source.get("candidate_commit_subject")
        self.expect(
            isinstance(candidate_subject, str)
            and f"Subject: [PATCH] {candidate_subject}\n" in format_patch_text,
            "AITER DCO format-patch: subject",
        )
        format_patch_paths = [
            left
            for left, right in re.findall(
                r"^diff --git a/(.+) b/(.+)$", format_patch_text, re.MULTILINE
            )
            if left == right
        ]
        self.expect_exact(
            format_patch_paths,
            patch_paths,
            "AITER DCO format-patch: path coverage/order",
        )
        format_diff_start = format_patch_text.find("diff --git ")
        format_diff_end = format_patch_text.rfind("\n-- \n")
        self.require(
            format_diff_start >= 0 and format_diff_end > format_diff_start,
            "AITER DCO format-patch: cannot isolate diff body",
        )
        self.expect_exact(
            format_patch_text[format_diff_start : format_diff_end + 1],
            patch_text,
            "AITER DCO format-patch: diff body equals authoritative patch",
        )

        strict_detail = "public patch-only"
        if self.aiter_source is not None:
            self.require(
                self.aiter_source.is_dir(),
                f"AITER strict source missing: {self.aiter_source}",
            )
            git_directory = self.aiter_source / ".git"
            self.require(
                git_directory.exists(),
                f"AITER strict source is not a Git worktree: {self.aiter_source}",
            )
            try:
                revision = subprocess.run(
                    ["git", "-C", str(self.aiter_source), "rev-parse", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError) as error:
                self.require(
                    False, f"AITER strict source: cannot resolve HEAD: {error}"
                )
                raise AssertionError("unreachable")
            upstream_base = metadata_source.get("upstream_base_commit")
            candidate = metadata_source.get("candidate_commit")
            self.expect(
                revision in {upstream_base, candidate},
                "AITER strict source: HEAD must be either the upstream base with the patch "
                "applied or the recorded candidate commit",
            )
            if revision == candidate:
                parent = subprocess.run(
                    ["git", "-C", str(self.aiter_source), "rev-parse", "HEAD^"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                self.expect_exact(
                    parent, upstream_base, "AITER strict source: candidate parent"
                )
                status = subprocess.run(
                    ["git", "-C", str(self.aiter_source), "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                self.expect_exact(
                    status, "", "AITER strict source: candidate worktree clean"
                )
                commit_message = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(self.aiter_source),
                        "show",
                        "-s",
                        "--format=%B",
                        "HEAD",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                self.expect(
                    isinstance(signed_off_by, str)
                    and f"Signed-off-by: {signed_off_by}" in commit_message,
                    "AITER strict source: DCO trailer",
                )
                commit_subject = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(self.aiter_source),
                        "show",
                        "-s",
                        "--format=%s",
                        "HEAD",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                self.expect_exact(
                    commit_subject,
                    candidate_subject,
                    "AITER strict source: candidate subject",
                )
                committed_diff = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(self.aiter_source),
                        "diff",
                        "--binary",
                        "--full-index",
                        str(upstream_base),
                        str(candidate),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                self.expect_exact(
                    committed_diff,
                    patch_text,
                    "AITER strict source: committed base-to-candidate diff equals patch",
                )
            else:
                status = subprocess.run(
                    ["git", "-C", str(self.aiter_source), "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                dirty_paths = {
                    line[3:]
                    for line in status.splitlines()
                    if len(line) >= 4 and " -> " not in line[3:]
                }
                self.expect_exact(
                    dirty_paths,
                    set(patch_paths),
                    "AITER strict patched-base source: no missing or extra dirty paths",
                )
            reverse_check = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.aiter_source),
                    "apply",
                    "--check",
                    "--reverse",
                    str(patch_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.expect(
                reverse_check.returncode == 0,
                "AITER strict source: patch is not the exact applied post-image: "
                + (reverse_check.stderr.strip() or reverse_check.stdout.strip()),
            )
            strict_detail = (
                "strict committed candidate"
                if revision == candidate
                else "strict patched-base post-image"
            )
        return f"{patch_sha256[:12]} ({len(patch_paths)} paths; {strict_detail})"

    def verify_aiter_zero_dimension_contract(
        self, directory: Path, source: dict[str, Any]
    ) -> None:
        def verify_reference_guards(text: str, filename: str, label: str) -> None:
            tree = ast.parse(text, filename=filename)
            functions = [
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "pack_group64"
            ]
            self.require(
                len(functions) == 1, f"{label}: expected one pack_group64 function"
            )
            function = functions[0]

            def has_guard(variable: str, message: str) -> bool:
                for node in function.body:
                    if not isinstance(node, ast.If) or not isinstance(
                        node.test, ast.Compare
                    ):
                        continue
                    test = node.test
                    predicate_matches = (
                        isinstance(test.left, ast.Name)
                        and test.left.id == variable
                        and len(test.ops) == 1
                        and isinstance(test.ops[0], ast.LtE)
                        and len(test.comparators) == 1
                        and isinstance(test.comparators[0], ast.Constant)
                        and test.comparators[0].value == 0
                    )
                    if not predicate_matches:
                        continue
                    for body_node in node.body:
                        if not isinstance(body_node, ast.Raise) or not isinstance(
                            body_node.exc, ast.Call
                        ):
                            continue
                        exception = body_node.exc
                        if (
                            isinstance(exception.func, ast.Name)
                            and exception.func.id == "ValueError"
                            and len(exception.args) == 1
                            and isinstance(exception.args[0], ast.Constant)
                            and exception.args[0].value == message
                        ):
                            return True
                return False

            self.expect(
                has_guard("n", "N must be positive"), f"{label}: explicit N <= 0 guard"
            )
            self.expect(
                has_guard("k", "K must be positive"), f"{label}: explicit K <= 0 guard"
            )

        evidence_reference = directory / "q4_group64_reference.py"
        self.require(
            evidence_reference.is_file(),
            "AITER zero contract: evidence reference missing",
        )
        verify_reference_guards(
            evidence_reference.read_text(encoding="utf-8"),
            str(evidence_reference),
            "AITER evidence reference packer",
        )

        if self.aiter_source is not None:
            external_reference = self.aiter_source / "op_tests/q4_group64_reference.py"
            self.require(
                external_reference.is_file(),
                f"AITER strict zero contract: missing {external_reference}",
            )
            external_reference_text = external_reference.read_text(encoding="utf-8")
            test_path = self.aiter_source / "op_tests/test_q4_group64_gemv.py"
            self.require(
                test_path.is_file(), f"AITER strict zero contract: missing {test_path}"
            )
            test_text = test_path.read_text(encoding="utf-8")
            reference_filename = str(external_reference)
            test_filename = str(test_path)
        else:
            external_reference_bytes = self.aiter_patch_postimages.get(
                "op_tests/q4_group64_reference.py"
            )
            test_bytes = self.aiter_patch_postimages.get(
                "op_tests/test_q4_group64_gemv.py"
            )
            self.require(
                external_reference_bytes is not None and test_bytes is not None,
                "AITER patch zero contract: reconstructed source/test missing",
            )
            external_reference_text = external_reference_bytes.decode("utf-8")
            test_text = test_bytes.decode("utf-8")
            reference_filename = "patch:op_tests/q4_group64_reference.py"
            test_filename = "patch:op_tests/test_q4_group64_gemv.py"
        verify_reference_guards(
            external_reference_text,
            reference_filename,
            "AITER source reference packer",
        )
        tree = ast.parse(test_text, filename=test_filename)
        functions = {
            node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        test = functions.get(
            "test_test_packer_rejects_non_tile_rows_and_non_group_columns"
        )
        self.require(
            isinstance(test, ast.FunctionDef),
            "AITER zero contract: source test missing",
        )
        rendered = ast.unparse(test)
        for fragment in (
            "match='N must be positive'",
            "torch.zeros((0, 64)",
            "match='K must be positive'",
            "torch.zeros((32, 0)",
        ):
            self.expect(
                fragment in rendered, f"AITER zero contract test: missing {fragment}"
            )

    def verify_aiter_public_source_contract(self) -> None:
        def source_text(relative_path: str) -> str:
            if self.aiter_source is not None:
                path = self.aiter_source / relative_path
                self.require(path.is_file(), f"AITER strict contract: missing {path}")
                return path.read_text(encoding="utf-8")
            content = self.aiter_patch_postimages.get(relative_path)
            self.require(
                content is not None,
                f"AITER patch contract: no reconstructed post-image for {relative_path}",
            )
            return content.decode("utf-8")

        python_source = source_text("aiter/ops/q4_group64_gemv.py")
        python_tree = ast.parse(python_source, filename="aiter/ops/q4_group64_gemv.py")
        python_functions = {
            node.name: node
            for node in python_tree.body
            if isinstance(node, ast.FunctionDef)
        }
        for function_name in (
            "q4_group64_gemv",
            "_q4_group64_gemv",
            "_require_experimental_enabled",
        ):
            self.expect(
                function_name in python_functions,
                f"AITER public API contract: missing {function_name}",
            )
        public_function = python_functions.get("q4_group64_gemv")
        if public_function is not None:
            self.expect_exact(
                [argument.arg for argument in public_function.args.args],
                ["x", "packed_weight"],
                "AITER public API contract: public positional arguments",
            )
            self.expect_exact(
                len(public_function.args.kwonlyargs),
                0,
                "AITER public API contract: no public mapping knob",
            )
        self.expect(
            '__all__ = ["q4_group64_gemv"]' in python_source,
            "AITER public API contract: __all__ export marker",
        )
        self.expect(
            "is_experimental_enabled" in python_source
            and "_require_experimental_enabled()" in python_source,
            "AITER public API contract: Python experimental gate",
        )

        header_source = source_text("csrc/include/q4_group64_gemv.h")
        exact_identity_cases = (
            'q4_group64_is_tuned_rx_9070_xt(0x7550, 32, "AMD Radeon RX 9070 XT")',
            '!q4_group64_is_tuned_rx_9070_xt(0x7550, 28, "AMD Radeon RX 9070")',
            '!q4_group64_is_tuned_rx_9070_xt(0x7550, 24, "AMD Radeon RX 9070 GRE")',
            '!q4_group64_is_tuned_rx_9070_xt(0x7551, 32, "AMD Radeon AI PRO R9700")',
            '!q4_group64_is_tuned_rx_9070_xt(0, 0, "unknown")',
        )
        for identity_case in exact_identity_cases:
            self.expect(
                identity_case in header_source,
                f"AITER exact RX 9070 XT identity contract: missing {identity_case}",
            )

        cpp_source = source_text("csrc/kernels/q4_group64_gemv.cu")
        for marker in (
            "experimental_runtime_enabled()",
            "hipDeviceAttributePciChipId",
            "HipDeviceGuard device_guard(x.device_id);",
            "cached_is_tuned_rx_9070_xt(x.device_id) ? select_mapping(rows, columns)",
            "aiter::getCurrentHIPStream()",
        ):
            self.expect(
                marker in cpp_source, f"AITER C++ public contract: missing {marker}"
            )
        self.expect(
            cpp_source.index("HipDeviceGuard device_guard(x.device_id);")
            < cpp_source.index("cached_is_tuned_rx_9070_xt(x.device_id)"),
            "AITER C++ public contract: device guard must precede identity lookup",
        )

        test_source = source_text("op_tests/test_q4_group64_gemv.py")
        required_tests = (
            "test_python_experimental_gate_rejects_unset_and_disabled",
            "test_cpp_auto_dispatch_has_exact_rx_9070_xt_guard",
            "test_benchmark_cli_rejects_when_no_requested_mapping_is_legal",
            "test_benchmark_auto_uses_public_call_and_controls_allocate_equally",
            "test_benchmark_pci_chip_id_uses_valid_aiter_helper_value",
            "test_benchmark_pci_chip_id_falls_back_from_rocm72_helper_value",
            "test_benchmark_pci_chip_id_fallback_fails_closed",
            "test_benchmark_requires_exact_rx_9070_xt_identity",
            "test_public_non_default_stream_and_preallocated_private_output",
            "test_runtime_gate_disables_loaded_public_and_direct_cpp_entries",
        )
        parsed_tests = {
            node.name
            for node in ast.parse(
                test_source, filename="op_tests/test_q4_group64_gemv.py"
            ).body
            if isinstance(node, ast.FunctionDef)
        }
        self.expect(
            set(required_tests).issubset(parsed_tests),
            "AITER public/device/gate source test coverage",
        )

        benchmark_source = source_text(
            "op_tests/op_benchmarks/hip/bench_q4_group64_gemv.py"
        )
        benchmark_tree = ast.parse(
            benchmark_source,
            filename="op_tests/op_benchmarks/hip/bench_q4_group64_gemv.py",
        )
        tolerance_pairs: list[tuple[float, float]] = []
        for node in ast.walk(benchmark_tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if node.func.attr != "assert_close":
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            try:
                rtol = float(ast.literal_eval(keywords["rtol"]))
                atol = float(ast.literal_eval(keywords["atol"]))
            except (KeyError, TypeError, ValueError):
                continue
            tolerance_pairs.append((rtol, atol))
        self.expect_exact(
            tolerance_pairs,
            [(5.0e-4, 5.0e-3)],
            "AITER public benchmark: rtol/atol correctness gate",
        )

    def aiter_dispatch_table(self) -> dict[tuple[int, int], str]:
        self.require(
            self.dispatch is not None,
            "AITER candidate: upstream dispatch document unavailable",
        )
        entries = [
            entry for entry in self.dispatch["entries"] if entry["mode"] == "plain"
        ]
        table = {
            (int(entry["rows"]), int(entry["columns"])): str(entry["mapping"])
            for entry in entries
        }
        self.expect_exact(
            len(entries), 14, "AITER candidate: plain dispatch entry count"
        )
        self.expect_exact(
            len(table), 14, "AITER candidate: unique plain dispatch shapes"
        )
        return table

    def verify_aiter_pytest(
        self, directory: Path, filename: str, expected_cases: set[str]
    ) -> int:
        path = directory / filename
        self.require(path.is_file(), f"AITER candidate: missing {path}")
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        self.require(
            len(suites) == 1, "AITER candidate: expected exactly one JUnit testsuite"
        )
        suite = suites[0]
        expected_totals = {
            "tests": len(expected_cases),
            "failures": 0,
            "errors": 0,
            "skipped": 0,
        }
        for field, expected in expected_totals.items():
            self.expect_exact(
                int(suite.attrib.get(field, -1)), expected, f"AITER pytest: {field}"
            )
        cases = suite.findall("testcase")
        self.expect_exact(
            len(cases), len(expected_cases), "AITER pytest: testcase element count"
        )
        names = {case.attrib.get("name") for case in cases}
        self.expect_exact(
            names, expected_cases, "AITER pytest: exact testcase coverage"
        )
        self.expect_exact(len(names), len(cases), "AITER pytest: unique testcase names")
        for case in cases:
            self.expect_exact(
                case.attrib.get("classname"),
                "op_tests.test_q4_group64_gemv",
                f"AITER pytest {case.attrib.get('name')}: classname",
            )
            self.expect(
                not list(case),
                f"AITER pytest {case.attrib.get('name')}: unexpected failure/skip child",
            )
        return len(cases)

    def verify_aiter_wrapper_results(
        self,
        directory: Path,
        dispatch: dict[tuple[int, int], str],
        stem: str,
        expected_schema: str,
        interleaved: bool,
    ) -> dict[str, dict[str, float]]:
        json_path = directory / f"{stem}_raw.json"
        csv_path = directory / f"{stem}_summary.csv"
        public_api_v3 = expected_schema == "aiter-q4-group64-benchmark-v3"
        label_prefix = (
            "AITER public-API v3"
            if public_api_v3
            else (
                "AITER interleaved wrapper"
                if interleaved
                else "AITER fixed-order wrapper"
            )
        )
        self.require(json_path.is_file(), f"{label_prefix}: missing {json_path}")
        document = read_json(json_path)
        self.expect_exact(
            document.get("schema"), expected_schema, f"{label_prefix}: schema"
        )
        configuration = document.get("configuration")
        rows = document.get("results")
        self.require(
            isinstance(configuration, dict), f"{label_prefix}: configuration missing"
        )
        self.require(isinstance(rows, list), f"{label_prefix}: results must be a list")
        expected_configuration = {
            "cache": "rotating",
            "rotate": 0,
            "automatic_minimum_packed_ring_bytes": 64 * 1024 * 1024,
            "warmup": 100,
            "samples": 30,
            "timing": "both",
            "calibration_iterations": 100,
            "target_sample_ms": 100.0,
        }
        if interleaved:
            expected_configuration.update(
                {
                    "mapping_execution_schedule": "cyclic_latin_by_sample_round",
                    "primary_latency_metric": "host wall-clock with a synchronized boundary",
                    "supplemental_latency_metric": "HIP event elapsed time",
                    "output_allocation_per_call": True,
                    "call_paths": {
                        "auto": "public:q4_group64_gemv",
                        "controls": "private-allocation-equivalent:_q4_group64_gemv(out=None)",
                    },
                    "device_identity_requirement": (
                        "gfx1201, name AMD Radeon RX 9070 XT, 32 HIP-reported "
                        "multiprocessors, PCI chip ID 0x7550; blank name is accepted "
                        "only for ROCm runtimes that do not populate it"
                    ),
                    "correctness_policy": {
                        "check": "torch.testing.assert_close against dequantized FP32 reference",
                        "rtol": 5.0e-4,
                        "atol": 5.0e-3,
                        "diagnostics": [
                            "correctness_max_abs",
                            "correctness_relative_l2",
                        ],
                    },
                }
            )
        compared_configuration = dict(configuration)
        device_identity = compared_configuration.pop("device_identity", None)
        self.expect_equal(
            compared_configuration,
            expected_configuration,
            f"{label_prefix}: configuration",
        )
        if interleaved:
            self.require(
                isinstance(device_identity, dict), f"{label_prefix}: device identity"
            )
            self.expect_exact(
                device_identity.get("device_index"), 0, f"{label_prefix}: device index"
            )
            self.expect_exact(
                device_identity.get("arch"), "gfx1201", f"{label_prefix}: device arch"
            )
            self.expect_exact(
                device_identity.get("pci_chip_id"),
                0x7550,
                f"{label_prefix}: PCI chip ID",
            )
            self.expect_exact(
                device_identity.get("pci_chip_id_hex"),
                "0x7550",
                f"{label_prefix}: PCI chip ID hex",
            )
            self.expect_exact(
                device_identity.get("hip_reported_multiprocessors"),
                32,
                f"{label_prefix}: HIP multiprocessors",
            )
            device_name = device_identity.get("name")
            self.expect(
                device_name in ("", "AMD Radeon RX 9070 XT"),
                f"{label_prefix}: exact RX 9070 XT name/blank compatibility",
            )
            self.expect_exact(
                device_identity.get("blank_name_compatibility_used"),
                device_name == "",
                f"{label_prefix}: blank-name compatibility marker",
            )
            pci_query = device_identity.get("pci_chip_id_query")
            self.require(
                isinstance(pci_query, dict), f"{label_prefix}: PCI query provenance"
            )
            self.expect_exact(
                pci_query.get("effective_value"),
                0x7550,
                f"{label_prefix}: effective PCI value",
            )
            self.expect_exact(
                pci_query.get("aiter_helper_raw"),
                0x100,
                f"{label_prefix}: ROCm 7.2 helper raw",
            )
            self.expect_exact(
                pci_query.get("fallback_attribute_id"),
                10020,
                f"{label_prefix}: numeric PCI fallback attribute",
            )
            self.expect_exact(
                pci_query.get("fallback_value"),
                0x7550,
                f"{label_prefix}: PCI fallback value",
            )
            self.expect_exact(
                pci_query.get("fallback_used"),
                True,
                f"{label_prefix}: PCI fallback marker",
            )
        self.expect_exact(len(rows), 84, f"{label_prefix}: result row count")
        combinations = {
            (int(row["n"]), int(row["k"]), str(row["requested"]), str(row["timing"]))
            for row in rows
        }
        expected_combinations = {
            (n, k, requested, timing)
            for n, k in dispatch
            for requested in ("old", "auto", "selected")
            for timing in ("integration", "batched")
        }
        self.expect_exact(
            combinations,
            expected_combinations,
            f"{label_prefix}: shape/request/timing matrix",
        )
        self.expect_exact(
            len(combinations), len(rows), f"{label_prefix}: duplicate rows"
        )
        baselines = {
            (int(row["n"]), int(row["k"]), str(row["timing"])): float(row["median_us"])
            for row in rows
            if row["requested"] == "old"
        }
        self.expect_exact(len(baselines), 28, f"{label_prefix}: old baseline count")
        auto_medians = {
            (int(row["n"]), int(row["k"]), str(row["timing"])): float(row["median_us"])
            for row in rows
            if row["requested"] == "auto"
        }
        self.expect_exact(len(auto_medians), 28, f"{label_prefix}: auto baseline count")
        wall_baselines = {
            (int(row["n"]), int(row["k"]), str(row["timing"])): float(
                row["median_wall_us"]
            )
            for row in rows
            if interleaved and row["requested"] == "old"
        }
        wall_auto_medians = {
            (int(row["n"]), int(row["k"]), str(row["timing"])): float(
                row["median_wall_us"]
            )
            for row in rows
            if interleaved and row["requested"] == "auto"
        }
        if interleaved:
            self.expect_exact(
                len(wall_baselines), 28, f"{label_prefix}: wall old baseline count"
            )
            self.expect_exact(
                len(wall_auto_medians), 28, f"{label_prefix}: wall auto baseline count"
            )
        correctness_by_case: dict[tuple[int, int, str], set[tuple[float, float]]] = {}
        for row in rows:
            n = int(row["n"])
            k = int(row["k"])
            requested = str(row["requested"])
            timing = str(row["timing"])
            label = f"{label_prefix} {n}x{k}/{requested}/{timing}"
            selected = dispatch[(n, k)]
            expected_mapping = (
                "old"
                if requested == "old"
                else "auto" if requested == "auto" else selected
            )
            expected_resolved = (
                "old"
                if requested == "old"
                else (
                    f"runtime-guarded:{selected}"
                    if public_api_v3 and requested == "auto"
                    else selected
                )
            )
            self.expect_exact(row.get("mapping"), expected_mapping, f"{label}: mapping")
            self.expect_exact(
                row.get("resolved"), expected_resolved, f"{label}: resolved mapping"
            )
            if public_api_v3:
                self.expect_exact(
                    row.get("candidate_mapping"),
                    selected if requested == "auto" else None,
                    f"{label}: candidate mapping",
                )
            self.expect_exact(row.get("cache"), "rotating", f"{label}: cache mode")
            raw = [float(value) for value in row.get("raw_us", [])]
            self.expect_exact(row.get("samples"), 30, f"{label}: samples field")
            self.expect_exact(len(raw), 30, f"{label}: raw sample count")
            self.expect(
                all(math.isfinite(value) and value > 0 for value in raw),
                f"{label}: timing samples must be positive and finite",
            )
            expected_stats = timing_statistics(raw)
            for field in ("median_us", "p10_us", "p90_us"):
                self.expect_equal(
                    row.get(field),
                    expected_stats[field],
                    f"{label}: recomputed {field}",
                )
            self.expect_exact(row.get("correct"), True, f"{label}: correctness")
            if interleaved:
                raw_event = [float(value) for value in row.get("raw_event_us", [])]
                raw_wall = [float(value) for value in row.get("raw_wall_us", [])]
                self.expect_equal(
                    raw_event, raw, f"{label}: raw_us aliases event samples"
                )
                self.expect_exact(len(raw_wall), 30, f"{label}: wall sample count")
                self.expect(
                    all(math.isfinite(value) and value > 0 for value in raw_wall),
                    f"{label}: wall samples must be positive and finite",
                )
                wall_stats = timing_statistics(raw_wall)
                for field, expected in (
                    ("median_event_us", expected_stats["median_us"]),
                    ("p10_event_us", expected_stats["p10_us"]),
                    ("p90_event_us", expected_stats["p90_us"]),
                    ("median_wall_us", wall_stats["median_us"]),
                    ("p10_wall_us", wall_stats["p10_us"]),
                    ("p90_wall_us", wall_stats["p90_us"]),
                ):
                    self.expect_equal(
                        row.get(field), expected, f"{label}: recomputed {field}"
                    )
                expected_call_path = (
                    "public:q4_group64_gemv"
                    if requested == "auto"
                    else "private-allocation-equivalent:_q4_group64_gemv(out=None)"
                )
                self.expect_exact(
                    row.get("call_path"), expected_call_path, f"{label}: call path"
                )
                max_abs = row.get("correctness_max_abs")
                relative_l2 = row.get("correctness_relative_l2")
                self.expect(
                    isinstance(max_abs, (int, float))
                    and math.isfinite(float(max_abs))
                    and float(max_abs) >= 0,
                    f"{label}: correctness max-absolute error",
                )
                self.expect(
                    isinstance(relative_l2, (int, float))
                    and math.isfinite(float(relative_l2))
                    and float(relative_l2) >= 0,
                    f"{label}: correctness relative-L2 diagnostic",
                )
                if isinstance(max_abs, (int, float)) and isinstance(
                    relative_l2, (int, float)
                ):
                    correctness_by_case.setdefault((n, k, requested), set()).add(
                        (float(max_abs), float(relative_l2))
                    )
            packed_bytes = n * k * 34 // 64
            rotations = max(2, (64 * 1024 * 1024) // packed_bytes + 1)
            packed_ring_bytes = packed_bytes * rotations
            traffic_bytes = packed_bytes + k * 4 + n * 4
            self.expect_exact(
                row.get("rotations"), rotations, f"{label}: rotation count"
            )
            self.expect_exact(
                row.get("packed_ring_bytes"),
                packed_ring_bytes,
                f"{label}: packed ring bytes",
            )
            self.expect(
                int(row["packed_ring_bytes"]) > 64 * 1024 * 1024,
                f"{label}: packed ring is not strictly larger than 64 MiB",
            )
            self.expect_exact(
                row.get("total_ring_bytes"),
                packed_ring_bytes + k * 4 + n * 4,
                f"{label}: total ring bytes",
            )
            median_us = float(expected_stats["median_us"])
            self.expect_equal(
                row.get("effective_gbps"),
                traffic_bytes / median_us / 1.0e3,
                f"{label}: effective GB/s",
            )
            self.expect_equal(
                row.get("effective_tflops"),
                (2.0 * n * k) / median_us / 1.0e6,
                f"{label}: effective TFLOP/s",
            )
            if interleaved:
                median_wall_us = float(row["median_wall_us"])
                self.expect_equal(
                    row.get("effective_wall_gbps"),
                    traffic_bytes / median_wall_us / 1.0e3,
                    f"{label}: effective wall GB/s",
                )
                self.expect_equal(
                    row.get("effective_wall_tflops"),
                    (2.0 * n * k) / median_wall_us / 1.0e6,
                    f"{label}: effective wall TFLOP/s",
                )
            if timing == "integration":
                self.expect_exact(
                    row.get("iterations_per_sample"),
                    1,
                    f"{label}: integration iterations",
                )
                self.expect_exact(
                    row.get("calibration_us"), None, f"{label}: integration calibration"
                )
                if interleaved:
                    self.expect_exact(
                        row.get("calibration_wall_us"),
                        None,
                        f"{label}: integration wall calibration",
                    )
                self.expect(
                    (
                        "single synchronized measurement around one public auto call"
                        if public_api_v3
                        else "single HIP-event interval"
                    )
                    in str(row.get("timing_boundary")),
                    f"{label}: integration timing boundary",
                )
            else:
                calibration = row.get("calibration_us")
                self.expect(
                    isinstance(calibration, (int, float))
                    and math.isfinite(float(calibration))
                    and float(calibration) > 0,
                    f"{label}: batched calibration",
                )
                iteration_calibration = (
                    row.get("calibration_wall_us") if interleaved else calibration
                )
                if interleaved:
                    self.expect(
                        isinstance(iteration_calibration, (int, float))
                        and math.isfinite(float(iteration_calibration))
                        and float(iteration_calibration) > 0,
                        f"{label}: batched wall calibration",
                    )
                expected_iterations = min(
                    2_000_000,
                    max(
                        10,
                        math.ceil(
                            float(configuration["target_sample_ms"])
                            * 1000.0
                            / float(iteration_calibration)
                        ),
                    ),
                )
                self.expect_exact(
                    row.get("iterations_per_sample"),
                    expected_iterations,
                    f"{label}: batched iterations",
                )
                self.expect(
                    int(row["iterations_per_sample"]) > 1,
                    f"{label}: batched iteration count",
                )
                expected_boundary_marker = (
                    "repeated public auto calls or allocation-equivalent private controls"
                    if public_api_v3
                    else (
                        "repeated public or public-equivalent Python/AITER calls"
                        if interleaved
                        else "repeated Python/AITER entry calls"
                    )
                )
                self.expect(
                    expected_boundary_marker in str(row.get("timing_boundary")),
                    f"{label}: batched timing boundary",
                )
            expected_speedup = baselines[(n, k, timing)] / median_us
            self.expect_equal(
                row.get("speedup_vs_old"), expected_speedup, f"{label}: speedup vs old"
            )
            if interleaved:
                expected_wall_speedup = wall_baselines[(n, k, timing)] / float(
                    row["median_wall_us"]
                )
                self.expect_equal(
                    row.get("speedup_vs_old_wall"),
                    expected_wall_speedup,
                    f"{label}: wall speedup vs old",
                )
                self.expect_exact(
                    row.get("execution_schedule"),
                    "cyclic_latin_by_sample_round",
                    f"{label}: execution schedule",
                )
                sample_records = row.get("sample_records")
                self.require(
                    isinstance(sample_records, list),
                    f"{label}: sample records must be a list",
                )
                self.expect_exact(
                    len(sample_records), 30, f"{label}: sample record count"
                )
                expected_orders = (
                    ["old", "auto", "selected"],
                    ["auto", "selected", "old"],
                    ["selected", "old", "auto"],
                )
                for round_index, (sample, record) in enumerate(
                    zip(raw, sample_records)
                ):
                    record_label = f"{label}: sample round {round_index}"
                    self.require(isinstance(record, dict), f"{record_label}: record")
                    order = expected_orders[round_index % 3]
                    self.expect_exact(
                        record.get("round"), round_index, f"{record_label}: round"
                    )
                    self.expect_exact(
                        record.get("execution_order"), order, f"{record_label}: order"
                    )
                    self.expect_exact(
                        record.get("position"),
                        order.index(requested),
                        f"{record_label}: position",
                    )
                    self.expect_equal(
                        record.get("latency_us"), sample, f"{record_label}: latency"
                    )
                    self.expect_equal(
                        record.get("event_us"), sample, f"{record_label}: event latency"
                    )
                    self.expect_equal(
                        record.get("wall_us"),
                        raw_wall[round_index],
                        f"{record_label}: wall latency",
                    )
                    self.expect_exact(
                        record.get("call_path"),
                        expected_call_path,
                        f"{record_label}: call path",
                    )
                self.expect_exact(
                    Counter(record.get("position") for record in sample_records),
                    Counter({0: 10, 1: 10, 2: 10}),
                    f"{label}: balanced execution positions",
                )
                expected_latency_ratio = median_us / auto_medians[(n, k, timing)]
                self.expect_equal(
                    row.get("latency_ratio_vs_auto"),
                    expected_latency_ratio,
                    f"{label}: latency ratio vs auto",
                )
                self.expect_equal(
                    row.get("latency_ratio_vs_auto_wall"),
                    float(row["median_wall_us"]) / wall_auto_medians[(n, k, timing)],
                    f"{label}: wall latency ratio vs auto",
                )

        if interleaved:
            self.expect_exact(
                len(correctness_by_case), 42, f"{label_prefix}: correctness case count"
            )
            self.expect(
                all(len(values) == 1 for values in correctness_by_case.values()),
                f"{label_prefix}: correctness diagnostics differ across timing boundaries",
            )

        self.verify_csv_projection(
            rows,
            csv_path,
            {"raw_us", "raw_event_us", "raw_wall_us", "sample_records"},
            label_prefix,
        )
        report: dict[str, dict[str, float]] = {}
        for timing in ("integration", "batched"):
            speedup_field = "speedup_vs_old_wall" if interleaved else "speedup_vs_old"
            speedups = [
                float(row[speedup_field])
                for row in rows
                if row["requested"] == "auto" and row["timing"] == timing
            ]
            self.expect_exact(
                len(speedups), 14, f"{label_prefix}: auto {timing} shape count"
            )
            report[timing] = {
                "median": statistics.median(speedups),
                "geomean": math.exp(
                    statistics.fmean(math.log(value) for value in speedups)
                ),
                "worst": min(speedups),
                "best": max(speedups),
            }
        return report

    def verify_aiter_kernel(
        self, directory: Path, dispatch: dict[tuple[int, int], str]
    ) -> dict[str, float]:
        json_path = directory / "kernel_raw.json"
        csv_path = directory / "kernel_summary.csv"
        driver_path = directory / "kernel_benchmark_driver.cu"
        self.require(driver_path.is_file(), f"AITER kernel: missing {driver_path}")
        driver_source = driver_path.read_text(encoding="utf-8")
        self.expect(
            driver_source.startswith('#include "csrc/kernels/q4_group64_gemv.cu"\n'),
            "AITER kernel: direct harness must use the AITER-root-relative include",
        )
        self.expect(
            "/home/" not in driver_source and "C:\\" not in driver_source,
            "AITER kernel: direct harness contains an absolute developer path",
        )
        self.expect(
            "int main(int argc, char** argv)" in driver_source
            and 'argc == 2 ? argv[1] : "."' in driver_source,
            "AITER kernel: portable output-directory argument",
        )
        self.require(json_path.is_file(), f"AITER kernel: missing {json_path}")
        document = read_json(json_path)
        self.expect_exact(document.get("schema"), 1, "AITER kernel: schema")
        self.expect_exact(document.get("arch"), "gfx1201", "AITER kernel: architecture")
        self.expect_exact(
            document.get("warmup_per_mapping"), 100, "AITER kernel: warmup"
        )
        self.expect_exact(
            document.get("calibration_launches"),
            100,
            "AITER kernel: calibration launches",
        )
        self.expect_exact(
            document.get("target_sample_us"), 100000, "AITER kernel: target sample us"
        )
        self.expect_exact(document.get("samples"), 30, "AITER kernel: sample count")
        self.expect_exact(
            document.get("weight_copies"), 72, "AITER kernel: weight copies"
        )
        shapes = document.get("shapes")
        self.require(isinstance(shapes, list), "AITER kernel: shapes must be a list")
        self.expect_exact(len(shapes), 14, "AITER kernel: shape count")
        self.expect_exact(
            {(int(shape["n"]), int(shape["k"])) for shape in shapes},
            set(dispatch),
            "AITER kernel: shape set vs dispatch",
        )
        csv_rows: list[dict[str, Any]] = []
        auto_speedups: list[float] = []
        for shape in shapes:
            n = int(shape["n"])
            k = int(shape["k"])
            label = f"AITER kernel {n}x{k}"
            selected = dispatch[(n, k)]
            self.expect_exact(
                shape.get("selected"), selected, f"{label}: selected mapping"
            )
            measurements = shape.get("measurements")
            self.require(
                isinstance(measurements, list), f"{label}: measurements must be a list"
            )
            self.expect_exact(len(measurements), 3, f"{label}: measurement count")
            self.expect_exact(
                {measurement["mapping"] for measurement in measurements},
                {"old", "auto", "selected"},
                f"{label}: mapping coverage",
            )
            by_mapping = {
                measurement["mapping"]: measurement for measurement in measurements
            }
            for mapping in ("old", "auto", "selected"):
                measurement = by_mapping[mapping]
                item_label = f"{label}/{mapping}"
                raw = [float(value) for value in measurement.get("raw_us", [])]
                self.expect_exact(len(raw), 30, f"{item_label}: raw sample count")
                self.expect(
                    all(math.isfinite(value) and value > 0 for value in raw),
                    f"{item_label}: raw samples",
                )
                stats = timing_statistics(raw)
                for field in ("median_us", "p10_us", "p90_us"):
                    self.expect_equal(
                        measurement.get(field),
                        stats[field],
                        f"{item_label}: recomputed {field}",
                    )
                calibration = measurement.get("calibration_us")
                self.expect(
                    isinstance(calibration, (int, float))
                    and math.isfinite(float(calibration))
                    and float(calibration) > 0,
                    f"{item_label}: calibration",
                )
                expected_iterations = min(
                    2_000_000,
                    max(
                        10,
                        math.ceil(
                            float(document["target_sample_us"]) / float(calibration)
                        ),
                    ),
                )
                self.expect_exact(
                    measurement.get("batch_iterations"),
                    expected_iterations,
                    f"{item_label}: batch iterations",
                )
                csv_rows.append(
                    {
                        "n": n,
                        "k": k,
                        "selected": selected,
                        "mapping": mapping,
                        "median_us": measurement["median_us"],
                        "p10_us": measurement["p10_us"],
                        "p90_us": measurement["p90_us"],
                        "calibration_us": measurement["calibration_us"],
                        "batch_iterations": measurement["batch_iterations"],
                        "weight_copies": document["weight_copies"],
                        "auto_over_old_speedup": shape["auto_over_old_speedup"],
                    }
                )
            speedup = float(by_mapping["old"]["median_us"]) / float(
                by_mapping["auto"]["median_us"]
            )
            auto_speedups.append(speedup)
            self.expect_equal(
                shape.get("auto_over_old_speedup"), speedup, f"{label}: auto speedup"
            )
        self.verify_csv_projection(csv_rows, csv_path, set(), "AITER direct kernel")
        return {
            "median": statistics.median(auto_speedups),
            "worst": min(auto_speedups),
            "best": max(auto_speedups),
        }

    def verify_aiter_legacy_integration(
        self,
        directory: Path,
        stem: str,
        dispatch: dict[tuple[int, int], str],
    ) -> None:
        json_path = directory / f"integration_{stem}_raw.json"
        csv_path = directory / f"integration_{stem}_summary.csv"
        self.require(
            json_path.is_file(), f"AITER integration {stem}: missing {json_path}"
        )
        document = read_json(json_path)
        self.expect_exact(
            document.get("schema"),
            "aiter-q4-group64-pybind-benchmark-v1",
            f"AITER integration {stem}: schema",
        )
        configuration = document.get("configuration")
        shapes = document.get("results")
        self.require(
            isinstance(configuration, dict),
            f"AITER integration {stem}: configuration missing",
        )
        self.require(
            isinstance(shapes, list),
            f"AITER integration {stem}: results must be a list",
        )
        self.expect_exact(
            configuration.get("processes"), 1, f"AITER integration {stem}: processes"
        )
        self.expect_exact(
            configuration.get("warmup_per_mapping"),
            30,
            f"AITER integration {stem}: warmup",
        )
        self.expect_exact(
            configuration.get("samples_per_mapping"),
            101,
            f"AITER integration {stem}: samples",
        )
        self.expect_exact(
            configuration.get("rotating_buffer_count"),
            16,
            f"AITER integration {stem}: rotating buffers",
        )
        self.expect_exact(len(shapes), 14, f"AITER integration {stem}: shape count")
        self.expect_exact(
            {(int(shape["n"]), int(shape["k"])) for shape in shapes},
            set(dispatch),
            f"AITER integration {stem}: shape set",
        )
        csv_rows: list[dict[str, Any]] = []
        for shape in shapes:
            n = int(shape["n"])
            k = int(shape["k"])
            label = f"AITER integration {stem} {n}x{k}"
            selected = dispatch[(n, k)]
            self.expect_exact(
                shape.get("selected_mapping"), selected, f"{label}: selected mapping"
            )
            self.expect_exact(
                shape.get("packed_bytes"), n * k * 34 // 64, f"{label}: packed bytes"
            )
            mappings = shape.get("mappings")
            self.require(
                isinstance(mappings, dict), f"{label}: mappings must be an object"
            )
            self.expect_exact(
                set(mappings), {"old", "auto", "selected"}, f"{label}: mapping set"
            )
            baseline_median = float(mappings["old"]["median_us"])
            for requested in ("old", "auto", "selected"):
                measurement = mappings[requested]
                raw = [float(value) for value in measurement.get("raw_us", [])]
                item_label = f"{label}/{requested}"
                self.expect_exact(measurement.get("count"), 101, f"{item_label}: count")
                self.expect_exact(len(raw), 101, f"{item_label}: raw sample count")
                self.expect(
                    all(math.isfinite(value) and value > 0 for value in raw),
                    f"{item_label}: raw samples",
                )
                stats = timing_statistics(raw)
                for field in (
                    "median_us",
                    "p10_us",
                    "p90_us",
                    "mean_us",
                    "sample_stddev_us",
                ):
                    self.expect_equal(
                        measurement.get(field),
                        stats[field],
                        f"{item_label}: recomputed {field}",
                    )
                traffic_bytes = int(shape["packed_bytes"]) + k * 4 + n * 4
                median_us = float(stats["median_us"])
                self.expect_equal(
                    measurement.get("effective_gbps"),
                    traffic_bytes / median_us / 1.0e3,
                    f"{item_label}: effective GB/s",
                )
                self.expect_equal(
                    measurement.get("effective_tflops"),
                    (2.0 * n * k) / median_us / 1.0e6,
                    f"{item_label}: effective TFLOP/s",
                )
                speedup = baseline_median / median_us
                resolved = "old" if requested == "old" else selected
                csv_rows.append(
                    {
                        "n": n,
                        "k": k,
                        "requested": requested,
                        "resolved": resolved,
                        "median_us": measurement["median_us"],
                        "p10_us": measurement["p10_us"],
                        "p90_us": measurement["p90_us"],
                        "mean_us": measurement["mean_us"],
                        "sample_stddev_us": measurement["sample_stddev_us"],
                        "effective_gbps": measurement["effective_gbps"],
                        "effective_tflops": measurement["effective_tflops"],
                        "speedup_vs_old": speedup,
                    }
                )
            self.expect_equal(
                shape.get("auto_speedup_vs_old"),
                baseline_median / float(mappings["auto"]["median_us"]),
                f"{label}: auto speedup",
            )
            self.expect_equal(
                shape.get("selected_speedup_vs_old"),
                baseline_median / float(mappings["selected"]["median_us"]),
                f"{label}: selected speedup",
            )
        self.verify_csv_projection(
            csv_rows, csv_path, set(), f"AITER integration {stem}"
        )

    def verify_aiter_rocm714_smoke(
        self,
        directory: Path,
        reference: Any,
        source: dict[str, Any],
        source_manifest: Sequence[tuple[str, str]],
    ) -> int:
        label = "AITER ROCm 7.14 supplemental native smoke"
        self.require(isinstance(reference, dict), f"{label}: validation reference")
        self.expect_exact(
            reference.get("role"),
            "supplemental_native_cross_validation_not_authoritative_public_api",
            f"{label}: role",
        )
        self.expect_exact(
            reference.get("authoritative_public_api_evidence"),
            False,
            f"{label}: non-authoritative marker",
        )

        def local_artifact(record: Any, item_label: str) -> Path:
            self.require(isinstance(record, dict), f"{item_label}: record")
            path_value = record.get("path")
            self.require(isinstance(path_value, str), f"{item_label}: path")
            path = self.safe_relative_path(self.repository_root, path_value, item_label)
            self.expect_exact(
                path.stat().st_size, record.get("size_bytes"), f"{item_label}: size"
            )
            self.expect_exact(
                self.cached_sha256(path), record.get("sha256"), f"{item_label}: SHA-256"
            )
            return path

        json_path = local_artifact(reference.get("json"), f"{label} JSON")
        log_path = local_artifact(reference.get("log"), f"{label} log")
        document = read_json(json_path)
        self.expect_exact(
            document.get("schema"),
            "aiter-q4-group64-rocm714-native-smoke-v1",
            f"{label}: schema",
        )
        self.expect_exact(document.get("status"), "passed", f"{label}: status")
        self.expect_exact(
            document.get("role"), reference.get("role"), f"{label}: role cross-check"
        )
        scope = document.get("scope")
        self.require(isinstance(scope, dict), f"{label}: scope")
        self.expect_exact(
            scope.get("authoritative_public_api_evidence"),
            False,
            f"{label}: scope authority",
        )
        self.expect(
            "Python public wrapper" in scope.get("excludes", [])
            and "performance measurement" in scope.get("excludes", []),
            f"{label}: scope exclusions",
        )

        provenance = document.get("source_provenance")
        self.require(isinstance(provenance, dict), f"{label}: source provenance")
        self.expect_exact(
            provenance.get("base_commit"),
            source.get("upstream_base_commit"),
            f"{label}: upstream base commit",
        )
        smoke_patch = provenance.get("portable_patch")
        source_patch = source.get("patch")
        self.require(isinstance(smoke_patch, dict), f"{label}: portable patch")
        self.require(isinstance(source_patch, dict), f"{label}: source patch")
        self.expect_exact(
            Path(str(smoke_patch.get("path"))).name,
            Path(str(source_patch.get("path"))).name,
            f"{label}: patch filename",
        )
        self.expect_exact(
            smoke_patch.get("sha256"),
            "bef225ba18c35243a7de9bc8b83a740c3bbca15eee02910a12c6b1fa622c8d7f",
            f"{label}: historical pre-roofline patch SHA-256",
        )
        self.expect(
            smoke_patch.get("sha256") != source_patch.get("sha256"),
            f"{label}: historical patch must remain distinct from the docs-only refreshed candidate patch",
        )

        manifest_hashes = dict(source_manifest)
        inputs = document.get("inputs")
        self.require(isinstance(inputs, dict), f"{label}: inputs")
        for input_name, expected_path in (
            ("kernel", "csrc/kernels/q4_group64_gemv.cu"),
            ("header", "csrc/include/q4_group64_gemv.h"),
        ):
            record = inputs.get(input_name)
            self.require(isinstance(record, dict), f"{label}: {input_name} input")
            self.expect_exact(
                record.get("path"), expected_path, f"{label}: {input_name} path"
            )
            self.expect_exact(
                record.get("path_semantics"),
                "AITER-repository-root-relative",
                f"{label}: {input_name} path semantics",
            )
            self.expect_exact(
                record.get("sha256"),
                manifest_hashes.get(expected_path),
                f"{label}: {input_name} SHA-256",
            )
        harness = inputs.get("standalone_harness")
        self.require(isinstance(harness, dict), f"{label}: harness input")
        self.expect_exact(
            harness.get("source_provenance"), False, f"{label}: harness provenance"
        )
        self.expect(
            Path(str(harness.get("path"))).is_absolute()
            and "ephemeral" in str(harness.get("path_semantics", "")),
            f"{label}: harness ephemeral-path disclosure",
        )

        environment = document.get("environment")
        self.require(isinstance(environment, dict), f"{label}: environment")
        self.expect_exact(
            environment.get("device_name"), "AMD Radeon RX 9070 XT", f"{label}: GPU"
        )
        self.expect_exact(
            environment.get("device_arch"), "gfx1201", f"{label}: architecture"
        )
        self.expect(
            str(environment.get("hip_version", "")).startswith("7.14."),
            f"{label}: HIP 7.14",
        )
        commands = document.get("commands")
        self.require(
            isinstance(commands, list)
            and all(isinstance(command, str) for command in commands),
            f"{label}: commands",
        )
        command_text = "\n".join(commands)
        for marker in (
            "--offload-arch=gfx1201",
            "llvm-nm",
            "AITER_ENABLE_EXPERIMENTAL=1",
        ):
            self.expect(marker in command_text, f"{label}: command marker {marker}")

        build_artifacts = document.get("build_artifacts")
        self.require(isinstance(build_artifacts, dict), f"{label}: build artifacts")
        for artifact_name in ("kernel_object", "harness_object", "binary"):
            record = build_artifacts.get(artifact_name)
            self.require(isinstance(record, dict), f"{label}: {artifact_name} record")
            artifact_path = Path(str(record.get("path", "")))
            self.expect(
                artifact_path.is_absolute(), f"{label}: {artifact_name} execution path"
            )
            self.expect(
                isinstance(record.get("size_bytes"), int) and record["size_bytes"] > 0,
                f"{label}: {artifact_name} size",
            )
            self.expect(
                isinstance(record.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
                f"{label}: {artifact_name} SHA-256",
            )
            if artifact_path.is_file():
                self.expect_exact(
                    artifact_path.stat().st_size,
                    record.get("size_bytes"),
                    f"{label}: {artifact_name} local size",
                )
                self.expect_exact(
                    self.cached_sha256(artifact_path),
                    record.get("sha256"),
                    f"{label}: {artifact_name} local SHA-256",
                )
            else:
                self.notes.append(
                    f"NOTE {label}: ephemeral {artifact_name} is no longer present"
                )

        symbol_validation = document.get("symbol_validation")
        self.require(isinstance(symbol_validation, dict), f"{label}: symbol validation")
        self.expect_exact(
            symbol_validation.get("passed"), True, f"{label}: symbols passed"
        )
        self.expect_exact(
            symbol_validation.get("public_native_entry"),
            "aiter::q4_group64_gemv_out",
            f"{label}: public native entry",
        )
        self.expect_exact(
            set(symbol_validation.get("kernel_families_found", [])),
            {
                "old<false>",
                "old<true>",
                "split<2>",
                "split<4>",
                "split<8>",
                "small<8,8>",
                "small<8,16>",
                "small<8,32>",
                "small<16,16>",
                "small<16,32>",
                "small<32,32>",
            },
            f"{label}: kernel symbol families",
        )
        self.expect_equal(
            document.get("correctness_policy"),
            {"atol": 5.0e-3, "rtol": 5.0e-4, "finite_required": True},
            f"{label}: AITER correctness gate",
        )
        runtime_gate = document.get("runtime_gate")
        self.require(isinstance(runtime_gate, dict), f"{label}: runtime gate")
        self.expect_exact(
            runtime_gate.get("environment_variable"),
            "AITER_ENABLE_EXPERIMENTAL",
            f"{label}: gate variable",
        )
        self.expect_exact(
            runtime_gate.get("unset_exit_code"), 134, f"{label}: gate rejection exit"
        )
        self.expect_exact(
            runtime_gate.get("expected_rejection_observed"),
            True,
            f"{label}: gate rejection marker",
        )

        cases = document.get("cases")
        self.require(isinstance(cases, list), f"{label}: cases")
        expected_labels = {f"mapping-{index}" for index in range(1, 11)} | {
            "non-default-stream",
            "unseen-auto",
            "unseen-auto-equals-old",
            "invalid-split2",
            "invalid-mapping",
            "known-auto",
            "known-auto-equals-small32x32",
        }
        self.expect_exact(
            {case.get("label") for case in cases}, expected_labels, f"{label}: case set"
        )
        self.expect_exact(len(cases), len(expected_labels), f"{label}: case count")
        self.expect(
            all(case.get("passed") is True for case in cases),
            f"{label}: all cases passed",
        )
        self.expect_exact(
            {
                case.get("mapping")
                for case in cases
                if str(case.get("label", "")).startswith("mapping-")
            },
            {
                "old",
                "split2",
                "split4",
                "split8",
                "small8x8",
                "small8x16",
                "small8x32",
                "small16x16",
                "small16x32",
                "small32x32",
            },
            f"{label}: explicit mapping coverage",
        )
        self.expect(
            all(
                math.isfinite(float(case["max_abs"])) and float(case["max_abs"]) >= 0
                for case in cases
                if "max_abs" in case
            ),
            f"{label}: finite maximum-absolute diagnostics",
        )
        warnings = document.get("warnings")
        self.require(isinstance(warnings, dict), f"{label}: warnings")
        self.expect_exact(
            warnings.get("new_kernel_warnings"), 0, f"{label}: new-kernel warnings"
        )

        full_log = document.get("full_log")
        self.expect_equal(
            full_log, reference.get("log"), f"{label}: log provenance cross-check"
        )
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        for marker in (
            "HIP version: 7.14.",
            "aiter::q4_group64_gemv_out",
            "experimental_gate_unset_exit=134",
            "known-auto-equals-small32x32 passed max_abs=0",
        ):
            self.expect(marker in log_text, f"{label}: log marker {marker}")
        self.expect(
            "supplements but does not replace" in str(document.get("verdict", "")),
            f"{label}: supplemental verdict",
        )
        return len(cases)

    def verify_aiter_container_validation(
        self, directory: Path
    ) -> tuple[int, int, list[tuple[str, str]], dict[str, Any]]:
        path = directory / "container_validation.json"
        self.require(path.is_file(), f"AITER container validation: missing {path}")
        document = read_json(path)
        self.expect_exact(
            document.get("schema"),
            "aiter-q4-group64-container-validation-v3",
            "AITER container validation: schema",
        )
        self.expect_exact(
            document.get("status"), "passed", "AITER container validation: status"
        )
        self.expect(
            isinstance(document.get("generated_at"), str)
            and bool(document["generated_at"]),
            "AITER container validation: generated_at",
        )

        scope = document.get("scope")
        self.require(
            isinstance(scope, dict), "AITER container validation: scope missing"
        )
        self.expect_exact(
            scope.get("operation"), "q4_group64_gemv", "AITER container: operation"
        )
        self.expect_exact(
            scope.get("public_python_wrapper_exercised"),
            True,
            "AITER container: public wrapper",
        )
        self.expect_exact(
            scope.get("jit_compiled_in_test_environment"),
            True,
            "AITER container: JIT exercised",
        )
        self.expect_exact(
            scope.get("gpu"), "AMD Radeon RX 9070 XT", "AITER container: GPU"
        )
        self.expect_exact(
            scope.get("architecture"), "gfx1201", "AITER container: architecture"
        )
        layout = scope.get("layout")
        self.expect(
            isinstance(layout, str)
            and all(
                marker in layout
                for marker in ("uint8", "1088", "32 FP16", "1024 signed-INT4")
            ),
            "AITER container: packed layout declaration",
        )

        source = document.get("aiter_source")
        self.require(
            isinstance(source, dict), "AITER container validation: source missing"
        )
        source_manifest = self.verify_aiter_source_manifest(
            source, "AITER container validation source"
        )
        self.expect_exact(
            source.get("repository"), "ROCm/aiter", "AITER container source: repository"
        )
        self.expect(
            isinstance(source.get("path_semantics"), str)
            and "relative to the AITER repository root" in source["path_semantics"],
            "AITER container source: relative-path semantics",
        )
        self.expect(
            isinstance(source.get("upstream_base_commit"), str)
            and re.fullmatch(r"[0-9a-f]{40}", source["upstream_base_commit"])
            is not None,
            "AITER container source: upstream base commit",
        )
        self.expect(
            isinstance(source.get("candidate_commit"), str)
            and re.fullmatch(r"[0-9a-f]{40}", source["candidate_commit"]) is not None,
            "AITER container source: candidate commit",
        )
        self.expect_exact(
            source.get("candidate_parent_commit"),
            source.get("upstream_base_commit"),
            "AITER container source: candidate parent/upstream base",
        )
        self.expect_exact(
            source.get("candidate_commit_dco_signed_off"),
            True,
            "AITER container source: DCO flag",
        )
        self.expect_exact(
            source.get("branch"),
            "perf/q4-group64-gemv",
            "AITER container source: branch",
        )
        self.expect_exact(
            source.get("candidate_commit_recorded"),
            True,
            "AITER container source: candidate commit recorded",
        )
        self.expect_exact(
            source.get("validation_predated_candidate_commit"),
            True,
            "AITER container source: validation predates candidate commit",
        )
        self.expect(
            "only post-run candidate change is the roofline documentation section"
            in str(source.get("validation_source_state", "")),
            "AITER container source: validation-to-candidate disclosure",
        )
        source_files = source.get("files")
        self.require(
            isinstance(source_files, list),
            "AITER container source: files must be a list",
        )
        expected_source_suffixes = {
            "aiter/__init__.py",
            "aiter/jit/optCompilerConfig.json",
            "aiter/ops/q4_group64_gemv.py",
            "csrc/include/q4_group64_gemv.h",
            "csrc/kernels/q4_group64_gemv.cu",
            "csrc/pybind/q4_group64_gemv_pybind.cu",
            "op_tests/test_q4_group64_gemv.py",
            "op_tests/op_benchmarks/hip/bench_q4_group64_gemv.py",
            "op_tests/q4_group64_reference.py",
            "docs/q4_group64_gemv.md",
        }
        self.expect_exact(
            len(source_files),
            len(expected_source_suffixes),
            "AITER container: source file count",
        )
        source_suffixes: set[str] = set()
        for index, record in enumerate(source_files):
            self.require(
                isinstance(record, dict), f"AITER container source file {index}: record"
            )
            value = record.get("path")
            self.require(
                isinstance(value, str), f"AITER container source file {index}: path"
            )
            self.expect(
                not Path(value).is_absolute(),
                f"AITER container source file {index}: path must be repository-relative",
            )
            matching_suffixes = [
                suffix
                for suffix in expected_source_suffixes
                if value == suffix or value.endswith("/" + suffix)
            ]
            self.expect_exact(
                len(matching_suffixes),
                1,
                f"AITER container source file {index}: recognized path",
            )
            if matching_suffixes:
                source_suffixes.add(matching_suffixes[0])
            self.expect(
                isinstance(record.get("size_bytes"), int)
                and not isinstance(record.get("size_bytes"), bool)
                and record["size_bytes"] >= 0,
                f"AITER container source {value}: size_bytes",
            )
            self.expect(
                isinstance(record.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
                f"AITER container source {value}: SHA-256",
            )
            if self.aiter_source is not None and matching_suffixes:
                self.verify_recorded_file(
                    path_value=str(self.aiter_source / matching_suffixes[0]),
                    size_value=record.get("size_bytes"),
                    sha256_value=record.get("sha256"),
                    label=f"AITER strict source {matching_suffixes[0]}",
                    missing_is_note=False,
                )
        self.expect_exact(
            source_suffixes,
            expected_source_suffixes,
            "AITER container: source file set",
        )
        worktree_status = source.get("worktree_status")
        self.require(
            isinstance(worktree_status, list), "AITER container source: worktree status"
        )
        status_paths = {
            str(entry).split(maxsplit=1)[1]
            for entry in worktree_status
            if isinstance(entry, str) and len(entry.split(maxsplit=1)) == 2
        }
        self.expect_exact(
            status_paths,
            expected_source_suffixes,
            "AITER container: source/status file set",
        )
        self.expect(
            "before the DCO commit" in str(source.get("worktree_status_context", "")),
            "AITER container source: historical worktree-status context",
        )

        container = document.get("container")
        self.require(
            isinstance(container, dict), "AITER container validation: container missing"
        )
        image = container.get("image")
        image_id = container.get("image_id")
        self.expect(
            isinstance(image, str) and bool(image),
            "AITER container validation: image name",
        )
        self.expect(
            isinstance(image_id, str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is not None,
            "AITER container validation: image ID",
        )
        image_leaf = str(image).rsplit("/", 1)[-1]
        image_repository = (
            str(image).rsplit(":", 1)[0] if ":" in image_leaf else str(image)
        )
        self.expect_exact(
            container.get("repo_digest"),
            f"{image_repository}@{image_id}",
            "AITER container: repo digest",
        )
        for field in ("created", "python", "torch", "torch_hip", "hipcc"):
            self.expect(
                isinstance(container.get(field), str) and bool(container[field]),
                f"AITER container: non-empty {field}",
            )
        self.expect_exact(
            container.get("gpu_visible"), True, "AITER container: GPU visible"
        )
        self.expect_exact(
            container.get("visible_device_count"), 1, "AITER container: device count"
        )
        self.expect_exact(
            container.get("device_gcn_arch_name"),
            "gfx1201",
            "AITER container: visible architecture",
        )
        self.expect(
            isinstance(container.get("device_total_memory_bytes"), int)
            and container["device_total_memory_bytes"] > 0,
            "AITER container: device memory",
        )
        self.expect_equal(
            container.get("controlled_environment"),
            {
                "HIP_VISIBLE_DEVICES": "0",
                "ROCR_VISIBLE_DEVICES": "0",
                "GPU_ARCHS": "gfx1201",
                "PYTORCH_ROCM_ARCH": "gfx1201",
                "AITER_ENABLE_EXPERIMENTAL": "1",
                "PYTHONPATH": "/workspace/aiter",
                "AITER_JIT_DIR": "/jit/aiter",
                "TORCH_EXTENSIONS_DIR": "/jit/torch",
            },
            "AITER container: controlled environment",
        )

        jit_modules = document.get("jit_modules")
        self.require(
            isinstance(jit_modules, list), "AITER container validation: JIT modules"
        )
        self.expect_exact(len(jit_modules), 2, "AITER container: JIT module count")
        self.expect_exact(
            {
                Path(str(record.get("path"))).name
                for record in jit_modules
                if isinstance(record, dict)
            },
            {"module_aiter_core.so", "module_q4_group64_gemv.so"},
            "AITER container: JIT module set",
        )
        checked_jit = 0
        for index, record in enumerate(jit_modules):
            self.require(
                isinstance(record, dict), f"AITER container JIT {index}: record"
            )
            checked = self.verify_recorded_file(
                path_value=record.get("path"),
                size_value=record.get("size_bytes"),
                sha256_value=record.get("sha256"),
                label=f"AITER container JIT {index}",
                missing_is_note=True,
            )
            checked_jit += checked is not None

        def verify_output_reference(reference: Any, label: str) -> Path:
            self.require(
                isinstance(reference, dict), f"{label}: reference must be an object"
            )
            relative = reference.get("path")
            self.require(isinstance(relative, str), f"{label}: path must be a string")
            output_path = self.safe_relative_path(self.repository_root, relative, label)
            self.expect_exact(
                output_path.stat().st_size,
                reference.get("size_bytes"),
                f"{label}: size",
            )
            self.expect_exact(
                self.cached_sha256(output_path),
                reference.get("sha256"),
                f"{label}: SHA-256",
            )
            return output_path

        pytest_record = document.get("pytest")
        self.require(
            isinstance(pytest_record, dict), "AITER container validation: pytest record"
        )
        self.expect_exact(
            pytest_record.get("command"),
            [
                "python",
                "-m",
                "pytest",
                "op_tests/test_q4_group64_gemv.py",
                "-q",
                "--junitxml=/evidence/container_public_api_pytest.xml",
            ],
            "AITER container pytest: command",
        )
        for field, expected in (
            ("exit_code", 0),
            ("tests", len(AITER_PYTEST_CASES)),
            ("passed", len(AITER_PYTEST_CASES)),
            ("failures", 0),
            ("errors", 0),
            ("skipped", 0),
        ):
            self.expect_exact(
                pytest_record.get(field), expected, f"AITER container pytest: {field}"
            )
        self.expect(
            isinstance(pytest_record.get("duration_seconds"), (int, float))
            and float(pytest_record["duration_seconds"]) > 0,
            "AITER container pytest: duration",
        )
        junit_path = verify_output_reference(
            pytest_record.get("junit"), "AITER container pytest JUnit"
        )
        verify_output_reference(pytest_record.get("log"), "AITER container pytest log")
        junit_root = ET.parse(junit_path).getroot()
        junit_suite = (
            junit_root
            if junit_root.tag == "testsuite"
            else next(iter(junit_root.findall("testsuite")), None)
        )
        self.require(
            junit_suite is not None, "AITER container pytest: missing JUnit suite"
        )
        self.expect_equal(
            float(junit_suite.attrib.get("time", "nan")),
            float(pytest_record["duration_seconds"]),
            "AITER container pytest: JUnit duration",
        )
        self.expect_exact(
            int(junit_suite.attrib.get("tests", -1)),
            len(AITER_PYTEST_CASES),
            "AITER container pytest: JUnit test count",
        )

        benchmark = document.get("benchmark")
        self.require(
            isinstance(benchmark, dict), "AITER container validation: benchmark record"
        )
        self.expect_exact(
            benchmark.get("exit_code"), 0, "AITER container benchmark: exit code"
        )
        self.expect_exact(
            benchmark.get("schema"),
            "aiter-q4-group64-benchmark-v3",
            "AITER container benchmark: schema",
        )
        self.expect(
            isinstance(benchmark.get("artifact_filesystem_mtime_utc"), str)
            and bool(benchmark["artifact_filesystem_mtime_utc"]),
            "AITER container benchmark: artifact filesystem mtime",
        )
        self.expect(
            "completed_at" not in benchmark and "hostname" not in benchmark,
            "AITER container benchmark: unrecorded completion time/hostname must not be invented",
        )
        self.expect_exact(
            benchmark.get("shape_count"), 14, "AITER container benchmark: shapes"
        )
        self.expect_exact(
            benchmark.get("requested_mappings"),
            ["old", "auto", "selected"],
            "AITER container benchmark: requested mappings",
        )
        self.expect_exact(
            benchmark.get("timing_boundaries"),
            ["integration", "batched"],
            "AITER container benchmark: timing boundaries",
        )
        self.expect_exact(
            benchmark.get("result_rows"), 84, "AITER container benchmark: rows"
        )
        self.expect_exact(
            benchmark.get("raw_samples_per_row"),
            30,
            "AITER container benchmark: samples per row",
        )
        self.expect(
            isinstance(benchmark.get("mapping_execution_schedule"), str)
            and "cyclic Latin" in benchmark["mapping_execution_schedule"]
            and "exactly 10 times" in benchmark["mapping_execution_schedule"],
            "AITER container benchmark: interleaved execution schedule",
        )
        self.expect_exact(
            benchmark.get("rotating_packed_ring_strictly_greater_than_bytes"),
            64 * 1024 * 1024,
            "AITER container benchmark: ring threshold",
        )
        self.expect_exact(
            benchmark.get("primary_latency_metric"),
            "host wall-clock with a synchronized boundary",
            "AITER container benchmark: primary latency metric",
        )
        self.expect_exact(
            benchmark.get("supplemental_latency_metric"),
            "HIP event elapsed time",
            "AITER container benchmark: supplemental latency metric",
        )
        self.expect_exact(
            benchmark.get("public_call_path"),
            "public:q4_group64_gemv",
            "AITER container benchmark: public call path",
        )
        self.expect_exact(
            benchmark.get("control_call_path"),
            "private-allocation-equivalent:_q4_group64_gemv(out=None)",
            "AITER container benchmark: control call path",
        )
        expected_benchmark_command = [
            "python",
            "op_tests/op_benchmarks/hip/bench_q4_group64_gemv.py",
            "--sweep",
            "--mappings",
            "old",
            "auto",
            "selected",
            "--cache",
            "rotating",
            "--rotate",
            "0",
            "--warmup",
            "100",
            "--samples",
            "30",
            "--timing",
            "both",
            "--calibration-iterations",
            "100",
            "--target-sample-ms",
            "100",
            "-o",
            "/evidence/container_public_api_summary.csv",
            "--json",
            "/evidence/container_public_api_raw.json",
        ]
        self.expect_exact(
            benchmark.get("command"),
            expected_benchmark_command,
            "AITER container benchmark: command",
        )
        command = benchmark.get("command", [])
        self.expect_exact(
            Path(command[command.index("-o") + 1]).name,
            Path(benchmark.get("summary_csv", {}).get("path", "")).name,
            "AITER container benchmark: -o output matches preserved CSV",
        )
        self.expect_exact(
            Path(command[command.index("--json") + 1]).name,
            Path(benchmark.get("raw_json", {}).get("path", "")).name,
            "AITER container benchmark: --json output matches preserved raw JSON",
        )
        wrapper_json_path = verify_output_reference(
            benchmark.get("raw_json"), "AITER container benchmark raw JSON"
        )
        verify_output_reference(
            benchmark.get("summary_csv"), "AITER container benchmark CSV"
        )
        verify_output_reference(benchmark.get("log"), "AITER container benchmark log")
        wrapper_document = read_json(wrapper_json_path)
        wrapper_rows = wrapper_document.get("results", [])
        self.expect_exact(
            wrapper_document.get("schema"),
            "aiter-q4-group64-benchmark-v3",
            "AITER container: authoritative wrapper schema",
        )
        self.expect_exact(
            wrapper_document.get("configuration", {}).get("mapping_execution_schedule"),
            "cyclic_latin_by_sample_round",
            "AITER container: authoritative wrapper schedule",
        )
        self.expect_exact(
            len(wrapper_rows),
            benchmark.get("result_rows"),
            "AITER container: wrapper row cross-check",
        )
        self.expect(
            all(
                len(row.get("raw_us", [])) == benchmark.get("raw_samples_per_row")
                and len(row.get("raw_event_us", []))
                == benchmark.get("raw_samples_per_row")
                and len(row.get("raw_wall_us", []))
                == benchmark.get("raw_samples_per_row")
                for row in wrapper_rows
            ),
            "AITER container: wrapper event/wall raw sample cross-check",
        )

        def recompute_metric_aggregates(
            median_field: str,
        ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
            medians = {
                (
                    int(row["n"]),
                    int(row["k"]),
                    str(row["timing"]),
                    str(row["requested"]),
                ): float(row[median_field])
                for row in wrapper_rows
            }
            auto_aggregates: dict[str, dict[str, Any]] = {}
            selected_aggregates: dict[str, dict[str, Any]] = {}
            for timing_name in ("integration", "batched"):
                measured_shapes = sorted(
                    {(int(row["n"]), int(row["k"])) for row in wrapper_rows}
                )
                speedups = [
                    medians[(n, k, timing_name, "old")]
                    / medians[(n, k, timing_name, "auto")]
                    for n, k in measured_shapes
                ]
                selected_ratios = [
                    medians[(n, k, timing_name, "selected")]
                    / medians[(n, k, timing_name, "auto")]
                    for n, k in measured_shapes
                ]
                self.expect_exact(
                    len(speedups),
                    14,
                    f"AITER container {median_field}: auto shape count",
                )
                self.expect_exact(
                    len(selected_ratios),
                    14,
                    f"AITER container {median_field}: selected shape count",
                )
                auto_aggregates[timing_name] = {
                    "median": statistics.median(speedups),
                    "geomean": math.exp(
                        statistics.fmean(math.log(value) for value in speedups)
                    ),
                    "range": [min(speedups), max(speedups)],
                }
                selected_aggregates[timing_name] = {
                    "range": [min(selected_ratios), max(selected_ratios)],
                    "max_absolute_deviation_from_one": max(
                        abs(value - 1.0) for value in selected_ratios
                    ),
                }
            return auto_aggregates, selected_aggregates

        for metric_label, median_field, auto_key, selected_key in (
            (
                "primary wall",
                "median_wall_us",
                "auto_over_old_primary_wall",
                "selected_over_auto_primary_wall",
            ),
            (
                "supplemental event",
                "median_event_us",
                "auto_over_old_supplemental_event",
                "selected_over_auto_supplemental_event",
            ),
        ):
            computed_auto, computed_selected = recompute_metric_aggregates(median_field)
            recorded_auto = benchmark.get(auto_key)
            recorded_selected = benchmark.get(selected_key)
            self.require(
                isinstance(recorded_auto, dict),
                f"AITER container: {metric_label} auto summary",
            )
            self.require(
                isinstance(recorded_selected, dict),
                f"AITER container: {metric_label} selected summary",
            )
            self.expect_equal(
                recorded_auto,
                computed_auto,
                f"AITER container: recomputed {metric_label} auto aggregates",
            )
            self.expect_equal(
                recorded_selected,
                computed_selected,
                f"AITER container: recomputed {metric_label} selected aggregates",
            )

        correctness = benchmark.get("correctness")
        self.require(
            isinstance(correctness, dict), "AITER container: correctness summary"
        )
        max_abs_values = [float(row["correctness_max_abs"]) for row in wrapper_rows]
        relative_l2_values = [
            float(row["correctness_relative_l2"]) for row in wrapper_rows
        ]
        self.expect_equal(
            correctness,
            {
                "policy": "torch.testing.assert_close against dequantized FP32 reference",
                "rtol": 5.0e-4,
                "atol": 5.0e-3,
                "all_rows_correct": all(
                    row.get("correct") is True for row in wrapper_rows
                ),
                "maximum_abs_error_observed": max(max_abs_values),
                "maximum_relative_l2_observed": max(relative_l2_values),
                "relative_l2_is_diagnostic_not_the_pass_gate": True,
            },
            "AITER container: recomputed correctness summary",
        )

        roofline = benchmark.get("roofline")
        self.require(isinstance(roofline, dict), "AITER container: roofline summary")
        measured_shapes = sorted(
            {(int(row["n"]), int(row["k"])) for row in wrapper_rows}
        )
        arithmetic_intensities = []
        for n, k in measured_shapes:
            packed_bytes = n * k * 34 // 64
            traffic_bytes = packed_bytes + 4 * k + 4 * n
            arithmetic_intensities.append((2.0 * n * k) / traffic_bytes)
        batched_auto_effective_wall_gbps = [
            float(row["effective_wall_gbps"])
            for row in wrapper_rows
            if row["requested"] == "auto" and row["timing"] == "batched"
        ]
        peak_fp32_tflops = 48.7
        nominal_bandwidth_gbps = 640.0
        weight_only_intensity = (2.0 * 2048) / 1088
        self.expect_equal(
            roofline,
            {
                "packed_tile_weights": 2048,
                "packed_tile_bytes": 1088,
                "flops_per_weight": 2,
                "weight_only_arithmetic_intensity_flops_per_byte": weight_only_intensity,
                "measured_shape_arithmetic_intensity_range_flops_per_byte": [
                    min(arithmetic_intensities),
                    max(arithmetic_intensities),
                ],
                "rx9070xt_peak_fp32_tflops": peak_fp32_tflops,
                "rx9070xt_nominal_memory_bandwidth_gbps": nominal_bandwidth_gbps,
                "rx9070xt_ridge_point_flops_per_byte": (
                    peak_fp32_tflops * 1000.0 / nominal_bandwidth_gbps
                ),
                "asymptotic_dram_roof_tflops": (
                    weight_only_intensity * nominal_bandwidth_gbps / 1000.0
                ),
                "batched_auto_effective_wall_gbps": {
                    "minimum": min(batched_auto_effective_wall_gbps),
                    "median": statistics.median(batched_auto_effective_wall_gbps),
                    "maximum": max(batched_auto_effective_wall_gbps),
                },
                "traffic_definition": (
                    "packed bytes plus FP32 activation bytes plus FP32 output bytes divided by "
                    "synchronized host-wall time"
                ),
                "interpretation": (
                    "timing-derived logical bandwidth is cache-sensitive and is not a physical "
                    "DRAM counter; no measured percent-of-peak bandwidth claim is made"
                ),
                "hardware_specification": (
                    "https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/"
                    "amd-radeon-rx-9070xt.html"
                ),
            },
            "AITER container: recomputed roofline summary",
        )

        superseded_runs = benchmark.get("superseded_runs")
        self.require(
            isinstance(superseded_runs, list) and len(superseded_runs) == 3,
            "AITER container benchmark: expected v2 plus two fixed-order superseded runs",
        )
        for index, superseded in enumerate(superseded_runs):
            self.require(
                isinstance(superseded, dict), f"AITER superseded run {index}: record"
            )
            self.expect(
                isinstance(superseded.get("reason"), str)
                and bool(superseded["reason"]),
                f"AITER superseded run {index}: reason",
            )
        actual_superseded_hashes = {
            (record.get("raw_json_sha256"), record.get("summary_csv_sha256"))
            for record in superseded_runs
        }
        expected_superseded_hashes = {
            (
                self.cached_sha256(
                    directory / "container_wrapper_interleaved_raw.json"
                ),
                self.cached_sha256(
                    directory / "container_wrapper_interleaved_summary.csv"
                ),
            ),
            (
                self.cached_sha256(directory / "container_wrapper_final_raw.json"),
                self.cached_sha256(directory / "container_wrapper_final_summary.csv"),
            ),
            (
                self.cached_sha256(directory / "container_wrapper_raw.json"),
                self.cached_sha256(directory / "container_wrapper_summary.csv"),
            ),
        }
        self.expect_exact(
            actual_superseded_hashes,
            expected_superseded_hashes,
            "AITER container benchmark: superseded artifact hashes",
        )

        native = document.get("native_direct_kernel_reference")
        self.require(
            isinstance(native, dict),
            "AITER container validation: native kernel reference",
        )
        self.expect(
            "direct kernel launch batches" in str(native.get("timing_boundary")),
            "AITER native kernel: timing boundary",
        )
        self.expect(
            isinstance(native.get("host_rocm_runtime_version"), int)
            and native["host_rocm_runtime_version"] > 0,
            "AITER native kernel: ROCm runtime version",
        )
        for field, expected in (
            ("shape_count", 14),
            ("weight_copies", 72),
            ("samples_per_mapping", 30),
            ("target_sample_microseconds", 100000),
        ):
            self.expect_exact(
                native.get(field), expected, f"AITER native kernel: {field}"
            )
        kernel_json_path = verify_output_reference(
            native.get("raw_json"), "AITER native kernel raw JSON"
        )
        verify_output_reference(native.get("summary_csv"), "AITER native kernel CSV")
        kernel_document = read_json(kernel_json_path)
        self.expect_exact(
            len(kernel_document.get("shapes", [])),
            native.get("shape_count"),
            "AITER native: shape cross-check",
        )
        self.expect_exact(
            kernel_document.get("weight_copies"),
            native.get("weight_copies"),
            "AITER native: copies cross-check",
        )
        self.expect_exact(
            kernel_document.get("samples"),
            native.get("samples_per_mapping"),
            "AITER native: sample cross-check",
        )
        self.expect_exact(
            kernel_document.get("target_sample_us"),
            native.get("target_sample_microseconds"),
            "AITER native: target-time cross-check",
        )
        executed_driver_sha = native.get("executed_driver_sha256")
        self.expect(
            isinstance(executed_driver_sha, str)
            and re.fullmatch(r"[0-9a-f]{64}", executed_driver_sha) is not None,
            "AITER native: executed driver SHA-256",
        )
        portable_driver = native.get("portable_reproduction_driver")
        portable_driver_path = verify_output_reference(
            portable_driver, "AITER native portable reproduction driver"
        )
        self.expect(
            isinstance(portable_driver, dict)
            and "AITER-root-relative" in str(portable_driver.get("normalization"))
            and "output directory" in str(portable_driver.get("normalization")),
            "AITER native: portable driver normalization disclosure",
        )
        portable_source = portable_driver_path.read_text(encoding="utf-8")
        self.expect(
            portable_source.startswith('#include "csrc/kernels/q4_group64_gemv.cu"\n')
            and "/home/" not in portable_source,
            "AITER native: portable driver source/include",
        )

        supplemental_cases = self.verify_aiter_rocm714_smoke(
            directory,
            document.get("rocm714_native_smoke_supplemental"),
            source,
            source_manifest,
        )
        self.expect_exact(
            supplemental_cases, 17, "AITER ROCm 7.14 supplemental case count"
        )
        notes = document.get("notes")
        self.expect(
            isinstance(notes, list)
            and bool(notes)
            and all(isinstance(note, str) and bool(note) for note in notes),
            "AITER container validation: notes",
        )
        return len(source_files), checked_jit, source_manifest, source

    def verify_aiter_candidate(self) -> str:
        directory = self.results_root / "aiter_candidate"
        self.require(
            directory.is_dir(), f"AITER candidate: missing directory {directory}"
        )
        dispatch = self.aiter_dispatch_table()
        tests = self.verify_aiter_pytest(
            directory, "container_public_api_pytest.xml", AITER_PYTEST_CASES
        )
        legacy_tests = self.verify_aiter_pytest(
            directory, "container_pytest.xml", AITER_PYTEST_CASES_LEGACY
        )
        wrapper = self.verify_aiter_wrapper_results(
            directory,
            dispatch,
            "container_public_api",
            "aiter-q4-group64-benchmark-v3",
            True,
        )
        self.verify_aiter_wrapper_results(
            directory,
            dispatch,
            "container_wrapper_final",
            "aiter-q4-group64-benchmark-v1",
            False,
        )
        kernel = self.verify_aiter_kernel(directory, dispatch)
        self.verify_aiter_legacy_integration(directory, "cached", dispatch)
        self.verify_aiter_legacy_integration(directory, "uncached", dispatch)
        source_files, checked_jit, validation_manifest, validation_source = (
            self.verify_aiter_container_validation(directory)
        )

        metadata_path = directory / "metadata.json"
        self.require(
            metadata_path.is_file(), f"AITER candidate: missing {metadata_path}"
        )
        metadata = read_json(metadata_path)
        self.expect_exact(
            metadata.get("schema"),
            "aiter-q4-group64-candidate-evidence-v1",
            "AITER candidate metadata: schema",
        )
        metadata_manifest = self.verify_aiter_source_manifest(
            metadata.get("source"), "AITER candidate metadata source"
        )
        self.expect_exact(
            metadata_manifest,
            validation_manifest,
            "AITER candidate: metadata/container canonical source manifest",
        )
        metadata_source = metadata.get("source")
        self.require(
            isinstance(metadata_source, dict), "AITER candidate metadata: source"
        )
        for field in (
            "upstream_base_commit",
            "candidate_commit",
            "candidate_parent_commit",
            "candidate_commit_subject",
            "candidate_commit_dco_signed_off",
            "validation_source_state",
        ):
            self.expect_exact(
                metadata_source.get(field),
                validation_source.get(field),
                f"AITER candidate: metadata/container {field}",
            )
        self.expect_exact(
            metadata_source.get("local_branch"),
            validation_source.get("branch"),
            "AITER candidate: metadata/container branch",
        )
        patch_detail = self.verify_aiter_patch(
            directory,
            metadata_source,
            validation_source,
            validation_manifest,
        )
        self.verify_aiter_public_source_contract()
        self.verify_aiter_zero_dimension_contract(directory, validation_source)
        status = metadata.get("status")
        self.require(
            isinstance(status, dict), "AITER candidate metadata: status missing"
        )
        self.expect_exact(
            status.get("integration_uncached"),
            "superseded",
            "AITER candidate: uncached status",
        )
        self.expect_exact(
            status.get("integration_cached"),
            "current-historical-pybind-boundary",
            "AITER candidate: cached status",
        )
        self.expect_exact(
            status.get("container_wrapper_fixed_order"),
            "superseded",
            "AITER candidate: fixed-order wrapper status",
        )
        self.expect_exact(
            status.get("container_wrapper_interleaved"),
            "superseded",
            "AITER candidate: interleaved wrapper status",
        )
        self.expect_exact(
            status.get("container_public_api"),
            "authoritative",
            "AITER candidate: public-API status",
        )
        self.expect_exact(
            status.get("kernel_batched"),
            "current-direct-kernel-reference",
            "AITER candidate: kernel status",
        )
        self.expect_exact(
            status.get("rocm714_native_smoke"),
            "supplemental-non-authoritative",
            "AITER candidate: ROCm 7.14 smoke status",
        )
        timing = metadata.get("timing", {})
        self.expect(
            "superseded" in str(timing.get("integration_uncached", "")).lower(),
            "AITER candidate: uncached timing note does not state superseded",
        )
        self.expect(
            "superseded" in str(timing.get("container_interleaved", "")).lower(),
            "AITER candidate: interleaved timing note does not state superseded",
        )
        self.expect(
            "authoritative v3" in str(timing.get("container_public_api", ""))
            and "wall-clock primary" in str(timing.get("container_public_api", ""))
            and "event supplemental" in str(timing.get("container_public_api", "")),
            "AITER candidate: public-API timing authority/metric declaration",
        )
        readme = (directory / "README.md").read_text(encoding="utf-8")
        self.expect(
            "integration_uncached_raw.json" in readme
            and "superseded" in readme.lower(),
            "AITER candidate: README does not identify uncached results as superseded",
        )
        self.expect(
            "container_wrapper_interleaved_raw.json" in readme
            and "superseded v2" in readme.lower()
            and "container_public_api_raw.json" in readme
            and "authoritative schema-v3" in readme.lower()
            and "superseded" in readme.lower(),
            "AITER candidate: README does not identify v3 authority/v2 supersession",
        )
        self.expect(
            "aiter-q4-group64-gemv.patch" in readme
            and "--aiter-source /path/to/aiter" in readme
            and "supplemental native ROCm 7.14" in readme,
            "AITER candidate: README portable patch/strict/supplemental disclosure",
        )
        self.expect(
            "AIT_SOURCE=/path/to/aiter" in readme
            and '-I "$AIT_SOURCE"' in readme
            and '-I "$AIT_SOURCE/csrc/include"' in readme
            and "/tmp/direct_kernel_benchmark /tmp/q4-group64-direct-results" in readme,
            "AITER candidate: portable direct-kernel reproduction command",
        )
        artifacts = metadata.get("artifacts")
        self.require(
            isinstance(artifacts, dict),
            "AITER candidate metadata: artifact hashes missing",
        )
        for filename, expected_hash in artifacts.items():
            artifact_path = directory / filename
            self.require(
                artifact_path.is_file(),
                f"AITER candidate metadata: missing artifact {filename}",
            )
            self.expect_exact(
                self.cached_sha256(artifact_path),
                expected_hash,
                f"AITER candidate metadata hash {filename}",
            )

        def format_stats(values: dict[str, float]) -> str:
            return (
                f"median={values['median']:.3f}x/worst={values['worst']:.3f}x/"
                f"best={values['best']:.3f}x"
            )

        return (
            f"pytest={tests}/{len(AITER_PYTEST_CASES)} "
            f"(legacy XML={legacy_tests}/{len(AITER_PYTEST_CASES_LEGACY)}); "
            f"authoritative public-API v3=84 rows; auto wall integration "
            f"{format_stats(wrapper['integration'])}; auto wall batched {format_stats(wrapper['batched'])}; "
            f"direct kernel {format_stats(kernel)}; provenance source={source_files}/10, "
            f"patch={patch_detail}, extant JIT={checked_jit}/2; "
            f"ROCm7.14 native smoke=17/17 supplemental; v2/fixed-order/uncached=superseded"
        )

    def verify_dynamic_profile(self) -> str:
        directory = self.results_root / "profile_dynamic"
        summary_path = directory / "summary.json"
        self.require(summary_path.is_file(), f"dynamic profile: missing {summary_path}")
        summary = read_json(summary_path)
        self.expect_exact(
            summary.get("schema"), DYNAMIC_PROFILE_SCHEMA, "dynamic profile: schema"
        )
        case_dirs = sorted(path.parent for path in directory.glob("*/harness.json"))
        self.require(bool(case_dirs), "dynamic profile: no case directories")
        expected_cases: list[dict[str, Any]] = []
        for case_dir in case_dirs:
            harness_path = case_dir / "harness.json"
            stats_path = case_dir / "trace_kernel_stats.csv"
            trace_path = case_dir / "trace_kernel_trace.csv"
            json_trace_path = case_dir / "trace_results.json"
            for artifact in (stats_path, trace_path, json_trace_path):
                self.require(
                    artifact.is_file(),
                    f"dynamic profile {case_dir.name}: missing {artifact.name}",
                )
            harness = read_json(harness_path)
            self.expect_exact(
                harness.get("schema"),
                HARNESS_SCHEMA,
                f"dynamic {case_dir.name}: harness schema",
            )
            self.expect_exact(
                harness["correctness"]["passed"],
                self.logical_correctness(harness),
                f"dynamic {case_dir.name}: correctness predicate",
            )
            stats_rows = [
                row for row in read_csv(stats_path) if "q4rdna_kernel::" in row["Name"]
            ]
            trace_rows = [
                row
                for row in read_csv(trace_path)
                if "q4rdna_kernel::" in row["Kernel_Name"]
            ]
            self.require(
                len(stats_rows) == len(trace_rows) == 1,
                f"dynamic {case_dir.name}: expected one Q4_RDNA stats and trace row",
            )
            stat = stats_rows[0]
            trace = trace_rows[0]
            duration = int(stat["TotalDurationNs"])
            trace_duration = int(trace["End_Timestamp"]) - int(trace["Start_Timestamp"])
            self.expect_exact(
                duration,
                trace_duration,
                f"dynamic {case_dir.name}: stats/CSV trace duration",
            )
            self.expect_exact(
                int(stat["Calls"]), 1, f"dynamic {case_dir.name}: kernel call count"
            )
            self.expect_exact(
                stat["Name"],
                trace["Kernel_Name"],
                f"dynamic {case_dir.name}: kernel name",
            )

            trace_json = read_json(json_trace_path)
            tool_records = trace_json.get("rocprofiler-sdk-tool")
            self.require(
                isinstance(tool_records, list) and len(tool_records) == 1,
                f"dynamic {case_dir.name}: malformed trace_results.json",
            )
            tool_record = tool_records[0]
            symbols = {
                int(symbol["kernel_id"]): symbol
                for symbol in tool_record["kernel_symbols"]
            }
            json_dispatches: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for dispatch_record in tool_record["buffer_records"]["kernel_dispatch"]:
                info = dispatch_record["dispatch_info"]
                symbol = symbols[int(info["kernel_id"])]
                if "q4rdna_kernel::" in symbol["formatted_kernel_name"]:
                    json_dispatches.append((dispatch_record, symbol))
            self.require(
                len(json_dispatches) == 1,
                f"dynamic {case_dir.name}: expected one Q4_RDNA JSON dispatch",
            )
            json_dispatch, json_symbol = json_dispatches[0]
            self.expect_exact(
                int(json_dispatch["end_timestamp"])
                - int(json_dispatch["start_timestamp"]),
                duration,
                f"dynamic {case_dir.name}: stats/JSON trace duration",
            )
            self.expect_exact(
                json_symbol["formatted_kernel_name"],
                stat["Name"],
                f"dynamic {case_dir.name}: JSON kernel name",
            )
            dispatch_info = json_dispatch["dispatch_info"]
            workgroup = [int(trace[f"Workgroup_Size_{axis}"]) for axis in "XYZ"]
            grid = [int(trace[f"Grid_Size_{axis}"]) for axis in "XYZ"]
            self.expect_exact(
                [dispatch_info["workgroup_size"][axis.lower()] for axis in "XYZ"],
                workgroup,
                f"dynamic {case_dir.name}: JSON/CSV workgroup",
            )
            self.expect_exact(
                [dispatch_info["grid_size"][axis.lower()] for axis in "XYZ"],
                grid,
                f"dynamic {case_dir.name}: JSON/CSV grid",
            )
            case = harness["case"]
            correctness = harness["correctness"]
            expected_cases.append(
                {
                    "rows": case["rows"],
                    "columns": case["columns"],
                    "mode": case["mode"],
                    "requested_mapping": case["requested_mapping"],
                    "resolved_mapping": case["resolved_mapping"],
                    "correctness_passed": correctness["passed"],
                    "relative_l2": correctness["relative_l2"],
                    "kernel_name": stat["Name"],
                    "duration_ns": duration,
                    "launch": {
                        "lds_block_bytes": int(trace["LDS_Block_Size"]),
                        "scratch_bytes": int(trace["Scratch_Size"]),
                        "vgpr_count": int(trace["VGPR_Count"]),
                        "accum_vgpr_count": int(trace["Accum_VGPR_Count"]),
                        "sgpr_count": int(trace["SGPR_Count"]),
                        "workgroup": workgroup,
                        "grid": grid,
                    },
                    "source_files": {
                        "harness": str(
                            harness_path.resolve().relative_to(self.repository_root)
                        ),
                        "kernel_stats": str(
                            stats_path.resolve().relative_to(self.repository_root)
                        ),
                        "kernel_trace": str(
                            trace_path.resolve().relative_to(self.repository_root)
                        ),
                    },
                }
            )

        grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
        for case in expected_cases:
            grouped.setdefault(
                (case["rows"], case["columns"], case["mode"]), []
            ).append(case)
        expected_comparisons: list[dict[str, Any]] = []
        for (rows, columns, mode), cases in sorted(grouped.items()):
            old = next(
                (case for case in cases if case["resolved_mapping"] == "old"), None
            )
            self.require(
                old is not None,
                f"dynamic {rows}x{columns}/{mode}: missing old baseline",
            )
            for candidate in sorted(cases, key=lambda item: item["resolved_mapping"]):
                if candidate is old:
                    continue
                expected_comparisons.append(
                    {
                        "rows": rows,
                        "columns": columns,
                        "mode": mode,
                        "candidate": candidate["resolved_mapping"],
                        "old_duration_ns": old["duration_ns"],
                        "candidate_duration_ns": candidate["duration_ns"],
                        "single_trace_old_over_candidate": old["duration_ns"]
                        / candidate["duration_ns"],
                    }
                )
        self.expect_equal(
            summary.get("cases"), expected_cases, "dynamic profile: rebuilt cases"
        )
        self.expect_equal(
            summary.get("comparisons"),
            expected_comparisons,
            "dynamic profile: rebuilt comparisons",
        )
        self.expect_exact(
            summary.get("case_count"),
            len(expected_cases),
            "dynamic profile: case count",
        )
        self.expect_exact(
            summary.get("all_correctness_passed"),
            all(case["correctness_passed"] for case in expected_cases),
            "dynamic profile: correctness aggregate",
        )
        self.expect_equal(
            summary.get("environment"),
            read_json(case_dirs[0] / "harness.json")["environment"],
            "dynamic profile: environment",
        )
        return f"{len(expected_cases)} one-launch traces, {len(expected_comparisons)} comparisons"

    def verify_static_profile(self) -> str:
        path = self.results_root / "profile_static.json"
        self.require(path.is_file(), f"static profile: missing {path}")
        document = read_json(path)
        self.expect_exact(
            document.get("schema"), STATIC_PROFILE_SCHEMA, "static profile: schema"
        )
        self.expect_exact(document.get("status"), "ok", "static profile: status")
        kernels = document.get("kernels")
        self.require(
            isinstance(kernels, list) and kernels,
            "static profile: kernels must be non-empty",
        )
        scope = document.get("scope", {})
        self.expect_exact(
            scope.get("kernel_instance_count"),
            len(kernels),
            "static profile: kernel count",
        )
        keys = [(kernel["mode"], kernel["mapping"]) for kernel in kernels]
        self.expect_exact(
            len(set(keys)), len(keys), "static profile: unique mode/mapping instances"
        )
        expected_keys = {
            (mode, mapping)
            for mode in scope.get("modes", [])
            for mapping in scope.get("mappings", [])
        }
        self.expect_exact(
            set(keys), expected_keys, "static profile: mode/mapping coverage"
        )
        for kernel in kernels:
            resources = kernel["resources"]
            execution = kernel["execution"]
            label = f"static {kernel['mode']}/{kernel['mapping']}"
            self.expect(
                int(resources["vgpr_count"]) > 0, f"{label}: non-positive VGPR count"
            )
            self.expect(
                int(resources["sgpr_count"]) > 0, f"{label}: non-positive SGPR count"
            )
            self.expect(
                int(resources["vgpr_spill_count"]) >= 0,
                f"{label}: negative VGPR spills",
            )
            self.expect(
                int(resources["sgpr_spill_count"]) >= 0,
                f"{label}: negative SGPR spills",
            )
            self.expect_exact(
                execution.get("wavefront_size"), 32, f"{label}: wavefront size"
            )

        source = document.get("source_executable", {})
        expected_hash = source.get("sha256")
        self.require(
            isinstance(expected_hash, str),
            "static profile: source executable SHA-256 missing",
        )
        validation = document.get("validation", {})
        self.expect_exact(
            validation.get("final_executable_sha256"),
            expected_hash,
            "static profile: validation executable hash",
        )
        self.expect_exact(
            validation.get("final_code_object_sha256"),
            document.get("code_object", {}).get("sha256"),
            "static profile: validation code-object hash",
        )
        self.expect_exact(
            validation.get("kernel_instance_count_compared"),
            len(kernels),
            "static profile: validation kernel count",
        )

        candidate_paths: list[Path] = []
        if self.binary_override is not None:
            candidate_paths.append(self.binary_override)
        configured = source.get("path")
        if configured:
            candidate_paths.append(Path(str(configured)))
        candidate_paths.append(self.repository_root / "build" / "q4rdna_splitk_bench")
        checked: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidate_paths:
            resolved = candidate.expanduser().resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            checked.append(resolved)
            self.expect_exact(
                sha256_file(resolved),
                expected_hash,
                f"static profile: binary hash {resolved}",
            )
            self.expect_exact(
                resolved.stat().st_size,
                source.get("size_bytes"),
                f"static profile: binary size {resolved}",
            )
        if not checked:
            self.notes.append(
                "static profile: local harness binary absent; binary hash check skipped"
            )

        code_object_path = Path(
            str(document.get("code_object", {}).get("temporary_path", ""))
        )
        if str(code_object_path) and code_object_path.is_file():
            self.expect_exact(
                sha256_file(code_object_path),
                document["code_object"]["sha256"],
                "static profile: extant code-object hash",
            )
        if self.microbench is not None:
            self.expect_exact(
                self.microbench.get("runtime_setup", {}).get("binary_sha256"),
                expected_hash,
                "static/microbench binary hash",
            )
        return f"{len(kernels)} kernel instances; local binaries checked={len(checked)}"

    def verify(self) -> bool:
        self.require(
            self.results_root.is_dir(),
            f"results root does not exist: {self.results_root}",
        )
        self.run_section("correctness", self.verify_correctness)
        self.run_section("microbench+dispatch", self.verify_microbench_and_dispatch)
        self.run_section("aggregate+environment", self.verify_aggregate_and_environment)
        self.run_section("end-to-end", self.verify_all_end_to_end)
        self.run_section("mistral-assets", self.verify_mistral_assets)
        self.run_section(
            "legacy-completion-equivalence", self.verify_completion_equivalence
        )
        self.run_section(
            "greedy-completion-equivalence-v2", self.verify_all_greedy_completions
        )
        self.run_section("aiter-candidate", self.verify_aiter_candidate)
        self.run_section("dynamic-profile", self.verify_dynamic_profile)
        self.run_section("static-profile", self.verify_static_profile)
        return not self.errors

    def report(self) -> None:
        for name, passed, detail in self.section_results:
            marker = "PASS" if passed else "FAIL"
            suffix = f": {detail}" if detail else ""
            print(f"[{marker}] {name}{suffix}")
        for note in self.notes:
            print(f"[NOTE] {note}")
        if self.errors:
            print("\nVerification errors:", file=sys.stderr)
            for error in self.errors:
                print(f"- {error}", file=sys.stderr)
            print(
                f"\nVERDICT: FAIL ({len(self.errors)} errors across {self.checks} checks)",
                file=sys.stderr,
            )
        else:
            print(f"\nVERDICT: PASS ({self.checks} checks)")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independently verify results/upstream_splitk using only Python's standard library."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"evidence directory (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=None,
        help="optional local q4rdna_splitk_bench path for static-profile hash verification",
    )
    parser.add_argument(
        "--aiter-source",
        type=Path,
        default=None,
        help=(
            "strictly verify either the clean recorded AITER candidate commit or the upstream "
            "base with its patch applied against the candidate-file manifest; missing or "
            "mismatched files fail"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    verifier = Verifier(arguments.results, arguments.binary, arguments.aiter_source)
    try:
        passed = verifier.verify()
    except SectionAbort:
        passed = False
    verifier.report()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
