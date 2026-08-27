#!/usr/bin/env python3
"""Run the reproducible Q4_RDNA old-mapping versus Split-K test suite.

The HIP harness remains the source of truth for input generation, correctness,
and timing.  This driver builds test matrices, preserves every harness JSON,
recomputes summary statistics from raw samples, and applies the documented
keep-better dispatch rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = REPOSITORY_ROOT / "build" / "q4rdna_splitk_bench"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "upstream_splitk"

HARNESS_SCHEMA = "q4rdna-splitk-bench-v1"
SUITE_SCHEMA = "q4rdna-splitk-suite-v1"
DISPATCH_SCHEMA = "q4rdna-splitk-dispatch-v1"

MODES = ("plain", "add", "gate-up")
PATTERNS = ("random", "zero", "zero-group", "alternating", "extreme", "group")
GENERAL_MAPPINGS = ("old", "split2", "split4", "split8")
SMALL_MAPPINGS = (
    "small8x8",
    "small8x16",
    "small8x32",
    "small16x16",
    "small16x32",
    "small32x32",
)
ALL_MAPPINGS = GENERAL_MAPPINGS + SMALL_MAPPINGS + ("auto",)

# The harness has a global 32-row alignment in addition to these mapping
# constraints.  Keeping both here makes invalid-case intent explicit.
MAPPING_ROW_MULTIPLE = {
    "old": 256,
    "split2": 128,
    "split4": 64,
    "split8": 32,
    "small8x8": 8,
    "small8x16": 16,
    "small8x32": 32,
    "small16x16": 16,
    "small16x32": 32,
    "small32x32": 32,
    "auto": 32,
}

MODEL_SHAPES: tuple[dict[str, Any], ...] = (
    {
        "model": "Qwen3-8B",
        "family": "Qwen3",
        "operators": (
            ("gate_up_single", 12288, 4096, "plain"),
            ("gate_up_fused", 12288, 4096, "gate-up"),
            ("down", 4096, 12288, "plain"),
            ("q_o", 4096, 4096, "plain"),
            ("k_v", 1024, 4096, "plain"),
        ),
    },
    {
        "model": "Mistral-7B-Instruct-v0.3",
        "family": "Mistral",
        "operators": (
            ("gate_up_single", 14336, 4096, "plain"),
            ("gate_up_fused", 14336, 4096, "gate-up"),
            ("down", 4096, 14336, "plain"),
            ("q_o", 4096, 4096, "plain"),
            ("k_v", 1024, 4096, "plain"),
        ),
    },
    {
        "model": "Qwen2.5-7B-Instruct",
        "family": "Qwen2.5",
        "operators": (
            ("gate_up_single", 18944, 3584, "plain"),
            ("gate_up_fused", 18944, 3584, "gate-up"),
            ("down", 3584, 18944, "plain"),
            ("q_o", 3584, 3584, "plain"),
            ("k_v", 512, 3584, "plain"),
        ),
    },
    {
        "model": "Phi-4-mini-instruct",
        "family": "Phi-4",
        "operators": (
            ("gate_up_single", 8192, 3072, "plain"),
            ("gate_up_fused", 8192, 3072, "gate-up"),
            ("down", 3072, 8192, "plain"),
            ("q_o", 3072, 3072, "plain"),
            ("k_v", 1024, 3072, "plain"),
        ),
    },
)


@dataclass
class HarnessCase:
    rows: int
    columns: int
    mode: str
    mapping: str
    pattern: str
    seed: int
    cache: str = "hot"
    tags: list[str] = field(default_factory=list)
    usages: list[dict[str, str]] = field(default_factory=list)

    def identity(self) -> tuple[Any, ...]:
        return (
            self.rows,
            self.columns,
            self.mode,
            self.mapping,
            self.pattern,
            self.seed,
            self.cache,
        )

    def slug(self) -> str:
        mode = self.mode.replace("-", "_")
        pattern = self.pattern.replace("-", "_")
        return (
            f"r{self.rows}_c{self.columns}_{mode}_{self.mapping}_{pattern}"
            f"_{self.cache}_s{self.seed}"
        )


@dataclass(frozen=True)
class InvalidProbe:
    name: str
    arguments: tuple[str, ...]
    expected_error: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object, base: int = 20260827) -> int:
    digest = hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).digest()
    return (base ^ int.from_bytes(digest[:4], "little")) & 0xFFFFFFFF


def mapping_is_legal(mapping: str, rows: int) -> bool:
    return rows > 0 and rows % 32 == 0 and rows % MAPPING_ROW_MULTIPLE[mapping] == 0


def linear_quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def sample_statistics(samples: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in samples if math.isfinite(float(value))]
    if len(finite) != len(samples) or not finite:
        return {
            "count": len(samples),
            "finite": len(finite) == len(samples),
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


def bootstrap_speedup_ci(
    baseline_samples: Sequence[float],
    candidate_samples: Sequence[float],
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any] | None:
    baseline = [float(value) for value in baseline_samples if float(value) > 0 and math.isfinite(float(value))]
    candidate = [float(value) for value in candidate_samples if float(value) > 0 and math.isfinite(float(value))]
    if not baseline or not candidate:
        return None

    generator = random.Random(seed)
    ratios: list[float] = []
    for _ in range(resamples):
        baseline_median = statistics.median(generator.choices(baseline, k=len(baseline)))
        candidate_median = statistics.median(generator.choices(candidate, k=len(candidate)))
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


def add_case(
    cases: dict[tuple[Any, ...], HarnessCase],
    rows: int,
    columns: int,
    mode: str,
    mapping: str,
    pattern: str,
    tag: str,
) -> None:
    seed = stable_seed("correctness", rows, columns, mode, mapping, pattern)
    case = HarnessCase(rows, columns, mode, mapping, pattern, seed, tags=[tag])
    existing = cases.get(case.identity())
    if existing is None:
        cases[case.identity()] = case
    elif tag not in existing.tags:
        existing.tags.append(tag)


def build_correctness_cases() -> list[HarnessCase]:
    cases: dict[tuple[Any, ...], HarnessCase] = {}
    boundary_columns = (64, 128, 192, 320, 576, 1088, 2112)
    row_options = (32, 64, 96, 128, 256, 512)

    # Full semantic coverage: every kernel mapping sees every operation and
    # every deterministic input pattern, including partial zero-scale groups.
    for mapping_index, mapping in enumerate(ALL_MAPPINGS):
        legal_rows = [row for row in row_options if mapping_is_legal(mapping, row)]
        for mode_index, mode in enumerate(MODES):
            for pattern_index, pattern in enumerate(PATTERNS):
                row = legal_rows[(mode_index + pattern_index) % len(legal_rows)]
                column_index = mapping_index * 5 + mode_index * 2 + pattern_index
                columns = boundary_columns[column_index % len(boundary_columns)]
                add_case(cases, row, columns, mode, mapping, pattern, "mapping-mode-pattern")

    # These shapes target partial row slices, K-group counts below split wave
    # counts, and group counts which do not divide 2/4/8/16/32 cleanly.
    boundary_shapes = (
        (32, 64),
        (32, 128),
        (32, 192),
        (64, 192),
        (64, 320),
        (96, 576),
        (128, 1088),
        (256, 2112),
        (512, 64),
        (1024, 128),
        (1024, 192),
    )
    for rows, columns in boundary_shapes:
        for mapping in ALL_MAPPINGS:
            if mapping_is_legal(mapping, rows):
                add_case(cases, rows, columns, "plain", mapping, "group", "split-boundary")

    # Large-K correctness uses modest row counts to cover real alignments
    # without duplicating the full model-shape performance sweep.
    large_k_shapes = (
        (512, 2560),
        (1024, 3072),
        (512, 3584),
        (1024, 4096),
        (256, 12288),
        (256, 14336),
        (256, 18944),
    )
    for rows, columns in large_k_shapes:
        for mapping in GENERAL_MAPPINGS + ("auto",):
            if mapping_is_legal(mapping, rows):
                add_case(cases, rows, columns, "plain", mapping, "random", "large-k")

    return sorted(cases.values(), key=lambda item: item.identity())


def invalid_correctness_probes() -> tuple[InvalidProbe, ...]:
    return (
        InvalidProbe(
            "rows-not-multiple-of-32",
            ("--rows", "33", "--columns", "64", "--mapping", "split8"),
            "rows must be a positive multiple of 32",
        ),
        InvalidProbe(
            "columns-not-multiple-of-64",
            ("--rows", "256", "--columns", "65", "--mapping", "old"),
            "columns must be a positive multiple of 64",
        ),
        InvalidProbe(
            "old-row-grid-incompatible",
            ("--rows", "32", "--columns", "192", "--mapping", "old"),
            "rows are incompatible with mapping old",
        ),
        InvalidProbe(
            "split2-row-grid-incompatible",
            ("--rows", "64", "--columns", "192", "--mapping", "split2"),
            "rows are incompatible with mapping split2",
        ),
    )


def model_usage_index() -> dict[tuple[int, int, str], list[dict[str, str]]]:
    result: dict[tuple[int, int, str], list[dict[str, str]]] = {}
    for model in MODEL_SHAPES:
        for role, rows, columns, mode in model["operators"]:
            result.setdefault((rows, columns, mode), []).append(
                {"model": model["model"], "family": model["family"], "role": role}
            )
    return result


def build_microbench_cases(cache_modes: Sequence[str]) -> list[HarnessCase]:
    cases: list[HarnessCase] = []
    for (rows, columns, mode), usages in sorted(model_usage_index().items()):
        mappings = list(GENERAL_MAPPINGS) + ["auto"]
        if rows <= 1024:
            mappings.extend(SMALL_MAPPINGS)
        for cache in cache_modes:
            seed = stable_seed("microbench", rows, columns, mode, cache)
            for mapping in mappings:
                if mapping_is_legal(mapping, rows):
                    cases.append(
                        HarnessCase(
                            rows=rows,
                            columns=columns,
                            mode=mode,
                            mapping=mapping,
                            pattern="random",
                            seed=seed,
                            cache=cache,
                            tags=["model-shape"],
                            usages=[dict(usage) for usage in usages],
                        )
                    )
    return cases


def prepare_runtime_environment() -> tuple[dict[str, str], dict[str, Any]]:
    environment = dict(os.environ)
    hipconfig = shutil.which("hipconfig")
    metadata: dict[str, Any] = {
        "hipconfig": hipconfig,
        "rocm_path": None,
        "prepended_library_path": None,
    }
    if hipconfig is None:
        return environment, metadata

    completed = subprocess.run(
        [hipconfig, "--path"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not lines:
        metadata["hipconfig_error"] = completed.stderr.strip()
        return environment, metadata

    rocm_path = Path(lines[-1]).expanduser()
    library_path = rocm_path / "lib"
    existing = [part for part in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
    combined: list[str] = []
    for part in (str(library_path), *existing):
        if part not in combined:
            combined.append(part)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(combined)
    metadata.update(
        {
            "rocm_path": str(rocm_path),
            "prepended_library_path": str(library_path),
            "effective_ld_library_path": combined,
        }
    )
    return environment, metadata


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_harness_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema") != HARNESS_SCHEMA:
        raise ValueError(f"unexpected harness schema in {path}: {document.get('schema')!r}")
    return document


def raw_matches(
    document: dict[str, Any],
    case: HarnessCase,
    warmup: int,
    samples: int,
    target_ms: float,
    weight_copies: int,
) -> bool:
    case_json = document.get("case", {})
    benchmark_json = document.get("benchmark", {})
    expected_copies = 1 if case.cache == "hot" else weight_copies
    return (
        case_json.get("rows") == case.rows
        and case_json.get("columns") == case.columns
        and case_json.get("mode") == case.mode
        and case_json.get("requested_mapping") == case.mapping
        and case_json.get("pattern") == case.pattern
        and case_json.get("seed") == case.seed
        and case_json.get("cache_mode") == case.cache
        and case_json.get("weight_copies") == expected_copies
        and benchmark_json.get("warmup") == warmup
        and len(benchmark_json.get("samples_us", [])) == samples
        and math.isclose(float(benchmark_json.get("target_sample_ms", -1.0)), target_ms)
    )


def run_harness_case(
    binary: Path,
    output_root: Path,
    raw_directory: Path,
    case: HarnessCase,
    warmup: int,
    samples: int,
    target_ms: float,
    weight_copies: int,
    runtime_environment: dict[str, str],
    resume: bool,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    raw_path = raw_directory / f"{case.slug()}.json"
    if resume and raw_path.is_file():
        try:
            document = load_harness_json(raw_path)
            if raw_matches(document, case, warmup, samples, target_ms, weight_copies):
                passed = bool(document.get("correctness", {}).get("passed"))
                return {
                    "id": case.slug(),
                    "request": case_to_json(case),
                    "raw_path": relative_path(raw_path, output_root),
                    "execution": {
                        "reused": True,
                        "returncode": 0 if passed else 1,
                        "duration_seconds": 0.0,
                    },
                    "result": document,
                    "passed": passed,
                }
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    raw_path.unlink(missing_ok=True)
    command = [
        str(binary),
        "--rows",
        str(case.rows),
        "--columns",
        str(case.columns),
        "--mode",
        case.mode,
        "--mapping",
        case.mapping,
        "--pattern",
        case.pattern,
        "--seed",
        str(case.seed),
        "--warmup",
        str(warmup),
        "--samples",
        str(samples),
        "--target-sample-ms",
        str(target_ms),
        "--cache",
        case.cache,
        "--weight-copies",
        str(weight_copies),
        "--output",
        str(raw_path),
    ]
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=runtime_environment,
            timeout=timeout_seconds,
        )
        returncode: int | None = completed.returncode
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        timed_out = False
    except subprocess.TimeoutExpired as error:
        returncode = None
        stdout = (error.stdout or "").strip() if isinstance(error.stdout, str) else ""
        stderr = (error.stderr or "").strip() if isinstance(error.stderr, str) else ""
        timed_out = True

    document: dict[str, Any] | None = None
    load_error: str | None = None
    if raw_path.is_file():
        try:
            document = load_harness_json(raw_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            load_error = str(error)
    else:
        load_error = "harness did not create the requested JSON output"

    request_matches = document is not None and raw_matches(
        document, case, warmup, samples, target_ms, weight_copies
    )
    if document is not None and not request_matches:
        load_error = "harness JSON does not match the requested case and timing protocol"
    passed = (
        returncode == 0
        and document is not None
        and request_matches
        and bool(document.get("correctness", {}).get("passed"))
    )
    execution: dict[str, Any] = {
        "reused": False,
        "returncode": returncode,
        "timed_out": timed_out,
        "duration_seconds": time.monotonic() - start,
        "command": command,
        "stdout": stdout,
        "stderr": stderr,
        "json_matches_request": request_matches,
    }
    if load_error is not None:
        execution["json_error"] = load_error
    return {
        "id": case.slug(),
        "request": case_to_json(case),
        "raw_path": relative_path(raw_path, output_root),
        "execution": execution,
        "result": document,
        "passed": passed,
    }


def case_to_json(case: HarnessCase) -> dict[str, Any]:
    return {
        "rows": case.rows,
        "columns": case.columns,
        "mode": case.mode,
        "mapping": case.mapping,
        "pattern": case.pattern,
        "seed": case.seed,
        "cache": case.cache,
        "tags": sorted(case.tags),
        "usages": case.usages,
    }


def run_invalid_probe(
    binary: Path,
    probe: InvalidProbe,
    runtime_environment: dict[str, str],
    timeout_seconds: float | None,
) -> dict[str, Any]:
    command = [str(binary), *probe.arguments]
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=runtime_environment,
            timeout=timeout_seconds,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        passed = completed.returncode == 2 and probe.expected_error in stderr
        returncode: int | None = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as error:
        stdout = (error.stdout or "").strip() if isinstance(error.stdout, str) else ""
        stderr = (error.stderr or "").strip() if isinstance(error.stderr, str) else ""
        passed = False
        returncode = None
        timed_out = True
    return {
        "name": probe.name,
        "command": command,
        "expected_returncode": 2,
        "expected_error": probe.expected_error,
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "duration_seconds": time.monotonic() - start,
        "passed": passed,
    }


def extract_environment(records: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    for record in records:
        result = record.get("result")
        if isinstance(result, dict) and isinstance(result.get("environment"), dict):
            return result["environment"]
    return None


def relative_l2_policy(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    policy: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        mode_results = [
            record["result"]["correctness"]
            for record in records
            if record["request"]["mode"] == mode and record.get("result") is not None
        ]
        limits = sorted(
            {
                float(result["relative_l2_limit"])
                for result in mode_results
                if result.get("relative_l2_limit") is not None
            }
        )
        errors = [
            float(result["relative_l2"])
            for result in mode_results
            if math.isfinite(float(result["relative_l2"]))
        ]
        policy[mode] = {
            "relative_l2_limits_reported_by_harness": limits,
            "maximum_relative_l2_observed": max(errors, default=None),
        }
    return policy


def aggregate_correctness(
    records: list[dict[str, Any]], invalid_records: list[dict[str, Any]], runtime: dict[str, Any]
) -> dict[str, Any]:
    passed = sum(bool(record["passed"]) for record in records)
    invalid_passed = sum(bool(record["passed"]) for record in invalid_records)
    relative_l2_values = [
        float(record["result"]["correctness"]["relative_l2"])
        for record in records
        if record.get("result") is not None
        and math.isfinite(float(record["result"]["correctness"]["relative_l2"]))
    ]
    max_absolute_values = [
        float(record["result"]["correctness"]["max_absolute"])
        for record in records
        if record.get("result") is not None
        and math.isfinite(float(record["result"]["correctness"]["max_absolute"]))
    ]
    coverage = {
        "mappings": sorted({record["request"]["mapping"] for record in records}),
        "modes": sorted({record["request"]["mode"] for record in records}),
        "patterns": sorted({record["request"]["pattern"] for record in records}),
        "rows": sorted({record["request"]["rows"] for record in records}),
        "columns": sorted({record["request"]["columns"] for record in records}),
    }
    status = passed == len(records) and invalid_passed == len(invalid_records)
    return {
        "schema": SUITE_SCHEMA,
        "kind": "correctness",
        "generated_at": utc_now(),
        "status": "passed" if status else "failed",
        "runtime_setup": runtime,
        "environment": extract_environment(records),
        "summary": {
            "positive_cases": len(records),
            "positive_passed": passed,
            "positive_failed": len(records) - passed,
            "invalid_cases": len(invalid_records),
            "invalid_passed": invalid_passed,
            "invalid_failed": len(invalid_records) - invalid_passed,
            "maximum_relative_l2": max(relative_l2_values, default=None),
            "maximum_absolute_error": max(max_absolute_values, default=None),
            "relative_l2_policy_by_mode": relative_l2_policy(records),
        },
        "coverage": coverage,
        "cases": records,
        "invalid_option_probes": invalid_records,
    }


def record_samples(record: dict[str, Any]) -> list[float]:
    result = record.get("result")
    if not isinstance(result, dict):
        return []
    return [float(value) for value in result.get("benchmark", {}).get("samples_us", [])]


def record_correct(record: dict[str, Any]) -> bool:
    result = record.get("result")
    return bool(
        record.get("passed")
        and isinstance(result, dict)
        and result.get("correctness", {}).get("passed")
    )


def comparison_rejection_reasons(
    correct: bool,
    speedup: float | None,
    confidence_interval: dict[str, Any] | None,
    minimum_speedup: float,
) -> list[str]:
    reasons: list[str] = []
    if not correct:
        reasons.append("correctness-failed")
    if speedup is None:
        reasons.append("missing-or-invalid-timing-samples")
    elif speedup < minimum_speedup:
        reasons.append(f"median-speedup-below-{minimum_speedup:.3f}")
    if confidence_interval is None:
        reasons.append("bootstrap-ci-unavailable")
    elif confidence_interval["lower"] <= 1.0:
        reasons.append("bootstrap-ci-lower-not-above-1.0")
    return reasons


def build_microbench_aggregate(
    records: list[dict[str, Any]],
    runtime: dict[str, Any],
    minimum_speedup: float,
    confidence: float,
    bootstrap_resamples: int,
    primary_cache: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    groups: dict[tuple[int, int, str, str], dict[str, dict[str, Any]]] = {}
    for record in records:
        request = record["request"]
        key = (request["rows"], request["columns"], request["mode"], request["cache"])
        groups.setdefault(key, {})[request["mapping"]] = record

    comparisons: list[dict[str, Any]] = []
    for (rows, columns, mode, cache), mapping_records in sorted(groups.items()):
        baseline = mapping_records.get("old")
        baseline_samples = record_samples(baseline) if baseline is not None else []
        baseline_stats = sample_statistics(baseline_samples)
        baseline_median = baseline_stats["median_us"]
        baseline_correct = baseline is not None and record_correct(baseline)
        candidate_rows: list[dict[str, Any]] = []

        for mapping, record in sorted(mapping_records.items()):
            if mapping == "old":
                continue
            samples = record_samples(record)
            stats = sample_statistics(samples)
            candidate_median = stats["median_us"]
            speedup = (
                baseline_median / candidate_median
                if baseline_median is not None and candidate_median is not None and candidate_median > 0
                else None
            )
            interval = bootstrap_speedup_ci(
                baseline_samples,
                samples,
                bootstrap_resamples,
                confidence,
                stable_seed("bootstrap", rows, columns, mode, cache, mapping),
            )
            correct = baseline_correct and record_correct(record)
            selection_candidate = mapping != "auto"
            reasons = comparison_rejection_reasons(correct, speedup, interval, minimum_speedup)
            qualifies = selection_candidate and not reasons
            if not selection_candidate:
                reasons = ["auto-is-observed-but-not-a-tuning-candidate", *reasons]
            result_case = record.get("result", {}).get("case", {}) if record.get("result") else {}
            candidate_rows.append(
                {
                    "mapping": mapping,
                    "resolved_mapping": result_case.get("resolved_mapping"),
                    "raw_path": record["raw_path"],
                    "correctness_passed": record_correct(record),
                    "statistics": stats,
                    "median_speedup_vs_old": speedup,
                    "bootstrap_speedup_ci": interval,
                    "selection_candidate": selection_candidate,
                    "qualifies": qualifies,
                    "selected": False,
                    "rejection_reasons": reasons,
                }
            )

        qualifying = [candidate for candidate in candidate_rows if candidate["qualifies"]]
        selected_candidate = max(
            qualifying,
            key=lambda candidate: float(candidate["median_speedup_vs_old"]),
            default=None,
        )
        if selected_candidate is not None:
            selected_candidate["selected"] = True
            selected_candidate["rejection_reasons"] = []
            for candidate in qualifying:
                if candidate is not selected_candidate:
                    candidate["rejection_reasons"] = ["slower-than-selected-qualified-candidate"]
            chosen_mapping = selected_candidate["mapping"]
            chosen_median = selected_candidate["statistics"]["median_us"]
            chosen_speedup = selected_candidate["median_speedup_vs_old"]
            chosen_interval = selected_candidate["bootstrap_speedup_ci"]
            decision = "split-k-selected"
        else:
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
            decision = "old-fallback"

        usages = baseline["request"]["usages"] if baseline is not None else []
        comparisons.append(
            {
                "rows": rows,
                "columns": columns,
                "mode": mode,
                "cache": cache,
                "usages": usages,
                "baseline": {
                    "mapping": "old",
                    "raw_path": baseline["raw_path"] if baseline is not None else None,
                    "correctness_passed": baseline_correct,
                    "statistics": baseline_stats,
                },
                "candidates": candidate_rows,
                "decision": {
                    "mapping": chosen_mapping,
                    "median_us": chosen_median,
                    "median_speedup_vs_old": chosen_speedup,
                    "bootstrap_speedup_ci": chosen_interval,
                    "reason": decision,
                },
            }
        )

    primary = [comparison for comparison in comparisons if comparison["cache"] == primary_cache]
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

    all_records_passed = all(record_correct(record) for record in records)
    aggregate_statistics = {
        "primary_cache": primary_cache,
        "shape_mode_count": len(primary),
        "selected_split_k_count": sum(
            comparison["decision"]["mapping"] != "old" for comparison in primary
        ),
        "old_fallback_count": sum(comparison["decision"]["mapping"] == "old" for comparison in primary),
        "median_dispatch_speedup": statistics.median(primary_speedups) if primary_speedups else None,
        "worst_dispatch_speedup": min(primary_speedups, default=None),
        "best_dispatch_speedup": max(primary_speedups, default=None),
        "baseline_latency_weighted_speedup": (
            weighted_baseline / weighted_selected if weighted_selected > 0 else None
        ),
        "weight_definition": "number of model-role references for each deduplicated shape and mode",
        "total_weight": total_weight,
        "worst_no_regression_threshold": 0.99,
        "worst_no_regression_passed": bool(primary_speedups) and min(primary_speedups) >= 0.99,
    }
    aggregate = {
        "schema": SUITE_SCHEMA,
        "kind": "microbench",
        "generated_at": utc_now(),
        "status": "passed" if all_records_passed else "failed",
        "runtime_setup": runtime,
        "environment": extract_environment(records),
        "models": list(MODEL_SHAPES),
        "selection_policy": {
            "minimum_median_speedup": minimum_speedup,
            "bootstrap_confidence": confidence,
            "minimum_ci_lower_bound_exclusive": 1.0,
            "fallback": "old",
            "unseen_shape_fallback": "old",
            "small_mapping_scope": "model-derived output rows <= 1024",
        },
        "summary": {
            "measurements": len(records),
            "measurements_passed": sum(record_correct(record) for record in records),
            "measurements_failed": sum(not record_correct(record) for record in records),
            "relative_l2_policy_by_mode": relative_l2_policy(records),
            **aggregate_statistics,
        },
        "measurements": records,
        "comparisons": comparisons,
    }

    dispatch_entries = []
    for comparison in primary:
        dispatch_entries.append(
            {
                "key": f"{comparison['rows']}x{comparison['columns']}:{comparison['mode']}",
                "rows": comparison["rows"],
                "columns": comparison["columns"],
                "mode": comparison["mode"],
                "mapping": comparison["decision"]["mapping"],
                "median_speedup_vs_old": comparison["decision"]["median_speedup_vs_old"],
                "bootstrap_speedup_ci": comparison["decision"]["bootstrap_speedup_ci"],
                "reason": comparison["decision"]["reason"],
                "usages": comparison["usages"],
                "candidates": comparison["candidates"],
            }
        )
    dispatch = {
        "schema": DISPATCH_SCHEMA,
        "generated_at": utc_now(),
        "architecture": "gfx1201",
        "cache_basis": primary_cache,
        "default_for_unseen_legal_shapes": "old",
        "selection_policy": aggregate["selection_policy"],
        "summary": aggregate_statistics,
        "entries": dispatch_entries,
    }
    return aggregate, dispatch


def json_write(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=False, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def markdown_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def markdown_l2_policy(policy_by_mode: dict[str, dict[str, Any]]) -> str:
    descriptions = []
    for mode, policy in policy_by_mode.items():
        limits = policy["relative_l2_limits_reported_by_harness"]
        limit_text = "/".join(f"{limit:.1e}" for limit in limits) if limits else "not-reported"
        descriptions.append(f"`{mode}`={limit_text}")
    return ", ".join(descriptions)


def write_markdown_summary(
    path: Path,
    suite: str,
    correctness: dict[str, Any] | None,
    microbench: dict[str, Any] | None,
    dispatch: dict[str, Any] | None,
) -> None:
    lines = [
        "# Q4_RDNA gfx1201 Split-K test summary",
        "",
        f"Generated: `{utc_now()}`  ",
        f"Requested suite: `{suite}`",
        "",
    ]
    if correctness is not None:
        summary = correctness["summary"]
        lines.extend(
            [
                "## Correctness",
                "",
                f"Status: **{correctness['status']}**. Positive cases: "
                f"{summary['positive_passed']}/{summary['positive_cases']}; expected-invalid "
                f"cases: {summary['invalid_passed']}/{summary['invalid_cases']}.",
                "",
                f"Maximum relative L2: `{markdown_float(summary['maximum_relative_l2'], 8)}`; "
                f"maximum absolute error: `{markdown_float(summary['maximum_absolute_error'], 8)}`.",
                "",
                "Relative-L2 acceptance limits are read from each harness result: "
                + markdown_l2_policy(summary["relative_l2_policy_by_mode"])
                + ". The fused `gate-up` path uses its separately reported limit because "
                "Split-K FP32 reduction order can be amplified by SiLU; finite/output/canary "
                "checks remain mandatory.",
                "",
                "Coverage: "
                f"{len(correctness['coverage']['mappings'])} mappings, "
                f"{len(correctness['coverage']['modes'])} modes, "
                f"{len(correctness['coverage']['patterns'])} patterns.",
                "",
            ]
        )
        failed = [record for record in correctness["cases"] if not record["passed"]]
        failed.extend(record for record in correctness["invalid_option_probes"] if not record["passed"])
        if failed:
            lines.extend(["Failed correctness records:", ""])
            for record in failed[:20]:
                lines.append(f"- `{record.get('id', record.get('name', 'unknown'))}`")
            if len(failed) > 20:
                lines.append(f"- ... and {len(failed) - 20} more")
            lines.append("")

    if microbench is not None and dispatch is not None:
        summary = microbench["summary"]
        lines.extend(
            [
                "## Model-derived microbenchmark",
                "",
                f"Status: **{microbench['status']}**. Measurements: "
                f"{summary['measurements_passed']}/{summary['measurements']} passed correctness.",
                "",
                f"Selection cache: `{summary['primary_cache']}`. Split-K selected for "
                f"{summary['selected_split_k_count']}/{summary['shape_mode_count']} deduplicated "
                "shape/mode entries.",
                "",
                f"Dispatch median/worst speedup: "
                f"`{markdown_float(summary['median_dispatch_speedup'])}x` / "
                f"`{markdown_float(summary['worst_dispatch_speedup'])}x`; "
                f"baseline-latency-weighted speedup: "
                f"`{markdown_float(summary['baseline_latency_weighted_speedup'])}x`.",
                "",
                "Harness correctness limits observed in this sweep: "
                + markdown_l2_policy(summary["relative_l2_policy_by_mode"])
                + ". The fused `gate-up` path reports a separate limit because Split-K FP32 "
                "reduction order can be amplified by SiLU; finite/output/canary checks remain mandatory.",
                "",
                "| Shape | Mode | Model roles | Chosen | Old us | Chosen us | Speedup | 95% CI | Decision |",
                "|---|---|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        comparison_index = {
            (item["rows"], item["columns"], item["mode"], item["cache"]): item
            for item in microbench["comparisons"]
        }
        for entry in dispatch["entries"]:
            comparison = comparison_index[
                (entry["rows"], entry["columns"], entry["mode"], dispatch["cache_basis"])
            ]
            roles = ", ".join(
                f"{usage['model']}:{usage['role']}" for usage in entry["usages"]
            )
            interval = entry["bootstrap_speedup_ci"]
            interval_text = (
                f"{markdown_float(interval.get('lower'))}-{markdown_float(interval.get('upper'))}"
                if interval is not None
                else "n/a"
            )
            lines.append(
                f"| {entry['rows']}x{entry['columns']} | {entry['mode']} | {roles} | "
                f"{entry['mapping']} | "
                f"{markdown_float(comparison['baseline']['statistics']['median_us'])} | "
                f"{markdown_float(comparison['decision']['median_us'])} | "
                f"{markdown_float(entry['median_speedup_vs_old'])}x | {interval_text} | "
                f"{entry['reason']} |"
            )
        lines.extend(
            [
                "",
                "Keep-better rule: select an explicit Split-K mapping only when median speedup "
                ">= 1.03x and the independently bootstrapped 95% confidence interval has a lower "
                "bound above 1.00x. All other and unseen legal shapes use `old`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Artifacts",
            "",
            "Raw per-invocation harness JSON is retained under `raw/`. Derived statistics are "
            "recomputed from `samples_us`; failed candidates remain in the aggregate and dispatch records.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Q4_RDNA Split-K correctness tests, four-model shape microbenchmarks, "
            "or both; preserve raw JSON and derive a conservative gfx1201 dispatch table."
        )
    )
    parser.add_argument("suite", choices=("correctness", "microbench", "all"))
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY, help="HIP benchmark executable")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="result directory"
    )
    parser.add_argument("--samples", type=int, default=30, help="microbenchmark samples per mapping")
    parser.add_argument("--warmup", type=int, default=100, help="microbenchmark warmup launches")
    parser.add_argument(
        "--target-ms", type=float, default=100.0, help="target duration for each timing sample"
    )
    parser.add_argument(
        "--cache",
        choices=("rotating", "hot", "both"),
        default="rotating",
        help="weight-cache protocol; rotating is the keep-better default",
    )
    parser.add_argument(
        "--weight-copies", type=int, default=8, help="weight buffers used in rotating-cache mode"
    )
    parser.add_argument(
        "--minimum-speedup", type=float, default=1.03, help="keep-better median speedup threshold"
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=5000, help="bootstrap draws per comparison"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95, help="two-sided bootstrap confidence level"
    )
    parser.add_argument(
        "--resume", action="store_true", help="reuse matching, valid raw harness JSON"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="per-process timeout in seconds; zero disables the timeout",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress per-case progress")
    return parser


def validate_arguments(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    if arguments.samples < 2 and arguments.suite in ("microbench", "all"):
        parser.error("--samples must be at least 2 for microbench confidence intervals")
    if arguments.warmup < 0:
        parser.error("--warmup must be non-negative")
    if arguments.target_ms <= 0:
        parser.error("--target-ms must be positive")
    if arguments.weight_copies <= 0:
        parser.error("--weight-copies must be positive")
    if arguments.minimum_speedup < 1.0:
        parser.error("--minimum-speedup must be at least 1.0")
    if arguments.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")
    if not 0.5 < arguments.confidence < 1.0:
        parser.error("--confidence must be between 0.5 and 1.0")
    if arguments.timeout < 0:
        parser.error("--timeout must be non-negative")


def announce(arguments: argparse.Namespace, index: int, total: int, case: HarnessCase) -> None:
    if not arguments.quiet:
        print(
            f"[{index}/{total}] {case.rows}x{case.columns} {case.mode} "
            f"{case.mapping} {case.pattern} cache={case.cache}",
            file=sys.stderr,
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    validate_arguments(parser, arguments)

    binary = arguments.binary.expanduser().resolve()
    output_dir = arguments.output_dir.expanduser().resolve()
    if not binary.is_file():
        parser.error(f"benchmark executable does not exist: {binary}")
    if not os.access(binary, os.X_OK):
        parser.error(f"benchmark is not executable: {binary}")

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_environment, runtime_metadata = prepare_runtime_environment()
    runtime_metadata["binary_sha256"] = sha256_file(binary)
    timeout_seconds = arguments.timeout if arguments.timeout > 0 else None
    correctness_document: dict[str, Any] | None = None
    microbench_document: dict[str, Any] | None = None
    dispatch_document: dict[str, Any] | None = None

    if arguments.suite in ("correctness", "all"):
        correctness_cases = build_correctness_cases()
        raw_directory = output_dir / "raw" / "correctness"
        raw_directory.mkdir(parents=True, exist_ok=True)
        correctness_records: list[dict[str, Any]] = []
        for index, case in enumerate(correctness_cases, start=1):
            announce(arguments, index, len(correctness_cases), case)
            correctness_records.append(
                run_harness_case(
                    binary=binary,
                    output_root=output_dir,
                    raw_directory=raw_directory,
                    case=case,
                    warmup=0,
                    samples=0,
                    target_ms=1.0,
                    weight_copies=1,
                    runtime_environment=runtime_environment,
                    resume=arguments.resume,
                    timeout_seconds=timeout_seconds,
                )
            )
        invalid_records: list[dict[str, Any]] = []
        for probe in invalid_correctness_probes():
            if not arguments.quiet:
                print(f"[invalid] {probe.name}", file=sys.stderr, flush=True)
            invalid_records.append(
                run_invalid_probe(binary, probe, runtime_environment, timeout_seconds)
            )
        correctness_document = aggregate_correctness(
            correctness_records, invalid_records, runtime_metadata
        )
        json_write(output_dir / "correctness.json", correctness_document)

    if arguments.suite in ("microbench", "all"):
        cache_modes = ("hot", "rotating") if arguments.cache == "both" else (arguments.cache,)
        primary_cache = "rotating" if "rotating" in cache_modes else "hot"
        microbench_cases = build_microbench_cases(cache_modes)
        raw_directory = output_dir / "raw" / "microbench"
        raw_directory.mkdir(parents=True, exist_ok=True)
        microbench_records: list[dict[str, Any]] = []
        for index, case in enumerate(microbench_cases, start=1):
            announce(arguments, index, len(microbench_cases), case)
            microbench_records.append(
                run_harness_case(
                    binary=binary,
                    output_root=output_dir,
                    raw_directory=raw_directory,
                    case=case,
                    warmup=arguments.warmup,
                    samples=arguments.samples,
                    target_ms=arguments.target_ms,
                    weight_copies=arguments.weight_copies,
                    runtime_environment=runtime_environment,
                    resume=arguments.resume,
                    timeout_seconds=timeout_seconds,
                )
            )
        microbench_document, dispatch_document = build_microbench_aggregate(
            records=microbench_records,
            runtime=runtime_metadata,
            minimum_speedup=arguments.minimum_speedup,
            confidence=arguments.confidence,
            bootstrap_resamples=arguments.bootstrap_resamples,
            primary_cache=primary_cache,
        )
        json_write(output_dir / "microbench.json", microbench_document)
        json_write(output_dir / "dispatch.json", dispatch_document)

    environment = None
    if microbench_document is not None:
        environment = microbench_document.get("environment")
    if environment is None and correctness_document is not None:
        environment = correctness_document.get("environment")
    json_write(
        output_dir / "environment.json",
        {
            "schema": SUITE_SCHEMA,
            "generated_at": utc_now(),
            "harness_environment": environment,
            "runtime_setup": runtime_metadata,
            "binary": str(binary),
        },
    )

    requested_statuses = []
    if correctness_document is not None:
        requested_statuses.append(correctness_document["status"])
    if microbench_document is not None:
        requested_statuses.append(microbench_document["status"])
    overall_status = "passed" if requested_statuses and all(item == "passed" for item in requested_statuses) else "failed"
    artifacts = {
        "environment": "environment.json",
        "correctness": "correctness.json" if correctness_document is not None else None,
        "microbench": "microbench.json" if microbench_document is not None else None,
        "dispatch": "dispatch.json" if dispatch_document is not None else None,
        "summary": "SUMMARY.md",
        "raw": "raw/",
    }
    json_write(
        output_dir / "aggregate.json",
        {
            "schema": SUITE_SCHEMA,
            "kind": "aggregate",
            "generated_at": utc_now(),
            "requested_suite": arguments.suite,
            "status": overall_status,
            "artifacts": artifacts,
            "correctness_summary": correctness_document.get("summary") if correctness_document else None,
            "microbench_summary": microbench_document.get("summary") if microbench_document else None,
            "dispatch_summary": dispatch_document.get("summary") if dispatch_document else None,
        },
    )
    write_markdown_summary(
        output_dir / "SUMMARY.md",
        arguments.suite,
        correctness_document,
        microbench_document,
        dispatch_document,
    )

    if not arguments.quiet:
        print(f"suite={arguments.suite} status={overall_status} output={output_dir}")
    return 0 if overall_status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
