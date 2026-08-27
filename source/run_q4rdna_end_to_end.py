#!/usr/bin/env python3
"""Run a rotated three-way Q4_RDNA llama-bench end-to-end comparison.

Every measured invocation uses ``llama-bench -r 1``.  The three routes are
rotated by round to reduce order bias, while one external warmup invocation per
route is retained as an artifact but excluded from all reported statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "results" / "upstream_splitk" / "end_to_end_qwen3"
SCHEMA = "q4rdna-end-to-end-v1"
Q4_ENV_PREFIX = "LLAMA_Q4_RDNA_"

LOAD_PATTERN = re.compile(
    r"^Q4_RDNA: loaded (?P<tensors>\d+) tensors, (?P<gib>[0-9.]+) GiB "
    r"on device (?P<device>\d+) from (?P<path>.+)$",
    re.MULTILINE,
)
LAUNCH_PATTERN = re.compile(r"^Q4_RDNA: launched (?P<count>\d+) decode GEMV kernels$", re.MULTILINE)
UNIQUE_PATTERN = re.compile(r"^Q4_RDNA: unique=(?P<count>\d+),", re.MULTILINE)
MAPPING_PATTERN = re.compile(
    r"^Q4_RDNA: (?:selected )?mapping(?:=|:| )+(?P<mapping>[A-Za-z0-9_-]+)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Route:
    key: str
    label: str
    expected_mapping: str
    uses_sidecar: bool


ROUTES = (
    Route("production_q4_k_m", "Production Q4_K_M", "production", False),
    Route("q4_rdna_old", "Q4_RDNA old", "old", True),
    Route("q4_rdna_split_auto", "Q4_RDNA split/auto", "auto", True),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_generations(value: str) -> tuple[int, ...]:
    try:
        generations = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("generations must be comma-separated integers") from error
    if not generations or any(generation <= 0 for generation in generations):
        raise argparse.ArgumentTypeError("generations must contain positive integers")
    if len(set(generations)) != len(generations):
        raise argparse.ArgumentTypeError("generations must not contain duplicates")
    return generations


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: Path) -> dict[str, Any]:
    status = path.stat()
    return {
        "path": str(path),
        "size_bytes": status.st_size,
        "sha256": sha256_file(path),
    }


def json_write(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


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

    try:
        completed = subprocess.run(
            [hipconfig, "--path"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        metadata["hipconfig_error"] = "hipconfig --path timed out"
        return environment, metadata
    paths = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not paths:
        metadata["hipconfig_error"] = completed.stderr.strip()
        return environment, metadata

    rocm_path = Path(paths[-1]).expanduser()
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


def route_environment(
    base_environment: dict[str, str], route: Route, sidecar: Path, visible_device: str
) -> tuple[dict[str, str], dict[str, Any]]:
    environment = dict(base_environment)
    cleared = sorted(key for key in environment if key.startswith(Q4_ENV_PREFIX))
    for key in cleared:
        environment.pop(key, None)

    environment["HIP_VISIBLE_DEVICES"] = visible_device
    environment["ROCR_VISIBLE_DEVICES"] = visible_device
    if route.uses_sidecar:
        environment["LLAMA_Q4_RDNA_SIDECAR"] = str(sidecar)
    if route.expected_mapping == "old":
        environment["LLAMA_Q4_RDNA_MAPPING"] = "old"

    controlled = {
        key: value
        for key, value in sorted(environment.items())
        if key.startswith(Q4_ENV_PREFIX)
        or key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "LD_LIBRARY_PATH")
    }
    return environment, {
        "cleared_inherited_q4rdna_variables": cleared,
        "controlled_environment": controlled,
    }


def expected_q4_environment(route: Route, sidecar: Path) -> dict[str, str]:
    if route.key == "production_q4_k_m":
        return {}
    if route.key == "q4_rdna_old":
        return {
            "LLAMA_Q4_RDNA_MAPPING": "old",
            "LLAMA_Q4_RDNA_SIDECAR": str(sidecar),
        }
    return {"LLAMA_Q4_RDNA_SIDECAR": str(sidecar)}


def build_command(
    binary: Path,
    model: Path,
    generations: Sequence[int],
    threads: int,
    batch: int,
    ubatch: int,
    gpu_layers: int,
) -> list[str]:
    return [
        str(binary),
        "-m",
        str(model),
        "-ngl",
        str(gpu_layers),
        "-p",
        "0",
        "-n",
        ",".join(str(generation) for generation in generations),
        "-b",
        str(batch),
        "-ub",
        str(ubatch),
        "-t",
        str(threads),
        "-r",
        "1",
        "-o",
        "json",
    ]


def decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def parse_stdout_json(stdout: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        document = json.loads(stdout)
    except json.JSONDecodeError as error:
        return None, f"stdout is not valid JSON: {error}"
    if not isinstance(document, list):
        return None, "llama-bench JSON root is not an array"
    if not all(isinstance(item, dict) for item in document):
        return None, "llama-bench JSON array contains a non-object item"
    return document, None


def validate_benchmark_rows(
    rows: list[dict[str, Any]] | None,
    generations: Sequence[int],
    model: Path,
    expected_model_family: str,
    threads: int,
    batch: int,
    ubatch: int,
    gpu_layers: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if rows is None:
        return [], errors

    by_generation: dict[int, dict[str, Any]] = {}
    for row in rows:
        try:
            generation = int(row["n_gen"])
        except (KeyError, TypeError, ValueError):
            errors.append("result row has no valid n_gen")
            continue
        if generation in by_generation:
            errors.append(f"duplicate result for n_gen={generation}")
        by_generation[generation] = row

        checks = {
            "n_prompt": 0,
            "n_threads": threads,
            "n_batch": batch,
            "n_ubatch": ubatch,
            "n_gpu_layers": gpu_layers,
        }
        for field_name, expected in checks.items():
            if row.get(field_name) != expected:
                errors.append(
                    f"n_gen={generation} field {field_name} is {row.get(field_name)!r}, expected {expected!r}"
                )
        if str(row.get("model_filename", "")) not in (str(model), model.name):
            errors.append(
                f"n_gen={generation} model_filename does not match requested model: "
                f"{row.get('model_filename')!r}"
            )
        model_type = str(row.get("model_type", "")).lower()
        if expected_model_family.lower() not in model_type:
            errors.append(
                f"n_gen={generation} model_type does not contain expected family "
                f"{expected_model_family!r}: {row.get('model_type')!r}"
            )
        if not ("q4_k_m" in model_type or ("q4_k" in model_type and "medium" in model_type)):
            errors.append(
                f"n_gen={generation} model_type is not identified as Q4_K_M: {row.get('model_type')!r}"
            )
        try:
            avg_ts = float(row["avg_ts"])
            avg_ns = float(row["avg_ns"])
            if not math.isfinite(avg_ts) or avg_ts <= 0:
                errors.append(f"n_gen={generation} avg_ts is not positive and finite")
            if not math.isfinite(avg_ns) or avg_ns <= 0:
                errors.append(f"n_gen={generation} avg_ns is not positive and finite")
        except (KeyError, TypeError, ValueError):
            errors.append(f"n_gen={generation} has invalid avg_ts or avg_ns")

        for samples_field in ("samples_ts", "samples_ns"):
            samples = row.get(samples_field)
            if samples is not None and (not isinstance(samples, list) or len(samples) != 1):
                errors.append(f"n_gen={generation} {samples_field} does not contain exactly one repetition")

    expected = set(generations)
    observed = set(by_generation)
    if observed != expected:
        errors.append(
            f"result generations are {sorted(observed)}, expected {sorted(expected)}"
        )
    return [by_generation[generation] for generation in generations if generation in by_generation], errors


def validate_runtime_evidence(
    route: Route, stderr: str, environment: dict[str, str], sidecar: Path
) -> dict[str, Any]:
    q4_environment = {
        key: value for key, value in environment.items() if key.startswith(Q4_ENV_PREFIX)
    }
    expected_environment = expected_q4_environment(route, sidecar)
    environment_matches = q4_environment == expected_environment

    load_matches = list(LOAD_PATTERN.finditer(stderr))
    launch_matches = list(LAUNCH_PATTERN.finditer(stderr))
    unique_matches = list(UNIQUE_PATTERN.finditer(stderr))
    mapping_matches = [match.group("mapping") for match in MAPPING_PATTERN.finditer(stderr)]
    error_lines = [
        line
        for line in stderr.splitlines()
        if line.startswith("Q4_RDNA:")
        and any(word in line.lower() for word in ("invalid", "truncated", "mismatch", "only supports"))
    ]

    loaded = [
        {
            "tensors": int(match.group("tensors")),
            "gib": float(match.group("gib")),
            "device": int(match.group("device")),
            "path": match.group("path").strip(),
        }
        for match in load_matches
    ]
    launch_counts = [int(match.group("count")) for match in launch_matches]
    unique_counts = [int(match.group("count")) for match in unique_matches]

    errors: list[str] = []
    if not environment_matches:
        errors.append(
            f"Q4_RDNA environment is {q4_environment!r}, expected {expected_environment!r}"
        )
    if error_lines:
        errors.append("Q4_RDNA emitted an error diagnostic")

    if route.uses_sidecar:
        if len(loaded) != 1:
            errors.append(f"expected exactly one sidecar load log, observed {len(loaded)}")
        elif loaded[0]["path"] != str(sidecar):
            errors.append(
                f"loaded sidecar path is {loaded[0]['path']!r}, expected {str(sidecar)!r}"
            )
        elif loaded[0]["tensors"] <= 0:
            errors.append("sidecar load log reports no tensors")
        if not launch_counts or max(launch_counts) <= 0:
            errors.append("no non-zero Q4_RDNA decode launch count was logged")
        if not unique_counts or max(unique_counts) <= 0:
            errors.append("no non-zero Q4_RDNA unique tensor hit count was logged")
    else:
        if loaded or launch_counts or unique_counts or "Q4_RDNA:" in stderr:
            errors.append("production baseline unexpectedly emitted Q4_RDNA runtime logs")

    if mapping_matches:
        accepted = {route.expected_mapping}
        if route.expected_mapping == "auto":
            accepted.update(("split", "split-k", "default"))
        if any(mapping not in accepted for mapping in mapping_matches):
            errors.append(
                f"runtime mapping marker is {mapping_matches!r}, expected one of {sorted(accepted)!r}"
            )

    return {
        "passed": not errors,
        "expected_mapping": route.expected_mapping,
        "mapping_contract": (
            "old is selected by LLAMA_Q4_RDNA_MAPPING=old; auto/default Split-K is selected "
            "by explicitly leaving LLAMA_Q4_RDNA_MAPPING unset"
        ),
        "mapping_marker_emitted": bool(mapping_matches),
        "mapping_markers": mapping_matches,
        "expected_q4rdna_environment": expected_environment,
        "observed_q4rdna_environment": q4_environment,
        "environment_matches": environment_matches,
        "sidecar_loads": loaded,
        "decode_launch_counts": launch_counts,
        "unique_tensor_counts": unique_counts,
        "q4rdna_error_lines": error_lines,
        "errors": errors,
    }


def run_once(
    run_id: str,
    phase: str,
    round_number: int | None,
    order_position: int,
    route: Route,
    binary: Path,
    model: Path,
    expected_model_family: str,
    sidecar: Path,
    output_root: Path,
    artifact_prefix: Path,
    generations: Sequence[int],
    threads: int,
    batch: int,
    ubatch: int,
    gpu_layers: int,
    visible_device: str,
    base_environment: dict[str, str],
    timeout_seconds: float | None,
) -> dict[str, Any]:
    environment, environment_summary = route_environment(
        base_environment, route, sidecar, visible_device
    )
    command = build_command(binary, model, generations, threads, batch, ubatch, gpu_layers)
    stdout_path = artifact_prefix.with_suffix(".stdout.json")
    stderr_path = artifact_prefix.with_suffix(".stderr.log")
    metadata_path = artifact_prefix.with_suffix(".meta.json")
    started_at = utc_now()
    start = time.monotonic()

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            cwd=output_root,
            timeout=timeout_seconds,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode: int | None = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as error:
        stdout = decode_timeout_output(error.stdout)
        stderr = decode_timeout_output(error.stderr)
        returncode = None
        timed_out = True

    duration_seconds = time.monotonic() - start
    text_write(stdout_path, stdout)
    text_write(stderr_path, stderr)

    parsed, parse_error = parse_stdout_json(stdout)
    rows, row_errors = validate_benchmark_rows(
        parsed,
        generations,
        model,
        expected_model_family,
        threads,
        batch,
        ubatch,
        gpu_layers,
    )
    runtime_evidence = validate_runtime_evidence(route, stderr, environment, sidecar)
    errors: list[str] = []
    if returncode != 0:
        errors.append(f"llama-bench return code is {returncode!r}, expected 0")
    if timed_out:
        errors.append("llama-bench timed out")
    if parse_error is not None:
        errors.append(parse_error)
    errors.extend(row_errors)
    errors.extend(runtime_evidence["errors"])

    record = {
        "id": run_id,
        "phase": phase,
        "round": round_number,
        "order_position": order_position,
        "route": route.key,
        "route_label": route.label,
        "expected_mapping": route.expected_mapping,
        "started_at": started_at,
        "duration_seconds": duration_seconds,
        "returncode": returncode,
        "timed_out": timed_out,
        "passed": not errors,
        "errors": errors,
        "command": command,
        "environment_summary": environment_summary,
        "artifacts": {
            "stdout_json": relative_path(stdout_path, output_root),
            "stderr": relative_path(stderr_path, output_root),
            "metadata": relative_path(metadata_path, output_root),
        },
        "runtime_evidence": runtime_evidence,
        "results": rows,
        "stderr_tail": stderr.splitlines()[-20:],
    }
    json_write(metadata_path, record)
    return record


def round_order(round_number: int) -> tuple[Route, ...]:
    offset = (round_number - 1) % len(ROUTES)
    return ROUTES[offset:] + ROUTES[:offset]


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


def result_for_generation(record: dict[str, Any], generation: int) -> dict[str, Any] | None:
    for result in record.get("results", []):
        if result.get("n_gen") == generation:
            return result
    return None


def build_series(
    measurements: Sequence[dict[str, Any]], generations: Sequence[int], rounds: int
) -> dict[str, Any]:
    series: dict[str, Any] = {}
    for route in ROUTES:
        route_results = [
            record for record in measurements if record["route"] == route.key and record["passed"]
        ]
        generation_results: dict[str, Any] = {}
        for generation in generations:
            rows = [result_for_generation(record, generation) for record in route_results]
            valid_rows = [row for row in rows if row is not None]
            generation_results[str(generation)] = {
                "throughput_tokens_per_second": sample_summary(
                    [float(row["avg_ts"]) for row in valid_rows], rounds
                ),
                "elapsed_nanoseconds": sample_summary(
                    [float(row["avg_ns"]) for row in valid_rows], rounds
                ),
            }
        series[route.key] = {
            "label": route.label,
            "expected_mapping": route.expected_mapping,
            "generations": generation_results,
        }
    return series


def paired_gain_samples(
    measurements: Sequence[dict[str, Any]],
    generation: int,
    numerator_route: str,
    denominator_route: str,
) -> tuple[list[float], list[float]]:
    by_round: dict[int, dict[str, dict[str, Any]]] = {}
    for record in measurements:
        if not record["passed"] or record["round"] is None:
            continue
        result = result_for_generation(record, generation)
        if result is not None:
            by_round.setdefault(int(record["round"]), {})[record["route"]] = result

    throughput_gains: list[float] = []
    elapsed_reductions: list[float] = []
    for round_results in by_round.values():
        numerator = round_results.get(numerator_route)
        denominator = round_results.get(denominator_route)
        if numerator is None or denominator is None:
            continue
        throughput_gains.append(
            (float(numerator["avg_ts"]) / float(denominator["avg_ts"]) - 1.0) * 100.0
        )
        elapsed_reductions.append(
            (1.0 - float(numerator["avg_ns"]) / float(denominator["avg_ns"])) * 100.0
        )
    return throughput_gains, elapsed_reductions


def build_gains(
    series: dict[str, Any],
    measurements: Sequence[dict[str, Any]],
    generations: Sequence[int],
    rounds: int,
) -> dict[str, Any]:
    comparisons = (
        ("split_auto_vs_old", "q4_rdna_split_auto", "q4_rdna_old"),
        ("split_auto_vs_production", "q4_rdna_split_auto", "production_q4_k_m"),
    )
    gains: dict[str, Any] = {}
    for label, numerator_route, denominator_route in comparisons:
        by_generation: dict[str, Any] = {}
        for generation in generations:
            numerator = series[numerator_route]["generations"][str(generation)]
            denominator = series[denominator_route]["generations"][str(generation)]
            numerator_ts = numerator["throughput_tokens_per_second"]["mean"]
            denominator_ts = denominator["throughput_tokens_per_second"]["mean"]
            numerator_ns = numerator["elapsed_nanoseconds"]["mean"]
            denominator_ns = denominator["elapsed_nanoseconds"]["mean"]
            paired_ts, paired_ns = paired_gain_samples(
                measurements, generation, numerator_route, denominator_route
            )
            by_generation[str(generation)] = {
                "gain_from_route_means_percent": (
                    (numerator_ts / denominator_ts - 1.0) * 100.0
                    if numerator_ts is not None and denominator_ts not in (None, 0)
                    else None
                ),
                "elapsed_reduction_from_route_means_percent": (
                    (1.0 - numerator_ns / denominator_ns) * 100.0
                    if numerator_ns is not None and denominator_ns not in (None, 0)
                    else None
                ),
                "paired_round_throughput_gain_percent": sample_summary(paired_ts, rounds),
                "paired_round_elapsed_reduction_percent": sample_summary(paired_ns, rounds),
            }
        gains[label] = {
            "numerator": numerator_route,
            "denominator": denominator_route,
            "generations": by_generation,
        }
    return gains


def run_git(source: Path, arguments: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def discover_source_provenance(binary: Path) -> dict[str, Any]:
    cmake_cache: Path | None = None
    for parent in binary.parents:
        candidate = parent / "CMakeCache.txt"
        if candidate.is_file():
            cmake_cache = candidate
            break
    source: Path | None = None
    if cmake_cache is not None:
        for line in cmake_cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
                source = Path(line.split("=", 1)[1]).expanduser().resolve()
                break
    if source is None:
        return {"cmake_cache": str(cmake_cache) if cmake_cache else None, "source": None}

    head = run_git(source, ("rev-parse", "HEAD"))
    status = run_git(source, ("status", "--porcelain"))
    return {
        "cmake_cache": str(cmake_cache),
        "source": str(source),
        "git_head": head,
        "git_worktree_clean": status == "" if status is not None else None,
        "git_status_porcelain": status,
    }


def build_commits(records: Iterable[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(result["build_commit"])
            for record in records
            for result in record.get("results", [])
            if result.get("build_commit")
        }
    )


def markdown_number(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def write_summary(path: Path, document: dict[str, Any]) -> None:
    protocol = document["protocol"]
    lines = [
        f"# Q4_RDNA {protocol['model_label']} end-to-end comparison",
        "",
        f"Status: **{document['status']}**  ",
        f"Generated: `{document['generated_at']}`",
        "",
        "## Protocol",
        "",
        f"Each route received one excluded external warmup. Then {protocol['rounds']} measured "
        "rounds used `llama-bench -r 1`; route order followed a three-way Latin rotation.",
        "",
        f"Generations: `{','.join(map(str, protocol['generations']))}`; prompt tokens: `0`; "
        f"GPU layers: `{protocol['n_gpu_layers']}`; batch/microbatch: "
        f"`{protocol['n_batch']}/{protocol['n_ubatch']}`; host threads: `{protocol['n_threads']}`.",
        "",
        "Before every process, all inherited `LLAMA_Q4_RDNA_*` variables are removed. "
        "Production leaves them unset; old sets only `SIDECAR` and `MAPPING=old`; split/auto "
        "sets only `SIDECAR` and deliberately leaves `MAPPING` unset.",
        "",
        "The current integration does not necessarily emit a mapping-name marker. Mapping "
        "validation therefore requires the exact environment contract, the expected sidecar "
        "load path, and non-zero Q4_RDNA decode-launch and tensor-hit logs. Production must emit "
        "no Q4_RDNA log.",
        "",
        "## Results",
        "",
        "| Generation | Route | Valid rounds | Mean tok/s | SD | Median | Min | Max | Mean elapsed ns |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for generation in protocol["generations"]:
        for route in ROUTES:
            result = document["series"][route.key]["generations"][str(generation)]
            throughput = result["throughput_tokens_per_second"]
            elapsed = result["elapsed_nanoseconds"]
            lines.append(
                f"| {generation} | {route.label} | {throughput['count']}/{throughput['expected_count']} | "
                f"{markdown_number(throughput['mean'])} | {markdown_number(throughput['sample_stddev'])} | "
                f"{markdown_number(throughput['median'])} | {markdown_number(throughput['minimum'])} | "
                f"{markdown_number(throughput['maximum'])} | {markdown_number(elapsed['mean'], 0)} |"
            )

    lines.extend(
        [
            "",
            "## Split/auto gains",
            "",
            "| Generation | Comparison | Gain from mean tok/s | Mean paired-round gain | "
            "Mean elapsed reduction |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for comparison_name, comparison in document["gains"].items():
        for generation in protocol["generations"]:
            result = comparison["generations"][str(generation)]
            paired = result["paired_round_throughput_gain_percent"]
            lines.append(
                f"| {generation} | {comparison_name} | "
                f"{markdown_number(result['gain_from_route_means_percent'])}% | "
                f"{markdown_number(paired['mean'])}% | "
                f"{markdown_number(result['elapsed_reduction_from_route_means_percent'])}% |"
            )

    lines.extend(["", "## Rotation", ""])
    for round_record in document["schedule"]:
        lines.append(
            f"- Round {round_record['round']}: " + " -> ".join(round_record["order"])
        )

    provenance = document["provenance"]
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- llama-bench build commit(s): `{', '.join(provenance['llama_build_commits']) or 'unavailable'}`",
            f"- Source git HEAD: `{provenance['llama_source'].get('git_head') or 'unavailable'}`",
            f"- Binary SHA-256: `{provenance['binary']['sha256']}`",
            f"- Model SHA-256: `{provenance['model']['sha256']}`",
            f"- Sidecar SHA-256: `{provenance['sidecar']['sha256']}`",
            "",
        ]
    )

    failures = [record for record in (*document["warmups"], *document["measurements"]) if not record["passed"]]
    lines.extend(["## Validation", ""])
    lines.append(
        f"Warmups passed: {sum(record['passed'] for record in document['warmups'])}/"
        f"{len(document['warmups'])}; measured invocations passed: "
        f"{sum(record['passed'] for record in document['measurements'])}/"
        f"{len(document['measurements'])}."
    )
    lines.append("")
    if failures:
        lines.append("Failed invocations (raw stdout, stderr, and metadata remain under `raw/`):")
        lines.append("")
        for record in failures:
            lines.append(f"- `{record['id']}`: {'; '.join(record['errors'])}")
        lines.append("")
    else:
        lines.append(
            "All process, JSON, expected-model-family, Q4_K_M, sidecar-load, "
            "mapping-contract, and Q4_RDNA hit checks passed."
        )
        lines.append("")
    text_write(path, "\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run production Q4_K_M, Q4_RDNA old, and Q4_RDNA split/auto through "
            "Latin-rotated llama-bench -r1 rounds."
        )
    )
    parser.add_argument("--binary", type=Path, required=True, help="clean llama-bench executable")
    parser.add_argument("--model", type=Path, required=True, help="Q4_K_M GGUF")
    parser.add_argument(
        "--expected-model-family",
        default="qwen3",
        help="case-insensitive substring required in llama-bench model_type (default: qwen3)",
    )
    parser.add_argument(
        "--model-label",
        default="Qwen3",
        help="human-readable model family used in SUMMARY.md (default: Qwen3)",
    )
    parser.add_argument("--sidecar", type=Path, required=True, help="matching Q4_RDNA sidecar")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rounds", type=int, default=10, help="measured rounds per route")
    parser.add_argument(
        "--generations",
        type=parse_generations,
        default=parse_generations("128,512"),
        metavar="N[,N...]",
        help="generated-token tests (default: 128,512)",
    )
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--ubatch", type=int, default=512)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument(
        "--visible-device",
        default="0",
        help="value assigned to both HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="per-invocation timeout in seconds; zero disables it",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


def validate_arguments(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    if arguments.rounds <= 0:
        parser.error("--rounds must be positive")
    if arguments.threads <= 0:
        parser.error("--threads must be positive")
    if arguments.batch <= 0 or arguments.ubatch <= 0:
        parser.error("--batch and --ubatch must be positive")
    if arguments.gpu_layers < 0:
        parser.error("--gpu-layers must be non-negative")
    if arguments.timeout < 0:
        parser.error("--timeout must be non-negative")
    if not arguments.visible_device:
        parser.error("--visible-device must not be empty")
    if not arguments.expected_model_family.strip():
        parser.error("--expected-model-family must not be empty")
    if not arguments.model_label.strip():
        parser.error("--model-label must not be empty")


def require_file(parser: argparse.ArgumentParser, path: Path, label: str, executable: bool = False) -> None:
    if not path.is_file():
        parser.error(f"{label} does not exist: {path}")
    if executable and not os.access(path, os.X_OK):
        parser.error(f"{label} is not executable: {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    validate_arguments(parser, arguments)

    binary = arguments.binary.expanduser().resolve()
    model = arguments.model.expanduser().resolve()
    sidecar = arguments.sidecar.expanduser().resolve()
    output_dir = arguments.output_dir.expanduser().resolve()
    require_file(parser, binary, "binary", executable=True)
    require_file(parser, model, "model")
    require_file(parser, sidecar, "sidecar")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_environment, runtime_setup = prepare_runtime_environment()
    timeout_seconds = arguments.timeout if arguments.timeout > 0 else None
    provenance: dict[str, Any] = {
        "binary": file_provenance(binary),
        "model": file_provenance(model),
        "sidecar": file_provenance(sidecar),
        "llama_source": discover_source_provenance(binary),
        "llama_build_commits": [],
    }
    protocol = {
        "runtime": "llama.cpp ROCm llama-bench",
        "expected_model_family": arguments.expected_model_family,
        "model_label": arguments.model_label,
        "rounds": arguments.rounds,
        "external_warmups_per_route": 1,
        "llama_bench_repetitions_per_process": 1,
        "generations": list(arguments.generations),
        "n_prompt": 0,
        "n_gpu_layers": arguments.gpu_layers,
        "n_batch": arguments.batch,
        "n_ubatch": arguments.ubatch,
        "n_threads": arguments.threads,
        "visible_device": arguments.visible_device,
        "profiler_attached": False,
        "rotation": "three-way cyclic Latin rotation",
    }

    warmups: list[dict[str, Any]] = []
    warmup_directory = output_dir / "raw" / "warmup"
    for position, route in enumerate(ROUTES, start=1):
        if not arguments.quiet:
            print(f"[warmup {position}/3] {route.key}", file=sys.stderr, flush=True)
        prefix = warmup_directory / f"position_{position:02d}_{route.key}"
        warmups.append(
            run_once(
                run_id=f"warmup-{route.key}",
                phase="warmup",
                round_number=None,
                order_position=position,
                route=route,
                binary=binary,
                model=model,
                expected_model_family=arguments.expected_model_family,
                sidecar=sidecar,
                output_root=output_dir,
                artifact_prefix=prefix,
                generations=arguments.generations,
                threads=arguments.threads,
                batch=arguments.batch,
                ubatch=arguments.ubatch,
                gpu_layers=arguments.gpu_layers,
                visible_device=arguments.visible_device,
                base_environment=base_environment,
                timeout_seconds=timeout_seconds,
            )
        )

    measurements: list[dict[str, Any]] = []
    schedule: list[dict[str, Any]] = []
    total_measurements = arguments.rounds * len(ROUTES)
    measurement_number = 0
    for round_number in range(1, arguments.rounds + 1):
        order = round_order(round_number)
        schedule.append({"round": round_number, "order": [route.key for route in order]})
        round_directory = output_dir / "raw" / f"round_{round_number:03d}"
        for position, route in enumerate(order, start=1):
            measurement_number += 1
            if not arguments.quiet:
                print(
                    f"[measure {measurement_number}/{total_measurements}] "
                    f"round={round_number} position={position} route={route.key}",
                    file=sys.stderr,
                    flush=True,
                )
            prefix = round_directory / f"position_{position:02d}_{route.key}"
            measurements.append(
                run_once(
                    run_id=f"round-{round_number:03d}-position-{position:02d}-{route.key}",
                    phase="measurement",
                    round_number=round_number,
                    order_position=position,
                    route=route,
                    binary=binary,
                    model=model,
                    expected_model_family=arguments.expected_model_family,
                    sidecar=sidecar,
                    output_root=output_dir,
                    artifact_prefix=prefix,
                    generations=arguments.generations,
                    threads=arguments.threads,
                    batch=arguments.batch,
                    ubatch=arguments.ubatch,
                    gpu_layers=arguments.gpu_layers,
                    visible_device=arguments.visible_device,
                    base_environment=base_environment,
                    timeout_seconds=timeout_seconds,
                )
            )

    all_records = [*warmups, *measurements]
    provenance["llama_build_commits"] = build_commits(all_records)
    series = build_series(measurements, arguments.generations, arguments.rounds)
    gains = build_gains(
        series, measurements, arguments.generations, arguments.rounds
    )
    expected_measurements = arguments.rounds * len(ROUTES)
    all_processes_passed = (
        len(warmups) == len(ROUTES)
        and len(measurements) == expected_measurements
        and all(record["passed"] for record in all_records)
    )
    series_complete = all(
        series[route.key]["generations"][str(generation)][metric]["complete"]
        for route in ROUTES
        for generation in arguments.generations
        for metric in ("throughput_tokens_per_second", "elapsed_nanoseconds")
    )
    build_commit_consistent = len(provenance["llama_build_commits"]) == 1
    status = "passed" if all_processes_passed and series_complete and build_commit_consistent else "failed"

    document = {
        "schema": SCHEMA,
        "generated_at": utc_now(),
        "status": status,
        "protocol": protocol,
        "runtime_setup": {
            **runtime_setup,
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "provenance": provenance,
        "validation_summary": {
            "warmups_passed": sum(record["passed"] for record in warmups),
            "warmups_expected": len(ROUTES),
            "measurements_passed": sum(record["passed"] for record in measurements),
            "measurements_expected": expected_measurements,
            "series_complete": series_complete,
            "single_build_commit": build_commit_consistent,
        },
        "schedule": schedule,
        "warmups": warmups,
        "measurements": measurements,
        "series": series,
        "gains": gains,
        "artifacts": {
            "raw_directory": "raw/",
            "summary": "SUMMARY.md",
            "aggregate": "end_to_end.json",
        },
    }
    json_write(output_dir / "end_to_end.json", document)
    write_summary(output_dir / "SUMMARY.md", document)
    if not arguments.quiet:
        print(f"status={status} output={output_dir}")
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
