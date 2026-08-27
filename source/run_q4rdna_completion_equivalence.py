#!/usr/bin/env python3
"""Compare deterministic llama-completion output for old and Split-K Q4_RDNA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "q4rdna-completion-equivalence-v2"
BASE_SCHEMA = "q4rdna-completion-equivalence-v1"
Q4_ENV_PREFIX = "LLAMA_Q4_RDNA_"
DEFAULT_PROMPTS = (
    "The first three prime numbers are",
    "Write one short sentence about the moon:",
    "In Python, a list comprehension for squares from 0 to 4 is",
)

LOAD_PATTERN = re.compile(
    r"^Q4_RDNA: loaded (?P<tensors>\d+) tensors, (?P<gib>[0-9.]+) GiB "
    r"on device (?P<device>\d+) from (?P<path>.+)$",
    re.MULTILINE,
)
LAUNCH_PATTERN = re.compile(
    r"^Q4_RDNA: launched (?P<count>\d+) decode GEMV kernels$", re.MULTILINE
)
UNIQUE_PATTERN = re.compile(
    r"^Q4_RDNA: unique=(?P<count>\d+)(?:, (?P<hits>.*))?$", re.MULTILINE
)
HIT_PATTERN = re.compile(r"(?P<shape>\d+x\d+|other)=(?P<count>\d+)")
VERSION_PATTERN = re.compile(r"^version:\s*(?P<version>.+)$", re.MULTILINE)
GPU_ARCH_PATTERN = re.compile(r"^\s*Name:\s*(?P<arch>gfx\d+)\s*$", re.MULTILINE)
GPU_NAME_PATTERN = re.compile(r"^\s*Marketing Name:\s*(?P<name>.+?)\s*$", re.MULTILINE)
SAMPLER_CHAIN_PATTERN = re.compile(
    r"^[^\r\n]*\bI sampler chain:\s*(?P<chain>.+?)\s*$", re.MULTILINE
)
SAMPLER_TEMP_PATTERN = re.compile(r"(?:^|\s)temp = (?P<temperature>-?[0-9.]+)")
SAMPLER_SEED_PATTERN = re.compile(r"\bI sampler seed:\s*(?P<seed>\d+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Route:
    key: str
    mapping: str | None


ROUTES = (Route("old", "old"), Route("split", None))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    status = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": status.st_size,
        "sha256": sha256_file(resolved),
    }


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, document: Any) -> None:
    write_text(path, json.dumps(document, indent=2, ensure_ascii=False) + "\n")


def decode_output(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def output_bytes(value: bytes | str | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8")


def prepend_unique_paths(existing: str, additions: Sequence[Path]) -> str:
    values: list[str] = []
    for path in additions:
        value = str(path)
        if value not in values:
            values.append(value)
    for value in existing.split(os.pathsep):
        if value and value not in values:
            values.append(value)
    return os.pathsep.join(values)


def detect_rocm_library_path(environment: dict[str, str]) -> Path | None:
    hipconfig = shutil.which("hipconfig", path=environment.get("PATH"))
    if hipconfig is None:
        return None
    try:
        completed = subprocess.run(
            [hipconfig, "--path"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if completed.returncode != 0 or not lines:
        return None
    candidate = Path(lines[-1]).expanduser().resolve() / "lib"
    return candidate if candidate.is_dir() else None


def prepare_base_environment(
    binary: Path, library_directories: Sequence[Path]
) -> tuple[dict[str, str], dict[str, Any]]:
    environment = dict(os.environ)
    cleared = sorted(key for key in environment if key.startswith(Q4_ENV_PREFIX))
    for key in cleared:
        environment.pop(key, None)

    additions = [binary.parent, *library_directories]
    rocm_library = detect_rocm_library_path(environment)
    if rocm_library is not None:
        additions.append(rocm_library)
    environment["LD_LIBRARY_PATH"] = prepend_unique_paths(
        environment.get("LD_LIBRARY_PATH", ""), additions
    )
    return environment, {
        "cleared_inherited_q4rdna_variables": cleared,
        "library_directories_prepended": [str(path) for path in additions],
        "effective_ld_library_path": environment["LD_LIBRARY_PATH"],
    }


def expected_q4_environment(sidecar: Path, route: Route) -> dict[str, str]:
    expected = {
        "LLAMA_Q4_RDNA_SIDECAR": str(sidecar),
        "LLAMA_Q4_RDNA_TRACE": "1",
    }
    if route.mapping is not None:
        expected["LLAMA_Q4_RDNA_MAPPING"] = route.mapping
    return expected


def route_environment(
    base_environment: dict[str, str], sidecar: Path, route: Route, visible_device: str
) -> tuple[dict[str, str], dict[str, str]]:
    environment = dict(base_environment)
    environment["HIP_VISIBLE_DEVICES"] = visible_device
    environment["ROCR_VISIBLE_DEVICES"] = visible_device
    environment.update(expected_q4_environment(sidecar, route))
    effective_q4 = {
        key: value
        for key, value in sorted(environment.items())
        if key.startswith(Q4_ENV_PREFIX)
    }
    return environment, effective_q4


def build_command(
    binary: Path,
    model: Path,
    prompt: str,
    seed: int,
    n_predict: int,
    threads: int,
    gpu_layers: str,
) -> list[str]:
    return [
        str(binary),
        "-m",
        str(model),
        "-ngl",
        gpu_layers,
        "-t",
        str(threads),
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
        "-p",
        prompt,
    ]


def parse_runtime_evidence(stderr: bytes, sidecar: Path) -> dict[str, Any]:
    text = stderr.decode("utf-8", errors="replace")
    loads = [
        {
            "tensors": int(match.group("tensors")),
            "device_gib": float(match.group("gib")),
            "device": int(match.group("device")),
            "path": match.group("path").strip(),
        }
        for match in LOAD_PATTERN.finditer(text)
    ]
    launches = [int(match.group("count")) for match in LAUNCH_PATTERN.finditer(text)]
    unique_matches = list(UNIQUE_PATTERN.finditer(text))
    unique_counts = [int(match.group("count")) for match in unique_matches]
    hit_summaries = [
        {
            match.group("shape"): int(match.group("count"))
            for match in HIT_PATTERN.finditer(unique_match.group("hits") or "")
        }
        for unique_match in unique_matches
    ]
    q4_error_lines = [
        line
        for line in text.splitlines()
        if line.startswith("Q4_RDNA:")
        and any(
            word in line.lower()
            for word in ("error", "failed", "invalid", "mismatch", "truncated", "only supports")
        )
    ]
    sampler_name_errors = [
        line
        for line in text.splitlines()
        if "unable to match sampler" in line.lower()
    ]
    sampler_chains = [
        match.group("chain").strip() for match in SAMPLER_CHAIN_PATTERN.finditer(text)
    ]
    sampler_temperatures = [
        float(match.group("temperature"))
        for match in SAMPLER_TEMP_PATTERN.finditer(text)
    ]
    sampler_seeds = [
        int(match.group("seed")) for match in SAMPLER_SEED_PATTERN.finditer(text)
    ]
    exact_load = (
        len(loads) == 1
        and loads[0]["path"] == str(sidecar)
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


def validate_runtime_evidence(
    evidence: dict[str, Any],
    expected_environment: dict[str, str],
    observed_environment: dict[str, str],
    expected_seed: int,
) -> list[str]:
    errors: list[str] = []
    if observed_environment != expected_environment:
        errors.append(
            f"Q4_RDNA environment is {observed_environment!r}, expected {expected_environment!r}"
        )
    if not evidence["exact_sidecar_loaded"]:
        errors.append("did not log exactly one positive sidecar load from the requested path")
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
    elif any(temperature != 0.0 for temperature in evidence["sampler_temperatures"]):
        errors.append(
            f"effective sampler temperature is not zero: {evidence['sampler_temperatures']!r}"
        )
    if evidence["sampler_seeds"] != [expected_seed]:
        errors.append(
            f"effective sampler seed is {evidence['sampler_seeds']!r}, expected [{expected_seed}]"
        )
    return errors


def run_route(
    route: Route,
    prompt_id: str,
    prompt: str,
    binary: Path,
    model: Path,
    sidecar: Path,
    raw_directory: Path,
    base_environment: dict[str, str],
    visible_device: str,
    seed: int,
    n_predict: int,
    threads: int,
    gpu_layers: str,
    timeout_seconds: float | None,
) -> dict[str, Any]:
    environment, observed_q4_environment = route_environment(
        base_environment, sidecar, route, visible_device
    )
    expected_environment = expected_q4_environment(sidecar, route)
    command = build_command(
        binary, model, prompt, seed, n_predict, threads, gpu_layers
    )
    stdout_path = raw_directory / f"{prompt_id}.{route.key}.stdout"
    stderr_path = raw_directory / f"{prompt_id}.{route.key}.stderr"
    started_at = utc_now()
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=environment,
            timeout=timeout_seconds,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode: int | None = completed.returncode
        timed_out = False
        execution_error: str | None = None
    except subprocess.TimeoutExpired as error:
        stdout = output_bytes(error.stdout)
        stderr = output_bytes(error.stderr)
        returncode = None
        timed_out = True
        execution_error = "llama-completion timed out"
    except OSError as error:
        stdout = b""
        stderr = str(error).encode("utf-8", errors="replace")
        returncode = None
        timed_out = False
        execution_error = f"could not execute llama-completion: {error}"

    duration_seconds = time.monotonic() - start
    write_bytes(stdout_path, stdout)
    write_bytes(stderr_path, stderr)
    evidence = parse_runtime_evidence(stderr, sidecar)
    errors = validate_runtime_evidence(
        evidence, expected_environment, observed_q4_environment, seed
    )
    if returncode != 0:
        errors.append(f"llama-completion exit code is {returncode!r}, expected 0")
    if execution_error is not None:
        errors.append(execution_error)
    if not stdout:
        errors.append("llama-completion stdout is empty")
    return {
        "route": route.key,
        "started_at": started_at,
        "duration_seconds": duration_seconds,
        "returncode": returncode,
        "timed_out": timed_out,
        "passed": not errors,
        "errors": errors,
        "command": command,
        "expected_q4rdna_environment": expected_environment,
        "observed_q4rdna_environment": observed_q4_environment,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "stdout": stdout,
        "stderr": stderr,
        "runtime_evidence": evidence,
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


def relative_artifact(path: Path, output_directory: Path) -> str:
    return str(path.relative_to(output_directory))


def combine_pair(
    prompt_id: str,
    prompt: str,
    old: dict[str, Any],
    split: dict[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    old_stdout = old["stdout"]
    split_stdout = split["stdout"]
    difference = first_byte_difference(old_stdout, split_stdout)
    pair_errors = [
        *(f"old: {error}" for error in old["errors"]),
        *(f"split: {error}" for error in split["errors"]),
    ]
    if difference is not None:
        pair_errors.append(f"stdout differs at byte offset {difference['offset']}")
    old_evidence = old["runtime_evidence"]
    split_evidence = split["runtime_evidence"]
    return {
        "id": prompt_id,
        "prompt": prompt,
        "old_exit_code": old["returncode"],
        "split_exit_code": split["returncode"],
        "byte_identical": difference is None,
        "first_difference": difference,
        "runtime_valid": old["passed"] and split["passed"],
        "passed": old["passed"] and split["passed"] and difference is None,
        "errors": pair_errors,
        "stdout_bytes": len(old_stdout),
        "split_stdout_bytes": len(split_stdout),
        "stdout_sha256": sha256_bytes(old_stdout),
        "old_stdout_sha256": sha256_bytes(old_stdout),
        "split_stdout_sha256": sha256_bytes(split_stdout),
        "old_stdout": relative_artifact(old["stdout_path"], output_directory),
        "split_stdout": relative_artifact(split["stdout_path"], output_directory),
        "old_stderr": relative_artifact(old["stderr_path"], output_directory),
        "split_stderr": relative_artifact(split["stderr_path"], output_directory),
        "old_stderr_sha256": sha256_bytes(old["stderr"]),
        "split_stderr_sha256": sha256_bytes(split["stderr"]),
        "old_q4rdna_launches": old_evidence["launch_total"],
        "split_q4rdna_launches": split_evidence["launch_total"],
        "old_unique_tensors": old_evidence["unique_tensor_maximum"],
        "split_unique_tensors": split_evidence["unique_tensor_maximum"],
        "old_sidecar_loaded": old_evidence["exact_sidecar_loaded"],
        "split_sidecar_loaded": split_evidence["exact_sidecar_loaded"],
        "old_runtime_validation": {
            key: value
            for key, value in old.items()
            if key
            in (
                "started_at",
                "duration_seconds",
                "timed_out",
                "passed",
                "errors",
                "command",
                "expected_q4rdna_environment",
                "observed_q4rdna_environment",
                "runtime_evidence",
            )
        },
        "split_runtime_validation": {
            key: value
            for key, value in split.items()
            if key
            in (
                "started_at",
                "duration_seconds",
                "timed_out",
                "passed",
                "errors",
                "command",
                "expected_q4rdna_environment",
                "observed_q4rdna_environment",
                "runtime_evidence",
            )
        },
    }


def discover_ggml_hip(binary: Path, configured: Path | None) -> Path | None:
    if configured is not None:
        return configured.expanduser().resolve()
    candidates = (
        binary.parent / "libggml-hip.so",
        binary.parent.parent / "lib" / "libggml-hip.so",
        binary.parent.parent / "lib64" / "libggml-hip.so",
    )
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def run_probe(command: Sequence[str], environment: dict[str, str]) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()


def detect_binary_version(binary: Path, environment: dict[str, str]) -> str | None:
    output = run_probe((str(binary), "--version"), environment)
    if not output:
        return None
    match = VERSION_PATTERN.search(output)
    if match is not None:
        return match.group("version").strip()
    return next((line.strip() for line in output.splitlines() if line.strip()), None)


def detect_gpu(environment: dict[str, str]) -> tuple[str | None, str | None]:
    rocminfo = shutil.which("rocminfo", path=environment.get("PATH"))
    if rocminfo is None:
        return None, None
    output = run_probe((rocminfo,), environment)
    if output is None:
        return None, None
    architectures = GPU_ARCH_PATTERN.findall(output)
    names = [name.strip() for name in GPU_NAME_PATTERN.findall(output)]
    gpu_names = [name for name in names if name and "AMD Ryzen" not in name]
    return (gpu_names[0] if gpu_names else None, architectures[0] if architectures else None)


def build_launch_summary(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    observations: list[dict[str, Any]] = []
    for run in runs:
        for route in ("old", "split"):
            validation = run[f"{route}_runtime_validation"]
            evidence = validation["runtime_evidence"]
            load = evidence["sidecar_loads"][0] if len(evidence["sidecar_loads"]) == 1 else None
            hits = evidence["hit_summaries"][0] if len(evidence["hit_summaries"]) == 1 else None
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
                "loaded_tensors": item["loaded_tensors"],
                "loaded_device_gib": item["loaded_device_gib"],
                "unique_tensors_hit": item["unique_tensors_hit"],
                "hits_by_shape": item["hits_by_shape"],
            },
            sort_keys=True,
        )
        for item in observations
    }
    first = observations[0] if observations else {}
    return {
        "loaded_tensors": first.get("loaded_tensors"),
        "loaded_device_gib": first.get("loaded_device_gib"),
        "unique_tensors_hit": first.get("unique_tensors_hit"),
        "hits_by_shape": first.get("hits_by_shape"),
        "consistent_across_runs": len(signatures) == 1,
        "observations": observations,
    }


def render_summary(document: dict[str, Any]) -> str:
    generation = document["generation"]
    runtime = document["runtime"]
    model_label = generation["model_label"]
    lines = [
        f"# {model_label} old vs Split-K completion equivalence",
        "",
        f"Result: **{document['verdict'].upper()}**. "
        + (
            "Every old/Split-K stdout pair is byte-for-byte identical and passed runtime validation."
            if document["verdict"] == "pass"
            else "At least one pair failed byte equality or runtime validation."
        ),
        "",
        "## Fixed setup",
        "",
        f"- Binary: `{runtime['binary']}`",
        f"- Binary SHA-256: `{runtime['binary_sha256']}`",
        f"- libggml-hip SHA-256: `{runtime['ggml_hip_sha256']}`",
        f"- Model: `{runtime['model']}`",
        f"- Model SHA-256: `{runtime['model_sha256']}`",
        f"- Sidecar: `{runtime['sidecar']}`",
        f"- Sidecar SHA-256: `{runtime['sidecar_sha256']}`",
        f"- Sampling: `--samplers temperature --temp 0 --seed {generation['seed']} -n {generation['n_predict']}` (greedy argmax)",
        "- Old route sets `LLAMA_Q4_RDNA_MAPPING=old`.",
        "- Split-K route leaves `LLAMA_Q4_RDNA_MAPPING` unset.",
        "- Both routes set only `LLAMA_Q4_RDNA_SIDECAR` and `LLAMA_Q4_RDNA_TRACE` in addition to old's mapping override.",
        "",
        "## Results",
        "",
        "| Prompt | Old bytes | Split bytes | Old SHA-256 | Split SHA-256 | Exact | Runtime |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for run in document["runs"]:
        prompt = run["prompt"].replace("|", "\\|")
        lines.append(
            f"| `{prompt}` | {run['stdout_bytes']} | {run['split_stdout_bytes']} | "
            f"`{run['old_stdout_sha256']}` | `{run['split_stdout_sha256']}` | "
            f"{'yes' if run['byte_identical'] else 'no'} | {'pass' if run['passed'] else 'fail'} |"
        )
    failures = [run for run in document["runs"] if not run["passed"]]
    lines.extend(["", "## Validation", ""])
    if failures:
        for run in failures:
            lines.append(f"- `{run['id']}`: {'; '.join(run['errors'])}")
    else:
        lines.append(
            "All processes exited with status 0, loaded the exact requested sidecar path, "
            "reported non-zero Q4_RDNA launch and unique-tensor counts, and produced identical stdout bytes."
        )
    lines.extend(
        [
            "",
            "Raw stdout and stderr bytes are retained in `raw/`. This checks these fixed prompts "
            "for this exact model, sidecar, and runtime; it is not a proof for every prompt.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare greedy llama-completion output for Q4_RDNA old and default Split-K mappings."
    )
    parser.add_argument("--binary", type=Path, required=True, help="llama-completion executable")
    parser.add_argument("--model", type=Path, required=True, help="GGUF model")
    parser.add_argument("--sidecar", type=Path, required=True, help="matching Q4_RDNA sidecar")
    parser.add_argument("--output-dir", type=Path, required=True, help="new empty evidence directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only a prior raw/, SUMMARY.md, and completion_equivalence.json in the output directory",
    )
    parser.add_argument(
        "--ggml-hip-library",
        type=Path,
        help="libggml-hip file; defaults to discovery beside the executable",
    )
    parser.add_argument(
        "--library-dir",
        action="append",
        type=Path,
        default=[],
        help="additional runtime library directory to prepend; may be repeated",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        help="prompt to test; may be repeated (default: three fixed ASCII prompts)",
    )
    parser.add_argument("--model-label", help="heading label (default: model filename stem)")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--gpu-layers", default="all")
    parser.add_argument(
        "--visible-device",
        default="0",
        help="value assigned to HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="timeout per completion in seconds; zero disables it (default: 600)",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    return parser


def require_file(
    parser: argparse.ArgumentParser, path: Path, label: str, executable: bool = False
) -> None:
    if not path.is_file():
        parser.error(f"{label} does not exist: {path}")
    if executable and not os.access(path, os.X_OK):
        parser.error(f"{label} is not executable: {path}")


def validate_arguments(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    if arguments.n_predict <= 0:
        parser.error("--n-predict must be positive")
    if arguments.threads <= 0:
        parser.error("--threads must be positive")
    if arguments.timeout < 0:
        parser.error("--timeout must be non-negative")
    if not arguments.gpu_layers:
        parser.error("--gpu-layers must not be empty")
    if not arguments.visible_device:
        parser.error("--visible-device must not be empty")
    prompts = arguments.prompt if arguments.prompt is not None else DEFAULT_PROMPTS
    if not prompts or any(not prompt for prompt in prompts):
        parser.error("prompts must not be empty")


def prepare_output_directory(
    parser: argparse.ArgumentParser, output_directory: Path, overwrite: bool
) -> None:
    if output_directory.exists() and not output_directory.is_dir():
        parser.error(f"output path is not a directory: {output_directory}")
    if not output_directory.exists():
        output_directory.mkdir(parents=True)
        return

    entries = list(output_directory.iterdir())
    if not entries:
        return
    if not overwrite:
        parser.error(
            f"output directory is not empty: {output_directory}; choose a new directory or pass --overwrite"
        )

    allowed = {"raw", "SUMMARY.md", "completion_equivalence.json"}
    unexpected = sorted(entry.name for entry in entries if entry.name not in allowed)
    if unexpected:
        parser.error(
            "--overwrite refuses an output directory with unexpected entries: "
            + ", ".join(unexpected)
        )
    for entry in entries:
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    validate_arguments(parser, arguments)

    binary = arguments.binary.expanduser().resolve()
    model = arguments.model.expanduser().resolve()
    sidecar = arguments.sidecar.expanduser().resolve()
    output_directory = arguments.output_dir.expanduser().resolve()
    library_directories = [path.expanduser().resolve() for path in arguments.library_dir]
    require_file(parser, binary, "binary", executable=True)
    require_file(parser, model, "model")
    require_file(parser, sidecar, "sidecar")
    for library_directory in library_directories:
        if not library_directory.is_dir():
            parser.error(f"library directory does not exist: {library_directory}")

    ggml_hip = discover_ggml_hip(binary, arguments.ggml_hip_library)
    if ggml_hip is None:
        parser.error("could not discover libggml-hip; pass --ggml-hip-library")
    require_file(parser, ggml_hip, "libggml-hip")
    prepare_output_directory(parser, output_directory, arguments.overwrite)
    raw_directory = output_directory / "raw"
    raw_directory.mkdir()

    prompts = tuple(arguments.prompt) if arguments.prompt is not None else DEFAULT_PROMPTS
    model_label = arguments.model_label or model.stem
    base_environment, environment_setup = prepare_base_environment(
        binary, library_directories
    )
    timeout_seconds = arguments.timeout if arguments.timeout > 0 else None
    gpu_name, gpu_arch = detect_gpu(base_environment)
    binary_version = detect_binary_version(binary, base_environment)

    runs: list[dict[str, Any]] = []
    first_difference: dict[str, Any] | None = None
    for prompt_number, prompt in enumerate(prompts, start=1):
        prompt_id = f"prompt{prompt_number}"
        route_results: dict[str, dict[str, Any]] = {}
        for route in ROUTES:
            if not arguments.quiet:
                print(
                    f"[{prompt_id}] route={route.key}", file=sys.stderr, flush=True
                )
            route_results[route.key] = run_route(
                route=route,
                prompt_id=prompt_id,
                prompt=prompt,
                binary=binary,
                model=model,
                sidecar=sidecar,
                raw_directory=raw_directory,
                base_environment=base_environment,
                visible_device=arguments.visible_device,
                seed=arguments.seed,
                n_predict=arguments.n_predict,
                threads=arguments.threads,
                gpu_layers=arguments.gpu_layers,
                timeout_seconds=timeout_seconds,
            )
        combined = combine_pair(
            prompt_id,
            prompt,
            route_results["old"],
            route_results["split"],
            output_directory,
        )
        if first_difference is None and combined["first_difference"] is not None:
            first_difference = {
                "run": prompt_id,
                **combined["first_difference"],
            }
        runs.append(combined)

    all_identical = all(run["byte_identical"] for run in runs)
    all_runtime_valid = all(run["runtime_valid"] for run in runs)
    verdict = "pass" if all_identical and all_runtime_valid else "fail"
    binary_info = file_provenance(binary)
    ggml_info = file_provenance(ggml_hip)
    model_info = file_provenance(model)
    sidecar_info = file_provenance(sidecar)
    runtime = {
        "os": " ".join(platform.uname()),
        "gpu": gpu_name,
        "gpu_arch": gpu_arch,
        "binary": binary_info["path"],
        "binary_version": binary_version,
        "binary_size_bytes": binary_info["size_bytes"],
        "binary_sha256": binary_info["sha256"],
        "ggml_hip": ggml_info["path"],
        "ggml_hip_size_bytes": ggml_info["size_bytes"],
        "ggml_hip_sha256": ggml_info["sha256"],
        "model": model_info["path"],
        "model_size_bytes": model_info["size_bytes"],
        "model_sha256": model_info["sha256"],
        "sidecar": sidecar_info["path"],
        "sidecar_size_bytes": sidecar_info["size_bytes"],
        "sidecar_sha256": sidecar_info["sha256"],
        "ld_library_path": base_environment.get("LD_LIBRARY_PATH", ""),
        "hip_visible_devices": arguments.visible_device,
        "rocr_visible_devices": arguments.visible_device,
    }
    generation = {
        "model_label": model_label,
        "prompts": list(prompts),
        "samplers": "greedy",
        "sampler_implementation": "temperature sampler with temperature 0 (argmax)",
        "temperature": 0.0,
        "seed": arguments.seed,
        "n_predict": arguments.n_predict,
        "threads": arguments.threads,
        "gpu_layers": arguments.gpu_layers,
        "common_arguments": [
            "--samplers",
            "temperature",
            "--temp",
            "0",
            "--seed",
            str(arguments.seed),
            "-n",
            str(arguments.n_predict),
            "--no-display-prompt",
            "--no-conversation",
            "--simple-io",
        ],
        "old_environment": {
            "LLAMA_Q4_RDNA_MAPPING": "old",
            "LLAMA_Q4_RDNA_SCOPE": None,
            "LLAMA_Q4_RDNA_COOP": None,
            "LLAMA_Q4_RDNA_SMALL_ROWS": None,
        },
        "split_environment": {
            "LLAMA_Q4_RDNA_MAPPING": None,
            "LLAMA_Q4_RDNA_SCOPE": None,
            "LLAMA_Q4_RDNA_COOP": None,
            "LLAMA_Q4_RDNA_SMALL_ROWS": None,
        },
        "shared_environment": {
            "LLAMA_Q4_RDNA_TRACE": "1",
            "LLAMA_Q4_RDNA_SIDECAR": str(sidecar),
        },
        "environment_contract": environment_setup,
    }
    document = {
        "schema": SCHEMA,
        "derived_from_schema": BASE_SCHEMA,
        "created_utc": utc_now(),
        "verdict": verdict,
        "all_byte_identical": all_identical,
        "all_runtime_valid": all_runtime_valid,
        "first_difference": first_difference,
        "runtime": runtime,
        "generation": generation,
        "runs": runs,
        "launch_summary_per_run": build_launch_summary(runs),
    }
    write_json(output_directory / "completion_equivalence.json", document)
    write_text(output_directory / "SUMMARY.md", render_summary(document))
    if not arguments.quiet:
        print(f"verdict={verdict} output={output_directory}")
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
