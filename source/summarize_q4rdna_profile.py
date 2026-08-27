#!/usr/bin/env python3
"""Summarize one-launch rocprofv3 traces for the Q4_RDNA Split-K harness.

The resulting durations are diagnostic trace samples, not benchmark estimates.
Use microbench.json for statistically sampled performance claims.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCHEMA = "q4rdna-rocprof-summary-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def as_int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def collect_case(case_dir: Path) -> dict[str, Any]:
    harness_path = case_dir / "harness.json"
    stats_path = case_dir / "trace_kernel_stats.csv"
    trace_path = case_dir / "trace_kernel_trace.csv"
    harness = read_json(harness_path)
    stats = read_csv(stats_path)
    traces = read_csv(trace_path)

    kernel_stats = [row for row in stats if "q4rdna_kernel::" in row["Name"]]
    kernel_traces = [row for row in traces if "q4rdna_kernel::" in row["Kernel_Name"]]
    if len(kernel_stats) != 1 or len(kernel_traces) != 1:
        raise ValueError(
            f"{case_dir}: expected exactly one Q4_RDNA kernel in stats and trace, "
            f"got {len(kernel_stats)} and {len(kernel_traces)}"
        )

    stat = kernel_stats[0]
    trace = kernel_traces[0]
    duration_ns = as_int(stat, "TotalDurationNs")
    trace_duration_ns = as_int(trace, "End_Timestamp") - as_int(
        trace, "Start_Timestamp"
    )
    if duration_ns != trace_duration_ns:
        raise ValueError(
            f"{case_dir}: stats duration {duration_ns} != trace duration {trace_duration_ns}"
        )

    case = harness["case"]
    correctness = harness["correctness"]
    return {
        "rows": case["rows"],
        "columns": case["columns"],
        "mode": case["mode"],
        "requested_mapping": case["requested_mapping"],
        "resolved_mapping": case["resolved_mapping"],
        "correctness_passed": correctness["passed"],
        "relative_l2": correctness["relative_l2"],
        "kernel_name": stat["Name"],
        "duration_ns": duration_ns,
        "launch": {
            "lds_block_bytes": as_int(trace, "LDS_Block_Size"),
            "scratch_bytes": as_int(trace, "Scratch_Size"),
            "vgpr_count": as_int(trace, "VGPR_Count"),
            "accum_vgpr_count": as_int(trace, "Accum_VGPR_Count"),
            "sgpr_count": as_int(trace, "SGPR_Count"),
            "workgroup": [
                as_int(trace, "Workgroup_Size_X"),
                as_int(trace, "Workgroup_Size_Y"),
                as_int(trace, "Workgroup_Size_Z"),
            ],
            "grid": [
                as_int(trace, "Grid_Size_X"),
                as_int(trace, "Grid_Size_Y"),
                as_int(trace, "Grid_Size_Z"),
            ],
        },
        "source_files": {
            "harness": str(harness_path),
            "kernel_stats": str(stats_path),
            "kernel_trace": str(trace_path),
        },
    }


def build_comparisons(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for case in cases:
        key = (case["rows"], case["columns"], case["mode"])
        grouped.setdefault(key, []).append(case)

    comparisons: list[dict[str, Any]] = []
    for (rows, columns, mode), group in sorted(grouped.items()):
        old = next((item for item in group if item["resolved_mapping"] == "old"), None)
        if old is None:
            raise ValueError(f"{rows}x{columns}/{mode}: missing old baseline")
        for candidate in sorted(group, key=lambda item: item["resolved_mapping"]):
            if candidate is old:
                continue
            comparisons.append(
                {
                    "rows": rows,
                    "columns": columns,
                    "mode": mode,
                    "candidate": candidate["resolved_mapping"],
                    "old_duration_ns": old["duration_ns"],
                    "candidate_duration_ns": candidate["duration_ns"],
                    "single_trace_old_over_candidate": (
                        old["duration_ns"] / candidate["duration_ns"]
                    ),
                }
            )
    return comparisons


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Q4_RDNA Split-K dynamic profile",
        "",
        "These are single, instrumented rocprofv3 kernel launches used to verify "
        "the selected kernel and launch geometry. They are not benchmark estimates; "
        "use `../microbench.json` for performance claims.",
        "",
        "| Shape | Mapping | Kernel duration | WG | Grid | LDS | VGPR | SGPR | Rel. L2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in summary["cases"]:
        launch = case["launch"]
        lines.append(
            f"| {case['rows']}x{case['columns']} | {case['resolved_mapping']} | "
            f"{case['duration_ns'] / 1000:.3f} us | "
            f"{'x'.join(map(str, launch['workgroup']))} | "
            f"{'x'.join(map(str, launch['grid']))} | "
            f"{launch['lds_block_bytes']} B | {launch['vgpr_count']} | "
            f"{launch['sgpr_count']} | {case['relative_l2']:.3e} |"
        )
    lines.extend(
        [
            "",
            "The trace confirms the general Split-K and small-tile kernels are "
            "actually dispatched. Every profiled output also passed the harness "
            "correctness checks.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    case_dirs = sorted(
        path.parent for path in args.input_dir.glob("*/harness.json") if path.is_file()
    )
    if not case_dirs:
        raise ValueError(f"no profile cases found under {args.input_dir}")
    cases = [collect_case(path) for path in case_dirs]
    if not all(case["correctness_passed"] for case in cases):
        raise ValueError("at least one profiled case failed correctness")
    environment = read_json(case_dirs[0] / "harness.json")["environment"]
    summary = {
        "schema": SCHEMA,
        "scope": "single instrumented launch per case; diagnostic, not benchmark",
        "benchmark_source": "results/upstream_splitk/microbench.json",
        "environment": environment,
        "case_count": len(cases),
        "all_correctness_passed": True,
        "cases": cases,
        "comparisons": build_comparisons(cases),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
