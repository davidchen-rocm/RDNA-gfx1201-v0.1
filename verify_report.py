#!/usr/bin/env python3
"""Verify Q4_RDNA report integrity and recompute headline arithmetic."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def close(actual: float, expected: float, tolerance: float = 1e-4) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"mismatch: actual={actual!r}, expected={expected!r}")


def verify_manifest() -> int:
    manifest = ROOT / "MANIFEST.sha256"
    entries = 0
    for line in manifest.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            raise AssertionError(f"manifest file is missing: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise AssertionError(f"SHA-256 mismatch for {relative}: {actual} != {digest}")
        entries += 1
    return entries


def verify_performance() -> list[str]:
    data = load_json("results/performance.json")
    messages = []
    for round_name, round_data in data["rounds"].items():
        for series_name, series in round_data["series"].items():
            for tokens, point in series.items():
                close(statistics.mean(point["samples"]), point["mean"])
                # Public per-run tok/s values are rounded by llama-bench, while
                # its reported standard deviation was computed pre-rounding.
                close(statistics.stdev(point["samples"]), point["sample_stddev"], tolerance=1e-3)
                for sample, elapsed in zip(point["samples"], point["elapsed_samples"]):
                    close(int(tokens) * 1e9 / elapsed, sample)
        baseline = round_data["series"]["production_q4_k_m"]
        if round_name == "mapping_ablation":
            old = round_data["series"]["q4_rdna_old_mapping"]
            split = round_data["series"]["q4_rdna_split_k"]
            derived = round_data["derived_gain_percent"]
            for tokens in ("128", "512"):
                close((old[tokens]["mean"] / baseline[tokens]["mean"] - 1) * 100,
                      derived[f"old_vs_q4_k_m_tg{tokens}"])
                close((split[tokens]["mean"] / baseline[tokens]["mean"] - 1) * 100,
                      derived[f"split_k_vs_q4_k_m_tg{tokens}"])
        else:
            full = round_data["series"]["q4_rdna_full_split_k"]
            hybrid = round_data["series"]["q4_rdna_qkv_fallback"]
            derived = round_data["derived_gain_percent"]
            for tokens in ("128", "512"):
                close((full[tokens]["mean"] / baseline[tokens]["mean"] - 1) * 100,
                      derived[f"full_vs_q4_k_m_tg{tokens}"])
                close((hybrid[tokens]["mean"] / baseline[tokens]["mean"] - 1) * 100,
                      derived[f"qkv_fallback_vs_q4_k_m_tg{tokens}"])
            messages.append(
                "Selected hybrid: "
                f"tg128={hybrid['128']['mean']:.2f} tok/s "
                f"({derived['qkv_fallback_vs_q4_k_m_tg128']:+.2f}%), "
                f"tg512={hybrid['512']['mean']:.2f} tok/s "
                f"({derived['qkv_fallback_vs_q4_k_m_tg512']:+.2f}%)"
            )
    return messages


def verify_quality() -> list[str]:
    data = load_json("results/quality.json")
    variants = data["variants"]
    for name, result in variants.items():
        close(math.exp(result["loss"]), result["perplexity"], tolerance=1e-10)
        close(result["math_correct"] / result["math_total"], result["math_accuracy"])
        if sum(result["by_subject_correct"].values()) != result["math_correct"]:
            raise AssertionError(f"subject totals do not match for {name}")

    baseline = variants["q4_k_m"]
    full = variants["q4_rdna_full"]
    hybrid = variants["q4_rdna_qkv_fallback"]
    full_cmp = data["comparisons"]["full_q4_rdna_vs_q4_k_m"]
    hybrid_cmp = data["comparisons"]["qkv_fallback_vs_q4_k_m"]
    close((full["perplexity"] / baseline["perplexity"] - 1) * 100,
          full_cmp["ppl_delta_percent"])
    close((hybrid["perplexity"] / baseline["perplexity"] - 1) * 100,
          hybrid_cmp["ppl_delta_percent"])
    close((hybrid["math_accuracy"] - baseline["math_accuracy"]) * 100,
          hybrid_cmp["math_delta_percentage_points"])
    if (hybrid_cmp["q4_k_m_correct_hybrid_wrong"]
            - hybrid_cmp["q4_k_m_wrong_hybrid_correct"]
            != baseline["math_correct"] - hybrid["math_correct"]):
        raise AssertionError("paired hybrid disagreements do not match aggregate counts")
    return [
        f"Quality: Q4_K_M PPL={baseline['perplexity']:.5f}, "
        f"hybrid PPL={hybrid['perplexity']:.5f}; "
        f"math={baseline['math_correct']}/{baseline['math_total']} vs "
        f"{hybrid['math_correct']}/{hybrid['math_total']}"
    ]


def verify_accounting() -> list[str]:
    profile = load_json("results/kernel_profile.json")
    screen = load_json("results/candidate_screen.json")
    accounting = profile["scoped_weight_accounting"]
    selected = screen["candidates"]["qkv_only"]
    weights = accounting["weights"]
    close(accounting["q4_rdna_packed_bytes"] * 8 / weights,
          accounting["q4_rdna_effective_bpw"])
    close(selected["packed_bytes"] * 8 / weights, selected["effective_bpw"])
    reduction = (1 - selected["packed_bytes"] / accounting["q4_k_m_packed_bytes"]) * 100
    close(reduction, 10.151708492839928)
    return [
        f"Accounting: selected hybrid={selected['effective_bpw']:.4f} bpw, "
        f"{reduction:.2f}% fewer scoped packed bytes than Q4_K_M"
    ]


def verify_project_update() -> list[str]:
    data = load_json("results/project_update.json")
    followup = data["mixed_bit_followup"]
    baseline = followup["baseline"]
    q5 = followup["q5_k_m"]
    q4 = followup["q4_k_m"]
    if not followup["no_requantization"]:
        raise AssertionError("mixed-bit follow-up must be direct from BF16")
    for candidate in (q5, q4):
        for tokens in ("128", "512"):
            expected = (
                candidate[f"tg{tokens}_tokens_per_second"]
                / baseline[f"tg{tokens}_tokens_per_second"]
                - 1
            ) * 100
            close(expected, candidate[f"tg{tokens}_gain_percent"])
        size_reduction = (
            1 - candidate["model_bytes"] / baseline["model_bytes"]
        ) * 100
        close(size_reduction, candidate["size_reduction_vs_q6_percent"])
    close((q5["perplexity"] / baseline["perplexity"] - 1) * 100,
          0.05555555555556424)
    if q5["math_correct"] - baseline["math_correct"] != 2:
        raise AssertionError("Q5 100-question delta is inconsistent")
    if q4["math_correct"] - baseline["math_correct"] != -3:
        raise AssertionError("Q4 100-question delta is inconsistent")
    q8 = data["q8_exploration"]
    if q8["decision"] != "REJECT" or q8["tg128_gain_percent"] >= 10:
        raise AssertionError("Q8 kernel-only decision is inconsistent")
    return [
        "Follow-up: Q5_K_M vs Q6_K "
        f"tg128={q5['tg128_gain_percent']:+.2f}%, "
        f"tg512={q5['tg512_gain_percent']:+.2f}%; "
        f"temporary math={q5['math_correct']}/{q5['math_total']}"
    ]


def main() -> None:
    count = verify_manifest()
    messages = (
        verify_performance()
        + verify_quality()
        + verify_accounting()
        + verify_project_update()
    )
    print(f"PASS: verified {count} report file hashes")
    for message in messages:
        print(message)
    print("PASS: report arithmetic is internally consistent")


if __name__ == "__main__":
    main()
