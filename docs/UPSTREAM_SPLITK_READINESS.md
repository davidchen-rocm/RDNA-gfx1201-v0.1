# gfx1201 Q4 group-64 Split-K upstream readiness

Date: 2026-08-27
Target hardware: AMD Radeon RX 9070 XT (`gfx1201`, wave32)

## Decision

The method is a credible upstream candidate as an **architecture- and
format-specific AITER operator**. It is not a new claim on the general idea of
Split-K. The contribution is the measured combination of:

- `gfx1201` wave32 execution;
- group-64 signed INT4 with FP16 per-row scales;
- the row-interleaved `[N/32,K/64,1088]` packed layout;
- block-local Split-K/LDS reduction and small-row variants;
- a shape-keyed keep-better dispatch with an `old` fallback.

The implementation is not keyed by model name. Four model families contributed
the dispatch shapes, and two different model families passed real end-to-end
performance tests. It remains specific to `gfx1201`, this packed format, FP32
activation/output, and the measured `(N,K)` table. Unseen legal shapes use the
old kernel.

A DCO-signed candidate commit is recorded as
`c60cc076871cc849d2f6e18d595beefbbf18e954`. Its parent and tested upstream
AITER base is `48718fa7bb1b73d0800130144449fca3c625aba1`. The portable
upstream-base-to-candidate patch is
`results/upstream_splitk/aiter_candidate/aiter-q4-group64-gemv.patch`, SHA-256
`3301a8a17dab5ec023d633c8fc0671f6e371f11230219830fab9ee2c4f018f2c`.

## Duplicate-work audit

AITER `main` was audited at
`48718fa7bb1b73d0800130144449fca3c625aba1`. It contains Split-K and multi-wave
decode kernels, but no equivalent operator combining the exact packed layout,
group size, signed INT4 interpretation, gfx1201 guard, wave32 block-local
reduction, and measured dispatch used here. The nearest conceptual code has
different types/layouts, and the custom wave-split GEMV path still has a NAVI
TODO.

The evidence protocol follows patterns seen in accepted or reviewed AITER work:
real model shapes, boundary correctness, explicit architecture guards,
keep-better fallback, complete distributions rather than best-case numbers,
and end-to-end results. Relevant examples are
[#3343](https://github.com/ROCm/aiter/pull/3343),
[#3759](https://github.com/ROCm/aiter/pull/3759), and
[#4781](https://github.com/ROCm/aiter/pull/4781).

## Qualification results

### Standalone shared-kernel correctness

- Positive cases: **336/336 passed**.
- Expected-invalid CLI probes: **4/4 passed**.
- Coverage: 11 mappings, 3 operation modes, and 6 input patterns.
- Boundary coverage includes non-divisible group/split counts, groups fewer
  than waves, minimum rows, large K, signed INT4 extremes, and zero scales.
- Maximum relative L2: `1.5543e-5` (fused gate/up, limit `2e-5`).
- Plain/add limit: `1e-5`; all finite/write/canary checks passed.

The standalone harness and the llama.cpp integration compile the same
`source/q4rdna_kernels.cuh` implementation.

### Four-model shape sweep

The rotating-cache suite used 72 distinct weight buffers, 100 warmups, 30 raw
samples, and at least 100 ms per calibrated sample. It covered 18 deduplicated
shape/mode signatures from Qwen3-8B, Mistral-7B-Instruct-v0.3,
Qwen2.5-7B-Instruct, and Phi-4-mini-instruct.

| Metric | Result |
|---|---:|
| Measurements passing correctness | 216/216 |
| Dispatch entries selecting Split-K | 18/18 |
| Median selected speedup | 2.384x |
| Worst selected speedup | 1.033x |
| Best selected speedup | 6.273x |
| Baseline-latency-weighted speedup | 1.733x |

Selection required median speedup at least `1.03x` and a bootstrapped 95%
median-ratio interval whose lower bound exceeded `1.00`. Unseen shapes fall
back to `old`.

### Real-model end-to-end decode

Each route received an excluded warmup followed by ten single-repetition rounds
in cyclic Latin order. `old` and `split/auto` used the exact same sidecar; the
production Q4_K_M route is a deployment reference, not the mapping ablation.

| Model | Generation | Old tok/s | Split/auto tok/s | Gain vs old | Gain vs production Q4_K_M |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B | 128 | 59.738 | 109.889 | +83.950% | +16.633% |
| Qwen3-8B | 512 | 61.137 | 110.795 | +81.225% | +15.606% |
| Mistral-7B-Instruct-v0.3 | 128 | 62.508 | 115.284 | +84.430% | +17.362% |
| Mistral-7B-Instruct-v0.3 | 512 | 67.452 | 124.078 | +83.950% | +17.933% |

All 60 measured invocations and all six excluded warmups passed process,
model, mapping-environment, exact sidecar-load, non-zero launch, and non-zero
tensor-hit validation.

### AITER public operator

The candidate adds a thin public
`q4_group64_gemv(x, packed_weight)` API. The explicit mapping knob is private;
the public call uses the measured auto table. Non-`gfx1201` calls reject before
launch. On `gfx1201`, the tuned table is enabled only when the device reports
PCI chip ID `0x7550` and 32 HIP-reported multiprocessors, plus the exact
`AMD Radeon RX 9070 XT` name when ROCm supplies a non-empty name. ROCm 7.2 may
leave that name blank, so blank-name compatibility still requires both numeric
identifiers. Other `gfx1201` identities use `old`; identity-query failures also
fail closed to `old`.

The experimental gate is checked independently at the public Python entry and
the loaded C++ entry on every call. Unset, zero, and malformed values reject,
including after a previously enabled call. Importing AITER does not query the
device, and architecture/identity lookups are cached by HIP device rather than
globally.

The real public wrapper was JIT-compiled and run in a hash-identified
ROCm-PyTorch container:

- PyTorch `2.11.0+gitd0c8b1f`, HIP `7.2.53211`;
- upstream AITER base `48718fa7...` and DCO-signed candidate `c60cc076...`;
- targeted pytest: **32/32 passed**, zero skip/failure/error;
- all explicit mappings, public known-shape auto, unseen fallback, non-default
  stream, extreme INT4, zero scales, invalid input, dynamic Python/C++
  experimental gates, exact-device policy, and Python/C++ dispatch-table parity
  were exercised.

The authoritative public-entry benchmark contains 84 rows
(`14 shapes × 3 requests × 2 timing boundaries`), 30 raw samples per row, and a
per-shape packed-weight ring strictly larger than 64 MiB. Within every shape
and sample round, `old`, `auto`, and the explicit selected mapping run in a
cyclic Latin order. Public `auto` calls the public API; `old` and `selected` are
allocation-equivalent private controls with `out=None`. Synchronized host wall
time is the primary metric, while HIP-event time is supplemental. This schema-v3
run supersedes both the older event-primary cyclic-Latin
`container_wrapper_interleaved_*` results and all fixed-order wrapper results.

| Public AITER host-wall boundary | Shapes faster | Geometric-mean speedup | Per-shape range |
|---|---:|---:|---:|
| Single-call integration | 14/14 | 1.6074x | 1.175–2.229x |
| Calibrated batched AITER entry | 14/14 | 1.9031x | 1.084–2.782x |

All 84 rows passed `torch.testing.assert_close` with `rtol=5e-4` and
`atol=5e-3`. The largest recorded absolute error was `4.9114e-5`; the largest
relative L2 diagnostic was `2.306e-6`. The single-call boundary includes output
allocation, validation, dispatch, and synchronization. The calibrated batched
boundary repeats the same public or allocation-equivalent call path and
normalizes by iteration count; neither is a pure-kernel timing.

ROCm 7.2 is therefore the authoritative public-API environment. A separate
host-ROCm 7.14 native smoke freshly compiled and linked the final kernel,
verified its symbols, and passed all ten explicit mappings, non-default stream,
known/unseen auto, gate rejection, and invalid-input checks. It is supplemental
native compatibility evidence only: it does not exercise the Python wrapper
and contains no performance claim.

## Numerical-output caveat

Parallel FP32 reduction changes addition order, so bit-exact generation is a
stronger condition than operator correctness.

- Qwen3 greedy (`temperature=0` argmax): **3/3** prompt pairs byte-identical.
- Mistral greedy: **2/3** prompt pairs byte-identical.
- All 12 old/split processes passed the sampler, sidecar, launch, and tensor-hit
  contract.
- The Mistral mismatch begins at stdout byte offset 11 after a near-tie can
  choose a different argmax; it is retained as a failed exactness experiment.

The earlier Qwen artifact that requested `--samplers greedy` is explicitly
marked legacy because that sampler name was not recognized by the pinned
llama.cpp revision; it proves fixed-seed distribution equality only. The v2
results use the recognized `--samplers temperature --temp 0` protocol and
validate the effective sampler chain from stderr.

This caveat must be disclosed in a PR. It does not invalidate an operator tested
with numerical tolerances, but the PR must not claim two-model bit-exact output.

## Recommended submission order

1. **AITER first:** submit the isolated experimental gfx1201 operator, tests,
   benchmark, and architecture guard. This is the cleanest upstream boundary.
2. **llama.cpp separately:** the runtime integration also needs acceptance of a
   custom sidecar/packing workflow, so it is a broader design discussion than
   the kernel optimization alone.

The AITER patch includes an exact CPU reference packer for tests and byte-format
documentation, but no production quantizer/packer, sidecar loader, vLLM call
site, or AITER model consumer. That missing consumer boundary remains the main
acceptance risk. The llama.cpp measurements are supplemental evidence for the
method and do not fill that AITER integration gap.

The repository pre-commit hook, a fresh 32/32 targeted GPU regression, and the
DCO-signed commit are complete. Public CI on other architectures should import
and skip the gfx1201 GPU tests without attempting to launch this kernel. The
tuned table should remain restricted to the known RX 9070 XT identity; other
`gfx1201` identities retain the safe `old` path.

## Evidence entry points

- Standalone summary: `results/upstream_splitk/SUMMARY.md`
- End-to-end: `results/upstream_splitk/end_to_end_qwen3/` and
  `results/upstream_splitk/end_to_end_mistral_v03/`
- Correct greedy checks: `results/upstream_splitk/completion_equivalence_qwen3_greedy/`
  and `results/upstream_splitk/completion_equivalence_mistral7b_v03_greedy/`
- AITER container validation:
  `results/upstream_splitk/aiter_candidate/container_validation.json`
- AITER final raw benchmark:
  `results/upstream_splitk/aiter_candidate/container_public_api_raw.json`
- Portable AITER patch:
  `results/upstream_splitk/aiter_candidate/aiter-q4-group64-gemv.patch`
- DCO Git email patch:
  `results/upstream_splitk/aiter_candidate/0001-Add-gfx1201-Q4-group-64-GEMV.patch`
- Formal PR body: `docs/AITER_Q4_GROUP64_PR.md`
- ROCm 7.14 supplemental native smoke:
  `results/upstream_splitk/aiter_candidate/rocm714_final_smoke.json`
- Independent verifier: `python3 verify_upstream_splitk.py`; add
  `--aiter-source /path/to/patched/aiter` for strict source/post-image checking.
