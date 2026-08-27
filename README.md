# Q4_RDNA on gfx1201

This technical report began as an experimental Qwen3-8B model-to-kernel
optimization on an AMD Radeon RX 9070 XT (`gfx1201`, wave32). It now also
contains a DCO-signed, standalone AITER Q4 group-64 GEMV candidate, multi-model
shape evidence, reproducible source patches, and independent verification.

## AITER upstream candidate

The formal candidate is commit
`c60cc076871cc849d2f6e18d595beefbbf18e954`, whose parent and tested upstream
AITER base is `48718fa7bb1b73d0800130144449fca3c625aba1`. The repository
pre-commit hook and a fresh targeted GPU regression passed; the commit contains
the required DCO `Signed-off-by` trailer.

On the RX 9070 XT, public `auto` beat `old` for all 14 model-derived shapes.
Synchronized host-wall geometric-mean speedup was 1.6074x at the single-call
boundary and 1.9031x for calibrated batched AITER entries. See the
[readiness report](docs/UPSTREAM_SPLITK_READINESS.md),
[formal PR body](docs/AITER_Q4_GROUP64_PR.md), and
[reproduction guide](docs/UPSTREAM_SPLITK_REPRODUCE.md). Verify the entire
bundle with:

```bash
python3 verify_upstream_splitk.py
python3 verify_upstream_splitk.py --aiter-source /path/to/aiter
```

## Latest project update

After the Q4_RDNA work, the project tested a Q8_0 kernel-only mapping and then
moved to mixed-bit model optimization. The Q8 kernel change produced only about
+1.9%, while an imatrix-calibrated Q5_K_M model improved decode throughput over
Q6_K by **+10.34% at tg128** and **+10.54% at tg512**. The temporary 100-question
math result was 64/100 for Q5 versus 62/100 for Q6.

Q5 was quantized directly from BF16; it was **not** created by requantizing Q8.
See [the short project update](docs/PROJECT_UPDATE.md) and
[`results/project_update.json`](results/project_update.json). The 100-question
test is provisional, so the Q5 result is reported as preliminary.

## Q4_RDNA results

All throughput numbers below are unprofiled, same-build `llama-bench` results
with three repetitions per point.

| Runtime path | Scoped weight bpw | tg128 tok/s | tg512 tok/s | 16K-token PPL | Math |
|---|---:|---:|---:|---:|---:|
| Production Q4_K_M | 4.791 | 96.48 | 96.13 | 3.43498 | 512/848 |
| Full Q4_RDNA split-K | 4.250 | 111.61 | 111.20 | 3.48450 | 499/848 |
| Q4_RDNA with Q/K/V fallback | **4.305** | **106.90** | **106.37** | **3.38712** | **509/848** |

The selected Q/K/V-fallback configuration delivered:

- **+10.80% tg128 throughput** versus production Q4_K_M.
- **+10.64% tg512 throughput** versus production Q4_K_M.
- **10.15% fewer packed bytes** over the 252 scoped transformer linear tensors.
- 509/848 versus 512/848 on the math evaluation, a difference of 0.354
  percentage points. The paired exact McNemar p-value is 0.852.
- On the 16K-token corpus, PPL was 3.38712 versus 3.43498 for Q4_K_M. No broader
  PPL comparison was performed.

The faster full-Q4_RDNA point reaches about **+15.7% throughput**, but has a
measured **+1.44% PPL** cost versus Q4_K_M. The Q/K/V fallback is the selected
quality/speed tradeoff.

## Reading guide

1. Run `python3 verify_report.py` from this directory. It verifies report
   hashes and recomputes means, standard deviations, gains, PPL, and accuracy.
2. Read [the optimization journey](docs/OPTIMIZATION_JOURNEY.md) to see the
   baseline, failed first integration, split-K fix, and quality repair.
3. Inspect [the raw timing samples](results/performance.json), not only the
   headline averages.
4. Read [the method](docs/METHOD.md) and [limitations](docs/LIMITATIONS.md) for
   the evaluation scope.
5. Use [the reproduction guide](docs/REPRODUCE.md) and the source snapshot if
   the required model assets and GPU are available.

## Validation

- The comparison target is llama.cpp's production Q4_K_M path, not a naive
  reference kernel.
- End-to-end throughput was measured without a profiler attached.
- The same packed sidecar was used for the old and split-K mapping ablation.
- A second clean benchmark round reproduced the full-Q4_RDNA result.
- Individual timing samples, quality counts, kernel medians, resource usage,
  source snapshots, and SHA-256 provenance are included.
- Unsupported `gfx1201` profiler counters are explicitly marked unavailable;
  they are not presented as zero traffic or zero utilization.
- Negative results are included: the first runtime mapping was roughly 36%
  slower even though it moved fewer weight bytes.

## Original llama.cpp report scope

The measurements cover Qwen3-8B single-batch decode on an RX 9070 XT. In this
setup, the Q4_RDNA split-K path improved end-to-end token generation throughput
over the tested production Q4_K_M build. The selected Q/K/V-fallback
configuration retained about a 10.7% throughput gain and scored 509/848 versus
512/848 on the math evaluation (paired exact McNemar p=0.852).

The measurements do not cover:

- a speedup on other models, GPUs, ROCm versions, runtimes, batch sizes, or
  prefill workloads;
- equal model quality on every task;
- measured DRAM traffic savings from hardware counters;
- production readiness or an upstream-compatible GGUF tensor type;
- independent third-party replication.

## Report contents

```text
README.md                         report summary and results
docs/METHOD.md                    exact scope and evaluation method
docs/OPTIMIZATION_JOURNEY.md      complete baseline-to-final process
docs/PROJECT_UPDATE.md            Q8 exploration and mixed-bit Q5 follow-up
docs/REPRODUCE.md                 startup and reproduction instructions
docs/LIMITATIONS.md               evaluation scope and caveats
docs/PRIVACY.md                   what was deliberately withheld
results/performance.json          individual llama-bench timing samples
results/kernel_profile.json       sanitized rocprof/ATT summary
results/quality.json              PPL, task counts, paired statistics
results/candidate_screen.json     five hybrid assignments tested
results/provenance.json           model, artifact, and source hashes
results/project_update.json       Q4, Q8, Q6/Q5/Q4 follow-up summary
results/upstream_splitk/           multi-model and AITER upstream evidence
docs/AITER_Q4_GROUP64_PR.md        formal upstream PR body
docs/UPSTREAM_SPLITK_READINESS.md  upstream decision and qualification
docs/UPSTREAM_SPLITK_REPRODUCE.md  full Split-K/AITER reproduction guide
source/                           experimental source snapshot and patch
verify_report.py                  local integrity and arithmetic verifier
verify_upstream_splitk.py          independent upstream-evidence verifier
MANIFEST.sha256                   hashes of report files
```

## Status

Research prototype. Q4_RDNA was measured on 2026-08-16 and the mixed-bit
follow-up on 2026-08-17. The llama.cpp base commit was
`a7a6d0d269c896218b6c78e0933bd6a17519d3f6`; the Q4_RDNA changes were uncommitted
experimental modifications captured in this report.
