# Q4_RDNA gfx1201 Split-K test summary

Generated: `2026-08-27T19:09:51.105692Z`  
Requested suite: `all`

## Correctness

Status: **passed**. Positive cases: 336/336; expected-invalid cases: 4/4.

Maximum relative L2: `0.00001554`; maximum absolute error: `0.01037598`.

Relative-L2 acceptance limits are read from each harness result: `plain`=1.0e-05, `add`=1.0e-05, `gate-up`=2.0e-05. The fused `gate-up` path uses its separately reported limit because Split-K FP32 reduction order can be amplified by SiLU; finite/output/canary checks remain mandatory.

Coverage: 11 mappings, 3 modes, 6 patterns.

## Model-derived microbenchmark

Status: **passed**. Measurements: 216/216 passed correctness.

Selection cache: `rotating`. Split-K selected for 18/18 deduplicated shape/mode entries.

Dispatch median/worst speedup: `2.384x` / `1.033x`; baseline-latency-weighted speedup: `1.733x`.

Harness correctness limits observed in this sweep: `plain`=1.0e-05, `add`=not-reported, `gate-up`=2.0e-05. The fused `gate-up` path reports a separate limit because Split-K FP32 reduction order can be amplified by SiLU; finite/output/canary checks remain mandatory.

| Shape | Mode | Model roles | Chosen | Old us | Chosen us | Speedup | 95% CI | Decision |
|---|---|---|---:|---:|---:|---:|---:|---|
| 512x3584 | plain | Qwen2.5-7B-Instruct:k_v | small32x32 | 37.804 | 6.027 | 6.273x | 6.252-6.327 | split-k-selected |
| 1024x3072 | plain | Phi-4-mini-instruct:k_v | small32x32 | 33.504 | 6.546 | 5.118x | 5.095-5.135 | split-k-selected |
| 1024x4096 | plain | Qwen3-8B:k_v, Mistral-7B-Instruct-v0.3:k_v | small32x32 | 43.230 | 7.194 | 6.009x | 5.970-6.023 | split-k-selected |
| 3072x3072 | plain | Phi-4-mini-instruct:q_o | split8 | 33.373 | 13.264 | 2.516x | 2.500-2.527 | split-k-selected |
| 3072x8192 | plain | Phi-4-mini-instruct:down | split8 | 84.352 | 29.722 | 2.838x | 2.817-2.856 | split-k-selected |
| 3584x3584 | plain | Qwen2.5-7B-Instruct:q_o | split8 | 38.251 | 15.604 | 2.451x | 2.436-2.458 | split-k-selected |
| 3584x18944 | plain | Qwen2.5-7B-Instruct:down | split8 | 187.378 | 68.368 | 2.741x | 2.724-2.753 | split-k-selected |
| 4096x4096 | plain | Qwen3-8B:q_o, Mistral-7B-Instruct-v0.3:q_o | split8 | 44.428 | 19.177 | 2.317x | 2.302-2.330 | split-k-selected |
| 4096x12288 | plain | Qwen3-8B:down | split8 | 122.358 | 49.623 | 2.466x | 2.452-2.495 | split-k-selected |
| 4096x14336 | plain | Mistral-7B-Instruct-v0.3:down | split8 | 140.526 | 56.567 | 2.484x | 2.478-2.493 | split-k-selected |
| 8192x3072 | gate-up | Phi-4-mini-instruct:gate_up_fused | split8 | 67.257 | 47.579 | 1.414x | 1.413-1.419 | split-k-selected |
| 8192x3072 | plain | Phi-4-mini-instruct:gate_up_single | split4 | 34.548 | 28.893 | 1.196x | 1.194-1.199 | split-k-selected |
| 12288x4096 | gate-up | Qwen3-8B:gate_up_fused | split8 | 114.050 | 94.004 | 1.213x | 1.212-1.214 | split-k-selected |
| 12288x4096 | plain | Qwen3-8B:gate_up_single | split8 | 58.069 | 51.980 | 1.117x | 1.116-1.118 | split-k-selected |
| 14336x4096 | gate-up | Mistral-7B-Instruct-v0.3:gate_up_fused | split8 | 121.011 | 105.820 | 1.144x | 1.143-1.144 | split-k-selected |
| 14336x4096 | plain | Mistral-7B-Instruct-v0.3:gate_up_single | split8 | 61.774 | 57.312 | 1.078x | 1.077-1.079 | split-k-selected |
| 18944x3584 | gate-up | Qwen2.5-7B-Instruct:gate_up_fused | split8 | 134.126 | 121.222 | 1.106x | 1.106-1.107 | split-k-selected |
| 18944x3584 | plain | Qwen2.5-7B-Instruct:gate_up_single | split8 | 70.581 | 68.355 | 1.033x | 1.032-1.033 | split-k-selected |

Keep-better rule: select an explicit Split-K mapping only when median speedup >= 1.03x and the independently bootstrapped 95% confidence interval has a lower bound above 1.00x. All other and unseen legal shapes use `old`.

## Artifacts

Raw per-invocation harness JSON is retained under `raw/`. Derived statistics are recomputed from `samples_us`; failed candidates remain in the aggregate and dispatch records.

## Additional upstream qualification

- Real decode: Qwen3-8B and Mistral-7B-Instruct-v0.3 both passed ten-round
  `old` versus automatic Split-K ablations at tg128 and tg512; automatic
  Split-K improved throughput by 81.225% to 84.430% versus `old` across the
  four model/generation pairs.
- AITER public path: the ROCm-PyTorch container test passed 32/32 tests. All
  14 measured dispatch shapes were faster through both the single-call and
  calibrated-batched public AITER timing boundaries in the authoritative
  cyclically interleaved run.
- Numerical caveat: true greedy completion was byte-identical for 3/3 Qwen
  prompts and 2/3 Mistral prompts. The retained Mistral mismatch means the
  submission must claim tolerance-based operator correctness, not two-model
  bit-exact generation.

The complete decision, environment distinction, limitations, and evidence
index are in `docs/UPSTREAM_SPLITK_READINESS.md`.
