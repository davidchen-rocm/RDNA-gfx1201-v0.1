# Q4_RDNA Qwen3 end-to-end comparison

Status: **passed**  
Generated: `2026-08-27T19:15:40.733353Z`

## Protocol

Each route received one excluded external warmup. Then 10 measured rounds used `llama-bench -r 1`; route order followed a three-way Latin rotation.

Generations: `128,512`; prompt tokens: `0`; GPU layers: `999`; batch/microbatch: `2048/512`; host threads: `12`.

Before every process, all inherited `LLAMA_Q4_RDNA_*` variables are removed. Production leaves them unset; old sets only `SIDECAR` and `MAPPING=old`; split/auto sets only `SIDECAR` and deliberately leaves `MAPPING` unset.

The current integration does not necessarily emit a mapping-name marker. Mapping validation therefore requires the exact environment contract, the expected sidecar load path, and non-zero Q4_RDNA decode-launch and tensor-hit logs. Production must emit no Q4_RDNA log.

## Results

| Generation | Route | Valid rounds | Mean tok/s | SD | Median | Min | Max | Mean elapsed ns |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | Production Q4_K_M | 10/10 | 94.217 | 1.977 | 95.296 | 90.213 | 95.634 | 1359110654 |
| 128 | Q4_RDNA old | 10/10 | 59.738 | 0.909 | 60.006 | 57.170 | 60.205 | 2143138048 |
| 128 | Q4_RDNA split/auto | 10/10 | 109.889 | 2.176 | 110.600 | 103.769 | 111.041 | 1165248416 |
| 512 | Production Q4_K_M | 10/10 | 95.838 | 0.353 | 95.934 | 94.855 | 96.050 | 5342400962 |
| 512 | Q4_RDNA old | 10/10 | 61.137 | 0.080 | 61.104 | 61.055 | 61.288 | 8374707206 |
| 512 | Q4_RDNA split/auto | 10/10 | 110.795 | 0.149 | 110.816 | 110.488 | 110.964 | 4621159962 |

## Split/auto gains

| Generation | Comparison | Gain from mean tok/s | Mean paired-round gain | Mean elapsed reduction |
|---:|---|---:|---:|---:|
| 128 | split_auto_vs_old | 83.950% | 83.995% | 45.629% |
| 512 | split_auto_vs_old | 81.225% | 81.225% | 44.820% |
| 128 | split_auto_vs_production | 16.633% | 16.665% | 14.264% |
| 512 | split_auto_vs_production | 15.606% | 15.608% | 13.500% |

## Rotation

- Round 1: production_q4_k_m -> q4_rdna_old -> q4_rdna_split_auto
- Round 2: q4_rdna_old -> q4_rdna_split_auto -> production_q4_k_m
- Round 3: q4_rdna_split_auto -> production_q4_k_m -> q4_rdna_old
- Round 4: production_q4_k_m -> q4_rdna_old -> q4_rdna_split_auto
- Round 5: q4_rdna_old -> q4_rdna_split_auto -> production_q4_k_m
- Round 6: q4_rdna_split_auto -> production_q4_k_m -> q4_rdna_old
- Round 7: production_q4_k_m -> q4_rdna_old -> q4_rdna_split_auto
- Round 8: q4_rdna_old -> q4_rdna_split_auto -> production_q4_k_m
- Round 9: q4_rdna_split_auto -> production_q4_k_m -> q4_rdna_old
- Round 10: production_q4_k_m -> q4_rdna_old -> q4_rdna_split_auto

## Provenance

- llama-bench build commit(s): `a7a6d0d`
- Source git HEAD: `a7a6d0d269c896218b6c78e0933bd6a17519d3f6`
- Binary SHA-256: `b7965ef122e72080a6f109c210a5d6ba7ba41805d27a0ddd6fe88dc6d9696f0f`
- Model SHA-256: `77aa55af4ee7d18a44d31eb2aadd2b6e9bf754b5ea3bed43d10101263ea48d10`
- Sidecar SHA-256: `68f086c9d4992f317d3737ccfd50e24da275420766531d6f47e414068f23e7b8`

## Validation

Warmups passed: 3/3; measured invocations passed: 30/30.

All process, JSON, Qwen3 model, sidecar-load, mapping-contract, and Q4_RDNA hit checks passed.
