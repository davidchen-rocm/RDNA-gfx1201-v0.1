# Q4_RDNA Mistral-7B-Instruct-v0.3 end-to-end comparison

Status: **passed**  
Generated: `2026-08-27T19:45:04.519986Z`

## Protocol

Each route received one excluded external warmup. Then 10 measured rounds used `llama-bench -r 1`; route order followed a three-way Latin rotation.

Generations: `128,512`; prompt tokens: `0`; GPU layers: `999`; batch/microbatch: `2048/512`; host threads: `12`.

Before every process, all inherited `LLAMA_Q4_RDNA_*` variables are removed. Production leaves them unset; old sets only `SIDECAR` and `MAPPING=old`; split/auto sets only `SIDECAR` and deliberately leaves `MAPPING` unset.

The current integration does not necessarily emit a mapping-name marker. Mapping validation therefore requires the exact environment contract, the expected sidecar load path, and non-zero Q4_RDNA decode-launch and tensor-hit logs. Production must emit no Q4_RDNA log.

## Results

| Generation | Route | Valid rounds | Mean tok/s | SD | Median | Min | Max | Mean elapsed ns |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | Production Q4_K_M | 10/10 | 98.229 | 0.385 | 98.394 | 97.392 | 98.586 | 1303098207 |
| 128 | Q4_RDNA old | 10/10 | 62.508 | 0.240 | 62.524 | 61.936 | 62.804 | 2047762982 |
| 128 | Q4_RDNA split/auto | 10/10 | 115.284 | 0.510 | 115.411 | 114.120 | 115.976 | 1110325742 |
| 512 | Production Q4_K_M | 10/10 | 105.211 | 0.191 | 105.244 | 104.876 | 105.474 | 4866434921 |
| 512 | Q4_RDNA old | 10/10 | 67.452 | 0.249 | 67.393 | 67.199 | 67.867 | 7590651790 |
| 512 | Q4_RDNA split/auto | 10/10 | 124.078 | 0.253 | 124.122 | 123.655 | 124.467 | 4126442211 |

## Split/auto gains

| Generation | Comparison | Gain from mean tok/s | Mean paired-round gain | Mean elapsed reduction |
|---:|---|---:|---:|---:|
| 128 | split_auto_vs_old | 84.430% | 84.434% | 45.779% |
| 512 | split_auto_vs_old | 83.950% | 83.952% | 45.638% |
| 128 | split_auto_vs_production | 17.362% | 17.364% | 14.793% |
| 512 | split_auto_vs_production | 17.933% | 17.933% | 15.206% |

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
- Model SHA-256: `1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6`
- Sidecar SHA-256: `9f1eedf1132e9ed2cc0e710249cf72dcd712f8801276f06e9cb2df292b0ed6b4`

## Validation

Warmups passed: 3/3; measured invocations passed: 30/30.

All process, JSON, expected-model-family, Q4_K_M, sidecar-load, mapping-contract, and Q4_RDNA hit checks passed.
