# Project Update: Q4_RDNA to Mixed-Bit Q5

This repository started with a custom Q4 representation and kernel for Qwen3-8B
decode on an RX 9070 XT. The main Q4 result remains the Q4_RDNA split-K path:

- the first execution mapping was about 36% slower than production Q4_K_M;
- replacing it with a wave32 split-K mapping produced a peak gain of about 15.7%;
- retaining production Q4_K_M for Q/K/V projections selected a more conservative
  result of +10.80% at tg128 and +10.64% at tg512.

## Follow-up optimization

The next experiments started from a Q8_0 model that fit comfortably in 16 GB.
A Q8 kernel-only VDR4 mapping improved tg128/tg512 by only 1.98%/1.87%, so it
was rejected against the 10% objective. This showed that the larger remaining
opportunity was reducing bytes per token rather than changing only execution
mapping.

The project then compared independently prepared Q6_K, Q5_K_M, and Q4_K_M
representations using the same llama.cpp runtime:

| Representation | tg128 | tg512 | Gain vs Q6 | Temporary math test | Decision |
|---|---:|---:|---:|---:|---|
| Q6_K baseline | 79.63 | 79.25 | baseline | 62/100 | baseline |
| Q5_K_M + imatrix | 87.86 | 87.60 | +10.34% / +10.54% | 64/100 | experimental accept |
| Q4_K_M + imatrix | 96.19 | 96.17 | +20.80% / +21.35% | 59/100 | reject |

Q5_K_M reduced model bytes by about 13.0% versus Q6_K and retained the desired
performance/quality balance in the temporary test. Q4_K_M was faster, but its
three-question drop exceeded the temporary two-question budget.

## Quantization provenance

“Q8 to Q5” describes the direction of the optimization project. It is not a
weight-conversion chain. Q5_K_M and Q4_K_M were each quantized directly from the
same BF16 GGUF with an independently recorded importance matrix. No Q8 or Q6
model was requantized to produce them.

The 100-question test is provisional and will be redesigned. For that reason,
the Q5 result is an experimental selection, not a production-readiness result.
Machine-readable values and source-artifact hashes are in
`results/project_update.json`.
