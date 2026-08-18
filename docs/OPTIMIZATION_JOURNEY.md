# Optimization Journey

This document includes the failed intermediate result because it shows that the
final gain came from an end-to-end optimization process, not from reporting only
a favorable microbenchmark.

## 1. Start from the production bottleneck

The target was single-batch Qwen3-8B decode on `gfx1201`. The dominant weight
matrix shapes were:

- fused gate/up `12288 x 4096`;
- plain `4096 x 4096`;
- plain `1024 x 4096`.

The required comparison was llama.cpp's production Q4_K_M implementation.
Naive Q4_K code was retained only as a correctness/reference tool.

## 2. Establish a fixed compact representation on CPU

The first representation used 64 INT4 weights plus one FP16 scale: 4.25 bpw.
The layout grouped 32 rows into a fixed 1,088-byte tile arranged for contiguous
wave32 access.

On one real `12288 x 4096` gate matrix, the initial weight MSE was 2.345 times
the Q4_K error. A bounded scale search reduced that ratio to approximately
1.631 without changing the packed layout.

This was used as a screening result, not as a model-quality conclusion.

## 3. Build an independent HIP microkernel

The microkernel proved that the layout could be consumed directly on GPU and
that unpack/dequantize/dot matched the CPU reference. An early comparison with
a naive Q4_K reference was encouraging, but it was not accepted as the final
performance gate.

## 4. Integrate early into the real runtime

A sidecar loader and experimental llama.cpp dispatch path were used instead of
first creating a formal GGUF tensor type. This reduced integration cost and made
it possible to test all 252 scoped tensors against production Q4_K_M quickly.

The first result was negative:

| Path | tg128 | tg512 |
|---|---:|---:|
| Production Q4_K_M | 95.97 | 95.60 |
| First Q4_RDNA mapping | 60.49 | 60.75 |

Moving 11.3% fewer scoped packed bytes did not automatically improve speed.
The first runtime path was approximately 36% slower.

## 5. Diagnose execution mapping

The first kernel assigned one lane to an entire output row and made that lane
perform the full 4,096-element dot product. This produced:

- a long dependent accumulation chain;
- 88-96 VGPR per thread;
- too few waves across the device, especially for small output shapes.

The problem was not simply high VGPR. On the tested GPU, the launch did not
provide enough device-wide wave supply to occupy all SIMDs.

## 6. Freeze the format and replace the mapping

Packing, scales, quantized values, and sidecar bytes were held fixed. The kernel
was redesigned so multiple wave32s split the K dimension for 32 output rows and
then combine partial results.

| Shape | Old Q4_RDNA | Split-K Q4_RDNA | Production Q4_K |
|---|---:|---:|---:|
| fused `12288 x 4096` | 108.64 us | 90.24 us | 92.04 us |
| `4096 x 4096` | 41.16 us | 16.88 us | 18.24 us |
| `1024 x 4096` | 41.88 us | 5.40 us | 5.88 us |

Resource and launch changes included:

| Shape | Old VGPR / waves | Split-K VGPR / waves |
|---|---:|---:|
| fused `12288 x 4096` | 96 / 384 | 48 / 3,072 |
| `4096 x 4096` | 88 / 128 | 32 / 1,024 |
| `1024 x 4096` | 88 / 32 | 32 / 1,024 |

In the clean ablation round, split-K improved tg128 from 61.05 to 111.56 tok/s
with the exact same sidecar. It was 15.32% faster than production Q4_K_M.

## 7. Measure model quality after performance passed

The complete 252-tensor Q4_RDNA configuration reached about 15.7% higher decode
throughput in the final benchmark round. Its quality result was usable but not
free:

- PPL: 3.48450 versus 3.43498 for Q4_K_M, a 1.44% increase;
- math: 499/848 versus 512/848.

The optimization therefore moved to selective tensor routing instead of
chasing additional throughput.

## 8. Test a small number of quality repairs

Five fallback assignments were screened while the Q4_RDNA representation and
kernel remained fixed:

- first 12 layers;
- last 12 layers;
- all FFN down projections;
- all attention projections;
- Q/K/V projections only.

Q/K/V fallback was the clear Pareto candidate. It retained 86.96% Q4_RDNA
weight coverage, used 4.305 effective bpw, and restored the disclosed quality
metrics close to Q4_K_M.

The final selected result was 106.90/106.37 tok/s at tg128/tg512, approximately
10.7% above production Q4_K_M.

## 9. Main lesson

The optimized object was not only a kernel. It was the combination:

```text
model tensor sensitivity
  x representation and packing
  x tensor routing
  x wave mapping and shape specialization
  x llama.cpp runtime integration
```

A format-only benchmark would have missed both major outcomes: the first slow
runtime integration and the final Q/K/V quality repair.
