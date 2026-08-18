# Method

## Hardware and software

- GPU: AMD Radeon RX 9070 XT
- ISA target: `gfx1201`
- Execution width: wave32
- Reported topology: 64 CUs, 128 SIMDs
- CPU: AMD Ryzen 9 7900X
- ROCm: 7.14.0
- HIP compiler: 7.14.60850, AMD clang 23.0.0
- Kernel: Linux 7.0.0-28-generic
- Profiler: rocprofv3 1.3.2
- Runtime: llama.cpp ROCm backend
- llama.cpp base commit: `a7a6d0d269c896218b6c78e0933bd6a17519d3f6`

No username, hostname, absolute local path, or full environment dump is needed
to interpret the results and none is included.

## Model and representation scope

The experiment used Qwen3-8B. Exact local model-file hashes are recorded in
`results/provenance.json`; model weights are not redistributed.

The scoped representation covers seven linear tensors in each of 36 transformer
blocks, 252 tensors total:

- attention Q, K, V, and output projections;
- MLP gate, up, and down projections.

Embeddings, normalization weights, and the language-model head remained outside
the Q4_RDNA scope.

Q4_RDNA is fixed at:

- 64 signed INT4 weights per group;
- one FP16 scale per group;
- quantized range `[-8, 7]`;
- 32 rows per tile;
- 1,088 bytes per tile;
- 4.25 bits per weight;
- fixed-size, row-interleaved wave32 layout.

The full sidecar contains 6,945,767,424 scoped weights and 3,689,938,944 packed
data bytes. The corresponding Q4_K_M GGUF tensors occupy 4,160,028,672 bytes,
or 4.79144 effective bits per scoped weight because Q4_K_M uses a mixture of
quantization types.

The selected hybrid keeps Q/K/V projection tensors on their original Q4_K_M
path and routes all other eligible tensors to Q4_RDNA:

- 108 Q/K/V fallback tensors;
- 144 Q4_RDNA tensors;
- 86.9565% of scoped weights use Q4_RDNA;
- 3,737,714,688 packed bytes;
- 4.30503 effective bits per scoped weight.

## Performance protocol

Throughput was measured with `llama-bench` using:

- full GPU offload (`n_gpu_layers=999`);
- one generation stream;
- no prompt tokens;
- 128 and 512 generated-token tests;
- batch 2,048 and microbatch 512;
- 12 host threads;
- three repetitions per point;
- the same executable within each comparison round;
- no profiler attached to the reported throughput runs.

Means and sample standard deviations are reported exactly as emitted by
`llama-bench`. Individual samples are in `results/performance.json`.

Two rounds are disclosed:

1. An ablation round comparing production Q4_K_M, the first Q4_RDNA mapping,
   and the split-K mapping.
2. A final tradeoff round comparing production Q4_K_M, full split-K Q4_RDNA,
   and the Q/K/V-fallback hybrid.

The two rounds were run in the same broader experiment session but should be
treated as separate same-build comparison groups. Ratios are calculated only
within a round.

## Kernel profiling protocol

rocprofv3 traces were used to compare median dispatch duration, launch wave
count, VGPR, and LDS usage for the three dominant shapes. These profiled kernel
durations explain the mechanism; they are not used to calculate the tok/s
result.

The old and split-K paths used the same sidecar SHA-256:

```text
68f086c9d4992f317d3737ccfd50e24da275420766531d6f47e414068f23e7b8
```

Therefore old-versus-split is a clean execution-mapping ablation: packing and
quantization did not change.

ATT stall samples are included only as qualitative support. ATT sampled one
target CU/shader-engine region, and one split-K decode produced a
stitch-incomplete warning.

On this ROCm/gfx1201 setup, `FetchSize`, `MemUnitBusy`, `OccupancyPercent`,
`WAVE_DEP_WAIT`, `WAVE_ISSUE_WAIT`, and tested GL2/TCP counters returned zero for
every selected dispatch. The report classifies them as unavailable and makes
no hardware-counter DRAM-byte result is reported.

## Quality protocol

Quality was evaluated through a BF16 Hugging Face model loaded on CPU. Target
weights were replaced with either:

- real Q4_K_M tensors dequantized from the tested GGUF; or
- Q4_RDNA values produced by the same fixed quantizer used by the sidecar packer.

The packed runtime was not used directly for these quality runs. A parity check
over 257 random 64-weight groups confirmed that the quality fake-quantizer and
sidecar packer produced bit-identical dequantized values for the checked groups.

Perplexity:

- first 16,384 scored tokens from deterministic concatenation of MATH-500 test
  problems and solutions;
- sequence length 256;
- batch size 4;
- identical token sequence for every variant.

Math evaluation:

- all 848 test questions from four MMLU math subjects;
- deterministic shuffle seed `20260815`;
- zero-shot A/B/C/D next-token scoring;
- identical question order for every variant.

The paired exact McNemar/binomial result compares which questions were correct
under Q4_K_M and the selected hybrid. It does not establish equivalence on tasks
outside this set.

## Derived arithmetic

Throughput gain is:

```text
candidate_mean_tok_s / baseline_mean_tok_s - 1
```

Packed-byte reduction is:

```text
1 - candidate_scoped_bytes / q4_k_m_scoped_bytes
```

PPL is `exp(mean cross-entropy loss)`. The verifier recomputes all disclosed
derived values from the machine-readable results.
