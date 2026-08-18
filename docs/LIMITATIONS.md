# Limitations

## Scope limitations

- One GPU model: AMD Radeon RX 9070 XT (`gfx1201`).
- One model: Qwen3-8B.
- One runtime: the disclosed experimental llama.cpp ROCm build.
- One primary workload: single-batch autoregressive decode.
- Two generation lengths: 128 and 512 tokens.
- Three benchmark repetitions per point.
- One PPL corpus and four math subjects.
- Measurements were performed by the project author, not an independent lab.

The result should not be generalized to prefill, batched serving, other model
families, other hidden sizes, other Radeon/Instinct GPUs, or later ROCm/runtime
versions without new measurements.

## Timing limitations

The report provides individual samples and a second clean benchmark round, but
three repetitions are still a small sample. GPU clocks were observed as part of
the working experiment but were not locked and logged into each benchmark JSON.
No confidence interval beyond the reported sample standard deviation was computed.

Within-round comparisons are the valid comparisons. Small differences between
the phase-3 and phase-4 baselines demonstrate normal run-to-run drift and should
not be combined as if every sample came from one randomized trial.

## Quality limitations

The quality evaluator used CPU-loaded BF16 weights with Q4_K_M tensors
dequantized from GGUF and Q4_RDNA fake-quantized values. It did not execute the
experimental HIP runtime during PPL or math evaluation.

Packer/evaluator parity was checked on 257 random groups rather than every
packed weight. The included source allows further verification.

The hybrid's lower PPL applies only to the disclosed corpus and should not be
generalized to other datasets. Math was three questions lower than Q4_K_M, with
paired p=0.852.

No broad instruction-following, coding, multilingual, safety, long-context, or
generation-preference benchmark is included. Production evaluation would
require broader quality coverage.

The mixed-bit Q5 follow-up uses a separate, provisional stratified 100-question
math test. It is included as an experiment-selection gate only. It is not
comparable to the original 848-question Q4_RDNA table and is not a production
quality result; it will be replaced with a broader evaluation.

## Profiler limitations

Several desired hardware metrics returned zero across every selected dispatch
on the tested ROCm 7.14/gfx1201 combination. They are treated as unsupported:

- `FetchSize`;
- `MemUnitBusy`;
- `OccupancyPercent`;
- `WAVE_DEP_WAIT`;
- `WAVE_ISSUE_WAIT`;
- tested raw GL2/TCP counters.

Consequently, the report does not include measured DRAM traffic or measured
MemUnitBusy reduction. Packed byte counts are format accounting, not a hardware
traffic measurement.

ATT samples one target CU/shader-engine region, and one split-K trace produced a
stitch-incomplete warning. ATT percentages are qualitative supporting data.

## Integration limitations

Q4_RDNA is loaded through an experimental sidecar and environment-variable
routing. It is not a formal GGUF type, has not been upstreamed, and has not been
tested for multi-GPU execution or broad error handling.

The working llama.cpp tree also contained unrelated local experimentation. That
unrelated code is intentionally excluded. The public source directory contains
only the Q4_RDNA implementation and its narrow llama.cpp integration patch.

## Reproducibility limitations

Model weights and datasets are not redistributed. Their identities, sources,
evaluation construction, and local file hashes are disclosed where useful, but
a reproducer must independently obtain assets under their original terms.

Exact numerical reproduction can be affected by runtime changes, compiler
changes, GPU firmware, clocks, thermals, and model-file variants. Reproduction
should first target the direction and approximate magnitude, then compare file
hashes and environment versions if results differ.
