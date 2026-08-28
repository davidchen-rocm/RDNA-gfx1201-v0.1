# Reproducing the upstream Split-K qualification

This guide reproduces the 2026-08-27 qualification suite under
`results/upstream_splitk/`. It is separate from `docs/REPRODUCE.md`, which
reproduces the earlier, hash-locked research report.

The standalone and llama.cpp measurements used an AMD Radeon RX 9070 XT
(`gfx1201`, wave32) and host ROCm 7.14. The authoritative AITER public-wrapper
pytest and schema-v3 benchmark used the same GPU in the hash-identified
ROCm-PyTorch 7.2 container. A final ROCm 7.14 native compile/launch smoke is
supplemental compatibility evidence only. The suite is architecture-specific:
a result on another GPU is useful, but it is not a reproduction of the tuned
dispatch table measured here.

## 1. Build the standalone harness

From the report root:

```bash
mkdir -p build
hipcc -O3 -std=c++17 --offload-arch=gfx1201 \
  source/q4rdna_splitk_bench.hip \
  -Wl,-rpath,/opt/rocm/core-7.14/lib \
  -o build/q4rdna_splitk_bench
```

The harness and llama.cpp integration both compile the kernels in
`source/q4rdna_kernels.cuh`. This prevents a benchmark-only implementation from
drifting away from the runtime implementation.

## 2. Run correctness and model-shape performance

The full suite runs 336 positive correctness cases, four expected-invalid
probes, and 216 performance measurements. Performance covers 18 deduplicated
shape/mode signatures from Qwen3-8B, Mistral-7B-Instruct-v0.3,
Qwen2.5-7B-Instruct, and Phi-4-mini-instruct. Each hot and rotating-cache
configuration records 30 raw samples after 100 warmups. Rotating mode uses 72
weight copies, enough for the smallest tested weight set to exceed the 64 MiB
last-level cache disclosed by `rocminfo`.

```bash
python3 source/run_q4rdna_splitk_suite.py all \
  --binary build/q4rdna_splitk_bench \
  --output-dir results/upstream_splitk \
  --cache both \
  --weight-copies 72 \
  --warmup 100 \
  --samples 30 \
  --target-sample-ms 100
```

Use `--resume` to reuse already completed raw cases. The driver recomputes the
mean, sample standard deviation, median, p10, p90, and an independent
nonparametric bootstrap confidence interval from `samples_us`. Its dispatch
rule enables an explicit Split-K mapping only when median speedup is at least
1.03x and the 95% interval lower bound is above 1.00x. Unseen shapes fall back
to `old`.

The plain and add paths require relative L2 at most `1e-5`. The fused gate/up
path records its separate `2e-5` limit because FP32 reduction-order changes can
be amplified by SiLU. Every path also requires finite output, complete writes,
and intact canaries.

## 3. Pack a model with the same native quantizer

The published Qwen3 sidecar was produced by the native OpenMP packer. On the
measured host, its first-tensor payload is byte-identical to the existing Qwen3
sidecar. A current PyTorch build can choose a different winning FP16 scale for
some groups because its tensor reductions do not necessarily use the same
floating-point order, so do not mix packers in a cross-model mapping ablation.

Create a raw-offset manifest from BF16 safetensors, then run the native packer:

```bash
python3 source/make_q4rdna_manifest.py \
  --model /path/to/model-hf \
  --output /path/to/model.q4rdna.manifest \
  --layers 32

g++ -O3 -std=c++17 -fopenmp source/q4rdna_sidecar_pack.cpp \
  -o build/q4rdna_sidecar_pack

OMP_NUM_THREADS=12 build/q4rdna_sidecar_pack \
  /path/to/model.q4rdna.manifest \
  /path/to/model.q4rdna

sha256sum /path/to/model.q4rdna
```

Use the architecture's actual layer count (`36` for the tested Qwen3-8B and
`32` for Mistral-7B-Instruct-v0.3). The manifest generator rejects non-BF16,
missing, mis-sized, or improperly aligned target tensors.

## 4. Rebuild the clean llama.cpp integration

Start from llama.cpp commit
`a7a6d0d269c896218b6c78e0933bd6a17519d3f6` in a clean checkout. Choose local
paths without reusing shell variables that have system meaning:

```bash
export Q4RDNA_LLAMA_SRC=/path/to/llama.cpp
export Q4RDNA_LLAMA_BUILD=/path/to/llama-build-q4rdna

cp source/q4rdna_splitk_runtime.cu \
  "$Q4RDNA_LLAMA_SRC/ggml/src/ggml-cuda/q4rdna.cu"
cp source/q4rdna.cuh \
  "$Q4RDNA_LLAMA_SRC/ggml/src/ggml-cuda/q4rdna.cuh"
cp source/q4rdna_kernels.cuh \
  "$Q4RDNA_LLAMA_SRC/ggml/src/ggml-cuda/q4rdna_kernels.cuh"
git -C "$Q4RDNA_LLAMA_SRC" apply \
  "$PWD/source/llama-cpp-q4rdna-integration.patch"

cmake -S "$Q4RDNA_LLAMA_SRC" -B "$Q4RDNA_LLAMA_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1201
cmake --build "$Q4RDNA_LLAMA_BUILD" \
  --target llama-bench llama-completion -j
```

Do not substitute the historical `source/q4rdna.cu` in this step. That file is
kept unchanged so the original report manifest remains verifiable; the new
runtime snapshot is `source/q4rdna_splitk_runtime.cu`.

## 5. Run a real-model three-way comparison

The runner removes every inherited `LLAMA_Q4_RDNA_*` variable before each
process. It performs one excluded external warmup per route, then ten
single-repetition rounds in cyclic Latin order:

1. production Q4_K_M with no sidecar;
2. the Q4_RDNA sidecar with `MAPPING=old`;
3. the same sidecar with the default Split-K mapping.

It rejects a run unless the GGUF is Q4_K_M, the expected model-family marker is
present, the exact sidecar path is loaded, Q4_RDNA launch and unique-hit counts
are non-zero, both requested generations appear exactly once, and all
provenance checks pass.

```bash
python3 source/run_q4rdna_end_to_end.py \
  --binary "$Q4RDNA_LLAMA_BUILD/bin/llama-bench" \
  --model /path/to/model-Q4_K_M.gguf \
  --sidecar /path/to/model.q4rdna \
  --expected-model-family qwen3 \
  --model-label Qwen3-8B \
  --output-dir results/upstream_splitk/end_to_end_qwen3 \
  --rounds 10 \
  --generations 128,512 \
  --threads 12 \
  --batch 2048 \
  --ubatch 512 \
  --gpu-layers 999
```

For Mistral, use the matching Mistral GGUF and sidecar, change the label to
`Mistral-7B-Instruct-v0.3`, and the output directory to
`results/upstream_splitk/end_to_end_mistral_v03`. At the pinned llama.cpp
commit this GGUF reports `model_type="llama 7B Q4_K - Medium"`, so pass
`--expected-model-family llama`; the argument validates llama-bench's emitted
model-type marker rather than the Hugging Face architecture name.

## 6. Check deterministic completion equivalence

Use the supplied runner for each real model:

```bash
python3 source/run_q4rdna_completion_equivalence.py \
  --binary "$Q4RDNA_LLAMA_BUILD/bin/llama-completion" \
  --ggml-hip-library "$Q4RDNA_LLAMA_BUILD/bin/libggml-hip.so.0.18.0" \
  --model /path/to/model-Q4_K_M.gguf \
  --sidecar /path/to/model.q4rdna \
  --model-label Model-Name \
  --output-dir results/upstream_splitk/completion_equivalence_model_greedy \
  --seed 20260827 \
  --n-predict 64 \
  --threads 12 \
  --gpu-layers all \
  --visible-device 0
```

For each of three fixed prompts, the runner launches `llama-completion` once
with `LLAMA_Q4_RDNA_MAPPING=old` and once with the mapping variable absent. It
uses `--samplers temperature --temp 0`, which is the pinned llama.cpp build's
recognized argmax/greedy protocol. A run is rejected if stderr reports an
unrecognized sampler, the effective sampler chain does not contain
temperature, the logged temperature is not zero, the exact sidecar is not
loaded, or Q4_RDNA launch/unique-hit counts are zero. It retains stdout and
stderr bytes separately and requires every old/Split-K stdout pair to match
byte for byte.

`results/upstream_splitk/completion_equivalence/` is an earlier audit artifact.
Its requested sampler name `greedy` was not recognized by this llama.cpp
revision, so it proves fixed-seed distribution equality only and is explicitly
marked legacy. The upstream evidence uses the v2 `*_greedy` directories.

## 7. Capture representative profiler evidence

Profiler durations are single instrumented launches and must not replace the
unprofiled suite statistics. A representative command is:

```bash
rocprofv3 --kernel-trace --stats --summary \
  --summary-output-file stdout -f csv json \
  -d results/upstream_splitk/profile_dynamic/4096x4096_split8 \
  -o trace -- \
  ./build/q4rdna_splitk_bench \
    --rows 4096 --columns 4096 --mode plain --mapping split8 \
    --pattern random --warmup 0 --samples 0 --cache hot \
    --weight-copies 1 \
    --output results/upstream_splitk/profile_dynamic/4096x4096_split8/harness.json
```

After capturing old and candidate cases, summarize and cross-check the trace:

```bash
python3 source/summarize_q4rdna_profile.py \
  --input-dir results/upstream_splitk/profile_dynamic \
  --output-json results/upstream_splitk/profile_dynamic/summary.json \
  --output-markdown results/upstream_splitk/profile_dynamic/SUMMARY.md
```

Static code-object metadata is retained in
`results/upstream_splitk/profile_static.json`; dynamic CSV and JSON traces are
retained under `results/upstream_splitk/profile_dynamic/`.

## 8. Reproduce the public AITER candidate

The authoritative source artifact is the portable patch
`results/upstream_splitk/aiter_candidate/aiter-q4-group64-gemv.patch`. It is
the diff from upstream AITER base commit
`48718fa7bb1b73d0800130144449fca3c625aba1` to the tree of DCO-signed
candidate commit `c60cc076871cc849d2f6e18d595beefbbf18e954`; its SHA-256 is
`3301a8a17dab5ec023d633c8fc0671f6e371f11230219830fab9ee2c4f018f2c`.
The adjacent `0001-Add-gfx1201-Q4-group-64-GEMV.patch` is a Git email patch
that preserves the candidate commit ID, author, subject, and `Signed-off-by`
trailer; its SHA-256 is
`be954fe6a593bc09f6fd2e3c7e04e9207331aa595a4a9cf65c7acfd9d2fe8dde`.

The upstream PR later added DCO-signed test-only commit
`f6b900dcbbedb557f4761723a951cfb525038621` to follow AITER's standard op-test
format. It changes only `op_tests/test_q4_group64_gemv.py`; the runtime, kernel,
benchmark, and reference sources remain byte-identical to the performance-tested
`c60cc076...` tree. The artifact below intentionally reproduces that measured
tree rather than rewriting historical performance provenance.

Apply it from the evidence-repository root:

```bash
export Q4RDNA_EVIDENCE_ROOT="$PWD"
export AITER_SOURCE=/path/to/aiter

git clone https://github.com/ROCm/aiter.git "$AITER_SOURCE"
git -C "$AITER_SOURCE" checkout 48718fa7bb1b73d0800130144449fca3c625aba1
sha256sum \
  "$Q4RDNA_EVIDENCE_ROOT/results/upstream_splitk/aiter_candidate/aiter-q4-group64-gemv.patch"
git -C "$AITER_SOURCE" apply --check \
  "$Q4RDNA_EVIDENCE_ROOT/results/upstream_splitk/aiter_candidate/aiter-q4-group64-gemv.patch"
git -C "$AITER_SOURCE" apply \
  "$Q4RDNA_EVIDENCE_ROOT/results/upstream_splitk/aiter_candidate/aiter-q4-group64-gemv.patch"
```

The public wrapper requires a ROCm-PyTorch environment. The authoritative run
used PyTorch `2.11.0+gitd0c8b1f`, HIP `7.2.53211`, one visible RX 9070 XT, and a
fresh JIT directory. From the patched AITER root, choose an output directory
and run:

```bash
export AITER_EVIDENCE=/path/to/aiter-output
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export GPU_ARCHS=gfx1201
export PYTORCH_ROCM_ARCH=gfx1201
export AITER_ENABLE_EXPERIMENTAL=1
export PYTHONPATH="$AITER_SOURCE"
export AITER_JIT_DIR=/path/to/fresh-jit/aiter
export TORCH_EXTENSIONS_DIR=/path/to/fresh-jit/torch

mkdir -p "$AITER_EVIDENCE" "$AITER_JIT_DIR" "$TORCH_EXTENSIONS_DIR"
cd "$AITER_SOURCE"

python -m pytest op_tests/test_q4_group64_gemv.py -q \
  --junitxml="$AITER_EVIDENCE/container_public_api_pytest.xml"

python op_tests/op_benchmarks/hip/bench_q4_group64_gemv.py \
  --sweep --mappings old auto selected \
  --cache rotating --rotate 0 \
  --warmup 100 --samples 30 --timing both \
  --calibration-iterations 100 --target-sample-ms 100 \
  -o "$AITER_EVIDENCE/container_public_api_summary.csv" \
  --json "$AITER_EVIDENCE/container_public_api_raw.json"
```

The benchmark calls public `q4_group64_gemv` for `auto`; `old` and `selected`
are allocation-equivalent private controls with `out=None`. Requests rotate in
cyclic Latin order within every sample round, and every packed-weight ring is
strictly larger than 64 MiB. Synchronized host wall-clock is primary; HIP-event
elapsed time is supplemental. The authoritative result is 32/32 pytest cases
and 84 benchmark rows (`14 × 3 × 2`), with 30 raw samples per row. Host-wall
Auto/Old geometric-mean speedup is `1.6074x` for single-call integration
(range `1.175–2.229x`) and `1.9031x` for calibrated batched calls (range
`1.084–2.782x`). Maximum absolute error is `4.9114e-5`; maximum relative L2 is
`2.306e-6`.

The exact tested container identity, source/JIT hashes, commands, JUnit result,
and output hashes are preserved in
`results/upstream_splitk/aiter_candidate/container_validation.json`. The older
`container_wrapper_interleaved_*` event-primary run and all fixed-order wrapper
runs are retained for provenance but superseded.

Auto tuning requires `gfx1201`, PCI chip ID `0x7550`, and 32 HIP-reported
multiprocessors. A non-empty device name must exactly equal
`AMD Radeon RX 9070 XT`. The blank-name path was exercised on ROCm 7.2; on any
runtime that returns a blank name, both numeric checks remain mandatory. Other
`gfx1201` identities use `old`, and non-`gfx1201` calls reject. Both Python and
C++ entries dynamically enforce `AITER_ENABLE_EXPERIMENTAL` on every call.

The final ROCm 7.14 native result is recorded in
`results/upstream_splitk/aiter_candidate/rocm714_final_smoke.json` with the full
execution log beside it. It covers native compile, link, symbols, gate, all ten
explicit mappings, non-default stream, auto fallback/selection, and invalid
requests. It does not exercise the Python public wrapper and records no
performance measurement, so it does not replace the ROCm 7.2 result.

The exact packed format and intended offline sidecar/loader ownership are in
the patched AITER file `docs/q4_group64_gemv.md`. A CPU reference packer exists
for conformance tests, but there is no production quantizer/packer, sidecar
loader, vLLM call site, or AITER model consumer yet. This remains an acceptance
risk; the separate llama.cpp integration is supplemental and does not close it.

## 9. Verify the evidence package

Run the historical verifier, then the Split-K verifier in both default
patch-only mode and strict source mode. Strict mode accepts either (a) the
upstream-base HEAD with the portable patch applied as uncommitted post-images,
as constructed above, or (b) a clean checkout of the recorded candidate
commit.

```bash
cd "$Q4RDNA_EVIDENCE_ROOT"
python3 verify_report.py
python3 verify_upstream_splitk.py
python3 verify_upstream_splitk.py --aiter-source "$AITER_SOURCE"
```

Default mode validates the explicit upstream-base-to-candidate binding, DCO
format-patch, reconstructable post-images, and all retained evidence. Strict
mode additionally checks the AITER HEAD, candidate parent when applicable,
DCO trailer for the committed candidate, worktree cleanliness for that mode,
and every source post-image. Candidate commit `c60cc076...` is the recorded
source identity for this evidence bundle.
