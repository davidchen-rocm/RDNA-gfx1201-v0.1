# Startup and Reproduction Guide

This guide separates a quick report check from a full GPU reproduction. The
first requires only Python 3. The second requires the disclosed model assets,
ROCm hardware, and a llama.cpp checkout.

## A. Verify the published results

From the report root:

```bash
python3 verify_report.py
```

The verifier checks every file listed in `MANIFEST.sha256`, recomputes timing
means and sample standard deviations, recomputes speedups, and validates PPL and
accuracy arithmetic.

This confirms internal integrity and arithmetic. It is not an independent GPU
replication.

## B. Required assets for GPU reproduction

Obtain independently:

- Qwen3-8B Hugging Face BF16 files;
- the Qwen3-8B Q4_K_M GGUF matching the hash in
  `results/provenance.json`;
- llama.cpp at base commit
  `a7a6d0d269c896218b6c78e0933bd6a17519d3f6`;
- ROCm 7.14 and a `gfx1201` GPU for the closest reproduction.

The report does not redistribute model or dataset content.

Use neutral local variables rather than copying any original machine path:

```bash
export LLAMA_DIR=/path/to/llama.cpp
export BUILD_DIR=/path/to/llama-build-q4rdna
export HF_MODEL_DIR=/path/to/qwen3-8b-hf
export Q4K_GGUF=/path/to/Qwen3-8B-Q4_K_M.gguf
export Q4RDNA_SIDECAR=/path/to/qwen3-8b.q4rdna
```

## C. Apply the experimental runtime source

```bash
cp source/q4rdna.cu "$LLAMA_DIR/ggml/src/ggml-cuda/q4rdna.cu"
cp source/q4rdna.cuh "$LLAMA_DIR/ggml/src/ggml-cuda/q4rdna.cuh"
git -C "$LLAMA_DIR" apply "$PWD/source/llama-cpp-q4rdna-integration.patch"
```

The llama.cpp CUDA/HIP CMake file uses a source glob, so the new `.cu` file is
picked up when configuring a fresh build directory.

Configure and build:

```bash
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1201
cmake --build "$BUILD_DIR" --target llama-bench -j
```

Exact build options can evolve with llama.cpp. Record the resolved CMake cache
and compiler version in any new reproduction.

## D. Generate the fixed Q4_RDNA sidecar

The Python packer requires PyTorch and Safetensors:

```bash
python3 source/pack_q4rdna_sidecar.py \
  --model "$HF_MODEL_DIR" \
  --output "$Q4RDNA_SIDECAR" \
  --layers 36
sha256sum "$Q4RDNA_SIDECAR"
```

The reproduction should produce this sidecar hash:

```text
68f086c9d4992f317d3737ccfd50e24da275420766531d6f47e414068f23e7b8
```

A different BF16 model snapshot, library rounding behavior, or packer source
will produce a different hash and is not an exact reproduction.

## E. Run the end-to-end benchmark

The command below is an equivalent reconstruction from the recorded
`llama-bench` configuration. Save JSON directly in a new results directory.

Production baseline:

```bash
env -u LLAMA_Q4_RDNA_SIDECAR \
    -u LLAMA_Q4_RDNA_SCOPE \
    -u LLAMA_Q4_RDNA_MAPPING \
  "$BUILD_DIR/bin/llama-bench" \
    -m "$Q4K_GGUF" -ngl 999 -p 0 -n 128,512 -b 2048 -ub 512 \
    -t 12 -r 3 -o json
```

Full Q4_RDNA split-K:

```bash
LLAMA_Q4_RDNA_SIDECAR="$Q4RDNA_SIDECAR" \
  "$BUILD_DIR/bin/llama-bench" \
    -m "$Q4K_GGUF" -ngl 999 -p 0 -n 128,512 -b 2048 -ub 512 \
    -t 12 -r 3 -o json
```

Selected Q/K/V fallback:

```bash
LLAMA_Q4_RDNA_SIDECAR="$Q4RDNA_SIDECAR" \
LLAMA_Q4_RDNA_SCOPE=qkv-fallback \
  "$BUILD_DIR/bin/llama-bench" \
    -m "$Q4K_GGUF" -ngl 999 -p 0 -n 128,512 -b 2048 -ub 512 \
    -t 12 -r 3 -o json
```

Old mapping ablation:

```bash
LLAMA_Q4_RDNA_SIDECAR="$Q4RDNA_SIDECAR" \
LLAMA_Q4_RDNA_MAPPING=old \
  "$BUILD_DIR/bin/llama-bench" \
    -m "$Q4K_GGUF" -ngl 999 -p 0 -n 128,512 -b 2048 -ub 512 \
    -t 12 -r 3 -o json
```

Run baseline and candidates close together, without other GPU workloads. Do not
use profiled timings as replacements for these uninstrumented results.

## F. Reproduce the quality evaluation

Install compatible versions of PyTorch, Transformers, Datasets, Safetensors,
NumPy, and llama.cpp's `gguf-py` package. The first dataset load requires access
to the original MMLU and MATH-500 sources.

Set `PYTHONPATH` so the `gguf` package can be imported, then run:

```bash
PYTHONPATH="source:$LLAMA_DIR/gguf-py" \
python3 source/q4rdna_threeway_quality_eval.py \
  --model "$HF_MODEL_DIR" \
  --q4-k "$Q4K_GGUF" \
  --ppl-tokens 16384 \
  --full-math \
  --variants bf16,q4_k_m,q4_rdna \
  --output quality-threeway-expanded.json
```

Selected hybrid:

```bash
PYTHONPATH="source:$LLAMA_DIR/gguf-py" \
python3 source/q4rdna_hybrid_quality_sweep.py \
  --model "$HF_MODEL_DIR" \
  --q4-k "$Q4K_GGUF" \
  --candidates qkv \
  --ppl-tokens 16384 \
  --full-math \
  --output qkv-full-quality.json
```

The evaluator uses `local_files_only=True` for the model, deterministic MMLU
shuffle seed `20260815`, and deterministic next-token A/B/C/D scoring.

## G. Recommended independent replication report

Publish at minimum:

- exact GPU, ROCm, compiler, runtime commit, and model hashes;
- all timing samples, not only the best run;
- Q4_K_M, old mapping, split-K, and selected-hybrid results;
- PPL token construction and task counts;
- any changed compiler flags or runtime defaults;
- failures or counters that could not be collected.
