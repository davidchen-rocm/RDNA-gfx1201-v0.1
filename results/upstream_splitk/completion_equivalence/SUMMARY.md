# Qwen3-8B old vs Split-K legacy seeded-sampling equivalence

Result: **PASS for the recorded fixed-seed protocol, not for greedy sampling**. All three old/Split-K stdout pairs are byte-for-byte identical.

This artifact is retained for audit history but is superseded by
`../completion_equivalence_qwen3_greedy/`. The pinned llama.cpp build did not
recognize the requested sampler name `greedy`; every raw stderr contains
`unable to match sampler by name 'greedy'` and reports the effective chain
`logits -> dist`. Seed `20260827` made that chain reproducible, but calling it
greedy would be inaccurate.

## Fixed setup

- GPU: AMD Radeon RX 9070 XT (`gfx1201`)
- Clean binary: `/tmp/q4rdna-llama-audit.2vKpBB/build/bin/llama-completion`
- Binary version: `1 (a7a6d0d)`
- Model: `Qwen3-8B-Q4_K_M.gguf`
- Q4_RDNA sidecar SHA256: `68f086c9d4992f317d3737ccfd50e24da275420766531d6f47e414068f23e7b8`
- Requested sampling: `--samplers greedy --seed 20260827 -n 64` (name rejected)
- Effective sampling: `logits -> dist`, seed `20260827`
- IO/mode: `--no-display-prompt --no-conversation --simple-io`
- Old run: `LLAMA_Q4_RDNA_MAPPING=old`
- Split-K run: `LLAMA_Q4_RDNA_MAPPING` unset
- Both runs: `LLAMA_Q4_RDNA_SCOPE`, `LLAMA_Q4_RDNA_COOP`, and `LLAMA_Q4_RDNA_SMALL_ROWS` unset

## Results

| Prompt | Output bytes | Old SHA256 | Split-K SHA256 | Exact match |
|---|---:|---|---|---|
| `The first three prime numbers are` | 237 | `b477f5b3e98ff09076cd9b59e0edb956fbe9db8cef43c086370b09d183c353df` | same | yes |
| `Write one short sentence about the moon:` | 296 | `8d3cafe44cd6c70ba781258ada2ed18c0e8f4e1868a9b27534a35a11363efab4` | same | yes |
| `In Python, a list comprehension for squares from 0 to 4 is` | 158 | `fa6c86ecfac6a06c973b7e2ff5de73b1bf7ea91380fbfd5cd91370819d91f838` | same | yes |

Each of the six runs exited with status 0, loaded all 252 sidecar tensors (3.44 GiB), and reported 436 Q4_RDNA decode GEMV launches. Each run hit all 252 unique tensors. The per-shape hit totals were identical: 148 for `12288x4096`, 74 for `4096x12288`, 144 for `4096x4096`, 144 for `1024x4096`, and 0 other.

Raw generated text and complete stderr logs are retained in [`raw/`](raw/). Machine-readable details are in [`completion_equivalence.json`](completion_equivalence.json).

This establishes fixed-seed completion equivalence for these three prompts and
this exact model/runtime. It is neither greedy evidence nor a proof that every
prompt is equivalent.
