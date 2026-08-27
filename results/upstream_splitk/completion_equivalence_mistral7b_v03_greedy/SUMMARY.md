# Mistral-7B-Instruct-v0.3 old vs Split-K completion equivalence

Result: **FAIL**. At least one pair failed byte equality or runtime validation.

## Fixed setup

- Binary: `/tmp/q4rdna-llama-audit.2vKpBB/build/bin/llama-completion`
- Binary SHA-256: `02adf6fbf0b6f8eb5c679e263d4334fa2c0c9a92f946838952ade915a8c93b17`
- libggml-hip SHA-256: `2424eb1ec0be45aec76f83eabfa38350b79c6e5dd205f8e1a775f37785f7ce83`
- Model: `/home/david/Desktop/model-library/derived/Mistral-7B-Instruct-v0.3/Q4_K_M/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`
- Model SHA-256: `1270d22c0fbb3d092fb725d4d96c457b7b687a5f5a715abe1e818da303e562b6`
- Sidecar: `/home/david/Desktop/model-library/derived/Mistral-7B-Instruct-v0.3/Q4_RDNA/Mistral-7B-Instruct-v0.3.q4rdna`
- Sidecar SHA-256: `9f1eedf1132e9ed2cc0e710249cf72dcd712f8801276f06e9cb2df292b0ed6b4`
- Sampling: `--samplers temperature --temp 0 --seed 20260827 -n 64` (greedy argmax)
- Old route sets `LLAMA_Q4_RDNA_MAPPING=old`.
- Split-K route leaves `LLAMA_Q4_RDNA_MAPPING` unset.
- Both routes set only `LLAMA_Q4_RDNA_SIDECAR` and `LLAMA_Q4_RDNA_TRACE` in addition to old's mapping override.

## Results

| Prompt | Old bytes | Split bytes | Old SHA-256 | Split SHA-256 | Exact | Runtime |
|---|---:|---:|---|---|---:|---:|
| `The first three prime numbers are` | 67 | 67 | `9a8eff6f063282986107757e0117d41e9157aca5bda5b2ad3e3acc15b7bf35fc` | `9a8eff6f063282986107757e0117d41e9157aca5bda5b2ad3e3acc15b7bf35fc` | yes | pass |
| `Write one short sentence about the moon:` | 120 | 120 | `f2f3c2b1a5315921df888c874b375192a199f8bcfac924df4f43b0a388f6dd66` | `f2f3c2b1a5315921df888c874b375192a199f8bcfac924df4f43b0a388f6dd66` | yes | pass |
| `In Python, a list comprehension for squares from 0 to 4 is` | 121 | 182 | `e3e1f76e3c8f4499c9acccb02a6ef147a8524f0d2642166eb5a6ecd57f03ba86` | `871ce59f34ceaab5d15d1cda1100db92c23faceade0048338b0cfc8e2935f46e` | no | fail |

## Validation

- `prompt3`: stdout differs at byte offset 11

Raw stdout and stderr bytes are retained in `raw/`. This checks these fixed prompts for this exact model, sidecar, and runtime; it is not a proof for every prompt.
