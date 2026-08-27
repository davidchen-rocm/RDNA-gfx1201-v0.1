# Qwen3-8B old vs Split-K completion equivalence

Result: **PASS**. Every old/Split-K stdout pair is byte-for-byte identical and passed runtime validation.

## Fixed setup

- Binary: `/tmp/q4rdna-llama-audit.2vKpBB/build/bin/llama-completion`
- Binary SHA-256: `02adf6fbf0b6f8eb5c679e263d4334fa2c0c9a92f946838952ade915a8c93b17`
- libggml-hip SHA-256: `2424eb1ec0be45aec76f83eabfa38350b79c6e5dd205f8e1a775f37785f7ce83`
- Model: `/media/david/9CCE793ACE790DB0/math-rule-loop/models/qwen3-8b-q4/Qwen3-8B-Q4_K_M.gguf`
- Model SHA-256: `77aa55af4ee7d18a44d31eb2aadd2b6e9bf754b5ea3bed43d10101263ea48d10`
- Sidecar: `/home/david/Desktop/math-rule-loop/artifacts/q4rdna-phase2/qwen3-8b.q4rdna`
- Sidecar SHA-256: `68f086c9d4992f317d3737ccfd50e24da275420766531d6f47e414068f23e7b8`
- Sampling: `--samplers temperature --temp 0 --seed 20260827 -n 64` (greedy argmax)
- Old route sets `LLAMA_Q4_RDNA_MAPPING=old`.
- Split-K route leaves `LLAMA_Q4_RDNA_MAPPING` unset.
- Both routes set only `LLAMA_Q4_RDNA_SIDECAR` and `LLAMA_Q4_RDNA_TRACE` in addition to old's mapping override.

## Results

| Prompt | Old bytes | Split bytes | Old SHA-256 | Split SHA-256 | Exact | Runtime |
|---|---:|---:|---|---|---:|---:|
| `The first three prime numbers are` | 85 | 85 | `7e0d5c84590f9655f75be4509280d91c05634261c2aca393bfe19969aa2de0e4` | `7e0d5c84590f9655f75be4509280d91c05634261c2aca393bfe19969aa2de0e4` | yes | pass |
| `Write one short sentence about the moon:` | 301 | 301 | `097e2a0a456f12b628b16a36e6a36e8058be80e8a17fcbc5944ea03b14ed36ed` | `097e2a0a456f12b628b16a36e6a36e8058be80e8a17fcbc5944ea03b14ed36ed` | yes | pass |
| `In Python, a list comprehension for squares from 0 to 4 is` | 249 | 249 | `a66a816a9b686e00bf493d1700e353454b34c6a96349fb1f732b14b8f4e565de` | `a66a816a9b686e00bf493d1700e353454b34c6a96349fb1f732b14b8f4e565de` | yes | pass |

## Validation

All processes exited with status 0, loaded the exact requested sidecar path, reported non-zero Q4_RDNA launch and unique-tensor counts, and produced identical stdout bytes.

Raw stdout and stderr bytes are retained in `raw/`. This checks these fixed prompts for this exact model, sidecar, and runtime; it is not a proof for every prompt.
