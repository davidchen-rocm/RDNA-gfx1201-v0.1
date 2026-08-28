## Summary

This change adds an experimental `gfx1201` operator for multiplying one FP32
activation vector by a prepacked group-64 signed-INT4 matrix:

```python
y = aiter.q4_group64_gemv(x, packed_weight)
```

The packed tensor has shape `[N/32, K/64, 1088]`. Each 32-row by 64-column
tile contains 32 FP16 row scales followed by 1024 bytes of row-interleaved
signed INT4 values. The public API uses a measured shape dispatch and exposes
no tuning knob. Unmeasured legal shapes use the conservative `old` kernel.

This PR does not claim to invent Split-K. Its contribution is the measured
combination of the group-64 layout, `gfx1201` wave32 execution, block-local
Split-K/LDS reduction, and small-row variants.

## Motivation and implementation

The conservative mapping assigns a wave to each 32-row tile. Dense decode
GEMV then has a long per-lane K dependency chain, while small-N projections do
not supply enough independent waves to the device.

The new kernels divide column groups across 2, 4, or 8 waves and reduce their
FP32 partial sums through LDS. Small-row variants use 8, 16, or 32 waves and
write 8, 16, or 32 rows per block. A measured `(N,K)` table selects a faster
variant for 14 model-derived shapes; no model name participates in dispatch.

Non-`gfx1201` calls reject before launch. On `gfx1201`, automatic tuned dispatch
requires PCI chip ID `0x7550` and 32 HIP-reported multiprocessors, plus the exact
`AMD Radeon RX 9070 XT` name when the runtime provides a non-empty name. ROCm
7.2 can report a blank name, so that compatibility case still requires both
numeric identifiers. Other `gfx1201` identities and failed identity queries
use the boundary-safe old kernel. An unseen shape also falls back to `old`.

Importing AITER does not query a GPU, and device identity is cached by HIP
device. The experimental opt-in is nevertheless re-read on every call at both
the public Python entry and the loaded C++ entry; unset, zero, or malformed
values reject even after an earlier enabled call.

## Supported contract

- GPU: `gfx1201`, wave32; the tuned automatic table is restricted to the known
  RX 9070 XT numeric identity described above.
- `x`: contiguous FP32 `[K]`.
- `packed_weight`: contiguous, 2-byte-aligned `uint8`
  `[N/32,K/64,1088]`.
- output: contiguous FP32 `[N]`.
- `N > 0`, `N % 32 == 0`, `K > 0`, `K % 64 == 0`.
- one FP16 scale for every row and 64-column group; quantized values are
  signed INT4 in `[-8,7]`.

The exact byte layout, CPU reference packer, and intended offline sidecar/loader
flow are documented in `docs/q4_group64_gemv.md`. The reference packer is for
tests and format conformance; this patch does not yet provide a production
quantizer/packer or loader. Quantization and packing must not run per token.
This initial experimental operator also has no vLLM or AITER model-level call
site, so a framework adapter must own and provide prepacked weights. That
missing producer/consumer path is an explicit acceptance risk.

## Correctness

The ROCm-PyTorch public-wrapper suite passed **32/32** tests on an RX 9070 XT.
It covers all ten explicit kernel mappings, known-shape automatic dispatch,
unseen-shape fallback, signed-INT4 extremes, zero scales, a non-default stream,
Python/C++ dispatch-table parity, the exact-device policy, dynamic Python/C++
experimental gates, and invalid dtype/shape/layout/alignment.

The authoritative benchmark retained 84 result rows and 30 raw samples per
row. Every row passed `torch.testing.assert_close` with `rtol=5e-4` and
`atol=5e-3`; the maximum absolute error was `4.9114e-5`, and the maximum
relative L2 diagnostic was `2.306e-6`.

The underlying technique also passed a separate 336-case qualification matrix
covering non-divisible group/split counts, groups fewer than waves, minimum
rows, large K, and plain/add/fused gate-up paths. Maximum relative L2 was
`1.5543e-5` in fused gate/up (`2e-5` limit); plain/add used a `1e-5` limit.

Parallel FP32 reduction changes summation order. True greedy completion was
byte-identical for 3/3 Qwen3 prompts and 2/3 Mistral prompts in an independent
runtime integration. Therefore this PR claims tolerance-based operator
correctness, not bit-exact generation across all models.

## Performance

Public-wrapper environment:

- AMD Radeon RX 9070 XT (`gfx1201`);
- PyTorch `2.11.0+gitd0c8b1f`, HIP `7.2.53211`;
- 14 model-derived shapes from Qwen3-8B, Mistral-7B v0.3,
  Qwen2.5-7B, and Phi-4-mini;
- 100 warmups and 30 raw samples per result;
- rotating packed-weight ring strictly larger than 64 MiB per shape;
- `old`, `auto`, and explicit-selected calls cyclically interleaved within
  every sample round.

| Primary synchronized host-wall boundary | Faster shapes | Geomean | Per-shape range |
|---|---:|---:|---:|
| Single-call integration | 14/14 | 1.6074x | 1.175–2.229x |
| Calibrated batched AITER entry | 14/14 | 1.9031x | 1.084–2.782x |

Roofline context: every 32×64 packed tile stores 2,048 weights in 1,088
bytes. At two FLOPs per weight, the weight-only arithmetic intensity is 3.765
FLOP/byte; including activation reads and output writes gives 3.702–3.756
FLOP/byte over these 14 shapes. AMD specifies up to 48.7 FP32 TFLOP/s and 640
GB/s for the RX 9070 XT, putting its ridge point near 76 FLOP/byte and the
operator's asymptotic DRAM roof near 2.41 TFLOP/s. The operation is therefore
bandwidth-bound. Batched `auto` reaches 58.9–696.7 GB/s of timing-derived
logical bandwidth (498.8 GB/s median). This is cache-sensitive logical traffic,
not a DRAM counter; values above 640 GB/s do not mean greater-than-100% physical
bandwidth utilization, and no measured percent-of-peak claim is made. See the
[official RX 9070 XT specifications](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html).

Public `auto` directly calls `q4_group64_gemv`; `old` and explicit-selected are
allocation-equivalent private controls with `out=None`. Both boundaries include
output allocation and a synchronization boundary. HIP-event samples are kept
as supplemental diagnostics, not substituted for the host-wall primary metric.
The schema-v3 result supersedes the earlier event-primary cyclic-Latin
`container_wrapper_interleaved_*` run and all earlier fixed-order runs.

The authoritative public-wrapper result uses PyTorch ROCm 7.2. A separate ROCm
7.14 native smoke freshly compiled and linked the final kernel, validated its
symbols, and passed every explicit mapping, a non-default stream, known/unseen
automatic dispatch, experimental-gate rejection, and invalid-input checks. It
does not exercise Python and contains no performance result, so it is
supplemental compatibility evidence rather than a second benchmark.

## Independent real-model check

An independent llama.cpp integration using the same packed format and mapping
strategy ran ten cyclically ordered rounds per route:

| Model | Generation | Old tok/s | Auto tok/s | Gain |
|---|---:|---:|---:|---:|
| Qwen3-8B | 128 | 59.738 | 109.889 | +83.950% |
| Qwen3-8B | 512 | 61.137 | 110.795 | +81.225% |
| Mistral-7B v0.3 | 128 | 62.508 | 115.284 | +84.430% |
| Mistral-7B v0.3 | 512 | 67.452 | 124.078 | +83.950% |

These numbers are supplemental method evidence from a separate custom sidecar
integration. They neither execute the AITER Python wrapper nor provide the
missing AITER production packer/consumer.

## Reproduction

The portable source artifact is
`results/upstream_splitk/aiter_candidate/aiter-q4-group64-gemv.patch`, bound to
upstream AITER base `48718fa7bb1b73d0800130144449fca3c625aba1` and producing
the tree of DCO-signed candidate commit
`c60cc076871cc849d2f6e18d595beefbbf18e954`; patch SHA-256 is
`3301a8a17dab5ec023d633c8fc0671f6e371f11230219830fab9ee2c4f018f2c`.
The accompanying Git email patch
`0001-Add-gfx1201-Q4-group-64-GEMV.patch` preserves the commit subject,
author, and `Signed-off-by` trailer.

The upstream PR later added DCO-signed test-only commit
`f6b900dcbbedb557f4761723a951cfb525038621` to follow AITER's standard op-test
format. It changes only `op_tests/test_q4_group64_gemv.py`; the measured runtime,
kernel, benchmark, and reference sources remain byte-identical to
`c60cc076...`.

From the evidence-repository root:

```bash
export AITER_SOURCE=/path/to/aiter
git clone https://github.com/ROCm/aiter.git "$AITER_SOURCE"
git -C "$AITER_SOURCE" checkout 48718fa7bb1b73d0800130144449fca3c625aba1
sha256sum results/upstream_splitk/aiter_candidate/aiter-q4-group64-gemv.patch
git -C "$AITER_SOURCE" apply --check \
  "$PWD/results/upstream_splitk/aiter_candidate/aiter-q4-group64-gemv.patch"
git -C "$AITER_SOURCE" apply \
  "$PWD/results/upstream_splitk/aiter_candidate/aiter-q4-group64-gemv.patch"

# Patch-only/public evidence verification, then strict patched-source checking.
python3 verify_upstream_splitk.py
python3 verify_upstream_splitk.py --aiter-source "$AITER_SOURCE"
```

The exact ROCm-PyTorch environment, pytest and benchmark commands, JIT hashes,
and output hashes are recorded in
`results/upstream_splitk/aiter_candidate/container_validation.json`.

## Known limitations

- `gfx1201` only; other architectures reject calls before launch.
- FP32 activation and output only.
- Shape-table dispatch; unseen shapes intentionally use `old`.
- The tuned Auto table is restricted to the known RX 9070 XT numeric identity;
  other `gfx1201` identities use `old`, while non-`gfx1201` calls reject.
- `small32x32` launches 1024 threads and is enabled only for three measured
  small-row shapes on `gfx1201`.
- The operator consumes an offline prepacked layout and currently has neither a
  production packer/loader nor a model-framework adapter in AITER.
- Non-`gfx1201` public CI can validate import/static behavior but cannot run
  the GPU kernels; the RX 9070 XT JUnit and raw benchmark evidence should be
  attached to the PR.

## Validation checklist

- [x] Targeted ROCm-PyTorch test: 32/32 passed.
- [x] Public benchmark: 84/84 rows with all raw samples retained.
- [x] Python/C++ dispatch-table parity.
- [x] Runtime experimental gate checked at both Python and C++ entries.
- [x] Upstream-base-to-candidate patch verified in default and strict source modes.
- [x] `gfx1201` implementation and non-target guard stub compile with warnings
  treated as errors (excluding existing warnings in common AITER headers).
- [x] Black, Ruff, py_compile, clang-format, JSON, and diff checks.
- [x] Current repository pre-commit hook.
- [x] DCO signed-off candidate commit.
- [ ] Upstream CI after opening the pull request.
