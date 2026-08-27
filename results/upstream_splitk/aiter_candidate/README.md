# AITER gfx1201 Q4 group-64 GEMV candidate measurements

This directory preserves the source patch, validation, and measurement sets for
DCO-signed candidate commit
`c60cc076871cc849d2f6e18d595beefbbf18e954`. Its parent and tested upstream
base is `48718fa7bb1b73d0800130144449fca3c625aba1`. The authoritative public-API
benchmark covers all 14 dispatch keys on an AMD Radeon RX 9070 XT (`gfx1201`)
inside the hash-identified ROCm-PyTorch container.

## Source provenance

`aiter-q4-group64-gemv.patch` is the authoritative, portable source artifact.
It is the 10-path `git diff --binary --full-index` from the upstream base to the
candidate tree and has SHA-256
`3301a8a17dab5ec023d633c8fc0671f6e371f11230219830fab9ee2c4f018f2c`.
`0001-Add-gfx1201-Q4-group-64-GEMV.patch` is the corresponding Git email
patch; it records candidate commit `c60cc076...`, subject, author, and the DCO
`Signed-off-by` trailer (SHA-256
`be954fe6a593bc09f6fd2e3c7e04e9207331aa595a4a9cf65c7acfd9d2fe8dde`).
All source paths in `metadata.json` and `container_validation.json` are relative
to the AITER repository root. To inspect or apply it:

```bash
git clone https://github.com/ROCm/aiter.git /path/to/aiter
git -C /path/to/aiter checkout 48718fa7bb1b73d0800130144449fca3c625aba1
git -C /path/to/aiter apply --check \
  "$PWD/results/upstream_splitk/aiter_candidate/aiter-q4-group64-gemv.patch"
git -C /path/to/aiter apply \
  "$PWD/results/upstream_splitk/aiter_candidate/aiter-q4-group64-gemv.patch"
python3 verify_upstream_splitk.py --aiter-source /path/to/aiter
```

Without `--aiter-source`, the verifier still strictly checks the patch hash,
upstream-base/candidate binding, DCO format-patch, ordered path set, full blob
IDs, and reconstructable new-file post-images. With `--aiter-source`, strict
mode accepts either the clean recorded candidate commit or the upstream-base
HEAD with the patch applied; missing or mismatched source files, a different
HEAD, or a patch/post-image mismatch is a failure.

The authoritative GPU run predates creation of the commit object. All nine
runtime, registration, test, reference, and benchmark files are byte-identical
to the candidate commit. The only post-run change is the documentation-only
roofline section in `docs/q4_group64_gemv.md`; the final committed source also
passed the repository pre-commit hook and a fresh 32/32 targeted GPU test.

## Result sets

- `integration_uncached_raw.json` and `integration_uncached_summary.csv` are
  the first-pass single-launch integration results. They are retained for
  provenance but are **superseded**: that source version queried device
  properties on every call.
- `integration_cached_raw.json` and `integration_cached_summary.csv` retain the
  historical pybind single-launch boundary after the per-device cache fix.
  They are not the authoritative public-API comparison.
- `kernel_raw.json` and `kernel_summary.csv` are the direct, batched kernel
  results. This is the appropriate set for kernel-to-kernel comparison.
- `container_public_api_raw.json` and `container_public_api_summary.csv` are
  the authoritative schema-v3 public-API results: 14 shapes × 3 requests × 2
  timing boundaries = 84 rows, with 30 samples per row. `auto` calls the public
  `q4_group64_gemv`; `old` and `selected` use the private ablation entry with
  `out=None`, so all paths allocate output. Mapping order rotates cyclically and
  every request occupies each position exactly 10 times.
- `container_wrapper_interleaved_raw.json` and
  `container_wrapper_interleaved_summary.csv` are the superseded v2
  cyclic-Latin run. It made HIP-event latency primary and did not explicitly
  prove the public versus allocation-equivalent control call paths. Both
  still-earlier fixed-order runs are also superseded because temporal drift
  could confound them.
- `container_validation.json` and `container_public_api_pytest.xml` record the
  exact container, source patch, JIT hashes, commands, and 32/32 pytest result.
- `rocm714_final_smoke.json` and its log are supplemental native ROCm 7.14
  compile/link/symbol/runtime-gate/dispatch/stream/correctness evidence. They
  exclude the Python public wrapper and performance measurement, so they do
  not replace the authoritative ROCm-PyTorch public-API run.

Every authoritative row contains raw synchronized host-wall and HIP-event
samples, p10/median/p90, correctness diagnostics, call path, candidate mapping,
ring size, and Latin-round provenance. Public `auto` is labelled
`runtime-guarded:<candidate>` because it selects the tuned mapping only when the
runtime identity gate confirms the exact RX 9070 XT.

## Timing boundaries

For schema v3, synchronized host wall-clock is the primary public-call metric;
HIP-event elapsed time is supplemental. The integration boundary covers one
public call or allocation-equivalent control and includes output allocation.
The batched boundary repeats the same call paths, divides by iterations, and
uses wall time for calibration. These are not strict kernel-only timings.

The packed format has a weight-only arithmetic intensity of 3.765 FLOP/byte;
activation/output traffic makes the measured-shape range 3.702–3.756
FLOP/byte. Against the RX 9070 XT's advertised 48.7 FP32 TFLOP/s and 640 GB/s,
the ridge point is about 76 FLOP/byte and the asymptotic DRAM roof is about 2.41
TFLOP/s, so this GEMV is bandwidth-bound. Batched `auto` reports a median 498.8
GB/s of timing-derived logical bandwidth. It is cache-sensitive rather than a
physical DRAM counter, and the evidence makes no measured percent-of-peak
claim.

The direct kernel run places HIP events around a calibrated batch of direct
kernel launches, targets approximately 100 ms per sample, and divides the
device timeline by the number of launches. It uses 100 warmups per mapping,
100 calibration launches, 30 samples, and 72 rotating physical weight copies.
It excludes Python, pybind, tensor validation, the architecture guard/cache,
and public `auto` lookup; host launch overhead and queue gaps are amortized over
the batch.

## Environment limitation

The host's only usable PyTorch installation is `torch 2.13.0+cu130` with
`torch.version.hip == None`, so the host integration run used the built AITER
pybind module, `aiter_tensor_t`, direct HIP allocations, and HIP events. The
normal public wrapper/JIT path then passed in the hash-identified
`math-rule-loop-training-rocm:hqq-0.2.8` container with PyTorch 2.11 and ROCm
7.2. Isolated HIP compilation, pybind ABI smoke tests, native C++ correctness,
and CPU-side Python tests were also run separately.

`metadata.json` records the patch, source post-image hashes, binaries,
environment, validation status, and checksums for preserved artifacts. The
current upstream-candidate benchmark is snapshotted exactly as
`upstream_benchmark_candidate.py`; it dynamically makes the packed ring larger
than 64 MiB and preserves both wall/event samples.

The AITER pass gate is `torch.testing.assert_close` against a dequantized FP32
reference with `rtol=5e-4` and `atol=5e-3`. Recorded maximum-absolute and
relative-L2 values are diagnostics. They are distinct from the standalone
Q4_RDNA harness's relative-L2 policy elsewhere in this evidence repository.

The container reported a blank device name and an AITER helper value `0x100` on
ROCm 7.2. The benchmark's recorded compatibility path queried numeric PCI
attribute 10020, which both recorded ROCm 7.2 and 7.14 headers resolve the
symbolic PCI-chip-ID enum to, and obtained `0x7550`; the production operator
uses the symbolic enum directly. Auto tuning additionally requires 32 HIP-reported
multiprocessors and the exact `AMD Radeon RX 9070 XT` name when a name is
populated. Other gfx1201 identities safely use `old`.

The shared CPU reference packer is preserved as `q4_group64_reference.py`.
`q4_group64_gemv.md` documents its exact byte layout, offline sidecar/loader
responsibilities, experimental status, and the absence of a current vLLM
callsite.

## Rebuild the direct-kernel harness

`kernel_benchmark_driver.cu` includes
`csrc/kernels/q4_group64_gemv.cu` relative to an AITER checkout; it contains no
developer-machine source path. From this evidence repository, set the checkout
explicitly and provide it through include flags:

```bash
AIT_SOURCE=/path/to/aiter
hipcc -std=c++17 -O3 --offload-arch=gfx1201 \
  -I "$AIT_SOURCE" -I "$AIT_SOURCE/csrc/include" \
  results/upstream_splitk/aiter_candidate/kernel_benchmark_driver.cu \
  -o /tmp/direct_kernel_benchmark
/tmp/direct_kernel_benchmark /tmp/q4-group64-direct-results
```

The executable writes `kernel_benchmark_raw.json` and
`kernel_benchmark_summary.csv` beneath the supplied output directory. The
portable driver differs from the executed evidence copy only in source/output
path handling; `container_validation.json` records both hashes. Building does
not require the AITER checkout to live at any particular absolute path.
