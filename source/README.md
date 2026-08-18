# Source Snapshot

This directory contains the source needed to inspect the disclosed experiment:

- `q4rdna.cu` / `q4rdna.cuh`: experimental llama.cpp ROCm runtime path;
- `llama-cpp-q4rdna-integration.patch`: narrow dispatcher integration against
  llama.cpp base commit `a7a6d0d269c896218b6c78e0933bd6a17519d3f6`;
- `pack_q4rdna_sidecar.py`: fixed 4.25-bpw sidecar generator;
- `q4rdna_sidecar_pack.cpp`: native sidecar packer used during development;
- `q4rdna_cpu_experiment.cpp`: single-matrix CPU representation experiment;
- `q4rdna_gemv_bench.hip`: independent HIP microbenchmark;
- `q4rdna_quality_smoke.py`, `q4rdna_threeway_quality_eval.py`, and
  `q4rdna_hybrid_quality_sweep.py`: quality evaluators;
- `LLAMA_CPP_LICENSE`: license from the tested llama.cpp source tree.

The working llama.cpp tree contained unrelated local experiments. They are not
part of the Q4_RDNA study and are intentionally excluded from this report.

The files were scanned for usernames, absolute local paths, credentials, and
private keys before export. Their SHA-256 values are recorded in the public
manifest and in `results/provenance.json`.

This is a research snapshot, not an upstream-ready patch set. In particular,
Q4_RDNA still uses an external sidecar and environment-variable routing.
