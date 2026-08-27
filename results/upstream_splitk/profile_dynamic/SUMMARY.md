# Q4_RDNA Split-K dynamic profile

These are single, instrumented rocprofv3 kernel launches used to verify the selected kernel and launch geometry. They are not benchmark estimates; use `../microbench.json` for performance claims.

| Shape | Mapping | Kernel duration | WG | Grid | LDS | VGPR | SGPR | Rel. L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024x4096 | old | 69.160 us | 256x1x1 | 1024x1x1 | 2048 B | 88 | 128 | 0.000e+00 |
| 1024x4096 | small32x32 | 8.080 us | 1024x1x1 | 32768x1x1 | 4096 B | 32 | 128 | 1.051e-06 |
| 18944x3584 | old | 113.921 us | 256x1x1 | 18944x1x1 | 2048 B | 88 | 128 | 0.000e+00 |
| 18944x3584 | split8 | 67.561 us | 256x1x1 | 151552x1x1 | 1024 B | 32 | 128 | 1.045e-06 |
| 4096x4096 | old | 60.441 us | 256x1x1 | 4096x1x1 | 2048 B | 88 | 128 | 0.000e+00 |
| 4096x4096 | split2 | 64.001 us | 256x1x1 | 8192x1x1 | 1024 B | 32 | 128 | 1.073e-06 |
| 4096x4096 | split8 | 27.120 us | 256x1x1 | 32768x1x1 | 1024 B | 32 | 128 | 1.068e-06 |

The trace confirms the general Split-K and small-tile kernels are actually dispatched. Every profiled output also passed the harness correctness checks.
