# Privacy and Disclosure Policy

This report uses minimum necessary disclosure.

## Included because it is necessary to assess the result

- GPU and CPU product names;
- gfx target, ROCm, compiler, kernel, profiler, and runtime versions;
- llama.cpp base commit;
- benchmark settings and individual timing samples;
- aggregate quality results and paired statistics;
- model/artifact/source SHA-256 hashes;
- Q4_RDNA source snapshot and the narrow integration patch;
- negative results, profiler caveats, and known limitations.

## Deliberately excluded

- username and hostname;
- absolute local filesystem paths;
- full process environment and unrelated environment variables;
- shell history;
- model weights and the 3.7 GB Q4_RDNA sidecar;
- dataset question/solution text;
- raw ATT/thread-trace directories containing local path metadata;
- unrelated source-tree modifications;
- unrelated workspace files, repositories, and experiment artifacts.

The published result JSON was transcribed from hashed source artifacts and omits
local path fields. `results/provenance.json` records the hashes of the original
private artifacts so they can be matched later without publishing their paths or
contents.

Before upload, run:

```bash
python3 verify_report.py
rg -n -i '/home/|username|hostname|password|secret|api[_-]?key|bearer' .
```

The expected search hits are explanatory words in this privacy document, not
actual credentials or local paths.
