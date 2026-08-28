# Privacy and Disclosure Policy

This repository uses minimum necessary disclosure, with two evidence layers:

- The original Q4_RDNA report files listed in `MANIFEST.sha256` were sanitized
  to omit the username, hostname, and absolute local filesystem paths.
- The later `results/upstream_splitk` qualification bundle preserves raw
  commands, logs, and metadata for auditability. Those artifacts intentionally
  retain the recorded username, hostname, model and sidecar locations, build
  directories, and temporary paths. They are provenance strings, not portable
  reproduction paths, and do not imply that the referenced files are included.

## Included because it is necessary to assess the result

- GPU and CPU product names;
- gfx target, ROCm, compiler, kernel, profiler, and runtime versions;
- llama.cpp base commit;
- benchmark settings and individual timing samples;
- aggregate quality results and paired statistics;
- model/artifact/source SHA-256 hashes;
- Q4_RDNA source snapshot and the narrow integration patch;
- negative results, profiler caveats, and known limitations.

## Deliberately excluded from both evidence layers

- full process environment and unrelated environment variables;
- shell history;
- model weights and the 3.7 GB Q4_RDNA sidecar;
- dataset question/solution text;
- raw ATT/thread-trace directories containing local path metadata;
- unrelated source-tree modifications;
- unrelated workspace files, repositories, and experiment artifacts.

The original report JSON was transcribed from hashed source artifacts and omits
local path fields. `results/provenance.json` records the hashes of the original
private artifacts so they can be matched later without publishing their paths
or contents. The later upstream qualification artifacts use a different policy:
raw runtime paths are retained, while credentials, model contents, and unrelated
workspace data remain excluded. Treat those paths as publicly disclosed.

Before upload, run:

```bash
python3 verify_report.py
python3 verify_upstream_splitk.py
git grep -n -I -i -E '/home/|username|hostname|password|secret|api[_-]?key|bearer'
```

Expected hits include this policy and retained provenance paths under
`results/upstream_splitk`. Review the matches before every release; credential
values are not expected anywhere in the repository.
