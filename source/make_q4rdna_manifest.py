#!/usr/bin/env python3
"""Write raw BF16 safetensors offsets for the native Q4_RDNA packer."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


TENSOR_NAMES = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--max-tensors", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.layers <= 0:
        raise ValueError("--layers must be positive")
    if args.max_tensors < 0:
        raise ValueError("--max-tensors must be non-negative")

    model = args.model.expanduser().resolve()
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    shard_headers: dict[str, tuple[Path, int, dict[str, object]]] = {}
    for shard_name in sorted(set(index.values())):
        path = model / shard_name
        with path.open("rb") as shard:
            encoded_header_size = shard.read(8)
            if len(encoded_header_size) != 8:
                raise ValueError(f"truncated safetensors header size: {path}")
            header_size = struct.unpack("<Q", encoded_header_size)[0]
            encoded_header = shard.read(header_size)
            if len(encoded_header) != header_size:
                raise ValueError(f"truncated safetensors header: {path}")
            header = json.loads(encoded_header)
        shard_headers[shard_name] = (path, 8 + header_size, header)

    lines: list[str] = []
    for layer in range(args.layers):
        for hf_suffix, gguf_suffix in TENSOR_NAMES.items():
            hf_name = f"model.layers.{layer}.{hf_suffix}"
            if hf_name not in index:
                raise ValueError(f"tensor missing from safetensors index: {hf_name}")
            shard_name = index[hf_name]
            path, data_start, header = shard_headers[shard_name]
            metadata = header.get(hf_name)
            if not isinstance(metadata, dict):
                raise ValueError(f"tensor missing from shard header: {hf_name}")
            if metadata.get("dtype") != "BF16":
                raise ValueError(f"expected BF16 tensor {hf_name}: {metadata}")
            shape = metadata.get("shape")
            offsets = metadata.get("data_offsets")
            if not isinstance(shape, list) or len(shape) != 2:
                raise ValueError(f"expected matrix tensor {hf_name}: {metadata}")
            if not isinstance(offsets, list) or len(offsets) != 2:
                raise ValueError(f"invalid data offsets for {hf_name}: {metadata}")
            rows, columns = map(int, shape)
            begin, end = map(int, offsets)
            if rows % 32 != 0 or columns % 64 != 0:
                raise ValueError(f"unsupported Q4_RDNA shape for {hf_name}: {shape}")
            if end - begin != rows * columns * 2:
                raise ValueError(f"bad BF16 byte count for {hf_name}")
            lines.append(
                f"{path}\t{data_start + begin}\t{rows}\t{columns}\t"
                f"blk.{layer}.{gguf_suffix}"
            )

    if args.max_tensors:
        lines = lines[: args.max_tensors]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} tensors to {args.output}")


if __name__ == "__main__":
    main()
