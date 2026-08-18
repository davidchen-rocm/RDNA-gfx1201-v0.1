#!/usr/bin/env python3
"""Pack Qwen3 decoder linear weights into the fixed Q4_RDNA sidecar format."""

import argparse
import json
import math
import struct
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open


MAGIC = b"Q4RDNA1"
VERSION = 1
GROUP = 64
TILE_ROWS = 32
TILE_BYTES = 1088
HEADER = struct.Struct("<8sIIIIIIQQ")
ENTRY = struct.Struct("<64sIIQQQ")
DATA_ALIGNMENT = 4096

TENSOR_NAMES = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def fp16_round(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float16).to(torch.float32)


@torch.inference_mode()
def quantize(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact FP16 scale and signed INT4 values used by quality eval."""
    positive_max = values.clamp_min(0).amax(dim=1, keepdim=True)
    negative_max = (-values).clamp_min(0).amax(dim=1, keepdim=True)
    positive_range = torch.maximum(positive_max / 7.0, negative_max / 8.0)
    negative_range = -torch.maximum(positive_max / 8.0, negative_max / 7.0)
    factors = (1.0, 0.9, 0.8, 0.7, 0.6)
    scales = fp16_round(torch.stack(
        [scale * factor for factor in factors for scale in (positive_range, negative_range)], dim=0))
    best_error = torch.full_like(scales[:, :, 0], math.inf)
    best_scale = scales.clone()

    for _ in range(12):
        quant = torch.round(values.unsqueeze(0) / scales).clamp(-8, 7)
        quant = torch.where(scales == 0, torch.zeros_like(quant), quant)
        error = ((values.unsqueeze(0) - quant * scales) ** 2).sum(dim=2)
        improved = error < best_error
        best_error = torch.where(improved, error, best_error)
        best_scale = torch.where(improved.unsqueeze(2), scales, best_scale)
        denominator = (quant * quant).sum(dim=2, keepdim=True)
        next_scale = (values.unsqueeze(0) * quant).sum(dim=2, keepdim=True) / denominator.clamp_min(1)
        scales = fp16_round(torch.where(denominator == 0, scales, next_scale))

    winner = best_error.argmin(dim=0)
    group_index = torch.arange(values.shape[0])
    scale = best_scale[winner, group_index]
    quant = torch.round(values / scale).clamp(-8, 7)
    quant = torch.where(scale == 0, torch.zeros_like(quant), quant).to(torch.int8)
    return scale.squeeze(1).to(torch.float16), quant


@torch.inference_mode()
def pack_tensor(output, weight: torch.Tensor, tile_chunk: int) -> None:
    if weight.ndim != 2:
        raise ValueError(f"expected matrix, got {tuple(weight.shape)}")
    rows, columns = weight.shape
    if rows % TILE_ROWS or columns % GROUP:
        raise ValueError(f"unsupported Q4_RDNA shape: {tuple(weight.shape)}")
    groups = columns // GROUP
    weight = weight.view(rows // TILE_ROWS, TILE_ROWS, groups, GROUP)

    for first in range(0, weight.shape[0], tile_chunk):
        source = weight[first:first + tile_chunk].to(torch.float32).contiguous()
        count = source.shape[0]
        scales, quants = quantize(source.view(-1, GROUP))
        scales = scales.view(count, TILE_ROWS, groups).permute(0, 2, 1).contiguous()
        quants = quants.view(count, TILE_ROWS, groups, GROUP)
        low = quants[..., 0::2].to(torch.int16) & 15
        high = (quants[..., 1::2].to(torch.int16) & 15) << 4
        packed = (low | high).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        tiles = torch.empty((count, groups, TILE_BYTES), dtype=torch.uint8)
        tiles[..., :64].copy_(scales.view(torch.uint8).reshape(count, groups, 64))
        tiles[..., 64:].copy_(packed.reshape(count, groups, 1024))
        output.write(tiles.numpy().tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen3-8b-hf")
    parser.add_argument("--output", default="artifacts/q4rdna-phase2/qwen3-8b.q4rdna")
    parser.add_argument("--tile-chunk", type=int, default=4)
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--max-tensors", type=int, default=0)
    args = parser.parse_args()

    model_dir = Path(args.model)
    output_path = Path(args.output)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    tensors = []
    for layer in range(args.layers):
        for hf_suffix, gguf_suffix in TENSOR_NAMES.items():
            hf_name = f"model.layers.{layer}.{hf_suffix}"
            tensors.append((hf_name, f"blk.{layer}.{gguf_suffix}", weight_map[hf_name]))
    if args.max_tensors:
        tensors = tensors[:args.max_tensors]

    with ExitStack() as stack:
        shards = {
            name: stack.enter_context(safe_open(str(model_dir / name), framework="pt", device="cpu"))
            for name in sorted({item[2] for item in tensors})
        }
        entries = []
        data_offset = align_up(HEADER.size + ENTRY.size * len(tensors), DATA_ALIGNMENT)
        offset = data_offset
        for hf_name, gguf_name, shard_name in tensors:
            rows, columns = shards[shard_name].get_slice(hf_name).get_shape()
            size = rows * columns * 34 // GROUP
            entries.append((hf_name, gguf_name, shard_name, rows, columns, offset, size))
            offset = align_up(offset + size, 256)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with output_path.open("wb") as output:
            output.write(HEADER.pack(MAGIC, VERSION, len(entries), GROUP, TILE_ROWS, TILE_BYTES, 0,
                                     data_offset, offset - data_offset))
            for _, name, _, rows, columns, entry_offset, size in entries:
                encoded = name.encode("ascii")
                if len(encoded) >= 64:
                    raise ValueError(f"tensor name too long: {name}")
                output.write(ENTRY.pack(encoded, rows, columns, entry_offset, size, 0))
            output.write(bytes(data_offset - output.tell()))

            for number, (hf_name, name, shard_name, rows, columns, entry_offset, size) in enumerate(entries, 1):
                if output.tell() != entry_offset:
                    output.write(bytes(entry_offset - output.tell()))
                tensor_started = time.monotonic()
                pack_tensor(output, shards[shard_name].get_tensor(hf_name), args.tile_chunk)
                if output.tell() != entry_offset + size:
                    raise RuntimeError(f"packed size mismatch for {name}")
                elapsed = time.monotonic() - started
                print(f"[{number:3d}/{len(entries)}] {name:31s} {rows:5d}x{columns:5d} "
                      f"{size / 1048576:7.1f} MiB {time.monotonic() - tensor_started:6.1f}s total {elapsed:7.1f}s",
                      flush=True)

            if output.tell() < offset:
                output.write(bytes(offset - output.tell()))

        print(f"wrote {output_path} ({output_path.stat().st_size} bytes) in {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
