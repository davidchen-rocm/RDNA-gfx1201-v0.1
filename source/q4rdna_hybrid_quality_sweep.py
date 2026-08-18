#!/usr/bin/env python3
"""Screen Q4_RDNA/Q4_K_M hybrid weight assignments on CPU."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from gguf import GGUFReader, dequantize
from transformers import AutoTokenizer

from q4rdna_quality_smoke import fake_quantize_weight
from q4rdna_threeway_quality_eval import (
    GGUF_TENSOR_NAMES,
    evaluate_math,
    evaluate_perplexity,
    load_math_items,
    load_model,
    load_perplexity_tokens,
)


def uses_q4_k_fallback(candidate: str, layer_id: int, hf_name: str) -> bool:
    if candidate == "first12":
        return layer_id < 12
    if candidate == "last12":
        return layer_id >= 24
    if candidate == "down":
        return hf_name == "mlp.down_proj.weight"
    if candidate == "attention":
        return hf_name.startswith("self_attn.")
    if candidate == "qkv":
        return hf_name in {
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
        }
    raise ValueError(f"unknown candidate: {candidate}")


@torch.inference_mode()
def apply_hybrid(model, candidate: str, tensors: dict, chunk_groups: int) -> dict:
    started = time.monotonic()
    q4_rdna_weights = 0
    q4_k_weights = 0
    q4_rdna_bytes = 0
    q4_k_bytes = 0
    fallback_tensors = 0

    for layer_id, layer in enumerate(model.model.layers):
        parameters = dict(layer.named_parameters())
        for hf_name, suffix in GGUF_TENSOR_NAMES.items():
            parameter = parameters[hf_name]
            if uses_q4_k_fallback(candidate, layer_id, hf_name):
                tensor = tensors[f"blk.{layer_id}.{suffix}"]
                values = dequantize(tensor.data, tensor.tensor_type)
                if tuple(values.shape) != tuple(parameter.shape):
                    raise RuntimeError(
                        f"shape mismatch for {tensor.name}: {values.shape} vs {tuple(parameter.shape)}"
                    )
                parameter.copy_(torch.from_numpy(values).to(torch.bfloat16))
                q4_k_weights += parameter.numel()
                q4_k_bytes += tensor.data.nbytes
                fallback_tensors += 1
                del values
            else:
                fake_quantize_weight(parameter, chunk_groups)
                q4_rdna_weights += parameter.numel()
                q4_rdna_bytes += parameter.numel() * 4.25 // 8

    total_weights = q4_rdna_weights + q4_k_weights
    total_bytes = q4_rdna_bytes + q4_k_bytes
    return {
        "candidate": candidate,
        "q4_rdna_weights": q4_rdna_weights,
        "q4_k_fallback_weights": q4_k_weights,
        "q4_k_fallback_tensors": fallback_tensors,
        "q4_rdna_coverage_fraction": q4_rdna_weights / total_weights,
        "packed_bytes": total_bytes,
        "effective_bpw": total_bytes * 8 / total_weights,
        "seconds": time.monotonic() - started,
    }


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen3-8b-hf")
    parser.add_argument("--q4-k", default="models/qwen3-8b-q4/Qwen3-8B-Q4_K_M.gguf")
    parser.add_argument("--candidates", default="first12,last12,down,attention,qkv")
    parser.add_argument("--ppl-tokens", type=int, default=4096)
    parser.add_argument("--ppl-sequence", type=int, default=256)
    parser.add_argument("--ppl-batch", type=int, default=4)
    parser.add_argument("--math-batch", type=int, default=16)
    parser.add_argument("--full-math", action="store_true")
    parser.add_argument("--chunk-groups", type=int, default=4096)
    parser.add_argument("--output", default="artifacts/q4rdna-phase4/hybrid-screen.json")
    args = parser.parse_args()

    torch.set_num_threads(12)
    output = Path(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    perplexity_tokens = load_perplexity_tokens(tokenizer, args.ppl_tokens)
    math_items = None
    math_counts = None
    if args.full_math:
        math_items, math_counts = load_math_items(True)

    reader = GGUFReader(args.q4_k, "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    candidates = [value for value in args.candidates.split(",") if value]
    result = {
        "status": "running",
        "protocol": {
            "candidates": candidates,
            "perplexity_tokens": args.ppl_tokens,
            "perplexity_sequence": args.ppl_sequence,
            "perplexity_batch": args.ppl_batch,
            "math_subjects": math_counts,
            "fallback": "real Q4_K_M tensor dequantization",
            "q4_rdna": "fixed 4.25 bpw representation and quantizer",
        },
        "candidates": {},
    }
    write_result(output, result)

    for candidate in candidates:
        model = load_model(args.model)
        entry = {"assignment": apply_hybrid(model, candidate, tensors, args.chunk_groups)}
        entry["perplexity"] = evaluate_perplexity(
            model, perplexity_tokens, args.ppl_sequence, args.ppl_batch
        )
        if math_items is not None:
            entry["math"] = evaluate_math(model, tokenizer, math_items, args.math_batch)
        result["candidates"][candidate] = entry
        write_result(output, result)
        print(json.dumps({candidate: entry}, ensure_ascii=False), flush=True)
        del model
        gc.collect()

    result["status"] = "complete"
    write_result(output, result)


if __name__ == "__main__":
    main()
