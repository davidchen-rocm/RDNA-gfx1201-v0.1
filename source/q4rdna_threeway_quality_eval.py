#!/usr/bin/env python3
"""Fixed BF16 vs Q4_K_M vs Q4_RDNA quality comparison."""

import argparse
import gc
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import disable_progress_bar, load_dataset
from gguf import GGUFReader, dequantize
from transformers import AutoModelForCausalLM, AutoTokenizer

from q4rdna_quality_smoke import fake_quantize_weight


MMLU_DEFAULT_COUNTS = {
    "abstract_algebra": 60,
    "college_mathematics": 60,
    "high_school_mathematics": 65,
    "elementary_mathematics": 65,
}

GGUF_TENSOR_NAMES = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}


def write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def load_model(path: str):
    return AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()


def format_mmlu(example: dict) -> str:
    labels = "ABCD"
    choices = "\n".join(f"{labels[index]}. {choice}" for index, choice in enumerate(example["choices"]))
    return f"Question: {example['question']}\n{choices}\nAnswer:"


def load_math_items(full: bool) -> tuple[list[dict], dict[str, int]]:
    items = []
    counts = {}
    for subject, default_count in MMLU_DEFAULT_COUNTS.items():
        dataset = load_dataset("cais/mmlu", subject, split="test").shuffle(seed=20260815)
        count = len(dataset) if full else default_count
        counts[subject] = count
        for example in dataset.select(range(count)):
            items.append({
                "subject": subject,
                "prompt": format_mmlu(example),
                "answer": "ABCD"[example["answer"]],
            })
    return items, counts


def load_perplexity_tokens(tokenizer, count: int) -> torch.Tensor:
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    text = "\n\n".join(f"Problem: {row['problem']}\nSolution: {row['solution']}" for row in dataset)
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if tokens.numel() < count + 1:
        raise RuntimeError("perplexity corpus is shorter than requested")
    return tokens[:count + 1]


@torch.inference_mode()
def evaluate_math(model, tokenizer, items: list[dict], batch_size: int) -> dict:
    started = time.monotonic()
    tokenizer.padding_side = "left"
    label_tokens = []
    for label in "ABCD":
        ids = tokenizer(" " + label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise RuntimeError(f"answer label {label} is not one token")
        label_tokens.append(ids[0])

    predictions = []
    for first in range(0, len(items), batch_size):
        batch = items[first:first + batch_size]
        encoded = tokenizer([item["prompt"] for item in batch], padding=True, return_tensors="pt")
        logits = model(**encoded, use_cache=False, logits_to_keep=1).logits[:, -1, label_tokens].float()
        predictions.extend("ABCD"[index] for index in logits.argmax(dim=1).tolist())

    by_subject = defaultdict(lambda: [0, 0])
    disagreements = []
    correct = 0
    for index, (item, prediction) in enumerate(zip(items, predictions)):
        matched = prediction == item["answer"]
        correct += matched
        by_subject[item["subject"]][0] += matched
        by_subject[item["subject"]][1] += 1
        if not matched:
            disagreements.append({"index": index, "subject": item["subject"], "prediction": prediction, "answer": item["answer"]})
    return {
        "correct": correct,
        "total": len(items),
        "accuracy": correct / len(items),
        "by_subject": {subject: {"correct": values[0], "total": values[1]} for subject, values in by_subject.items()},
        "wrong": disagreements,
        "predictions": "".join(predictions),
        "seconds": time.monotonic() - started,
    }


@torch.inference_mode()
def evaluate_perplexity(model, tokens: torch.Tensor, sequence_length: int, batch_size: int) -> dict:
    started = time.monotonic()
    sequences = []
    for first in range(0, tokens.numel() - 1, sequence_length):
        sequence = tokens[first:min(first + sequence_length + 1, tokens.numel())]
        if sequence.numel() > 1:
            sequences.append(sequence)

    total_loss = 0.0
    total_tokens = 0
    for first in range(0, len(sequences), batch_size):
        batch = sequences[first:first + batch_size]
        width = max(sequence.numel() for sequence in batch)
        input_ids = torch.zeros((len(batch), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        valid = 0
        for row, sequence in enumerate(batch):
            input_ids[row, :sequence.numel()] = sequence
            attention_mask[row, :sequence.numel()] = 1
            labels[row, :sequence.numel()] = sequence
            valid += sequence.numel() - 1
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False).loss
        total_loss += float(loss) * valid
        total_tokens += valid
    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss, "perplexity": math.exp(mean_loss), "tokens": total_tokens, "seconds": time.monotonic() - started}


def evaluate_and_save(result, key, model, tokenizer, math_items, perplexity_tokens, args, output) -> None:
    result[key] = {"math": evaluate_math(model, tokenizer, math_items, args.math_batch)}
    write_result(output, result)
    result[key]["perplexity"] = evaluate_perplexity(model, perplexity_tokens, args.ppl_sequence, args.ppl_batch)
    write_result(output, result)


@torch.inference_mode()
def apply_q4_k_m(model, gguf_path: str) -> dict:
    started = time.monotonic()
    reader = GGUFReader(gguf_path, "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    quantized_weights = 0
    packed_bytes = 0
    types = Counter()
    for layer_id, layer in enumerate(model.model.layers):
        parameters = dict(layer.named_parameters())
        for hf_name, suffix in GGUF_TENSOR_NAMES.items():
            parameter = parameters[hf_name]
            tensor = tensors[f"blk.{layer_id}.{suffix}"]
            values = dequantize(tensor.data, tensor.tensor_type)
            if tuple(values.shape) != tuple(parameter.shape):
                raise RuntimeError(f"shape mismatch for {tensor.name}: {values.shape} vs {tuple(parameter.shape)}")
            parameter.copy_(torch.from_numpy(values).to(torch.bfloat16))
            quantized_weights += parameter.numel()
            packed_bytes += tensor.data.nbytes
            types[tensor.tensor_type.name] += parameter.numel()
            del values
    return {
        "quantized_weights": quantized_weights,
        "packed_bytes": packed_bytes,
        "effective_bpw": packed_bytes * 8 / quantized_weights,
        "weight_types": dict(types),
        "seconds": time.monotonic() - started,
    }


@torch.inference_mode()
def apply_q4_rdna(model, chunk_groups: int) -> dict:
    started = time.monotonic()
    quantized_weights = 0
    for layer in model.model.layers:
        parameters = dict(layer.named_parameters())
        for hf_name in GGUF_TENSOR_NAMES:
            parameter = parameters[hf_name]
            fake_quantize_weight(parameter, chunk_groups)
            quantized_weights += parameter.numel()
    return {
        "quantized_weights": quantized_weights,
        "effective_bpw": 4.25,
        "seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen3-8b-hf")
    parser.add_argument("--q4-k", default="models/qwen3-8b-q4/Qwen3-8B-Q4_K_M.gguf")
    parser.add_argument("--ppl-tokens", type=int, default=4096)
    parser.add_argument("--ppl-sequence", type=int, default=256)
    parser.add_argument("--ppl-batch", type=int, default=4)
    parser.add_argument("--math-batch", type=int, default=16)
    parser.add_argument("--full-math", action="store_true",
                        help="evaluate every test item in the four MMLU math subjects")
    parser.add_argument("--chunk-groups", type=int, default=4096)
    parser.add_argument("--variants", default="bf16,q4_k_m,q4_rdna")
    parser.add_argument("--output", default="artifacts/q4rdna-phase1/quality-threeway.json")
    args = parser.parse_args()

    disable_progress_bar()
    torch.set_num_threads(12)
    output = Path(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    math_items, math_counts = load_math_items(args.full_math)
    perplexity_tokens = load_perplexity_tokens(tokenizer, args.ppl_tokens)
    protocol_hash = hashlib.sha256(
        ("\n".join(item["prompt"] + item["answer"] for item in math_items)).encode()
        + perplexity_tokens.numpy().tobytes()
    ).hexdigest()
    protocol = {
            "math": f"{len(math_items)} deterministic zero-shot MMLU math questions",
            "math_subjects": math_counts,
            "perplexity_corpus":
                f"first {args.ppl_tokens} tokens of deterministic MATH-500 problem+solution concatenation",
            "perplexity_tokens": args.ppl_tokens,
            "scope": "252 linear weights in all 36 transformer blocks; embedding, lm_head, and norms remain BF16",
            "hash": protocol_hash,
    }
    if output.exists():
        result = json.loads(output.read_text())
        if result.get("protocol", {}).get("hash") != protocol_hash:
            raise RuntimeError("existing output uses a different protocol")
    else:
        result = {"status": "running", "protocol": protocol}

    variants = {name for name in args.variants.split(",") if name}
    if "bf16" in variants:
        model = load_model(args.model)
        evaluate_and_save(result, "bf16", model, tokenizer, math_items, perplexity_tokens, args, output)
        del model
        gc.collect()

    if "q4_k_m" in variants:
        model = load_model(args.model)
        result["q4_k_m_quantization"] = apply_q4_k_m(model, args.q4_k)
        write_result(output, result)
        evaluate_and_save(result, "q4_k_m", model, tokenizer, math_items, perplexity_tokens, args, output)
        del model
        gc.collect()

    if "q4_rdna" in variants:
        model = load_model(args.model)
        result["q4_rdna_quantization"] = apply_q4_rdna(model, args.chunk_groups)
        write_result(output, result)
        evaluate_and_save(result, "q4_rdna", model, tokenizer, math_items, perplexity_tokens, args, output)
        del model
        gc.collect()

    if all(name in result for name in ("bf16", "q4_k_m", "q4_rdna")):
        result["status"] = "complete"
    write_result(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
