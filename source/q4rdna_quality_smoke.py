#!/usr/bin/env python3
"""Short, CPU-only Q4_RDNA model-quality smoke test.

The test keeps the deployed representation fixed: 64 weights share one FP16
scale and each weight is a signed INT4 in [-8, 7]. It fake-quantizes selected
Qwen decoder layers in place, then compares teacher-forced loss and generation
against the same BF16 model before mutation.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TEXT = """Mathematics is useful because it gives precise ways to describe patterns. A proof does more than report that a statement seems true: it explains why the statement must be true. For example, the sum of the first n positive integers is n times n plus one, divided by two. This identity can be shown by pairing the first and last terms, then the second and second-to-last terms. Scientific models use the same discipline. They simplify reality, state assumptions, and produce predictions that can be checked against observations. When a prediction fails, the model or its assumptions must be revised. Careful measurement therefore matters as much as elegant theory."""

MATH_ITEMS = [
    ("Solve 3x + 7 = 25. Answer: ", ["4", "5", "6", "7"], "6"),
    ("Compute 17 times 6. Answer: ", ["92", "96", "102", "112"], "102"),
    ("What is 15 percent of 240? Answer: ", ["24", "32", "36", "40"], "36"),
    ("A square has side length 9. Its area is: ", ["18", "36", "72", "81"], "81"),
    ("The next prime number after 31 is: ", ["32", "33", "35", "37"], "37"),
    ("If y/4 = 7, then y is: ", ["11", "21", "28", "32"], "28"),
    ("The average of 8, 12, and 16 is: ", ["10", "12", "14", "16"], "12"),
    ("A fair coin is flipped twice. The probability of two heads is: ", ["1/2", "1/3", "1/4", "1/8"], "1/4"),
    ("Simplify 2 to the fifth power. Answer: ", ["10", "16", "25", "32"], "32"),
    ("A triangle has base 10 and height 6. Its area is: ", ["16", "30", "60", "120"], "30"),
    ("Compute 144 divided by 12. Answer: ", ["10", "11", "12", "14"], "12"),
    ("If a sequence starts 5, 9, 13, 17, the next term is: ", ["19", "20", "21", "22"], "21"),
]


def fp16_round(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float16).to(torch.float32)


@torch.inference_mode()
def quantize_chunk(values: torch.Tensor) -> torch.Tensor:
    """FP16-aware multi-start least-squares scale for [groups, 64]."""
    positive_max = values.clamp_min(0).amax(dim=1, keepdim=True)
    negative_max = (-values).clamp_min(0).amax(dim=1, keepdim=True)
    positive_range = torch.maximum(positive_max / 7.0, negative_max / 8.0)
    negative_range = -torch.maximum(positive_max / 8.0, negative_max / 7.0)

    factors = (1.0, 0.9, 0.8, 0.7, 0.6)
    starts = torch.stack(
        [scale * factor for factor in factors for scale in (positive_range, negative_range)],
        dim=0,
    )
    scales = fp16_round(starts)
    best_error = torch.full_like(scales[:, :, 0], math.inf)
    best_scale = scales.clone()

    for _ in range(12):
        quant = torch.round(values.unsqueeze(0) / scales.clamp(min=-math.inf, max=math.inf)).clamp(-8, 7)
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
    quant = torch.where(scale == 0, torch.zeros_like(quant), quant)
    return quant * scale


@torch.inference_mode()
def fake_quantize_weight(weight: torch.Tensor, chunk_groups: int) -> None:
    if weight.ndim != 2 or weight.shape[1] % 64:
        raise ValueError(f"unsupported Q4_RDNA tensor shape: {tuple(weight.shape)}")
    flat = weight.view(-1)
    groups = flat.numel() // 64
    for first in range(0, groups, chunk_groups):
        last = min(first + chunk_groups, groups)
        source = flat[first * 64:last * 64].to(torch.float32).view(-1, 64)
        flat[first * 64:last * 64].copy_(quantize_chunk(source).reshape(-1).to(weight.dtype))


@torch.inference_mode()
def measure_loss(model, input_ids: torch.Tensor) -> dict:
    started = time.monotonic()
    logits = model(input_ids=input_ids, use_cache=False).logits[:, :-1].float()
    labels = input_ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    value = float(loss)
    return {"loss": value, "perplexity": math.exp(value), "seconds": time.monotonic() - started}


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, tokens: int) -> dict:
    encoded = tokenizer(prompt, return_tensors="pt")
    started = time.monotonic()
    output = model.generate(
        **encoded,
        max_new_tokens=tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    return {
        "text": tokenizer.decode(output[0, encoded.input_ids.shape[1]:], skip_special_tokens=True),
        "seconds": time.monotonic() - started,
    }


@torch.inference_mode()
def math_accuracy(model, tokenizer) -> dict:
    started = time.monotonic()
    correct = 0
    predictions = []
    for prompt, choices, answer in MATH_ITEMS:
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        sequences = []
        answer_starts = []
        for choice in choices:
            choice_ids = tokenizer(choice, add_special_tokens=False).input_ids
            sequences.append(prompt_ids + choice_ids)
            answer_starts.append(len(prompt_ids))
        width = max(len(sequence) for sequence in sequences)
        input_ids = torch.full((len(sequences), width), tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, sequence in enumerate(sequences):
            input_ids[row, :len(sequence)] = torch.tensor(sequence)
            attention_mask[row, :len(sequence)] = 1
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1)
        scores = []
        for row, sequence in enumerate(sequences):
            score = 0.0
            for position in range(answer_starts[row], len(sequence)):
                score += float(log_probs[row, position - 1, sequence[position]])
            scores.append(score)
        prediction = choices[max(range(len(choices)), key=scores.__getitem__)]
        correct += prediction == answer
        predictions.append({"prompt": prompt, "prediction": prediction, "answer": answer})
    return {"correct": correct, "total": len(MATH_ITEMS), "accuracy": correct / len(MATH_ITEMS), "seconds": time.monotonic() - started, "predictions": predictions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/qwen3-8b-hf")
    parser.add_argument("--layers", default="0,12,24,35")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--generate-tokens", type=int, default=8)
    parser.add_argument("--chunk-groups", type=int, default=4096)
    parser.add_argument("--output", default="artifacts/q4rdna-phase1/quality-smoke.json")
    args = parser.parse_args()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    layer_ids = [int(value) for value in args.layers.split(",") if value]
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    input_ids = tokenizer(DEFAULT_TEXT, return_tensors="pt").input_ids[:, :args.tokens]
    prompt = "Solve carefully: If 3x + 7 = 25, then x ="

    result = {
        "model": args.model,
        "layers": layer_ids,
        "eval_tokens": int(input_ids.shape[1]),
        "format": {"group": 64, "scale": "FP16", "quant": "signed INT4 [-8,7]", "bpw": 4.25},
    }
    result["bf16"] = {
        "metrics": measure_loss(model, input_ids),
        "math": math_accuracy(model, tokenizer),
        "generation": generate(model, tokenizer, prompt, args.generate_tokens),
    }

    quant_started = time.monotonic()
    quantized_weights = 0
    quantized_names = []
    for layer_id in layer_ids:
        layer = model.model.layers[layer_id]
        for name, parameter in layer.named_parameters():
            if parameter.ndim != 2:
                continue
            fake_quantize_weight(parameter.data, args.chunk_groups)
            quantized_weights += parameter.numel()
            quantized_names.append(f"model.layers.{layer_id}.{name}")
    result["q4_rdna"] = {
        "quantized_weights": quantized_weights,
        "quantized_tensors": quantized_names,
        "quantize_seconds": time.monotonic() - quant_started,
        "metrics": measure_loss(model, input_ids),
        "math": math_accuracy(model, tokenizer),
        "generation": generate(model, tokenizer, prompt, args.generate_tokens),
    }
    result["delta"] = {
        "loss": result["q4_rdna"]["metrics"]["loss"] - result["bf16"]["metrics"]["loss"],
        "perplexity_fraction": result["q4_rdna"]["metrics"]["perplexity"] / result["bf16"]["metrics"]["perplexity"] - 1,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
