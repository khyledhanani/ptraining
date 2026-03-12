#!/usr/bin/env python3
"""Evaluate and compare two GRPO checkpoints on a held-out split.

Default setup matches this repo's training scripts:
- baseline model dir:   grpo-qwen-math-baseline
- arithmetic model dir: grpo-qwen-math-arithmetic
- dataset:              trl-lib/DeepMath-103K
- held-out slice:       train[5000:6000]

Metrics:
- pass@1
- pass@k (k = --num-samples)
- mean/std completion reward
- avg unique completions per prompt
- duplicate rate within prompt samples
- new tokens/sec and prompts/sec
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two trained GRPO models on held-out data.")
    parser.add_argument("--baseline-model", default="grpo-qwen-math-baseline", help="Path or HF id for baseline model")
    parser.add_argument("--arithmetic-model", default="grpo-qwen-math-arithmetic", help="Path or HF id for arithmetic model")

    parser.add_argument("--dataset", default="trl-lib/DeepMath-103K", help="Dataset name")
    parser.add_argument("--dataset-config", default=None, help="Optional dataset config")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--start", type=int, default=5000, help="Start index of held-out slice")
    parser.add_argument("--count", type=int, default=1000, help="Number of held-out examples")
    parser.add_argument("--prompt-column", default=None, help="Override prompt column name")
    parser.add_argument("--answer-column", default=None, help="Override answer column name")

    parser.add_argument("--batch-size", type=int, default=8, help="Prompts per generation batch")
    parser.add_argument("--num-samples", type=int, default=8, help="Completions per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generated tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p nucleus sampling")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 disables)")

    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto", help="Model dtype")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code")

    parser.add_argument("--output-json", default="eval_compare_results.json", help="Where to write metrics JSON")
    parser.add_argument("--log-wandb", action="store_true", help="Log summary metrics to W&B")
    parser.add_argument("--wandb-project", default="grpo-eval", help="W&B project name")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity")
    parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


PROMPT_CANDIDATES = [
    "prompt",
    "problem",
    "question",
    "query",
    "input",
    "instruction",
    "messages",
]
ANSWER_CANDIDATES = [
    "answer",
    "solution",
    "target",
    "final_answer",
    "ground_truth",
    "output",
]


def _infer_column(columns: list[str], explicit: str | None, candidates: list[str], kind: str) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"{kind} column '{explicit}' not in dataset columns: {columns}")
        return explicit

    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    for col in columns:
        col_l = col.lower()
        if any(candidate in col_l for candidate in candidates):
            return col

    raise ValueError(
        f"Could not infer {kind} column. Available columns={columns}. "
        f"Pass --{kind}-column explicitly."
    )


def _as_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    # Common chat format: [{"role": ..., "content": ...}, ...]
    if isinstance(value, list) and value and isinstance(value[0], dict) and "content" in value[0]:
        chunks = [str(item.get("content", "")) for item in value]
        return "\n".join(c for c in chunks if c)

    if isinstance(value, dict):
        for key in ("content", "text", "value", "answer", "solution"):
            if key in value:
                return _as_text(value[key])

    if isinstance(value, list):
        return "\n".join(_as_text(v) for v in value)

    return str(value)


def load_eval_slice(args: argparse.Namespace):
    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.dataset_config, split=args.split)
    end = min(len(ds), args.start + args.count)
    if args.start >= end:
        raise ValueError(f"Empty eval slice: start={args.start}, count={args.count}, split size={len(ds)}")

    ds = ds.select(range(args.start, end))

    prompt_col = _infer_column(ds.column_names, args.prompt_column, PROMPT_CANDIDATES, "prompt")
    answer_col = _infer_column(ds.column_names, args.answer_column, ANSWER_CANDIDATES, "answer")

    prompts: list[str] = []
    answers: list[str] = []
    for row in ds:
        prompts.append(_as_text(row[prompt_col]))
        answers.append(_as_text(row[answer_col]))

    return prompts, answers, prompt_col, answer_col


def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\\boxed{", "")
    s = s.replace("}", "")
    return s


def _extract_last_number(s: str) -> str | None:
    # Handles integers, decimals, and simple scientific notation.
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return matches[-1] if matches else None


def _fallback_score(completion: str, answer: str) -> float:
    c_norm = _normalize_text(completion)
    a_norm = _normalize_text(answer)

    if c_norm == a_norm and a_norm:
        return 1.0

    c_num = _extract_last_number(c_norm)
    a_num = _extract_last_number(a_norm)
    if c_num is not None and a_num is not None:
        try:
            return 1.0 if math.isclose(float(c_num), float(a_num), rel_tol=1e-9, abs_tol=1e-9) else 0.0
        except Exception:
            return 0.0

    return 0.0


def build_reward_scorer() -> tuple[str, Callable[[str, str, str], float]]:
    """Build a scalar reward scorer with graceful fallback.

    Returns:
        (backend_name, scorer)
        scorer(prompt, completion, answer) -> float in [0, 1]
    """

    try:
        from trl.rewards import accuracy_reward

        def trl_scorer(prompt: str, completion: str, answer: str) -> float:
            completion_chat = [[{"role": "assistant", "content": completion}]]

            call_variants = [
                {"completions": completion_chat, "solution": [answer], "prompts": [prompt]},
                {"completions": completion_chat, "solutions": [answer], "prompts": [prompt]},
                {"completions": completion_chat, "answer": [answer], "prompts": [prompt]},
                {"completions": completion_chat, "answers": [answer], "prompts": [prompt]},
                {"completions": completion_chat, "solution": [answer]},
                {"completions": completion_chat, "solutions": [answer]},
                {"completion": [completion], "answer": [answer], "prompt": [prompt]},
                {"completion": [completion], "solution": [answer], "prompt": [prompt]},
            ]

            for kwargs in call_variants:
                try:
                    sig = inspect.signature(accuracy_reward)
                    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
                    if not supported:
                        continue
                    out = accuracy_reward(**supported)
                    if isinstance(out, (list, tuple)) and out:
                        return float(out[0])
                    return float(out)
                except Exception:
                    continue

            # If signature probing fails, fallback for this sample.
            return _fallback_score(completion, answer)

        return "trl.accuracy_reward", trl_scorer

    except Exception:

        def fallback(prompt: str, completion: str, answer: str) -> float:
            del prompt
            return _fallback_score(completion, answer)

        return "fallback_numeric_exact", fallback


@dataclass
class ModelEvalResult:
    label: str
    model_path: str
    num_prompts: int
    num_samples: int
    pass_at_1: float
    pass_at_k: float
    reward_mean: float
    reward_std: float
    avg_unique_completions: float
    duplicate_rate: float
    total_new_tokens: int
    wall_time_sec: float
    new_tokens_per_sec: float
    prompts_per_sec: float


def _torch_dtype(dtype_name: str):
    import torch

    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    return None


def evaluate_model(
    model_path: str,
    label: str,
    prompts: list[str],
    answers: list[str],
    args: argparse.Namespace,
    scorer: Callable[[str, str, str], float],
) -> ModelEvalResult:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if use_cuda else None,
    )
    model.eval()
    if not use_cuda:
        model.to("cpu")

    k = args.num_samples
    batch_size = args.batch_size

    pass1_hits = 0
    passk_hits = 0
    all_scores: list[float] = []
    unique_counts: list[int] = []
    total_new_tokens = 0

    start_time = time.perf_counter()

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_answers = answers[i : i + batch_size]

        tokenized = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if use_cuda:
            tokenized = {k_: v.cuda() for k_, v in tokenized.items()}

        generate_kwargs = {
            "do_sample": True,
            "num_return_sequences": k,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if args.top_k > 0:
            generate_kwargs["top_k"] = args.top_k

        with torch.no_grad():
            generated = model.generate(**tokenized, **generate_kwargs)

        input_len = tokenized["input_ids"].shape[1]
        completions = tokenizer.batch_decode(generated[:, input_len:], skip_special_tokens=True)

        # HF outputs sequences grouped by each prompt, each with k completions.
        for j, (prompt, answer) in enumerate(zip(batch_prompts, batch_answers, strict=True)):
            group = completions[j * k : (j + 1) * k]
            group_scores = [float(scorer(prompt, c, answer)) for c in group]
            all_scores.extend(group_scores)

            if group_scores and group_scores[0] > 0.5:
                pass1_hits += 1
            if any(s > 0.5 for s in group_scores):
                passk_hits += 1

            normalized = [_normalize_text(c) for c in group]
            unique_counts.append(len(set(normalized)))

        # Track generated token count directly from token ids.
        generated_len = generated.shape[1]
        total_new_tokens += generated_len * generated.shape[0] - input_len * generated.shape[0]

    wall = max(time.perf_counter() - start_time, 1e-9)
    n_prompts = len(prompts)

    avg_unique = statistics.mean(unique_counts) if unique_counts else 0.0
    dup_rate = 1.0 - (avg_unique / max(k, 1))

    return ModelEvalResult(
        label=label,
        model_path=model_path,
        num_prompts=n_prompts,
        num_samples=k,
        pass_at_1=pass1_hits / n_prompts,
        pass_at_k=passk_hits / n_prompts,
        reward_mean=statistics.mean(all_scores) if all_scores else 0.0,
        reward_std=statistics.pstdev(all_scores) if len(all_scores) > 1 else 0.0,
        avg_unique_completions=avg_unique,
        duplicate_rate=dup_rate,
        total_new_tokens=total_new_tokens,
        wall_time_sec=wall,
        new_tokens_per_sec=total_new_tokens / wall,
        prompts_per_sec=n_prompts / wall,
    )


def print_summary(baseline: ModelEvalResult, arithmetic: ModelEvalResult, reward_backend: str) -> None:
    print("\n=== Evaluation Summary ===")
    print(f"Reward backend: {reward_backend}")

    headers = [
        "metric",
        "baseline",
        "arithmetic",
        "delta (arith - base)",
    ]
    rows = [
        ("pass@1", baseline.pass_at_1, arithmetic.pass_at_1),
        ("pass@k", baseline.pass_at_k, arithmetic.pass_at_k),
        ("reward_mean", baseline.reward_mean, arithmetic.reward_mean),
        ("reward_std", baseline.reward_std, arithmetic.reward_std),
        ("avg_unique_completions", baseline.avg_unique_completions, arithmetic.avg_unique_completions),
        ("duplicate_rate", baseline.duplicate_rate, arithmetic.duplicate_rate),
        ("new_tokens_per_sec", baseline.new_tokens_per_sec, arithmetic.new_tokens_per_sec),
        ("prompts_per_sec", baseline.prompts_per_sec, arithmetic.prompts_per_sec),
        ("wall_time_sec", baseline.wall_time_sec, arithmetic.wall_time_sec),
    ]

    print(" | ".join(headers))
    print("-" * 92)
    for name, b, a in rows:
        print(f"{name} | {b:.6f} | {a:.6f} | {a - b:+.6f}")


def maybe_log_wandb(
    args: argparse.Namespace,
    baseline: ModelEvalResult,
    arithmetic: ModelEvalResult,
    reward_backend: str,
) -> None:
    if not args.log_wandb:
        return

    try:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "dataset": args.dataset,
                "split": args.split,
                "start": args.start,
                "count": args.count,
                "num_samples": args.num_samples,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "reward_backend": reward_backend,
                "baseline_model": args.baseline_model,
                "arithmetic_model": args.arithmetic_model,
            },
        )

        metrics: dict[str, float] = {}
        for prefix, result in (("baseline", baseline), ("arithmetic", arithmetic)):
            rd = asdict(result)
            for key, value in rd.items():
                if isinstance(value, (int, float)):
                    metrics[f"eval/{prefix}/{key}"] = float(value)

        # Direct deltas for quick inspection.
        metrics.update(
            {
                "eval/delta/pass_at_1": arithmetic.pass_at_1 - baseline.pass_at_1,
                "eval/delta/pass_at_k": arithmetic.pass_at_k - baseline.pass_at_k,
                "eval/delta/reward_mean": arithmetic.reward_mean - baseline.reward_mean,
                "eval/delta/reward_std": arithmetic.reward_std - baseline.reward_std,
                "eval/delta/avg_unique_completions": arithmetic.avg_unique_completions - baseline.avg_unique_completions,
                "eval/delta/duplicate_rate": arithmetic.duplicate_rate - baseline.duplicate_rate,
                "eval/delta/new_tokens_per_sec": arithmetic.new_tokens_per_sec - baseline.new_tokens_per_sec,
            }
        )

        run.log(metrics)
        run.finish()
    except Exception as exc:
        print(f"[warn] W&B logging failed: {type(exc).__name__}: {exc}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    prompts, answers, prompt_col, answer_col = load_eval_slice(args)
    print(
        f"Loaded eval slice: dataset={args.dataset} split={args.split}[{args.start}:{args.start + len(prompts)}] "
        f"prompt_col={prompt_col} answer_col={answer_col}"
    )

    reward_backend, scorer = build_reward_scorer()
    print(f"Using reward scorer: {reward_backend}")

    baseline = evaluate_model(
        model_path=args.baseline_model,
        label="baseline",
        prompts=prompts,
        answers=answers,
        args=args,
        scorer=scorer,
    )
    arithmetic = evaluate_model(
        model_path=args.arithmetic_model,
        label="arithmetic",
        prompts=prompts,
        answers=answers,
        args=args,
        scorer=scorer,
    )

    print_summary(baseline, arithmetic, reward_backend)

    output = {
        "meta": {
            "dataset": args.dataset,
            "dataset_config": args.dataset_config,
            "split": args.split,
            "start": args.start,
            "count": len(prompts),
            "prompt_column": prompt_col,
            "answer_column": answer_col,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "reward_backend": reward_backend,
        },
        "baseline": asdict(baseline),
        "arithmetic": asdict(arithmetic),
        "delta": {
            "pass_at_1": arithmetic.pass_at_1 - baseline.pass_at_1,
            "pass_at_k": arithmetic.pass_at_k - baseline.pass_at_k,
            "reward_mean": arithmetic.reward_mean - baseline.reward_mean,
            "reward_std": arithmetic.reward_std - baseline.reward_std,
            "avg_unique_completions": arithmetic.avg_unique_completions - baseline.avg_unique_completions,
            "duplicate_rate": arithmetic.duplicate_rate - baseline.duplicate_rate,
            "new_tokens_per_sec": arithmetic.new_tokens_per_sec - baseline.new_tokens_per_sec,
            "prompts_per_sec": arithmetic.prompts_per_sec - baseline.prompts_per_sec,
            "wall_time_sec": arithmetic.wall_time_sec - baseline.wall_time_sec,
        },
    }

    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote results to: {out_path.resolve()}")

    maybe_log_wandb(args, baseline, arithmetic, reward_backend)


if __name__ == "__main__":
    main()
