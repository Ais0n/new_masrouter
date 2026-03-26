#!/usr/bin/env python3
"""
Offline Evaluation for SDPO v3 LLMRouter.

Metrics
───────
  format_valid    – output contains at least one "Step N (Role):" line
  judge_correct   – GPT-4o-mini judges whether the output shows CONTAINMENT or
                    PROPAGATION; compared to the ground-truth outcome label
                    (standalone/received → expect containment;
                     propagated → expect recovery/containment)
  containment_kw  – keyword heuristic: recovery words vs propagation words
                    in [-1, +1]; used as a cheap backup when no judge key

Usage (run from masrouter/):
  python LLMRouter/eval_offline.py \\
    --adapter  checkpoints/llm_router_sdpo_v3/final \\
    --data     ../MAST-Data/output/dpo_pairs/sdpo_pairs_v3.jsonl \\
    [--judge-model openai/gpt-4o-mini] \\
    [--judge-key  $OPENROUTER_API_KEY] \\
    [--no-judge] [--val-ratio 0.1] [--seed 42] [--output eval_sdpo_results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

try:
    from LLMRouter.train_dpo import SYSTEM_PROMPT
except ModuleNotFoundError:
    from train_dpo import SYSTEM_PROMPT


# ── LLM-as-judge ──────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are evaluating a multi-agent system trajectory continuation. "
    "Your only job is to classify whether the trajectory shows ERROR CONTAINMENT "
    "(an agent catches, flags, or corrects the error before it reaches the final output) "
    "or ERROR PROPAGATION (the error passes through agents unchanged and reaches the output). "
    'Reply with exactly one word: "containment" or "propagation".'
)

JUDGE_USER_TMPL = """\
[Error pattern]: {error_pattern}
[Trajectory continuation to classify]:
{output}
"""


def judge_one(output: str, error_pattern: str,
              model: str, api_key: str,
              retries: int = 3) -> Optional[str]:
    """Call OpenRouter to judge containment vs propagation. Returns 'containment'/'propagation'/None."""
    import requests
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model":       model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": JUDGE_USER_TMPL.format(
                error_pattern=error_pattern,
                output=output[:800],
            )},
        ],
        "max_tokens":  5,
        "temperature": 0.0,
    }
    for attempt in range(retries):
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers, json=payload, timeout=30,
            )
            resp.raise_for_status()
            verdict = resp.json()["choices"][0]["message"]["content"].strip().lower()
            if "containment" in verdict:
                return "containment"
            if "propagation" in verdict:
                return "propagation"
            return None
        except Exception as e:
            time.sleep(2 ** attempt)
    return None


# ── Keyword fallback ───────────────────────────────────────────────────────────

STEP_RE = re.compile(r"Step\s+\d+\s*\(", re.IGNORECASE)

CONTAINMENT_KW = {
    "verify", "verif", "correct", "correcting", "correction", "fix", "fixing",
    "catch", "catching", "detect", "error caught", "mistake", "wrong answer",
    "re-check", "recheck", "review", "confirms", "accurate", "confirmed",
    "right answer", "actual answer", "should be", "incorrect",
}
PROPAGATION_KW = {
    "passes", "passing", "forwarding", "forward", "unchanged",
    "submit", "submitting", "final answer", "finaliz", "output the answer",
    "accepts", "accepted", "uses the result", "sends the result",
    "does not check", "without checking", "without verification",
}


def containment_score(text: str) -> float:
    t = text.lower()
    pos = sum(1 for kw in CONTAINMENT_KW if kw in t)
    neg = sum(1 for kw in PROPAGATION_KW if kw in t)
    total = pos + neg
    return 0.0 if total == 0 else (pos - neg) / total


def format_valid(text: str) -> bool:
    return bool(STEP_RE.search(text or ""))


# Outcomes where the model SHOULD generate containment
CONTAINMENT_EXPECTED = {"standalone", "received"}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_val_split(data_path: str, val_ratio: float, seed: int) -> List[dict]:
    random.seed(seed)
    pairs = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    return pairs[:n_val]


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(pair: dict, router) -> str:
    import torch
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": pair["prompt"]},
    ]
    text = router.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = router.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=3200
    ).to(router.model.device)
    with torch.inference_mode():
        outputs = router.model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            pad_token_id=router.tokenizer.pad_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return router.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Display helpers ────────────────────────────────────────────────────────────

def acc_bar(correct: int, total: int, width: int = 28) -> str:
    if total == 0:
        return "N/A"
    frac = correct / total
    bar = "█" * int(frac * width) + "░" * (width - int(frac * width))
    return f"{bar}  {correct}/{total}  ({100*frac:.1f}%)"


def float_bar(value: float, width: int = 28) -> str:
    frac = (value + 1) / 2
    bar = "█" * int(frac * width) + "░" * (width - int(frac * width))
    return f"{bar}  {value:+.3f}"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",      required=True)
    parser.add_argument("--data",         required=True)
    parser.add_argument("--base-model",   default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--val-ratio",    type=float, default=0.1)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--output",       default="eval_sdpo_results.json")
    parser.add_argument("--max-samples",  type=int,   default=0)
    parser.add_argument("--judge-model",  default="openai/gpt-4o-mini")
    parser.add_argument("--judge-key",
                        default=os.getenv("OPENROUTER_API_KEY", ""),
                        help="OpenRouter API key for LLM-as-judge (or set OPENROUTER_API_KEY).")
    parser.add_argument("--no-judge",     action="store_true",
                        help="Skip LLM judge; use keyword heuristic only.")
    args = parser.parse_args()

    use_judge = not args.no_judge and bool(args.judge_key)
    if not use_judge:
        print("LLM judge disabled — using keyword heuristic only.")

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading val split from {args.data} ...")
    val_pairs = load_val_split(args.data, args.val_ratio, args.seed)
    if args.max_samples > 0:
        val_pairs = val_pairs[:args.max_samples]
    print(f"Val set size: {len(val_pairs)}")
    print(f"Outcome dist : {dict(Counter(p['meta'].get('outcome','?') for p in val_pairs))}")

    # ── Load router ──────────────────────────────────────────────────────────
    try:
        from LLMRouter.router_inference import LLMRouter
    except ModuleNotFoundError:
        from router_inference import LLMRouter
    print(f"\nLoading LLMRouter from {args.adapter} ...")
    router = LLMRouter(adapter_path=args.adapter, base_model=args.base_model)

    # ── Inference + evaluation ───────────────────────────────────────────────
    print(f"\nRunning inference on {len(val_pairs)} samples ...")
    results = []
    for i, pair in enumerate(val_pairs):
        meta         = pair.get("meta", {})
        outcome      = meta.get("outcome", "?")
        coarse       = meta.get("coarse_type", "?")
        error_code   = meta.get("error_code", "?")
        expect_containment = outcome in CONTAINMENT_EXPECTED

        raw = run_inference(pair, router)

        fv = format_valid(raw)
        cs = containment_score(raw)

        # LLM judge
        judge_verdict = None
        judge_correct = None
        if use_judge:
            judge_verdict = judge_one(
                raw,
                error_pattern=f"{error_code} ({coarse})",
                model=args.judge_model,
                api_key=args.judge_key,
            )
            if judge_verdict is not None:
                judge_correct = (judge_verdict == "containment") == expect_containment

        # Keyword fallback correctness (for display when no judge)
        kw_correct = (cs >= 0) == expect_containment

        correct_flag = judge_correct if judge_correct is not None else kw_correct
        status = "✓" if correct_flag else "✗"
        judge_str = f"judge={judge_verdict}" if judge_verdict else f"kw_correct={'Y' if kw_correct else 'N'}"
        print(f"  [{i+1:>3}/{len(val_pairs)}] {status}  "
              f"outcome={outcome}  coarse={coarse}  "
              f"fmt={'ok' if fv else 'BAD'}  "
              f"cs={cs:+.2f}  {judge_str}")

        results.append({
            "idx":             i,
            "outcome":         outcome,
            "coarse_type":     coarse,
            "format_valid":    fv,
            "containment_score": round(cs, 4),
            "judge_verdict":   judge_verdict,
            "judge_correct":   judge_correct,
            "kw_correct":      kw_correct,
            "correct":         correct_flag,
            "raw_output":      raw[:400],
        })

    # ── Aggregate ────────────────────────────────────────────────────────────
    n = len(results)
    n_fmt     = sum(r["format_valid"] for r in results)
    n_correct = sum(r["correct"]      for r in results)
    mean_cs   = sum(r["containment_score"] for r in results) / n
    n_judged  = sum(1 for r in results if r["judge_verdict"] is not None)

    by_outcome: dict = defaultdict(lambda: {"correct": 0, "total": 0, "cs_sum": 0.0})
    for r in results:
        oc = r["outcome"]
        by_outcome[oc]["total"]  += 1
        by_outcome[oc]["cs_sum"] += r["containment_score"]
        if r["correct"]:
            by_outcome[oc]["correct"] += 1

    # ── Report ───────────────────────────────────────────────────────────────
    SEP = "─" * 62
    judge_label = f"judge ({n_judged}/{n} judged)" if use_judge else "kw-heuristic"
    print(f"\n{'═'*62}")
    print(f"  SDPO OFFLINE EVAL — {args.adapter}")
    print(f"{'═'*62}")
    print(f"  Val samples       : {n}")
    print()
    print(f"  Format valid      : {acc_bar(n_fmt, n)}")
    print(f"  Correct ({judge_label:18s}): {acc_bar(n_correct, n)}")
    print(f"  Containment score : {float_bar(mean_cs)}  (mean; +1=containment, -1=propagation)")
    print()
    print(f"{SEP}")
    print("  Per-outcome breakdown:")
    for oc, d in sorted(by_outcome.items()):
        exp = "→ containment expected" if oc in CONTAINMENT_EXPECTED else "→ recovery expected"
        print(f"    {oc:14s} {exp}: {acc_bar(d['correct'], d['total'])}  "
              f"cs={d['cs_sum']/d['total']:+.2f}")

    print(f"\n{SEP}")
    print("  Sample outputs (first 3):")
    for r in results[:3]:
        verdict = f"judge={r['judge_verdict']}" if r["judge_verdict"] else f"cs={r['containment_score']:+.2f}"
        print(f"\n  ── sample {r['idx']} ({r['outcome']}, {r['coarse_type']}, {verdict}) ──")
        print(f"  {r['raw_output'][:300]}")

    # ── Save ─────────────────────────────────────────────────────────────────
    summary = {
        "adapter":              args.adapter,
        "n_val_samples":        n,
        "format_valid_rate":    round(n_fmt     / n, 4),
        "correct_rate":         round(n_correct / n, 4),
        "judge_coverage":       round(n_judged  / n, 4) if use_judge else 0.0,
        "mean_containment_score": round(mean_cs, 4),
        "per_outcome": {
            oc: {
                "correct_rate":     round(d["correct"] / d["total"], 4),
                "mean_containment": round(d["cs_sum"] / d["total"], 4),
            }
            for oc, d in by_outcome.items()
        },
        "per_sample": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  Full results → {out_path}")
    print(f"{'═'*62}\n")

    # ── Diagnostics ──────────────────────────────────────────────────────────
    print("  DIAGNOSTIC NOTES:")
    if n_fmt / n < 0.7:
        print("  ⚠  <70% format valid — model is not following Step N (Role): format.")
    if n_correct / n < 0.55:
        print("  ⚠  Correct rate < 55% — model containment/propagation direction is off.")
        print("     Try more epochs or higher beta.")
    if mean_cs < 0.0:
        print("  ⚠  Negative mean containment score — outputs lean toward propagation.")
    if n_correct / n >= 0.65 and mean_cs > 0.1 and n_fmt / n >= 0.8:
        print("  ✓  Routing looks healthy — proceed to end-to-end eval (Milestone 2).")


if __name__ == "__main__":
    main()
