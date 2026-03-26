#!/usr/bin/env python3
"""
Offline Evaluation for SDPO v3 LLMRouter.

In SDPO v3 chosen/rejected are plain trajectory-segment strings, not routing
config JSON.  This script evaluates whether the fine-tuned model generates
trajectory continuations that look like the CHOSEN (containment) side rather
than the REJECTED (propagation) side.

Metrics
───────
  format_valid    – output contains at least one "Step N (Role):" line
  containment_kw  – output uses recovery language (verifier catching error)
                    vs propagation language (passing error unchanged)
  prefer_chosen   – output is lexically closer to chosen than rejected
                    (bigram Jaccard similarity)
  repair_rate     – fraction of outputs that were empty or malformed

Usage (run from masrouter/):
  python LLMRouter/eval_offline.py \\
    --adapter  checkpoints/llm_router_sdpo_v3/final \\
    --data     ../MAST-Data/output/dpo_pairs/sdpo_pairs_v3.jsonl \\
    [--val-ratio 0.1] [--seed 42] [--output eval_sdpo_results.json]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

try:
    from LLMRouter.train_dpo import SYSTEM_PROMPT
except ModuleNotFoundError:
    from train_dpo import SYSTEM_PROMPT


# ── Text-similarity helpers ────────────────────────────────────────────────────

def bigrams(text: str) -> set:
    tokens = re.findall(r"\w+", text.lower())
    return set(zip(tokens, tokens[1:])) if len(tokens) > 1 else set()


def jaccard(a: str, b: str) -> float:
    ba, bb = bigrams(a), bigrams(b)
    if not ba and not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


# ── SDPO-specific metrics ──────────────────────────────────────────────────────

STEP_RE = re.compile(r"Step\s+\d+\s*\(", re.IGNORECASE)

CONTAINMENT_KW = {
    "verify", "verif", "correct", "correcting", "correction", "fix", "fixing",
    "catch", "catching", "detect", "error caught", "mistake", "wrong answer",
    "re-check", "recheck", "review", "confirms", "accurate", "confirmed",
    "right answer", "actual answer", "should be",
}

PROPAGATION_KW = {
    "passes", "passing", "forwarding", "forward", "unchanged",
    "submit", "submitting", "final answer", "finaliz", "output the answer",
    "accepts", "accepted", "uses the result", "sends the result",
    "does not check", "without checking", "without verification",
}


def containment_score(text: str) -> float:
    """
    +1 for each containment keyword hit, -1 for each propagation keyword hit.
    Normalised to [-1, +1]. +1 = pure containment, -1 = pure propagation.
    """
    t = text.lower()
    pos = sum(1 for kw in CONTAINMENT_KW if kw in t)
    neg = sum(1 for kw in PROPAGATION_KW if kw in t)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def format_valid(text: str) -> bool:
    return bool(STEP_RE.search(text or ""))


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


# ── Accuracy bar ──────────────────────────────────────────────────────────────

def acc_bar(correct: int, total: int, width: int = 28) -> str:
    if total == 0:
        return "N/A"
    frac = correct / total
    filled = int(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar}  {correct}/{total}  ({100*frac:.1f}%)"


def float_bar(value: float, width: int = 28) -> str:
    """Bar for a [-1,+1] value, centred at 0."""
    frac = (value + 1) / 2          # map [-1,1] → [0,1]
    filled = int(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar}  {value:+.3f}"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",    required=True)
    parser.add_argument("--data",       required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--val-ratio",  type=float, default=0.1)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--output",     default="eval_sdpo_results.json")
    parser.add_argument("--max-samples", type=int,  default=0)
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading val split from {args.data} ...")
    val_pairs = load_val_split(args.data, args.val_ratio, args.seed)
    if args.max_samples > 0:
        val_pairs = val_pairs[:args.max_samples]
    print(f"Val set size: {len(val_pairs)}")
    outcome_dist = Counter(p.get("meta", {}).get("outcome", "?") for p in val_pairs)
    print(f"Outcome distribution: {dict(outcome_dist)}")

    # ── Load router ──────────────────────────────────────────────────────────
    try:
        from LLMRouter.router_inference import LLMRouter
    except ModuleNotFoundError:
        from router_inference import LLMRouter
    print(f"\nLoading LLMRouter from {args.adapter} ...")
    router = LLMRouter(adapter_path=args.adapter, base_model=args.base_model)

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"\nRunning inference on {len(val_pairs)} samples ...")
    results = []
    for i, pair in enumerate(val_pairs):
        outcome   = pair.get("meta", {}).get("outcome", "?")
        coarse    = pair.get("meta", {}).get("coarse_type", "?")
        chosen    = pair["chosen"]
        rejected  = pair["rejected"]

        raw = run_inference(pair, router)

        fv  = format_valid(raw)
        cs  = containment_score(raw)
        sim_chosen   = jaccard(raw, chosen)
        sim_rejected = jaccard(raw, rejected)
        prefer_chosen = sim_chosen > sim_rejected

        status = "✓" if prefer_chosen else "✗"
        print(f"  [{i+1:>3}/{len(val_pairs)}] {status}  "
              f"outcome={outcome}  coarse={coarse}  "
              f"fmt={'ok' if fv else 'BAD'}  "
              f"containment={cs:+.2f}  "
              f"sim(c={sim_chosen:.2f}, r={sim_rejected:.2f})")

        results.append({
            "idx":             i,
            "outcome":         outcome,
            "coarse_type":     coarse,
            "format_valid":    fv,
            "containment_score": round(cs, 4),
            "sim_chosen":      round(sim_chosen,   4),
            "sim_rejected":    round(sim_rejected, 4),
            "prefer_chosen":   prefer_chosen,
            "raw_output":      raw[:400],
        })

    # ── Aggregate ────────────────────────────────────────────────────────────
    n = len(results)
    n_fmt    = sum(r["format_valid"]  for r in results)
    n_prefer = sum(r["prefer_chosen"] for r in results)
    mean_cs  = sum(r["containment_score"] for r in results) / n

    # Per-outcome breakdown
    by_outcome: dict = defaultdict(lambda: {"prefer": 0, "total": 0, "cs_sum": 0.0})
    for r in results:
        oc = r["outcome"]
        by_outcome[oc]["total"]  += 1
        by_outcome[oc]["cs_sum"] += r["containment_score"]
        if r["prefer_chosen"]:
            by_outcome[oc]["prefer"] += 1

    # ── Report ───────────────────────────────────────────────────────────────
    SEP = "─" * 60
    print(f"\n{'═'*60}")
    print(f"  SDPO OFFLINE EVAL — {args.adapter}")
    print(f"{'═'*60}")
    print(f"  Val samples    : {n}")
    print()
    print(f"  Format valid   : {acc_bar(n_fmt, n)}")
    print(f"  Prefer chosen  : {acc_bar(n_prefer, n)}")
    print(f"  Containment    : {float_bar(mean_cs)}  (mean; +1=recovery, -1=propagation)")
    print()
    print(f"{SEP}")
    print("  Per-outcome breakdown:")
    for oc, d in sorted(by_outcome.items()):
        mean_oc = d["cs_sum"] / d["total"]
        print(f"    {oc:12s}: prefer_chosen={acc_bar(d['prefer'], d['total'])}  "
              f"containment={mean_oc:+.3f}")

    print(f"\n{SEP}")
    print("  Sample outputs (first 3):")
    for r in results[:3]:
        print(f"\n  ── sample {r['idx']} ({r['outcome']}, {r['coarse_type']}) ──")
        print(f"  {r['raw_output'][:300]}")

    # ── Save ─────────────────────────────────────────────────────────────────
    summary = {
        "adapter":           args.adapter,
        "n_val_samples":     n,
        "format_valid_rate": round(n_fmt    / n, 4),
        "prefer_chosen_rate": round(n_prefer / n, 4),
        "mean_containment_score": round(mean_cs, 4),
        "per_outcome": {
            oc: {
                "prefer_chosen_rate": round(d["prefer"] / d["total"], 4),
                "mean_containment":   round(d["cs_sum"] / d["total"], 4),
            }
            for oc, d in by_outcome.items()
        },
        "per_sample": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  Full results → {out_path}")
    print(f"{'═'*60}\n")

    # ── Diagnostics ──────────────────────────────────────────────────────────
    print("  DIAGNOSTIC NOTES:")
    if n_fmt / n < 0.7:
        print("  ⚠  <70% format valid — model output is not following Step N (Role): format.")
        print("     Consider adding few-shot step examples to the system prompt.")
    if n_prefer / n < 0.55:
        print("  ⚠  prefer_chosen < 55% — model output is closer to rejected than chosen.")
        print("     More epochs or a higher beta may help.")
    if mean_cs < 0.0:
        print("  ⚠  Negative mean containment score — outputs lean toward propagation language.")
    if n_prefer / n >= 0.65 and mean_cs > 0.1 and n_fmt / n >= 0.8:
        print("  ✓  Routing looks healthy — proceed to end-to-end eval (Milestone 2).")


if __name__ == "__main__":
    main()
