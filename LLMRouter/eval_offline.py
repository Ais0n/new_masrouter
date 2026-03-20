#!/usr/bin/env python3
"""
Offline Routing Accuracy Evaluation for LLMRouter.

Holds out the same 10 % validation split used during DPO training (seed=42),
runs LLMRouter on every prompt (blind — no access to chosen/rejected),
then compares the predicted routing to the oracle (the 'chosen' label).

Metrics reported
────────────────
  • Mode accuracy          – predicted collaboration_mode == oracle mode
  • n_agents accuracy      – predicted num_agents == oracle num_agents
  • Joint accuracy         – both mode and n_agents correct
  • Per-coarse accuracy    – breakdown by coarse_type
  • Confusion matrix       – oracle mode (row) × predicted mode (col)
  • Mode distribution      – oracle vs predicted frequency per mode
  • Repair rate            – fraction of outputs that needed post-hoc repair

Usage (run from new_masrouter/):
  python LLMRouter/eval_offline.py \\
    --adapter  checkpoints/llm_router_dpo/final \\
    --data     ../MAST-Data/output/dpo_pairs/dpo_pairs_combined.jsonl \\
    [--val-ratio 0.1] [--seed 42] [--output eval_offline_results.json]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

# ── import router ────────────────────────────────────────────────────────────
try:
    from LLMRouter.router_inference import LLMRouter, parse_routing_decision
except ModuleNotFoundError:
    from router_inference import LLMRouter, parse_routing_decision


# ── helpers ──────────────────────────────────────────────────────────────────

ALL_MODES = ["IO", "CoT", "Chain", "FullConnected", "Debate", "Reflection"]


def load_val_split(data_path: str, val_ratio: float, seed: int) -> List[dict]:
    """Reproduce the exact val split used by train_dpo.py."""
    random.seed(seed)
    pairs = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    return pairs[:n_val]


def extract_prompt_text(pair: dict) -> str:
    """Return the raw prompt string from a DPO pair."""
    return pair["prompt"]


def oracle_mode(pair: dict) -> str:
    return json.loads(pair["chosen"])["collaboration_mode"]


def oracle_n(pair: dict) -> int:
    return json.loads(pair["chosen"])["num_agents"]


def confusion_matrix_str(matrix: dict, modes: List[str]) -> str:
    """Pretty-print confusion matrix (oracle rows × predicted cols)."""
    active = [m for m in modes if any(matrix[m][p] for p in modes)]
    if not active:
        return "(empty)"
    col_w = max(len(m) for m in active) + 2
    header = " " * col_w + "".join(f"{m:>{col_w}}" for m in active) + "  ← PREDICTED"
    rows = [header, "-" * len(header)]
    for oracle_m in active:
        row = f"{oracle_m:<{col_w}}" + "".join(
            f"{matrix[oracle_m][p]:>{col_w}}" for p in active
        )
        rows.append(row)
    rows.append("↑ ORACLE")
    return "\n".join(rows)


def accuracy_bar(correct: int, total: int, width: int = 30) -> str:
    if total == 0:
        return "N/A"
    frac = correct / total
    filled = int(frac * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar}  {correct}/{total}  ({100*frac:.1f}%)"


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",   required=True,
                        help="Path to saved LoRA adapter dir.")
    parser.add_argument("--data",      required=True,
                        help="Path to dpo_pairs_combined.jsonl (full dataset).")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--output",    default="eval_offline_results.json",
                        help="Where to write the JSON results file.")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Truncate val set (0=all). For quick smoke-test.")
    args = parser.parse_args()

    # ── Load val split ───────────────────────────────────────────────────────
    print(f"Loading val split from {args.data} ...")
    val_pairs = load_val_split(args.data, args.val_ratio, args.seed)
    if args.max_samples > 0:
        val_pairs = val_pairs[:args.max_samples]
    print(f"Val set size: {len(val_pairs)}")
    print(f"Oracle mode distribution: "
          f"{dict(Counter(oracle_mode(p) for p in val_pairs))}")

    # ── Load router ──────────────────────────────────────────────────────────
    print(f"\nLoading LLMRouter from {args.adapter} ...")
    router = LLMRouter(adapter_path=args.adapter, base_model=args.base_model)

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"\nRunning inference on {len(val_pairs)} samples ...")
    results = []
    for i, pair in enumerate(val_pairs):
        prompt_text = extract_prompt_text(pair)
        oracle_m    = oracle_mode(pair)
        oracle_na   = oracle_n(pair)
        coarse      = pair["meta"].get("coarse_type", "?")
        source      = pair["meta"].get("source", "MAST")

        # Feed the raw prompt directly to the model (no cheating)
        # LLMRouter.route() builds its own messages, so we call the model
        # directly using the same tokenizer + generate pipeline.
        from transformers import AutoTokenizer
        try:
            from LLMRouter.train_dpo import SYSTEM_PROMPT
        except ModuleNotFoundError:
            from train_dpo import SYSTEM_PROMPT

        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": prompt_text},
        ]
        text = router.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        import torch
        inputs = router.tokenizer(text, return_tensors="pt",
                                  truncation=True, max_length=1024).to(router.model.device)
        with torch.inference_mode():
            outputs = router.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,         # greedy for determinism in eval
                pad_token_id=router.tokenizer.pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw = router.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Parse and repair
        raw_parsed = None
        try:
            raw_parsed = json.loads(raw.strip())
        except Exception:
            pass
        repaired = parse_routing_decision(raw)

        pred_mode = repaired["collaboration_mode"]
        pred_na   = repaired["num_agents"]
        was_repaired = (raw_parsed is None or
                        raw_parsed.get("collaboration_mode") != pred_mode or
                        raw_parsed.get("num_agents") != pred_na)

        mode_correct   = (pred_mode == oracle_m)
        n_correct      = (pred_na   == oracle_na)
        joint_correct  = mode_correct and n_correct

        results.append({
            "idx":           i,
            "coarse_type":   coarse,
            "source":        source,
            "oracle_mode":   oracle_m,
            "oracle_n":      oracle_na,
            "pred_mode":     pred_mode,
            "pred_n":        pred_na,
            "mode_correct":  mode_correct,
            "n_correct":     n_correct,
            "joint_correct": joint_correct,
            "was_repaired":  was_repaired,
            "raw_output":    raw[:300],
        })

        status = "✓" if mode_correct else "✗"
        print(f"  [{i+1:>3}/{len(val_pairs)}] {status}  "
              f"oracle={oracle_m}/{oracle_na}  pred={pred_mode}/{pred_na}  "
              f"coarse={coarse}  src={source}")

    # ── Aggregate metrics ────────────────────────────────────────────────────
    n = len(results)
    mode_acc  = sum(r["mode_correct"]  for r in results) / n
    n_acc     = sum(r["n_correct"]     for r in results) / n
    joint_acc = sum(r["joint_correct"] for r in results) / n
    repair_rate = sum(r["was_repaired"] for r in results) / n

    # Per-coarse accuracy
    by_coarse: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        ct = r["coarse_type"]
        by_coarse[ct]["total"] += 1
        if r["mode_correct"]:
            by_coarse[ct]["correct"] += 1

    # Confusion matrix
    confusion: dict = {m: Counter() for m in ALL_MODES}
    for r in results:
        confusion[r["oracle_mode"]][r["pred_mode"]] += 1

    # Mode distributions
    oracle_dist = Counter(r["oracle_mode"] for r in results)
    pred_dist   = Counter(r["pred_mode"]   for r in results)

    # ── Print report ─────────────────────────────────────────────────────────
    sep = "─" * 60
    print(f"\n{'═'*60}")
    print(f"  OFFLINE ROUTING ACCURACY REPORT")
    print(f"{'═'*60}")
    print(f"  Val samples : {n}")
    print(f"  Adapter     : {args.adapter}")
    print()
    print(f"  Mode accuracy   : {accuracy_bar(sum(r['mode_correct']  for r in results), n)}")
    print(f"  n_agents acc    : {accuracy_bar(sum(r['n_correct']     for r in results), n)}")
    print(f"  Joint accuracy  : {accuracy_bar(sum(r['joint_correct'] for r in results), n)}")
    print(f"  Repair rate     : {repair_rate*100:.1f}%  (outputs needing post-hoc fix)")

    print(f"\n{sep}")
    print("  Per-coarse-type mode accuracy:")
    for ct, d in sorted(by_coarse.items()):
        print(f"    {ct:25s}: {accuracy_bar(d['correct'], d['total'])}")

    print(f"\n{sep}")
    print("  Mode distribution  (oracle vs predicted):")
    all_active = sorted(set(list(oracle_dist.keys()) + list(pred_dist.keys())))
    print(f"  {'Mode':<15} {'Oracle':>8} {'Predicted':>10}")
    print(f"  {'-'*35}")
    for m in all_active:
        o = oracle_dist.get(m, 0)
        p = pred_dist.get(m, 0)
        flag = "  ← missing!" if o > 0 and p == 0 else ""
        print(f"  {m:<15} {o:>8} {p:>10}{flag}")

    print(f"\n{sep}")
    print("  Confusion matrix (oracle=rows, predicted=cols):")
    print()
    cm_str = confusion_matrix_str(confusion, ALL_MODES)
    for line in cm_str.split("\n"):
        print(f"    {line}")

    # ── Save ─────────────────────────────────────────────────────────────────
    summary = {
        "adapter":        args.adapter,
        "n_val_samples":  n,
        "mode_accuracy":  round(mode_acc,  4),
        "n_agents_acc":   round(n_acc,     4),
        "joint_accuracy": round(joint_acc, 4),
        "repair_rate":    round(repair_rate, 4),
        "per_coarse_accuracy": {
            ct: round(d["correct"] / d["total"], 4)
            for ct, d in by_coarse.items()
        },
        "oracle_mode_dist":    dict(oracle_dist),
        "predicted_mode_dist": dict(pred_dist),
        "confusion_matrix": {
            oracle_m: dict(pred_counts)
            for oracle_m, pred_counts in confusion.items()
            if any(pred_counts.values())
        },
        "per_sample": results,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  Full results → {out_path}")
    print(f"{'═'*60}\n")

    # ── Suggestions based on results ─────────────────────────────────────────
    print("  DIAGNOSTIC NOTES:")
    if mode_acc < 0.6:
        print("  ⚠  Mode accuracy < 60% — consider re-training with more epochs or "
              "adding chain-of-thought rationale to the prompt.")
    missing = [m for m in ["Chain","FullConnected","Debate","Reflection"]
               if pred_dist.get(m, 0) == 0]
    if missing:
        print(f"  ⚠  Model never predicts: {missing}. "
              "Check training data balance for these modes.")
    if repair_rate > 0.3:
        print("  ⚠  High repair rate (>30%) — model output format is unreliable. "
              "Consider adding few-shot JSON examples to the system prompt.")
    if mode_acc >= 0.75 and not missing:
        print("  ✓  Routing looks healthy — proceed to end-to-end eval (Milestone 2).")


if __name__ == "__main__":
    main()
