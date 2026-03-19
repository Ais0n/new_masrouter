#!/usr/bin/env python3
"""
Compute token cost (USD) and token counts for both experiments.

The 'cost' field in the JSONL is the dollar cost computed by MasRouter's
price.py: sum of (prompt_tokens × input_price + completion_tokens × output_price)
for every LLM call during graph execution.

Time cost is NOT part of the reward function. The reward is:
    utility = is_solved − β·cost − γ·error_penalty
where cost is dollar cost only.

This script reports total and per-query cost for train and test phases.
"""

import json, os
from collections import defaultdict
from pathlib import Path

FILES = {
    "Ours (γ=0.1)":   "gsm8k_error_responses_2026-02-24-15-45-52.jsonl",
    "Baseline (γ=0)": "gsm8k_error_responses_2026-02-25-11-16-41.jsonl",
}

LOG_DIR = Path(__file__).parent

def analyze(label, filepath):
    records = {"train": [], "test": []}
    with open(filepath) as f:
        for line in f:
            rec = json.loads(line)
            phase = rec.get("phase", "unknown")
            if phase in records:
                records[phase].append(rec)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    for phase in ["train", "test"]:
        recs = records[phase]
        if not recs:
            continue
        costs = [r["cost"] for r in recs]
        total_cost = sum(costs)
        avg_cost = total_cost / len(costs) if costs else 0
        min_cost = min(costs) if costs else 0
        max_cost = max(costs) if costs else 0
        median_cost = sorted(costs)[len(costs)//2] if costs else 0

        # Per-epoch breakdown (train only)
        epoch_costs = defaultdict(list)
        for r in recs:
            epoch_costs[r.get("epoch", 0)].append(r["cost"])

        solved = sum(1 for r in recs if r.get("is_solved"))

        print(f"\n  ── {phase.upper()} ({len(recs)} samples) ──")
        print(f"    Total dollar cost:   ${total_cost:.4f}")
        print(f"    Avg cost/query:      ${avg_cost:.6f}")
        print(f"    Median cost/query:   ${median_cost:.6f}")
        print(f"    Min cost/query:      ${min_cost:.6f}")
        print(f"    Max cost/query:      ${max_cost:.6f}")
        print(f"    Solve rate:          {solved}/{len(recs)} = {100*solved/len(recs):.1f}%")

        # β·cost contribution to reward
        beta = 200.0
        avg_beta_cost = beta * avg_cost
        print(f"    β·avg_cost (β=200):  {avg_beta_cost:.4f}")

        if phase == "train" and len(epoch_costs) > 1:
            print(f"\n    Per-epoch breakdown:")
            print(f"    {'Epoch':<8} {'Samples':>8} {'Total $':>10} {'Avg $/query':>12}")
            print(f"    {'-'*42}")
            for ep in sorted(epoch_costs.keys()):
                ec = epoch_costs[ep]
                print(f"    {ep:<8} {len(ec):>8} {sum(ec):>10.4f} {sum(ec)/len(ec):>12.6f}")

    # Cost by LLM (test phase)
    print(f"\n  ── COST BREAKDOWN BY LLM (test, estimated from trace) ──")
    # We don't have per-LLM cost directly, but we can count agent steps per LLM
    llm_steps = defaultdict(int)
    for r in records["test"]:
        for step in r.get("trace", []):
            llm = step.get("llm_name", "unknown")
            llm_steps[llm] += 1
    total_steps = sum(llm_steps.values())
    print(f"    {'LLM':<45} {'Steps':>6} {'%':>7}")
    print(f"    {'-'*60}")
    for llm, cnt in sorted(llm_steps.items(), key=lambda x: -x[1]):
        print(f"    {llm:<45} {cnt:>6} {100*cnt/total_steps:>6.1f}%")


def compare(results_data):
    print(f"\n{'='*70}")
    print(f"  COST COMPARISON (test phase)")
    print(f"{'='*70}")

    rows = []
    for label, data in results_data.items():
        costs = [r["cost"] for r in data["test"]]
        total = sum(costs)
        avg = total / len(costs)
        rows.append((label, len(costs), total, avg))

    print(f"\n  {'Experiment':<22} {'Samples':>8} {'Total $':>10} {'Avg $/query':>12} {'β·avg (β=200)':>14}")
    print(f"  {'-'*70}")
    for label, n, total, avg in rows:
        print(f"  {label:<22} {n:>8} {total:>10.4f} {avg:>12.6f} {200*avg:>14.4f}")

    if len(rows) == 2:
        delta_avg = rows[0][3] - rows[1][3]
        pct = 100 * delta_avg / rows[1][3] if rows[1][3] != 0 else 0
        print(f"\n  Δ avg $/query (Ours − Baseline): ${delta_avg:.6f} ({pct:+.1f}%)")
        print(f"  Δ β·cost contribution:           {200*delta_avg:.4f}")

    # Train costs
    print(f"\n  {'Experiment':<22} {'Train total $':>14} {'Train avg $/q':>14} {'Test total $':>12} {'Combined $':>12}")
    print(f"  {'-'*78}")
    for label, data in results_data.items():
        train_costs = [r["cost"] for r in data["train"]]
        test_costs = [r["cost"] for r in data["test"]]
        tt = sum(train_costs)
        ta = tt / len(train_costs) if train_costs else 0
        te = sum(test_costs)
        print(f"  {label:<22} {tt:>14.4f} {ta:>14.6f} {te:>12.4f} {tt+te:>12.4f}")


if __name__ == "__main__":
    os.chdir(LOG_DIR)
    all_data = {}
    for label, fname in FILES.items():
        records = {"train": [], "test": []}
        with open(fname) as f:
            for line in f:
                rec = json.loads(line)
                phase = rec.get("phase", "unknown")
                if phase in records:
                    records[phase].append(rec)
        all_data[label] = records
        analyze(label, fname)

    compare(all_data)
