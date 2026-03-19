#!/usr/bin/env python3
"""
Unified metrics for all experiments.

For each experiment, computes (test phase only):
  a) Test accuracy
  b) Total test errors
  c) Samples with errors
  d) Error propagation edges
  e) Total token cost (dollar)

Handles crash/recovery by merging multiple JSONL files per experiment.
For γ=0.2 (gpt-4o-mini): train from 00-50-44, test from 23-48-37
For gpt-5-mini:           complete in 03-07-12-15-40
"""

import json, os
from collections import defaultdict
from pathlib import Path

LOG_DIR = Path(__file__).parent

# ── Experiment definitions ─────────────────────────────────────────────────
# Each experiment maps to a list of JSONL files to merge.
# Order matters: later files can override earlier ones if there's overlap.
EXPERIMENTS = {
    "γ=0 (baseline)": {
        "files": ["gsm8k_error_responses_2026-02-25-11-16-41.jsonl"],
        "gamma": 0.0,
        "eval_llm": "gpt-4o-mini",
    },
    "γ=0.1": {
        "files": ["gsm8k_error_responses_2026-02-24-15-45-52.jsonl"],
        "gamma": 0.1,
        "eval_llm": "gpt-4o-mini",
    },
    "γ=0.2": {
        "files": [
            "gsm8k_error_responses_2026-03-03-00-50-44.jsonl",   # train + partial test
            "gsm8k_error_responses_2026-03-03-23-48-37.jsonl",   # full test (recovery)
        ],
        "gamma": 0.2,
        "eval_llm": "gpt-4o-mini",
    },
    "γ=0.4": {
        "files": ["gsm8k_error_responses_2026-03-04-12-23-35.jsonl"],
        "gamma": 0.4,
        "eval_llm": "gpt-4o-mini",
    },
    "γ=0.8": {
        "files": ["gsm8k_error_responses_2026-03-05-12-52-50.jsonl"],
        "gamma": 0.8,
        "eval_llm": "gpt-4o-mini",
    },
    "γ=1.6": {
        "files": ["gsm8k_error_responses_2026-03-06-12-21-42.jsonl"],
        "gamma": 1.6,
        "eval_llm": "gpt-4o-mini",
    },
    "γ=0.2 (gpt-5-mini)": {
        "files": ["gsm8k_error_responses_2026-03-07-12-15-40.jsonl"],
        "gamma": 0.2,
        "eval_llm": "gpt-5-mini",
    },
}


def load_test_records(file_list):
    """Load and merge test records from multiple JSONL files.
    
    For crash/recovery, the later file's test records replace earlier ones.
    We use the complete test set from the last file that has test records.
    """
    test_records = []
    train_records = []
    
    for fname in file_list:
        fpath = LOG_DIR / fname
        if not fpath.exists():
            print(f"  WARNING: {fname} not found, skipping")
            continue
        
        file_test = []
        file_train = []
        with open(fpath) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("phase") == "test":
                    file_test.append(rec)
                elif rec.get("phase") == "train":
                    file_train.append(rec)
        
        # For test: if this file has a complete test set (1056), use it
        # Otherwise keep partial as fallback
        if file_test:
            if len(file_test) >= len(test_records):
                test_records = file_test
        
        train_records.extend(file_train)
    
    return test_records, train_records


def compute_metrics(test_records):
    """Compute all requested metrics from test records."""
    n_total = len(test_records)
    if n_total == 0:
        return None
    
    n_solved = sum(1 for r in test_records if r.get("is_solved"))
    
    total_errors = 0
    n_with_errors = 0
    total_prop_edges = 0
    n_with_propagation = 0
    total_cost = 0.0
    
    for rec in test_records:
        errors = rec.get("errors", []) or []
        prop = rec.get("propagation", {}) or {}
        trajectory = prop.get("trajectory", []) or []
        cost = rec.get("cost", 0.0)
        
        total_errors += len(errors)
        if errors:
            n_with_errors += 1
        
        total_prop_edges += len(trajectory)
        if trajectory:
            n_with_propagation += 1
        
        total_cost += cost
    
    return {
        "n_total": n_total,
        "n_solved": n_solved,
        "accuracy": 100.0 * n_solved / n_total,
        "total_errors": total_errors,
        "n_with_errors": n_with_errors,
        "pct_with_errors": 100.0 * n_with_errors / n_total,
        "total_prop_edges": total_prop_edges,
        "n_with_propagation": n_with_propagation,
        "total_cost": total_cost,
        "avg_cost": total_cost / n_total,
    }


def main():
    os.chdir(LOG_DIR)
    
    all_metrics = {}
    
    for label, config in EXPERIMENTS.items():
        test_recs, train_recs = load_test_records(config["files"])
        metrics = compute_metrics(test_recs)
        if metrics:
            all_metrics[label] = metrics
        else:
            print(f"  WARNING: No test records for {label}")
    
    # ── Detailed per-experiment output ──────────────────────────────────
    for label, config in EXPERIMENTS.items():
        if label not in all_metrics:
            continue
        m = all_metrics[label]
        print(f"\n{'─'*60}")
        print(f"  {label}  (γ={config['gamma']}, judge={config['eval_llm']})")
        print(f"{'─'*60}")
        print(f"  a) Test accuracy:        {m['n_solved']:>5} / {m['n_total']}  = {m['accuracy']:.2f}%")
        print(f"  b) Total test errors:    {m['total_errors']:>5}")
        print(f"  c) Samples with errors:  {m['n_with_errors']:>5}  ({m['pct_with_errors']:.1f}%)")
        print(f"  d) Propagation edges:    {m['total_prop_edges']:>5}  (in {m['n_with_propagation']} samples)")
        print(f"  e) Total token cost:     ${m['total_cost']:.4f}  (avg ${m['avg_cost']:.6f}/query)")
    
    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n\n{'='*110}")
    print(f"  SUMMARY TABLE (all test phase)")
    print(f"{'='*110}")
    
    header = (f"  {'Experiment':<24} {'Judge':<12} {'Accuracy':>10} {'Errors':>8} "
              f"{'Samples w/ err':>15} {'Prop edges':>11} {'Cost ($)':>10} {'Avg $/q':>10}")
    print(header)
    print(f"  {'-'*108}")
    
    for label in EXPERIMENTS:
        if label not in all_metrics:
            continue
        m = all_metrics[label]
        config = EXPERIMENTS[label]
        judge = config["eval_llm"]
        print(f"  {label:<24} {judge:<12} "
              f"{m['accuracy']:>9.2f}% {m['total_errors']:>8} "
              f"{m['n_with_errors']:>8} ({m['pct_with_errors']:>4.1f}%) "
              f"{m['total_prop_edges']:>11} "
              f"{m['total_cost']:>10.4f} {m['avg_cost']:>10.6f}")
    
    # ── Relative to baseline ────────────────────────────────────────────
    if "γ=0 (baseline)" in all_metrics:
        base = all_metrics["γ=0 (baseline)"]
        print(f"\n  {'─'*108}")
        print(f"  {'Experiment':<24} {'ΔAccuracy':>10} {'ΔErrors':>9} {'ΔErrors%':>9} "
              f"{'ΔSamples w/e':>13} {'ΔProp edges':>12} {'ΔCost%':>8}")
        print(f"  {'─'*108}")
        
        for label in EXPERIMENTS:
            if label not in all_metrics or label == "γ=0 (baseline)":
                continue
            m = all_metrics[label]
            da = m['accuracy'] - base['accuracy']
            de = m['total_errors'] - base['total_errors']
            de_pct = 100.0 * de / base['total_errors'] if base['total_errors'] else 0
            ds = m['n_with_errors'] - base['n_with_errors']
            dp = m['total_prop_edges'] - base['total_prop_edges']
            dc_pct = 100.0 * (m['total_cost'] - base['total_cost']) / base['total_cost'] if base['total_cost'] else 0
            
            print(f"  {label:<24} {da:>+9.2f}pp {de:>+9d} {de_pct:>+8.1f}% "
                  f"{ds:>+13d} {dp:>+12d} {dc_pct:>+7.1f}%")


if __name__ == "__main__":
    main()
