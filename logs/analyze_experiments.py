#!/usr/bin/env python3
"""Comprehensive analysis of error-augmented MasRouter experiments."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

OURS_JSONL = Path(__file__).parent / "gsm8k_error_responses_2026-02-24-15-45-52.jsonl"
BASE_JSONL = Path(__file__).parent / "gsm8k_error_responses_2026-02-25-11-16-41.jsonl"


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze(records, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    # Split by phase
    train = [r for r in records if r["phase"] == "train"]
    test = [r for r in records if r["phase"] == "test"]
    print(f"Total records: {len(records)} (train={len(train)}, test={len(test)})")

    # --- Per-epoch training stats ---
    epochs = sorted(set(r["epoch"] for r in train))
    print(f"\n--- Training Per-Epoch Summary ---")
    print(f"{'Epoch':>5} {'Samples':>8} {'Solved':>7} {'Acc':>8} {'Errors':>7} {'ErrSamples':>11} {'AvgPenalty':>11} {'AvgCost':>10}")
    for ep in epochs:
        ep_records = [r for r in train if r["epoch"] == ep]
        n = len(ep_records)
        solved = sum(1 for r in ep_records if r["is_solved"])
        total_errors = sum(len(r.get("errors", [])) for r in ep_records)
        err_samples = sum(1 for r in ep_records if len(r.get("errors", [])) > 0)
        avg_penalty = sum(r.get("error_penalty", 0) for r in ep_records) / max(n, 1)
        avg_cost = sum(r.get("cost", 0) for r in ep_records) / max(n, 1)
        print(f"{ep:>5} {n:>8} {solved:>7} {solved/n:>8.4f} {total_errors:>7} {err_samples:>11} {avg_penalty:>11.4f} {avg_cost:>10.6f}")

    # --- Test stats ---
    print(f"\n--- Test Summary ---")
    n_test = len(test)
    solved_test = sum(1 for r in test if r["is_solved"])
    total_test_errors = sum(len(r.get("errors", [])) for r in test)
    test_err_samples = sum(1 for r in test if len(r.get("errors", [])) > 0)
    avg_test_penalty = sum(r.get("error_penalty", 0) for r in test) / max(n_test, 1)
    avg_test_cost = sum(r.get("cost", 0) for r in test) / max(n_test, 1)
    print(f"Samples: {n_test}")
    print(f"Solved:  {solved_test} ({solved_test/n_test:.4f})")
    print(f"Total errors: {total_test_errors}")
    print(f"Samples with errors: {test_err_samples} ({test_err_samples/n_test:.4f})")
    print(f"Avg error penalty: {avg_test_penalty:.4f}")
    print(f"Avg cost: {avg_test_cost:.6f}")

    # --- Error type distribution (test) ---
    print(f"\n--- Test Error Type Distribution ---")
    error_type_counter = Counter()
    for r in test:
        for e in r.get("errors", []):
            error_type_counter[e["error_type"]] += 1
    for etype, count in error_type_counter.most_common():
        print(f"  {etype:45s} {count:>5}")
    print(f"  {'TOTAL':45s} {sum(error_type_counter.values()):>5}")

    # --- Error type distribution (all train) ---
    print(f"\n--- Train Error Type Distribution (all epochs) ---")
    error_type_counter_train = Counter()
    for r in train:
        for e in r.get("errors", []):
            error_type_counter_train[e["error_type"]] += 1
    for etype, count in error_type_counter_train.most_common():
        print(f"  {etype:45s} {count:>5}")
    print(f"  {'TOTAL':45s} {sum(error_type_counter_train.values()):>5}")

    # --- Per-agent-role error distribution (test) ---
    print(f"\n--- Test Errors by Agent Role ---")
    role_counter = Counter()
    for r in test:
        for e in r.get("errors", []):
            role_counter[e.get("agent_role", "unknown")] += 1
    for role, count in role_counter.most_common():
        print(f"  {role:35s} {count:>5}")

    # --- Per-LLM error distribution (test) ---
    print(f"\n--- Test Errors by LLM ---")
    llm_counter = Counter()
    for r in test:
        for e in r.get("errors", []):
            llm_counter[e.get("agent_llm", "unknown")] += 1
    for llm, count in llm_counter.most_common():
        print(f"  {llm:45s} {count:>5}")

    # --- Propagation analysis (test) ---
    print(f"\n--- Test Error Propagation ---")
    samples_with_prop = sum(1 for r in test if r.get("propagation", {}).get("trajectory", []))
    total_prop_edges = sum(len(r.get("propagation", {}).get("trajectory", [])) for r in test)
    print(f"Samples with propagation: {samples_with_prop}")
    print(f"Total propagation edges: {total_prop_edges}")

    # --- Trace length distribution (test) ---
    print(f"\n--- Test Trace Length Distribution ---")
    trace_lens = [len(r.get("trace", [])) for r in test]
    trace_counter = Counter(trace_lens)
    for tl in sorted(trace_counter):
        print(f"  Length {tl}: {trace_counter[tl]:>5} samples")
    avg_trace = sum(trace_lens) / max(len(trace_lens), 1)
    print(f"  Average trace length: {avg_trace:.2f}")

    # --- Number of agents (unique roles per sample, test) ---
    print(f"\n--- Test Num-Agents Distribution ---")
    num_agents = []
    for r in test:
        roles = set()
        for step in r.get("trace", []):
            roles.add(step.get("role", ""))
        num_agents.append(len(roles))
    agent_counter = Counter(num_agents)
    for na in sorted(agent_counter):
        print(f"  {na} agents: {agent_counter[na]:>5} samples")

    # --- Collaboration topology (test) ---
    print(f"\n--- Test Collaboration Topology ---")
    # Check whether samples have spatial predecessors (collaborative) or not
    collab_count = 0
    independent_count = 0
    for r in test:
        has_spatial = False
        for step in r.get("trace", []):
            if step.get("spatial_predecessors", []):
                has_spatial = True
                break
        if has_spatial:
            collab_count += 1
        else:
            independent_count += 1
    print(f"  Collaborative (has spatial edges): {collab_count}")
    print(f"  Independent (no spatial edges):    {independent_count}")

    # --- Cost distribution (test) ---
    costs = [r.get("cost", 0) for r in test]
    print(f"\n--- Test Cost Distribution ---")
    print(f"  Min:  {min(costs):.6f}")
    print(f"  Max:  {max(costs):.6f}")
    print(f"  Mean: {sum(costs)/len(costs):.6f}")
    sorted_costs = sorted(costs)
    print(f"  P50:  {sorted_costs[len(sorted_costs)//2]:.6f}")
    print(f"  P90:  {sorted_costs[int(len(sorted_costs)*0.9)]:.6f}")

    # --- Penalty distribution (test, among samples with errors) ---
    penalties = [r.get("error_penalty", 0) for r in test if r.get("error_penalty", 0) > 0]
    if penalties:
        print(f"\n--- Test Penalty Distribution (among erroneous samples) ---")
        print(f"  N:    {len(penalties)}")
        print(f"  Min:  {min(penalties):.4f}")
        print(f"  Max:  {max(penalties):.4f}")
        print(f"  Mean: {sum(penalties)/len(penalties):.4f}")
        sorted_p = sorted(penalties)
        print(f"  P50:  {sorted_p[len(sorted_p)//2]:.4f}")
        print(f"  P90:  {sorted_p[int(len(sorted_p)*0.9)]:.4f}")

    # --- Solved but has errors vs not solved ---
    print(f"\n--- Test: Solved vs Error Presence ---")
    solved_w_errors = sum(1 for r in test if r["is_solved"] and len(r.get("errors", [])) > 0)
    solved_no_errors = sum(1 for r in test if r["is_solved"] and len(r.get("errors", [])) == 0)
    unsolved_w_errors = sum(1 for r in test if not r["is_solved"] and len(r.get("errors", [])) > 0)
    unsolved_no_errors = sum(1 for r in test if not r["is_solved"] and len(r.get("errors", [])) == 0)
    print(f"  Solved   + Errors:    {solved_w_errors}")
    print(f"  Solved   + No Errors: {solved_no_errors}")
    print(f"  Unsolved + Errors:    {unsolved_w_errors}")
    print(f"  Unsolved + No Errors: {unsolved_no_errors}")

    # --- LLM usage distribution (test) ---
    print(f"\n--- Test LLM Usage (across all trace steps) ---")
    llm_usage = Counter()
    for r in test:
        for step in r.get("trace", []):
            llm_usage[step.get("llm_name", "unknown")] += 1
    for llm, count in llm_usage.most_common():
        print(f"  {llm:45s} {count:>5}")

    # --- Role usage distribution (test) ---
    print(f"\n--- Test Role Usage (across all trace steps) ---")
    role_usage = Counter()
    for r in test:
        for step in r.get("trace", []):
            role_usage[step.get("role", "unknown")] += 1
    for role, count in role_usage.most_common():
        print(f"  {role:35s} {count:>5}")

    return {
        "n_test": n_test,
        "solved_test": solved_test,
        "test_acc": solved_test / n_test,
        "total_test_errors": total_test_errors,
        "test_err_samples": test_err_samples,
        "avg_test_penalty": avg_test_penalty,
        "avg_test_cost": avg_test_cost,
        "error_types": error_type_counter,
        "llm_errors": llm_counter,
        "role_errors": role_counter,
    }


def compare(ours_stats, base_stats):
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Ours (γ=0.1) vs Baseline (γ=0)")
    print(f"{'='*70}")

    print(f"\n{'Metric':35s} {'Ours':>12} {'Baseline':>12} {'Δ':>12} {'Δ%':>10}")
    print("-" * 85)

    metrics = [
        ("Test Accuracy", "test_acc", "{:.4f}"),
        ("Test Solved Count", "solved_test", "{:d}"),
        ("Total Test Errors", "total_test_errors", "{:d}"),
        ("Samples with Errors", "test_err_samples", "{:d}"),
        ("Avg Error Penalty", "avg_test_penalty", "{:.4f}"),
        ("Avg Cost", "avg_test_cost", "{:.6f}"),
    ]

    for name, key, fmt in metrics:
        ov = ours_stats[key]
        bv = base_stats[key]
        delta = ov - bv
        pct = (delta / bv * 100) if bv != 0 else float("nan")
        ov_s = fmt.format(ov)
        bv_s = fmt.format(bv)
        d_s = fmt.format(delta) if isinstance(delta, float) else f"{delta:+d}"
        print(f"  {name:33s} {ov_s:>12} {bv_s:>12} {d_s:>12} {pct:>+9.1f}%")

    # --- Error type comparison ---
    print(f"\n--- Error Type Comparison (Test) ---")
    all_types = sorted(set(list(ours_stats["error_types"].keys()) + list(base_stats["error_types"].keys())))
    print(f"  {'Error Type':45s} {'Ours':>6} {'Base':>6} {'Δ':>6} {'Δ%':>8}")
    print(f"  {'-'*73}")
    for et in all_types:
        ov = ours_stats["error_types"].get(et, 0)
        bv = base_stats["error_types"].get(et, 0)
        delta = ov - bv
        pct = (delta / bv * 100) if bv != 0 else float("nan")
        print(f"  {et:45s} {ov:>6} {bv:>6} {delta:>+6} {pct:>+7.1f}%")

    # --- LLM error comparison ---
    print(f"\n--- LLM Error Comparison (Test) ---")
    all_llms = sorted(set(list(ours_stats["llm_errors"].keys()) + list(base_stats["llm_errors"].keys())))
    print(f"  {'LLM':45s} {'Ours':>6} {'Base':>6} {'Δ':>6} {'Δ%':>8}")
    print(f"  {'-'*73}")
    for llm in all_llms:
        ov = ours_stats["llm_errors"].get(llm, 0)
        bv = base_stats["llm_errors"].get(llm, 0)
        delta = ov - bv
        pct = (delta / bv * 100) if bv != 0 else float("nan")
        print(f"  {llm:45s} {ov:>6} {bv:>6} {delta:>+6} {pct:>+7.1f}%")

    # --- Role error comparison ---
    print(f"\n--- Role Error Comparison (Test) ---")
    all_roles = sorted(set(list(ours_stats["role_errors"].keys()) + list(base_stats["role_errors"].keys())))
    print(f"  {'Role':35s} {'Ours':>6} {'Base':>6} {'Δ':>6} {'Δ%':>8}")
    print(f"  {'-'*63}")
    for role in all_roles:
        ov = ours_stats["role_errors"].get(role, 0)
        bv = base_stats["role_errors"].get(role, 0)
        delta = ov - bv
        pct = (delta / bv * 100) if bv != 0 else float("nan")
        print(f"  {role:35s} {ov:>6} {bv:>6} {delta:>+6} {pct:>+7.1f}%")


if __name__ == "__main__":
    print("Loading JSONL files...")
    ours_records = load_jsonl(OURS_JSONL)
    base_records = load_jsonl(BASE_JSONL)
    print(f"Ours: {len(ours_records)} records")
    print(f"Baseline: {len(base_records)} records")

    ours_stats = analyze(ours_records, "OURS (γ=0.1, error-augmented)")
    base_stats = analyze(base_records, "BASELINE (γ=0, no error penalty)")
    compare(ours_stats, base_stats)
