#!/usr/bin/env python3
"""Q1: LLM selection shift per epoch for both experiments.
   Q3: Detailed error type appearance table per epoch.
   Q4: Examples where ours/baseline diverge on correctness."""

import json
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

def llm_usage_per_epoch(records, label):
    """Q1: per-epoch LLM usage (# trace steps assigned to each LLM)."""
    print(f"\n{'='*80}")
    print(f"  Q1: LLM Usage Per Epoch — {label}")
    print(f"{'='*80}")
    
    # Collect all LLMs
    all_llms = set()
    for r in records:
        for step in r.get("trace", []):
            all_llms.add(step.get("llm_name", "unknown"))
    all_llms = sorted(all_llms)
    
    phases = ["train", "test"]
    for phase in phases:
        phase_recs = [r for r in records if r["phase"] == phase]
        epochs = sorted(set(r.get("epoch", 0) for r in phase_recs))
        
        print(f"\n  Phase: {phase}")
        header = f"  {'Epoch':>5}"
        for llm in all_llms:
            short = llm.split("/")[-1] if "/" in llm else llm
            header += f"  {short:>18}"
        header += f"  {'Total':>7}"
        print(header)
        print(f"  {'-'*len(header)}")
        
        for ep in epochs:
            ep_recs = [r for r in phase_recs if r.get("epoch", 0) == ep]
            llm_count = Counter()
            for r in ep_recs:
                for step in r.get("trace", []):
                    llm_count[step.get("llm_name", "unknown")] += 1
            row = f"  {ep:>5}"
            total = 0
            for llm in all_llms:
                c = llm_count.get(llm, 0)
                total += c
                row += f"  {c:>18}"
            row += f"  {total:>7}"
            print(row)
    
    # Also show per-epoch LLM error counts (train)
    print(f"\n  LLM Errors Per Epoch (train) — {label}")
    train_recs = [r for r in records if r["phase"] == "train"]
    epochs = sorted(set(r.get("epoch", 0) for r in train_recs))
    header = f"  {'Epoch':>5}"
    for llm in all_llms:
        short = llm.split("/")[-1] if "/" in llm else llm
        header += f"  {short:>18}"
    header += f"  {'Total':>7}"
    print(header)
    print(f"  {'-'*len(header)}")
    for ep in epochs:
        ep_recs = [r for r in train_recs if r.get("epoch", 0) == ep]
        llm_err = Counter()
        for r in ep_recs:
            for e in r.get("errors", []):
                llm_err[e.get("agent_llm", "unknown")] += 1
        row = f"  {ep:>5}"
        total = 0
        for llm in all_llms:
            c = llm_err.get(llm, 0)
            total += c
            row += f"  {c:>18}"
        row += f"  {total:>7}"
        print(row)


def error_type_per_epoch(records, label):
    """Q3: Detailed error type appearance table."""
    print(f"\n{'='*80}")
    print(f"  Q3: Error Type Appearances Per Epoch — {label}")
    print(f"{'='*80}")
    
    # Collect all error types
    all_types = set()
    for r in records:
        for e in r.get("errors", []):
            all_types.add(e["error_type"])
    all_types = sorted(all_types)
    
    for phase in ["train", "test"]:
        phase_recs = [r for r in records if r["phase"] == phase]
        epochs = sorted(set(r.get("epoch", 0) for r in phase_recs))
        
        print(f"\n  Phase: {phase}")
        header = f"  {'Error Type':45s}"
        for ep in epochs:
            header += f"  {'E'+str(ep):>5}"
        header += f"  {'Total':>7}"
        print(header)
        print(f"  {'-'*len(header)}")
        
        for et in all_types:
            row = f"  {et:45s}"
            total = 0
            for ep in epochs:
                ep_recs = [r for r in phase_recs if r.get("epoch", 0) == ep]
                c = sum(1 for r in ep_recs for e in r.get("errors", []) if e["error_type"] == et)
                total += c
                row += f"  {c:>5}"
            row += f"  {total:>7}"
            print(row)
        
        # Total row
        row = f"  {'TOTAL':45s}"
        grand = 0
        for ep in epochs:
            ep_recs = [r for r in phase_recs if r.get("epoch", 0) == ep]
            c = sum(len(r.get("errors", [])) for r in ep_recs)
            grand += c
            row += f"  {c:>5}"
        row += f"  {grand:>7}"
        print(row)


def find_divergent_examples(ours_records, base_records):
    """Q4: Find examples where one experiment solves but the other doesn't, 
    or both solve but error profiles differ significantly."""
    print(f"\n{'='*80}")
    print(f"  Q4: Divergent Examples (Accuracy Analysis)")
    print(f"{'='*80}")
    
    # Build index by (phase, epoch, global_sample_idx)
    def index_by_query(records):
        idx = {}
        for r in records:
            if r["phase"] == "test":
                key = r["global_sample_idx"]
                idx[key] = r
        return idx
    
    ours_idx = index_by_query(ours_records)
    base_idx = index_by_query(base_records)
    
    common_keys = set(ours_idx.keys()) & set(base_idx.keys())
    
    # Category 1: Ours solves, baseline doesn't
    ours_only = []
    for k in common_keys:
        if ours_idx[k]["is_solved"] and not base_idx[k]["is_solved"]:
            ours_only.append(k)
    
    # Category 2: Baseline solves, ours doesn't  
    base_only = []
    for k in common_keys:
        if not ours_idx[k]["is_solved"] and base_idx[k]["is_solved"]:
            base_only.append(k)
    
    # Category 3: Both fail
    both_fail = []
    for k in common_keys:
        if not ours_idx[k]["is_solved"] and not base_idx[k]["is_solved"]:
            both_fail.append(k)
    
    # Category 4: Both solve, but ours has fewer errors
    both_solve_ours_fewer = []
    for k in common_keys:
        if ours_idx[k]["is_solved"] and base_idx[k]["is_solved"]:
            oe = len(ours_idx[k].get("errors", []))
            be = len(base_idx[k].get("errors", []))
            if oe < be:
                both_solve_ours_fewer.append((k, oe, be))
    
    # Category 5: Both solve, but baseline has fewer errors
    both_solve_base_fewer = []
    for k in common_keys:
        if ours_idx[k]["is_solved"] and base_idx[k]["is_solved"]:
            oe = len(ours_idx[k].get("errors", []))
            be = len(base_idx[k].get("errors", []))
            if oe > be:
                both_solve_base_fewer.append((k, oe, be))
    
    print(f"\n  Divergence Statistics:")
    print(f"  Total test samples matched: {len(common_keys)}")
    print(f"  Ours solves, baseline doesn't: {len(ours_only)}")
    print(f"  Baseline solves, ours doesn't: {len(base_only)}")
    print(f"  Both fail: {len(both_fail)}")
    print(f"  Both solve, ours fewer errors: {len(both_solve_ours_fewer)}")
    print(f"  Both solve, base fewer errors: {len(both_solve_base_fewer)}")
    
    def print_example(r_ours, r_base, category):
        print(f"\n  --- {category} ---")
        q = r_ours.get("query", "")[:120]
        print(f"  Query: {q}...")
        print(f"  True answer: {r_ours['true_answer']}")
        print(f"  Ours:     solved={r_ours['is_solved']}, errors={len(r_ours.get('errors',[]))}, penalty={r_ours.get('error_penalty',0):.2f}")
        
        # Ours agent config
        ours_agents = [(s["role"], s["llm_name"].split("/")[-1]) for s in r_ours.get("trace", [])]
        print(f"  Ours agents: {ours_agents}")
        ours_etypes = [e["error_type"] for e in r_ours.get("errors", [])]
        if ours_etypes:
            print(f"  Ours errors: {ours_etypes}")
        ours_has_collab = any(s.get("spatial_predecessors", []) for s in r_ours.get("trace", []))
        print(f"  Ours collaborative: {ours_has_collab}")
        
        print(f"  Baseline: solved={r_base['is_solved']}, errors={len(r_base.get('errors',[]))}, penalty={r_base.get('error_penalty',0):.2f}")
        base_agents = [(s["role"], s["llm_name"].split("/")[-1]) for s in r_base.get("trace", [])]
        print(f"  Baseline agents: {base_agents}")
        base_etypes = [e["error_type"] for e in r_base.get("errors", [])]
        if base_etypes:
            print(f"  Baseline errors: {base_etypes}")
        base_has_collab = any(s.get("spatial_predecessors", []) for s in r_base.get("trace", []))
        print(f"  Baseline collaborative: {base_has_collab}")
    
    # Print examples
    print(f"\n  ============ Examples: Ours solves, baseline doesn't ============")
    for k in sorted(ours_only)[:3]:
        print_example(ours_idx[k], base_idx[k], f"Ours-only-solves (idx={k})")
    
    print(f"\n  ============ Examples: Baseline solves, ours doesn't ============")
    for k in sorted(base_only)[:3]:
        print_example(ours_idx[k], base_idx[k], f"Baseline-only-solves (idx={k})")
    
    print(f"\n  ============ Examples: Both fail ============")
    for k in sorted(both_fail)[:3]:
        print_example(ours_idx[k], base_idx[k], f"Both-fail (idx={k})")
    
    print(f"\n  ============ Examples: Both solve, ours fewer errors ============")
    both_solve_ours_fewer.sort(key=lambda x: x[2]-x[1], reverse=True)
    for k, oe, be in both_solve_ours_fewer[:3]:
        print_example(ours_idx[k], base_idx[k], f"Both-solve-ours-fewer (idx={k}, ours_err={oe}, base_err={be})")
    
    # Also: analyze the overlap - how many failures are shared?
    ours_fail_keys = set(k for k in common_keys if not ours_idx[k]["is_solved"])
    base_fail_keys = set(k for k in common_keys if not base_idx[k]["is_solved"])
    overlap = ours_fail_keys & base_fail_keys
    print(f"\n  ============ Failure Overlap Analysis ============")
    print(f"  Ours failures: {len(ours_fail_keys)}")
    print(f"  Baseline failures: {len(base_fail_keys)}")
    print(f"  Shared failures (both fail): {len(overlap)}")
    print(f"  Ours-only failures: {len(ours_fail_keys - base_fail_keys)}")
    print(f"  Baseline-only failures: {len(base_fail_keys - ours_fail_keys)}")
    print(f"  Jaccard similarity of failure sets: {len(overlap)/len(ours_fail_keys | base_fail_keys):.3f}")


if __name__ == "__main__":
    ours = load_jsonl(OURS_JSONL)
    base = load_jsonl(BASE_JSONL)
    
    llm_usage_per_epoch(ours, "OURS (γ=0.1)")
    llm_usage_per_epoch(base, "BASELINE (γ=0)")
    
    error_type_per_epoch(ours, "OURS (γ=0.1)")
    error_type_per_epoch(base, "BASELINE (γ=0)")
    
    find_divergent_examples(ours, base)
