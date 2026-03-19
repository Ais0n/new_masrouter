#!/usr/bin/env python3
"""Re-do the divergent example analysis by matching on QUERY TEXT, not global_sample_idx."""
import json

OURS_JSONL = "/Users/yanwei/work/nips2026/masrouter/logs/gsm8k_error_responses_2026-02-24-15-45-52.jsonl"
BASE_JSONL = "/Users/yanwei/work/nips2026/masrouter/logs/gsm8k_error_responses_2026-02-25-11-16-41.jsonl"

def load_jsonl(path):
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip():
                r = json.loads(line.strip())
                r["_line_number"] = i + 1
                records.append(r)
    return records

def main():
    ours_all = load_jsonl(OURS_JSONL)
    base_all = load_jsonl(BASE_JSONL)
    
    # Build test indices by query text
    ours_test = {r["query"]: r for r in ours_all if r["phase"] == "test"}
    base_test = {r["query"]: r for r in base_all if r["phase"] == "test"}
    
    common_queries = set(ours_test.keys()) & set(base_test.keys())
    ours_only_queries = set(ours_test.keys()) - set(base_test.keys())
    base_only_queries = set(base_test.keys()) - set(ours_test.keys())
    
    print(f"Ours test: {len(ours_test)}, Baseline test: {len(base_test)}")
    print(f"Common queries: {len(common_queries)}")
    print(f"Ours-only queries: {len(ours_only_queries)}")
    print(f"Base-only queries: {len(base_only_queries)}")
    
    # Categorize
    ours_solves = []  # ours solves, base doesn't
    base_solves = []  # base solves, ours doesn't
    both_fail = []
    both_solve_ours_fewer = []
    both_solve = 0
    
    for q in common_queries:
        ro = ours_test[q]
        rb = base_test[q]
        if ro["is_solved"] and not rb["is_solved"]:
            ours_solves.append(q)
        elif not ro["is_solved"] and rb["is_solved"]:
            base_solves.append(q)
        elif not ro["is_solved"] and not rb["is_solved"]:
            both_fail.append(q)
        else:
            both_solve += 1
            oe = len(ro.get("errors", []))
            be = len(rb.get("errors", []))
            if oe < be:
                both_solve_ours_fewer.append((q, oe, be))
    
    ours_fail_set = set(q for q in common_queries if not ours_test[q]["is_solved"])
    base_fail_set = set(q for q in common_queries if not base_test[q]["is_solved"])
    overlap = ours_fail_set & base_fail_set
    
    print(f"\n--- Divergence Statistics (matched by query text) ---")
    print(f"Ours solves, baseline doesn't: {len(ours_solves)}")
    print(f"Baseline solves, ours doesn't: {len(base_solves)}")
    print(f"Both fail:                     {len(both_fail)}")
    print(f"Both solve:                    {both_solve}")
    print(f"  - Ours fewer errors:         {len(both_solve_ours_fewer)}")
    print(f"\nOurs total failures:   {len(ours_fail_set)}")
    print(f"Base total failures:   {len(base_fail_set)}")
    print(f"Shared failures:       {len(overlap)}")
    print(f"Ours-only failures:    {len(ours_fail_set - base_fail_set)}")
    print(f"Base-only failures:    {len(base_fail_set - ours_fail_set)}")
    if ours_fail_set | base_fail_set:
        print(f"Jaccard similarity:    {len(overlap)/len(ours_fail_set | base_fail_set):.3f}")
    
    def print_example(q, tag):
        ro = ours_test[q]
        rb = base_test[q]
        print(f"\n  --- {tag} ---")
        print(f"  Query: {q[:120]}...")
        print(f"  True answer: {ro['true_answer']}")
        print(f"")
        print(f"  OURS:")
        print(f"    JSONL line: {ro['_line_number']}")
        print(f"    global_sample_idx: {ro['global_sample_idx']}")
        print(f"    solved={ro['is_solved']}, errors={len(ro.get('errors',[]))}, penalty={ro.get('error_penalty',0):.2f}")
        agents = [(s["role"], s["llm_name"].split("/")[-1]) for s in ro.get("trace", [])]
        print(f"    agents: {agents}")
        if ro.get("errors"):
            print(f"    error_types: {[e['error_type'] for e in ro['errors']]}")
        has_collab = any(s.get("spatial_predecessors", []) for s in ro.get("trace", []))
        print(f"    collaborative: {has_collab}")
        print(f"")
        print(f"  BASELINE:")
        print(f"    JSONL line: {rb['_line_number']}")
        print(f"    global_sample_idx: {rb['global_sample_idx']}")
        print(f"    solved={rb['is_solved']}, errors={len(rb.get('errors',[]))}, penalty={rb.get('error_penalty',0):.2f}")
        agents = [(s["role"], s["llm_name"].split("/")[-1]) for s in rb.get("trace", [])]
        print(f"    agents: {agents}")
        if rb.get("errors"):
            print(f"    error_types: {[e['error_type'] for e in rb['errors']]}")
        has_collab = any(s.get("spatial_predecessors", []) for s in rb.get("trace", []))
        print(f"    collaborative: {has_collab}")
    
    print(f"\n{'='*70}")
    print(f"  EXAMPLES: Ours solves, baseline doesn't")
    print(f"{'='*70}")
    for q in sorted(ours_solves, key=lambda q: ours_test[q]["_line_number"])[:3]:
        print_example(q, "Ours-only-solves")
    
    print(f"\n{'='*70}")
    print(f"  EXAMPLES: Baseline solves, ours doesn't")
    print(f"{'='*70}")
    for q in sorted(base_solves, key=lambda q: ours_test[q]["_line_number"])[:3]:
        print_example(q, "Base-only-solves")
    
    print(f"\n{'='*70}")
    print(f"  EXAMPLES: Both fail")
    print(f"{'='*70}")
    for q in sorted(both_fail, key=lambda q: ours_test[q]["_line_number"])[:3]:
        print_example(q, "Both-fail")
    
    print(f"\n{'='*70}")
    print(f"  EXAMPLES: Both solve, ours fewer errors")
    print(f"{'='*70}")
    both_solve_ours_fewer.sort(key=lambda x: x[2]-x[1], reverse=True)
    for q, oe, be in both_solve_ours_fewer[:3]:
        print_example(q, f"Both-solve (ours_err={oe}, base_err={be})")

if __name__ == "__main__":
    main()
