#!/usr/bin/env python3
"""
Analyze error propagation in MasRouter test traces.

For each experiment (ours γ=0.1  vs  baseline γ=0), reports:
  1. Number of test samples with propagated errors
  2. Propagation chain length distribution
  3. Propagation vs collaboration mode (Chain/FullConnected have inter-agent
     edges, so propagation is architecturally possible; CoT/IO/Debate do not)
  4. Relationship between propagation and correctness
  5. Error types most frequently involved in propagation chains

A propagation chain is a connected path E_i → E_j → … through the
propagation trajectory.  Chain length = number of edges in the longest
connected component (≥1 means at least one from→to link).
"""

import json, sys, os
from collections import Counter, defaultdict
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────
FILES = {
    "Ours (γ=0.1)":   "gsm8k_error_responses_2026-02-24-15-45-52.jsonl",
    "Baseline (γ=0)": "gsm8k_error_responses_2026-02-25-11-16-41.jsonl",
}

LOG_DIR = Path(__file__).parent


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def load_test_records(filepath):
    records = []
    with open(filepath) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("phase") == "test":
                records.append(rec)
    return records


def get_propagation_chains(trajectory):
    """
    Given a propagation trajectory (list of {from, to, explanation}),
    return a list of chains where each chain is a list of error IDs
    forming a connected path.
    
    E.g. E1→E2, E2→E3, E4→E5  →  [[E1,E2,E3], [E4,E5]]
    """
    if not trajectory:
        return []
    
    # Build adjacency list
    successors = defaultdict(list)
    all_nodes = set()
    targets = set()
    for link in trajectory:
        src = link.get("from", "")
        dst = link.get("to", "")
        if src and dst:
            successors[src].append(dst)
            all_nodes.add(src)
            all_nodes.add(dst)
            targets.add(dst)
    
    if not all_nodes:
        return []
    
    # Find roots (nodes that are sources but not targets)
    roots = all_nodes - targets
    if not roots:
        # Cycle or all are targets; just pick any
        roots = {next(iter(all_nodes))}
    
    # BFS from each root to find chains
    chains = []
    visited = set()
    for root in sorted(roots):
        if root in visited:
            continue
        chain = []
        queue = [root]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            chain.append(node)
            for succ in successors.get(node, []):
                if succ not in visited:
                    queue.append(succ)
        if len(chain) >= 2:  # at least one propagation edge
            chains.append(chain)
    
    return chains


def chain_lengths(trajectory):
    """Return list of chain lengths (num edges) for each connected component."""
    chains = get_propagation_chains(trajectory)
    return [len(c) - 1 for c in chains]  # edges = nodes - 1


def infer_collab_mode(trace):
    """
    Infer the collaboration mode from the trace's spatial_predecessors:
      - All empty → IO/CoT/Reflection (independent)
      - Chain pattern (each sees only previous) → Chain
      - Full connectivity → FullConnected
      - Otherwise → Other
    Also returns whether any agent has spatial predecessors.
    """
    n = len(trace)
    if n == 0:
        return "unknown", False
    
    has_spatial = any(
        len(step.get("spatial_predecessors", [])) > 0 for step in trace
    )
    
    if not has_spatial:
        return "independent", False
    
    # Check for chain pattern
    # In Chain mode: step[0] has no predecessors, step[i] has exactly step[i-1]
    is_chain = True
    node_ids = [step["node_id"] for step in trace]
    for i, step in enumerate(trace):
        sp = step.get("spatial_predecessors", [])
        if i == 0:
            if len(sp) != 0:
                is_chain = False
                break
        else:
            if sp != [node_ids[i-1]]:
                is_chain = False
                break
    if is_chain:
        return "chain", True
    
    # Check for full connectivity: step[i] has all step[0..i-1] as predecessors
    is_full = True
    for i, step in enumerate(trace):
        sp = set(step.get("spatial_predecessors", []))
        expected = set(node_ids[:i])
        if sp != expected:
            is_full = False
            break
    if is_full:
        return "full_connected", True
    
    return "other_connected", True


# ────────────────────────────────────────────────────────────────────────────
# Main analysis
# ────────────────────────────────────────────────────────────────────────────

def analyze(label, filepath):
    records = load_test_records(filepath)
    n_total = len(records)
    
    # Core counters
    n_with_errors = 0
    n_with_propagation = 0
    
    all_chain_lengths = []  # one entry per chain
    all_sample_max_chain_len = []  # max chain length per sample (for samples with propagation)
    all_num_edges_per_sample = []  # total propagation edges per sample
    
    # By mode
    mode_counts = Counter()
    mode_with_prop = Counter()
    mode_with_errors = Counter()
    
    # By correctness
    prop_solved = 0
    prop_unsolved = 0
    no_prop_solved = 0
    no_prop_unsolved = 0
    
    # Error types involved in propagation
    prop_source_types = Counter()
    prop_target_types = Counter()
    
    # Propagation across vs within agents
    cross_agent_prop = 0
    same_agent_prop = 0
    
    for rec in records:
        errors = rec.get("errors", []) or []
        prop = rec.get("propagation", {}) or {}
        trajectory = prop.get("trajectory", []) or []
        trace = rec.get("trace", []) or []
        is_solved = rec.get("is_solved", False)
        
        has_errors = len(errors) > 0
        has_prop = len(trajectory) > 0
        
        mode, has_spatial = infer_collab_mode(trace)
        mode_counts[mode] += 1
        
        if has_errors:
            n_with_errors += 1
            mode_with_errors[mode] += 1
        
        if has_prop:
            n_with_propagation += 1
            mode_with_prop[mode] += 1
            
            lengths = chain_lengths(trajectory)
            all_chain_lengths.extend(lengths)
            
            if lengths:
                all_sample_max_chain_len.append(max(lengths))
            
            all_num_edges_per_sample.append(len(trajectory))
            
            if is_solved:
                prop_solved += 1
            else:
                prop_unsolved += 1
            
            # Analyze error types in propagation
            error_map = {e.get("error_id"): e for e in errors}
            for link in trajectory:
                src_id = link.get("from", "")
                dst_id = link.get("to", "")
                if src_id in error_map:
                    etype = error_map[src_id].get("error_type", "Unknown")
                    prop_source_types[etype] += 1
                if dst_id in error_map:
                    etype = error_map[dst_id].get("error_type", "Unknown")
                    prop_target_types[etype] += 1
                
                # Cross-agent vs same-agent propagation
                src_error = error_map.get(src_id, {})
                dst_error = error_map.get(dst_id, {})
                src_step = src_error.get("error_start_step", -1)
                dst_step = dst_error.get("error_start_step", -1)
                if src_step >= 0 and dst_step >= 0:
                    # Check if they are the same agent
                    src_agent = None
                    dst_agent = None
                    for step in trace:
                        if step["step_id"] == src_step:
                            src_agent = step.get("node_id")
                        if step["step_id"] == dst_step:
                            dst_agent = step.get("node_id")
                    if src_agent and dst_agent:
                        if src_agent != dst_agent:
                            cross_agent_prop += 1
                        else:
                            same_agent_prop += 1
        else:
            if is_solved:
                no_prop_solved += 1
            else:
                no_prop_unsolved += 1
    
    # ── Print report ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    
    print(f"\n1. OVERVIEW")
    print(f"   Total test samples:                {n_total:>6}")
    print(f"   Samples with errors:               {n_with_errors:>6}  ({100*n_with_errors/n_total:.1f}%)")
    print(f"   Samples with propagated errors:     {n_with_propagation:>6}  ({100*n_with_propagation/n_total:.1f}%)")
    print(f"   Propagation rate (among errored):   {n_with_propagation:>6} / {n_with_errors:>4}  "
          f"= {100*n_with_propagation/max(n_with_errors,1):.1f}%")
    
    print(f"\n2. PROPAGATION LENGTH DISTRIBUTION  (per chain)")
    len_counter = Counter(all_chain_lengths)
    if len_counter:
        print(f"   {'Chain len (edges)':<22} {'Count':>6}  {'%':>6}")
        print(f"   {'-'*40}")
        for length in sorted(len_counter.keys()):
            cnt = len_counter[length]
            print(f"   {length:<22} {cnt:>6}  {100*cnt/sum(len_counter.values()):>5.1f}%")
        avg_len = sum(all_chain_lengths) / len(all_chain_lengths)
        max_len = max(all_chain_lengths) if all_chain_lengths else 0
        print(f"   {'':22} ------")
        print(f"   Total chains:         {sum(len_counter.values()):>6}")
        print(f"   Mean chain length:    {avg_len:>6.2f} edges")
        print(f"   Max chain length:     {max_len:>6} edges")
    else:
        print(f"   (no propagation chains found)")
    
    print(f"\n3. PROPAGATION EDGES PER SAMPLE  (among samples with propagation)")
    if all_num_edges_per_sample:
        edge_counter = Counter(all_num_edges_per_sample)
        print(f"   {'# edges':<22} {'Count':>6}  {'%':>6}")
        print(f"   {'-'*40}")
        for n_edges in sorted(edge_counter.keys()):
            cnt = edge_counter[n_edges]
            print(f"   {n_edges:<22} {cnt:>6}  {100*cnt/sum(edge_counter.values()):>5.1f}%")
        avg_edges = sum(all_num_edges_per_sample) / len(all_num_edges_per_sample)
        print(f"   Mean edges/sample:    {avg_edges:>6.2f}")
    
    print(f"\n4. PROPAGATION BY COLLABORATION MODE")
    print(f"   {'Mode':<22} {'Total':>6} {'w/ errors':>10} {'w/ prop':>10} {'prop rate':>10}")
    print(f"   {'-'*62}")
    for mode in sorted(mode_counts.keys()):
        total = mode_counts[mode]
        err = mode_with_errors.get(mode, 0)
        prp = mode_with_prop.get(mode, 0)
        rate = f"{100*prp/max(err,1):.1f}%" if err > 0 else "n/a"
        print(f"   {mode:<22} {total:>6} {err:>10} {prp:>10} {rate:>10}")
    
    print(f"\n5. CROSS-AGENT vs SAME-AGENT PROPAGATION")
    total_prop_edges = cross_agent_prop + same_agent_prop
    if total_prop_edges > 0:
        print(f"   Cross-agent propagation:  {cross_agent_prop:>5}  ({100*cross_agent_prop/total_prop_edges:.1f}%)")
        print(f"   Same-agent propagation:   {same_agent_prop:>5}  ({100*same_agent_prop/total_prop_edges:.1f}%)")
    else:
        print(f"   (no propagation edges with matched agents)")
    
    print(f"\n6. PROPAGATION vs CORRECTNESS")
    print(f"   {'':30} {'Solved':>8} {'Unsolved':>10} {'Solve rate':>12}")
    print(f"   {'-'*62}")
    total_prop = prop_solved + prop_unsolved
    total_no = no_prop_solved + no_prop_unsolved
    print(f"   {'With propagation':<30} {prop_solved:>8} {prop_unsolved:>10} "
          f"{100*prop_solved/max(total_prop,1):>10.1f}%")
    print(f"   {'Without propagation':<30} {no_prop_solved:>8} {no_prop_unsolved:>10} "
          f"{100*no_prop_solved/max(total_no,1):>10.1f}%")
    
    print(f"\n7. ERROR TYPES MOST OFTEN AT SOURCE OF PROPAGATION  (top 10)")
    print(f"   {'Error type':<45} {'Count':>6}")
    print(f"   {'-'*55}")
    for etype, cnt in prop_source_types.most_common(10):
        print(f"   {etype:<45} {cnt:>6}")
    
    print(f"\n8. ERROR TYPES MOST OFTEN AS TARGET (propagated-to)  (top 10)")
    print(f"   {'Error type':<45} {'Count':>6}")
    print(f"   {'-'*55}")
    for etype, cnt in prop_target_types.most_common(10):
        print(f"   {etype:<45} {cnt:>6}")
    
    return {
        "n_total": n_total,
        "n_with_errors": n_with_errors,
        "n_with_propagation": n_with_propagation,
        "chain_lengths": all_chain_lengths,
        "edges_per_sample": all_num_edges_per_sample,
        "cross_agent": cross_agent_prop,
        "same_agent": same_agent_prop,
        "prop_source_types": prop_source_types,
    }


# ────────────────────────────────────────────────────────────────────────────
# Comparative summary
# ────────────────────────────────────────────────────────────────────────────

def compare(results):
    ours = results["Ours (γ=0.1)"]
    base = results["Baseline (γ=0)"]
    
    print(f"\n{'='*70}")
    print(f"  COMPARATIVE SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n   {'Metric':<45} {'Ours':>10} {'Baseline':>10} {'Δ':>10}")
    print(f"   {'-'*78}")
    
    def row(label, v_ours, v_base, fmt=".1f", is_pct=False):
        suffix = "%" if is_pct else ""
        delta = v_ours - v_base
        sign = "+" if delta > 0 else ""
        print(f"   {label:<45} {v_ours:>9{fmt}}{suffix} {v_base:>9{fmt}}{suffix} {sign}{delta:>8{fmt}}{suffix}")
    
    row("Test samples with errors",
        ours["n_with_errors"], base["n_with_errors"], fmt="d")
    row("Test samples with propagation",
        ours["n_with_propagation"], base["n_with_propagation"], fmt="d")
    
    ours_prop_rate = 100 * ours["n_with_propagation"] / max(ours["n_with_errors"], 1)
    base_prop_rate = 100 * base["n_with_propagation"] / max(base["n_with_errors"], 1)
    row("Propagation rate (among errored)", ours_prop_rate, base_prop_rate, is_pct=True)
    
    ours_avg_chain = sum(ours["chain_lengths"]) / max(len(ours["chain_lengths"]), 1)
    base_avg_chain = sum(base["chain_lengths"]) / max(len(base["chain_lengths"]), 1)
    row("Mean chain length (edges)", ours_avg_chain, base_avg_chain, fmt=".2f")
    
    ours_avg_edges = sum(ours["edges_per_sample"]) / max(len(ours["edges_per_sample"]), 1)
    base_avg_edges = sum(base["edges_per_sample"]) / max(len(base["edges_per_sample"]), 1)
    row("Mean edges/sample (w/ prop)", ours_avg_edges, base_avg_edges, fmt=".2f")
    
    ours_cross_pct = 100 * ours["cross_agent"] / max(ours["cross_agent"] + ours["same_agent"], 1)
    base_cross_pct = 100 * base["cross_agent"] / max(base["cross_agent"] + base["same_agent"], 1)
    row("Cross-agent propagation %", ours_cross_pct, base_cross_pct, is_pct=True)


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(LOG_DIR)
    results = {}
    for label, fname in FILES.items():
        results[label] = analyze(label, fname)
    compare(results)
