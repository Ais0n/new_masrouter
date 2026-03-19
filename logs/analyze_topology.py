#!/usr/bin/env python3
"""Analyze topology distribution from traces."""
import json
from collections import Counter

def analyze_topology(path, label):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    
    test = [r for r in records if r["phase"] == "test"]
    
    topologies = Counter()
    for r in test:
        trace = r.get("trace", [])
        n_steps = len(trace)
        has_spatial = any(s.get("spatial_predecessors", []) for s in trace)
        has_temporal = any(s.get("temporal_predecessors", []) for s in trace)
        n_rounds = max((s.get("round", 0) for s in trace), default=0) + 1
        spatial_edges = sum(len(s.get("spatial_predecessors", [])) for s in trace)
        per_round_steps = n_steps // max(n_rounds, 1)
        
        if n_steps <= 1:
            topo = "IO/CoT (single)"
        elif not has_spatial and not has_temporal and n_rounds == 1:
            topo = "Independent (no edges, 1 round)"
        elif has_spatial and not has_temporal and n_rounds == 1:
            if spatial_edges <= per_round_steps:
                topo = "Chain (linear spatial)"
            else:
                topo = "FullConnected/Star (dense spatial)"
        elif not has_spatial and has_temporal and n_rounds > 1:
            topo = "Debate (temporal only, %d rounds)" % n_rounds
        elif has_spatial and has_temporal:
            topo = "Mixed spatial+temporal (%d rounds)" % n_rounds
        else:
            topo = "Other"
        topologies[topo] += 1
    
    print(f"\n{label} (N={len(test)}):")
    for t, c in topologies.most_common():
        print(f"  {t:50s} {c:>5} ({c/len(test)*100:.1f}%)")

analyze_topology("gsm8k_error_responses_2026-02-24-15-45-52.jsonl", "OURS (gamma=0.1)")
analyze_topology("gsm8k_error_responses_2026-02-25-11-16-41.jsonl", "BASELINE (gamma=0)")
