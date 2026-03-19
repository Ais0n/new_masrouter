#!/usr/bin/env python3
"""Extract per-sample collaboration mode from text logs and correlate with JSONL."""
import re
from collections import Counter

def extract_reasoning_modes(log_path):
    """Extract all 'Reasoning: <Mode>' lines from a log file, in order."""
    modes = []
    pattern = re.compile(r"Reasoning:\s+(\w+)")
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                modes.append(m.group(1))
    return modes

def count_modes_by_phase(log_path, n_train, n_test):
    """
    The log runs: train epoch 0 (n_train samples), ..., epoch 4 (n_train), then test (n_test).
    But some train samples may share lines. Let's just count all and also
    split by looking for phase markers.
    """
    modes = []
    phase_markers = []  # (mode, phase)
    # Only match lines from mas_router*.forward which log the collab mode
    pattern_reasoning = re.compile(r"mas_router[^:]*:forward:\d+ - Reasoning:\s+(\w+)")
    pattern_epoch = re.compile(r"Epoch (\d+) ")
    pattern_test = re.compile(r"Start testing|Test Batch|phase.*test", re.IGNORECASE)

    current_phase = "train"
    with open(log_path) as f:
        for line in f:
            if "Start testing" in line or "Testing" in line and "Epoch" not in line:
                current_phase = "test"
            m = pattern_reasoning.search(line)
            if m:
                phase_markers.append((m.group(1), current_phase))

    train_modes = [m for m, p in phase_markers if p == "train"]
    test_modes = [m for m, p in phase_markers if p == "test"]
    return train_modes, test_modes

def main():
    logs = [
        ("OURS (gamma=0.1)", 
         "/Users/yanwei/work/nips2026/masrouter/logs/gsm8k_error_2026-02-24-15-45-52.txt"),
        ("BASELINE (gamma=0)", 
         "/Users/yanwei/work/nips2026/masrouter/logs/gsm8k_error_2026-02-25-11-16-41.txt"),
    ]

    for label, log_path in logs:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        train_modes, test_modes = count_modes_by_phase(log_path, 1280, 1056)
        
        print(f"\n  Total reasoning mode entries: train={len(train_modes)}, test={len(test_modes)}")

        # --- Train per-epoch breakdown ---
        # 256 samples per epoch, 5 epochs = 1280
        print(f"\n  --- Training: Per-Epoch Collaboration Mode ---")
        epoch_size = 256
        n_epochs = 5
        header = f"  {'Mode':20s}"
        for e in range(n_epochs):
            header += f"  {'E'+str(e):>6}"
        header += f"  {'Total':>7}  {'%':>6}"
        print(header)
        print(f"  {'-'*len(header)}")

        all_mode_names = set()
        epoch_counters = []
        for e in range(n_epochs):
            start = e * epoch_size
            end = start + epoch_size
            ep_modes = train_modes[start:end]
            c = Counter(ep_modes)
            epoch_counters.append(c)
            all_mode_names.update(c.keys())

        all_mode_names = sorted(all_mode_names)
        train_total = Counter(train_modes)
        
        for mode in all_mode_names:
            row = f"  {mode:20s}"
            for e in range(n_epochs):
                row += f"  {epoch_counters[e].get(mode, 0):>6}"
            row += f"  {train_total[mode]:>7}"
            pct = train_total[mode] / max(len(train_modes), 1) * 100
            row += f"  {pct:>5.1f}%"
            print(row)

        total_row = f"  {'TOTAL':20s}"
        for e in range(n_epochs):
            total_row += f"  {sum(epoch_counters[e].values()):>6}"
        total_row += f"  {len(train_modes):>7}"
        print(total_row)

        # --- Test breakdown ---
        print(f"\n  --- Test: Collaboration Mode Distribution ---")
        test_counter = Counter(test_modes)
        print(f"  {'Mode':20s}  {'Count':>6}  {'%':>6}")
        print(f"  {'-'*40}")
        for mode in sorted(test_counter.keys()):
            c = test_counter[mode]
            pct = c / max(len(test_modes), 1) * 100
            print(f"  {mode:20s}  {c:>6}  {pct:>5.1f}%")
        print(f"  {'TOTAL':20s}  {len(test_modes):>6}")

    # --- Side-by-side comparison (test) ---
    print(f"\n{'='*60}")
    print(f"  COMPARISON: Test Collaboration Modes")
    print(f"{'='*60}")
    
    results = {}
    for label, log_path in logs:
        _, test_modes = count_modes_by_phase(log_path, 1280, 1056)
        results[label] = Counter(test_modes)
    
    all_modes = sorted(set().union(*(r.keys() for r in results.values())))
    totals = {label: sum(c.values()) for label, c in results.items()}
    
    print(f"\n  {'Mode':20s}  {'Ours':>6} {'Ours%':>6}  {'Base':>6} {'Base%':>6}  {'Δ':>6} {'Δ%':>7}")
    print(f"  {'-'*70}")
    for mode in all_modes:
        ov = results["OURS (gamma=0.1)"].get(mode, 0)
        bv = results["BASELINE (gamma=0)"].get(mode, 0)
        op = ov / max(totals["OURS (gamma=0.1)"], 1) * 100
        bp = bv / max(totals["BASELINE (gamma=0)"], 1) * 100
        delta = ov - bv
        dpct = (delta / bv * 100) if bv else float('nan')
        print(f"  {mode:20s}  {ov:>6} {op:>5.1f}%  {bv:>6} {bp:>5.1f}%  {delta:>+6} {dpct:>+6.1f}%")

if __name__ == "__main__":
    main()
