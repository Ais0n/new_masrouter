# Experiment Logs & Analysis Scripts

This directory contains experiment logs (gitignored) and analysis scripts for reproducing the results reported in the main README.

## Log File Naming Convention

- **Text logs:** `gsm8k_error_YYYY-MM-DD-HH-MM-SS.txt` — human-readable training/test output
- **JSONL logs:** `gsm8k_error_responses_YYYY-MM-DD-HH-MM-SS.jsonl` — structured per-query records with routing decisions, error labels, and rewards

## Experiment Index

| Timestamp | γ | Judge | Notes |
|---|---|---|---|
| `2026-02-25-11-16-41` | 0.0 | gpt-4o-mini | **Baseline** — no error penalty |
| `2026-02-24-15-45-52` | 0.1 | gpt-4o-mini | First error-augmented run |
| `2026-03-03-00-50-44` | 0.2 | gpt-4o-mini | Train only (crash recovery) |
| `2026-03-03-23-48-37` | 0.2 | gpt-4o-mini | Test-only rerun (1056 records) |
| `2026-03-04-12-23-35` | 0.4 | gpt-4o-mini | **Best** — sweet-spot γ |
| `2026-03-05-12-52-50` | 0.8 | gpt-4o-mini | Over-penalization starts |
| `2026-03-06-12-21-42` | 1.6 | gpt-4o-mini | Strong over-penalization |
| `2026-03-07-12-15-40` | 0.2 | gpt-5-mini | Judge ablation |

## Analysis Scripts

### `analyze_all_experiments.py`
**Unified metrics across all γ values.** Reads every JSONL file, computes accuracy, total errors, error-sample count, propagation edges, and per-query cost. Produces the main results table.

```bash
python analyze_all_experiments.py
```

### `analyze_propagation.py`
**Error propagation chain analysis.** Extracts propagation graphs from JSONL records, computes chain lengths, cross-agent vs intra-agent ratios, and the correlation between propagation and solve rate.

```bash
python analyze_propagation.py
```

### `analyze_cost.py`
**Token cost breakdown.** Computes per-query and per-LLM cost from the JSONL records. Confirms that cost is NOT in the reward function and reports the marginal overhead of the error judge.

```bash
python analyze_cost.py
```

### `analyze_collab_modes.py`
**Collaboration mode distribution.** Parses text logs to extract the frequency of each collaboration mode (IO, CoT, Chain, FullConnected, Debate, Reflection) per experiment.

```bash
python analyze_collab_modes.py
```

### `analyze_experiments.py`
**Original headline metrics.** Extracts accuracy and error counts from a pair of experiments (baseline vs ours). The first analysis script written.

### `analyze_followup.py`
**Per-epoch LLM selection & error type tables.** Tracks how LLM routing distributions shift across training epochs and produces per-error-type frequency breakdowns.

### `analyze_topology.py`
**Topology analysis.** Examines the graph topologies selected by the controller and their relationship to error rates.

### `locate_examples_v2.py`
**Query-text-matched example locator.** Finds specific query examples across experiments using text matching (not index-based) to handle shuffled datasets correctly.

### `experiment_report.md`
**Original compiled report.** The first experiment report covering baseline vs γ=0.1.
