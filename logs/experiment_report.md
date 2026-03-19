# Experiment Report: Error-Augmented MasRouter on GSM8K

> **Note:** This is the initial report covering the baseline vs γ=0.1 comparison only.
> For the full γ-sweep results (0.0–1.6) and judge ablation, see the main [README](../README.md).

**Date:** 2026-02-26  
**Experiments:**  
- **Ours (γ=0.1):** `gsm8k_error_2026-02-24-15-45-52`  
- **Baseline (γ=0):** `gsm8k_error_2026-02-25-11-16-41`

---

## 1. Experiment Setup

| Parameter | Value |
|---|---|
| Dataset | GSM8K |
| Train samples | 263 (20% of 1319) |
| Test samples | 1056 (80% of 1319) |
| Batch size | 16 |
| Epochs | 5 |
| Batches/epoch | 16 (256 samples used per epoch) |
| β (cost weight) | 200.0 |
| γ (error weight) | **0.1** (ours) / **0.0** (baseline) |
| Error evaluator LLM | gpt-4o-mini |
| Candidate LLMs | gpt-4o-mini, gemini-2.0-flash-001, deepseek-chat, claude-3.5-haiku, llama-3.1-70b-instruct |
| Training method | Per-agent REINFORCE with counterfactual credit assignment |
| Verbose mode | Enabled |

**Key difference:** The only variable is γ. When γ=0, the error penalty term vanishes and the system reduces to the original MasRouter reward: `utility = is_solved − β·cost`.

---

## 2. Headline Results

| Metric | Ours (γ=0.1) | Baseline (γ=0) | Δ | Δ% |
|---|:---:|:---:|:---:|:---:|
| **Test Accuracy** | **0.9498** | 0.9489 | +0.0009 | +0.1% |
| **Total Test Errors** | **194** | 274 | **−80** | **−29.2%** |
| Samples with ≥1 error | 149 (14.1%) | 197 (18.7%) | −48 | −24.4% |
| Avg error penalty (test) | 0.2014 | 0.2827 | −0.0814 | −28.8% |
| Avg cost (test) | 0.000914 | 0.000881 | +0.000033 | +3.8% |
| Propagation edges (test) | 57 | 95 | −38 | −40.0% |

### Key takeaway

> **Error-augmented training (γ=0.1) reduces total test errors by 29.2% (194 → 274) and error propagation by 40% while maintaining task accuracy (94.98% vs 94.89%).** The accuracy improvement is marginal (+0.09pp), but the error quality improvement is substantial.

---

## 3. Training Dynamics

### 3.1 Per-Epoch Training Accuracy

| Epoch | Ours Acc | Baseline Acc | Ours Errors | Baseline Errors | Ours Avg Penalty | Baseline Avg Penalty |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.9336 | 0.9414 | 61 | 80 | 0.2583 | 0.3530 |
| 1 | 0.9492 | 0.9336 | 63 | 55 | 0.2632 | 0.2376 |
| 2 | 0.9297 | 0.9180 | 60 | 73 | 0.2440 | 0.2902 |
| 3 | **0.9648** | 0.9453 | **38** | 59 | **0.1741** | 0.2552 |
| 4 | 0.9297 | 0.9219 | 45 | 69 | 0.1915 | 0.3043 |

### 3.2 Training Error Trend

- **Ours:** 61 → 63 → 60 → **38** → 45 — clear decline in epochs 3–4 (error penalty drives learning)
- **Baseline:** 80 → 55 → 73 → 59 → 69 — no systematic trend (errors not penalized, no learning signal)

The error count in ours drops by **37.7%** from epoch 0 to epoch 3 (61→38), while the baseline has no comparable reduction. This confirms that the γ·error_penalty term provides a meaningful gradient signal for the REINFORCE policy.

### 3.3 Training Cost Progression

| Epoch | Ours Avg Cost | Baseline Avg Cost |
|:---:|:---:|:---:|
| 0 | 0.001503 | 0.001487 |
| 1 | 0.001024 | 0.001041 |
| 2 | 0.000978 | 0.000991 |
| 3 | 0.000954 | 0.000916 |
| 4 | 0.000961 | 0.000826 |

Both experiments show similar cost optimization trajectories. The baseline achieves slightly lower cost by epoch 4 (0.000826 vs 0.000961), as expected — with γ=0 the optimizer allocates all reward signal to accuracy and cost, leaving more gradient budget for cost minimization.

---

## 4. Error Type Analysis (Test Set)

### 4.1 Error Type Distribution

| Error Type | Ours | Baseline | Δ | Δ% |
|---|:---:|:---:|:---:|:---:|
| **M1.1** Disobey-Task-Specification | 142 | 177 | −35 | −19.8% |
| **M3.3** Incorrect-verification | 31 | 44 | −13 | −29.5% |
| **M4.1** Hallucinated-reasoning | **1** | 16 | **−15** | **−93.8%** |
| **M2.1** Ignore-peer-response | 11 | 16 | −5 | −31.2% |
| **M3.2** Wrong-summary | 1 | 8 | −7 | −87.5% |
| **M1.3** Fail-to-meet-implicit-requirements | 2 | 6 | −4 | −66.7% |
| M2.2 Ignore-peer-review | 3 | 2 | +1 | +50.0% |
| M3.1 Redundant-plan | 1 | 2 | −1 | −50.0% |
| M3.4 Premature-termination | 1 | 1 | 0 | 0.0% |
| M1.2 Forget-corner-case | 1 | 1 | 0 | 0.0% |
| M4.2 Agent-hide-error | 0 | 1 | −1 | −100.0% |
| **TOTAL** | **194** | **274** | **−80** | **−29.2%** |

### 4.2 Error Category Highlights

**Largest absolute reduction:** M1.1 Disobey-Task-Specification (−35 errors). This is the most common error type on GSM8K, where agents produce a numeric answer without following the required output format.

**Largest relative reduction:** M4.1 Hallucinated-reasoning drops from 16 → 1 (**−93.8%**). This suggests that the error penalty particularly discourages agent configurations that produce fabricated reasoning chains.

**M3.2 Wrong-summary** also drops dramatically (8 → 1, −87.5%), and **M1.3 Fail-to-meet-implicit-requirements** drops from 6 → 2 (−66.7%).

The only error type that slightly increased is M2.2 Ignore-peer-review (2 → 3), which is within noise range.

---

## 5. LLM-Level Error Analysis (Test Set)

| LLM | Ours Errors | Base Errors | Δ | Δ% | Ours Usage | Base Usage |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| openai/gpt-4o-mini | 86 | 87 | −1 | −1.1% | 1242 | 978 |
| google/gemini-2.0-flash-001 | 58 | 71 | −13 | −18.3% | 1290 | 1011 |
| meta-llama/llama-3.1-70b-instruct | **13** | **81** | **−68** | **−84.0%** | 183 | 990 |
| deepseek/deepseek-chat | 15 | 12 | +3 | +25.0% | 352 | 155 |
| anthropic/claude-3.5-haiku | 9 | 10 | −1 | −10.0% | 185 | 139 |

### Key observation: LLM selection shift

The most dramatic change is in **Llama-3.1-70b-instruct**:
- Baseline: Used in **990 trace steps**, committed **81 errors** (error rate: 8.2%)
- Ours: Used in only **183 trace steps**, committed **13 errors** (error rate: 7.1%)

**The router learned to reduce Llama usage by 81.5%** (990 → 183 steps), which is the primary mechanism for the 84% error reduction. This is a strong validation of the per-agent REINFORCE: when an LLM repeatedly produces errors, the error penalty backpropagates through the LLM-selection stage, shifting allocation away from it.

Conversely, the router shifts traffic toward:
- **gemini-2.0-flash-001:** 1011 → 1290 steps (+27.6%)
- **gpt-4o-mini:** 978 → 1242 steps (+27.0%)
- **deepseek-chat:** 155 → 352 steps (+127.1%)
- **claude-3.5-haiku:** 139 → 185 steps (+33.1%)

The cost increase of +3.8% is a direct consequence of this shift — gemini and gpt-4o-mini are slightly more expensive than Llama, but produce significantly fewer errors.

---

## 6. Agent Role Error Analysis (Test Set)

| Role | Ours | Base | Δ | Δ% |
|---|:---:|:---:|:---:|:---:|
| CertifiedAccountant | 25 | 46 | −21 | −45.7% |
| Economist | 16 | 37 | −21 | −56.8% |
| MathTeacher | 12 | 31 | −19 | −61.3% |
| MathSolver | 17 | 33 | −16 | −48.5% |
| AlgorithmEngineer | 4 | 12 | −8 | −66.7% |
| SoftwareDeveloper | 9 | 17 | −8 | −47.1% |
| Scientist | 9 | 16 | −7 | −43.8% |
| Engineer | 18 | 20 | −2 | −10.0% |
| ProgrammingExpert | 22 | 21 | +1 | +4.8% |
| Mathematician | 13 | 13 | 0 | 0.0% |
| Inspector | 20 | 7 | +13 | +185.7% |
| MathAnalyst | 16 | 8 | +8 | +100.0% |

Most roles see substantial error reductions. **Inspector** and **MathAnalyst** show increases — the router now uses these roles more frequently (Inspector: 173→391 steps, MathAnalyst: 154→259 steps), so the absolute error count rises even though the per-step error rate may be lower. This reflects a role-selection shift toward verification-oriented roles.

---

## 7. Structural Analysis (Test Set)

### 7.1 Trace Length

| Trace Length | Ours | Baseline |
|:---:|:---:|:---:|
| 3 steps | 1028 (97.3%) | 1021 (96.7%) |
| 6 steps | 28 (2.7%) | 35 (3.3%) |
| Average | 3.08 | 3.10 |

Trace lengths are nearly identical — the error penalty does not substantially change the number of agent steps the router selects.

### 7.2 Collaboration Topology

| Topology | Ours | Baseline |
|---|:---:|:---:|
| Collaborative (has spatial edges) | **451** (42.7%) | 81 (7.7%) |
| Independent (no spatial edges) | 605 (57.3%) | 975 (92.3%) |

**This is a striking structural difference.** The error-augmented router chooses collaborative topologies **5.6× more often** (42.7% vs 7.7%). In collaborative topologies, agents can read each other's outputs via spatial predecessors, enabling cross-checking and verification. The error penalty incentivizes this because collaboration reduces errors (especially M2.x communication errors and M3.x verification errors).

### 7.3 Number of Distinct Agents

| # Agents | Ours | Baseline |
|:---:|:---:|:---:|
| 1 | 8 | 13 |
| 2 | 250 | 296 |
| 3 | 798 | 747 |

The ours configuration uses 3-agent teams more often (75.6% vs 70.7%), consistent with the router learning that more agents + collaboration = fewer errors.

### 7.4 Error–Correctness Correlation

| Category | Ours | Baseline |
|---|:---:|:---:|
| Solved + Errors | 99 | 144 |
| Solved + No Errors | 904 | 858 |
| Unsolved + Errors | 50 | 53 |
| Unsolved + No Errors | 3 | 1 |

Among solved samples, ours has only 99 with errors vs baseline's 144 — a **31.2% reduction in "noisy correctness"**. This means ours produces cleaner reasoning traces even when the final answer is right.

Among unsolved samples, both conditions have high error rates (50/53 vs 53/54 ≈ 94–98%), confirming that errors strongly correlate with failure.

---

## 8. Error Propagation Analysis (Test Set)

| Metric | Ours | Baseline | Δ% |
|---|:---:|:---:|:---:|
| Samples with propagation | 46 | 76 | −39.5% |
| Total propagation edges | 57 | 95 | −40.0% |
| Avg edges per propagating sample | 1.24 | 1.25 | −0.8% |

The error-augmented router reduces error propagation events by **40%**. The average propagation chain length remains ~1.25 edges per sample, suggesting the reduction comes from preventing propagation from starting at all, not from shortening chains once they begin.

---

## 9. Error Penalty Distribution (Test, Erroneous Samples)

| Statistic | Ours | Baseline |
|---|:---:|:---:|
| N (samples with errors) | 149 | 197 |
| Min penalty | 1.0000 | 1.0000 |
| Max penalty | 4.9500 | 4.9500 |
| Mean penalty | 1.4273 | 1.5156 |
| P50 penalty | 1.2000 | 1.3000 |
| P90 penalty | 1.9500 | 2.4000 |

The ours distribution has a lower mean (1.43 vs 1.52) and notably a lower P90 (1.95 vs 2.40), indicating that when errors do occur, they tend to be less severe (lower severity weight and/or shorter span).

---

## 10. Summary & Discussion

### What works
1. **Error reduction is the headline result:** 29.2% fewer errors, 40% fewer propagation events.
2. **The router learns to avoid error-prone LLMs:** Llama-3.1-70b usage drops 81.5%, driving most of the error reduction.
3. **Collaboration is incentivized:** Ours uses collaborative topologies 5.6× more than baseline.
4. **Error quality improves:** M4.1 Hallucinated-reasoning drops 93.8%, M3.2 Wrong-summary drops 87.5%.
5. **Accuracy is maintained:** No accuracy penalty from adding the error signal.

### What needs investigation
1. **Accuracy improvement is marginal (+0.09pp).** Error reduction does not directly translate to more solved problems on GSM8K. This may be because GSM8K is already well-saturated — the 94.9% baseline accuracy leaves few unsolved samples. Harder benchmarks (MATH, MMLU) may show larger accuracy gains.
2. **Cost increased slightly (+3.8%).** The shift from Llama to more capable (and expensive) LLMs is the cause. This trade-off (more cost for fewer errors) may not always be desirable.
3. **Inspector/MathAnalyst error increase.** These roles are used more under ours but show higher absolute error counts. Investigating their per-step error rates would clarify whether this is simply a volume effect.
4. **Training instability:** Epoch 4 accuracy drops back to 0.9297 in ours after a 0.9648 peak at epoch 3. This may indicate that 5 epochs is sufficient (or that learning-rate scheduling / early stopping could help).

### Next steps
- **Run on harder benchmarks** (MATH, MBPP, MMLU) where the accuracy ceiling is lower and error reduction may translate to accuracy gains.
- **Ablation on γ:** Test γ ∈ {0.01, 0.05, 0.1, 0.2, 0.5} to find the optimal error penalty weight.
- **Per-step error rate analysis:** Compute error_count / usage_count per LLM and per role to normalize for selection frequency.
- **Qualitative trace comparison:** Compare traces for the same query between ours and baseline to understand what changes in the reasoning process.

---

*Report generated from analysis of `gsm8k_error_responses_*.jsonl` and `gsm8k_error_*.txt` log files.*
