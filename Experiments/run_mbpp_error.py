"""
Error-augmented MasRouter training on MBPP.

Compared to the baseline run_mbpp.py, this script:
  1. Uses ErrorAwareMasRouter with robustness-augmented embeddings
  2. Runs an LLM-based error evaluator (MAST codebook, 28 error types)
  3. Adds an error penalty to the REINFORCE reward:
       utility = is_solved − β·cost − γ·error_penalty
  4. Updates the RobustnessTracker after each episode
  5. Logs per-category error statistics for analysis

Usage:
    python Experiments/run_mbpp_error.py --gamma 0.1
    python Experiments/run_mbpp_error.py --gamma 0.0   # ablation: no error penalty
"""

import sys
import os
import io
import argparse
import json
import re
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

import torch
import torch.nn.functional as F
from loguru import logger

from MAR.MasRouter.mas_router_error import ErrorAwareMasRouter
from MAR.LLM.llm_profile import llm_profile
from MAR.Agent.reasoning_profile import reasoning_profile
from MAR.Prompts.tasks_profile import tasks_profile
from MAR.Tools.coding.python_executor import PyExecutor
from MAR.Utils.utils import fix_random_seed
from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens
from MAR.Utils.log import configure_logging
from MAR.ErrorAnalysis.robustness_tracker import RobustnessTracker
from MAR.ErrorAnalysis.error_evaluator import LLMErrorEvaluator
from MAR.ErrorAnalysis.error_reward import ErrorRewardComputer
from Datasets.mbpp_dataset import MbppDataset, MbppDataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Error-augmented MasRouter on MBPP"
    )
    # ── Dataset / model args (same as baseline) ───────────────────────────
    parser.add_argument("--dataset_json", type=str,
                        default="Datasets/mbpp/mbpp.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--domain', type=str, default="mbpp")
    parser.add_argument('--decision_method', type=str, default='FinalRefer')
    parser.add_argument('--prompt_file', type=str,
                        default='MAR/Roles/FinalNode/mbpp.json')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--cost_rate', type=float, default=400.0)
    parser.add_argument('--max_agent', type=int, default=6)

    # ── Error-augmented reward args ───────────────────────────────────────
    parser.add_argument('--beta', type=float, default=None,
                        help="Weight for cost in reward (if None, uses cost_rate)")
    parser.add_argument('--gamma', type=float, default=0.1,
                        help="Weight for error penalty in reward")
    parser.add_argument('--eval_llm', type=str, default='gpt-4o-mini',
                        help="LLM model name for error evaluation")
    parser.add_argument('--source_multiplier', type=float, default=1.0,
                        help="Reward multiplier for source errors")
    parser.add_argument('--propagated_multiplier', type=float, default=0.3,
                        help="Reward multiplier for propagated errors")
    parser.add_argument('--tracker_save_path', type=str,
                        default='logs/robustness_tracker_mbpp.json',
                        help="Path to save/load robustness tracker state")

    # ── Response logging ──────────────────────────────────────────────────
    parser.add_argument('--response_log', type=str, default=None,
                        help="Path to JSONL file for per-sample response logging. "
                             "Defaults to logs/mbpp_error_responses_<timestamp>.jsonl")
    parser.add_argument('--no_response_log', action='store_true',
                        help="Disable per-sample response logging entirely")

    args = parser.parse_args()
    if args.beta is None:
        args.beta = args.cost_rate
    return args


def append_response_log(path: str, record: dict):
    """Append one JSON record (one line) to the response log file.
    Safe to call after each sample — a crash loses nothing already written."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    args = parse_args()
    fix_random_seed(1234)

    train_dataset = MbppDataset('train')
    test_dataset = MbppDataset('test')

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = f"mbpp_error_{current_time}.txt"
    configure_logging(log_name=log_file)

    # ── Response log file ─────────────────────────────────────────────────
    if args.no_response_log:
        response_log_path = None
    else:
        response_log_path = (
            args.response_log
            or f"logs/mbpp_error_responses_{current_time}.jsonl"
        )
        logger.info(f"Response log: {response_log_path}")

    # ── Initialize components ──────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tracker = RobustnessTracker()
    if os.path.exists(args.tracker_save_path):
        tracker.load(args.tracker_save_path)
        logger.info(f"Loaded robustness tracker from {args.tracker_save_path}")

    router = ErrorAwareMasRouter(
        max_agent=args.max_agent,
        device=device,
        robustness_tracker=tracker,
    ).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=args.lr)

    # Error evaluator (LLM-based, MAST codebook)
    evaluator = LLMErrorEvaluator(llm_name=args.eval_llm)
    reward_computer = ErrorRewardComputer(
        source_multiplier=args.source_multiplier,
        propagated_multiplier=args.propagated_multiplier,
    )

    tasks = tasks_profile
    llms = llm_profile
    reasonings = reasoning_profile

    logger.info("=" * 80)
    logger.info("Error-Augmented MasRouter Training on MBPP")
    logger.info(f"  beta={args.beta}, gamma={args.gamma}")
    logger.info(f"  eval_llm={args.eval_llm}")
    logger.info(f"  source_multiplier={args.source_multiplier}, "
                f"propagated_multiplier={args.propagated_multiplier}")
    logger.info("=" * 80)

    # ══════════════════════════════════════════════════════════════════════════
    #  TRAINING
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Start training...")

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch} " + 80 * '-')
        total_solved, total_executed = 0, 0
        epoch_errors_total = 0
        epoch_error_details = {
            'm1_task_spec': 0.0, 'm2_communication': 0.0,
            'm3_plan_verify': 0.0, 'm4_reasoning': 0.0,
            'm5_infra': 0.0, 'tool_errors': 0.0, 'user_errors': 0.0,
        }

        train_loader = MbppDataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )

        if epoch < args.start_epoch:
            router.load_state_dict(torch.load(
                f"mbpp_error_router_epoch{epoch}.pth",
                map_location=device
            ))
            continue

        for i_batch, current_batch in enumerate(train_loader):
            logger.info(f"Batch {i_batch} " + 80 * '-')
            start_ts = time.time()

            queries = [item['task'] for item in current_batch]
            tests = [item['test_list'] for item in current_batch]
            task_labels = [2 for _ in current_batch]  # 2 = Code
            tasks_y = torch.tensor(task_labels).to(device)

            optimizer.zero_grad()

            # ── Forward pass (with trace collection) ───────────────────────
            results, costs, log_probs, tasks_probs, vae_loss, agents_num, \
                traces, per_agent_info = router.forward(
                    queries, tasks, llms, reasonings,
                    task_labels, prompt_file=args.prompt_file,
                )
            collab_log_probs = per_agent_info['collab_log_probs']
            role_lps_per_pos = per_agent_info['role_lps_per_pos']
            llm_lps_per_pos = per_agent_info['llm_lps_per_pos']
            role_names_pq = per_agent_info['role_names_per_query']

            task_loss = F.cross_entropy(tasks_probs, tasks_y)

            # ── Per-sample reward computation ──────────────────────────────
            utilities = []
            answers_loss = []
            is_solved_list = []
            batch_error_penalties = []
            batch_raw_utilities = []  # (raw_global_u, [raw_agent_u_j, ...]) per sample
            pattern = r'```python.*```'

            for sample_i, (query, result, test, cost, trace) in enumerate(
                zip(queries, results, tests, costs, traces)
            ):
                collab_lp = collab_log_probs[sample_i]
                role_lps = role_lps_per_pos[sample_i]
                llm_lps = llm_lps_per_pos[sample_i]
                role_names = role_names_pq[sample_i]

                # Task success
                match = re.search(pattern, result, re.DOTALL | re.MULTILINE)
                if match:
                    answer = match.group(0).lstrip("```python\n").rstrip("\n```")
                    is_solved, _, _ = PyExecutor().execute(answer, test, timeout=100)
                else:
                    is_solved = 0
                    answer = ""

                total_solved += is_solved
                total_executed += 1
                is_solved_list.append(is_solved)

                # ── Error evaluation (LLM-based, MAST codebook) ────────────
                errors, propagation = evaluator.evaluate(
                    trace=trace,
                    query=query,
                    ground_truth="[test cases]",
                    final_answer=str(result)[:200],
                    is_solved=bool(is_solved),
                    task_domain='Code',
                )

                num_steps = len(trace)
                error_penalty = reward_computer.compute_penalty(
                    errors, propagation, num_steps=num_steps
                )
                batch_error_penalties.append(error_penalty)

                # Per-agent penalty decomposition
                per_agent_penalties = reward_computer.compute_per_agent_penalty(
                    errors, propagation, trace
                )

                details = reward_computer.compute_detailed_penalty(
                    errors, propagation, num_steps=num_steps
                )
                epoch_errors_total += details['error_count']
                for cat in epoch_error_details:
                    epoch_error_details[cat] += details.get(cat, 0.0)

                # Update robustness tracker
                tracker.update(trace, errors, propagation)

                # ── Persist LLM responses for post-experiment diagnosis ────
                if response_log_path:
                    append_response_log(response_log_path, {
                        'phase': 'train',
                        'epoch': epoch,
                        'batch': i_batch,
                        'sample_in_batch': sample_i,
                        'global_sample_idx': total_executed,
                        'query': query,
                        'is_solved': bool(is_solved),
                        'cost': cost,
                        'error_penalty': error_penalty,
                        'trace': trace,
                        'errors': errors,
                        'propagation': propagation,
                    })

                # ── Error-augmented reward ─────────────────────────────────
                global_utility = (is_solved
                                  - args.beta * cost
                                  - args.gamma * error_penalty)
                utility = global_utility
                utilities.append(utility)

                # Store raw per-agent utilities for batch normalization (loss deferred)
                num_agents = min(len(role_lps), len(llm_lps))
                raw_agent_utilities = []
                for j in range(num_agents):
                    agent_name = role_names[j] if j < len(role_names) else ''
                    agent_penalty = per_agent_penalties.get(agent_name, 0.0)
                    raw_agent_utilities.append(
                        is_solved - args.beta * cost - args.gamma * agent_penalty
                    )
                batch_raw_utilities.append((global_utility, raw_agent_utilities))

            # ── Batch advantage normalization ──────────────────────────────
            # Collect all utility values (global + per-agent across all samples),
            # then normalise to zero-mean / unit-variance before computing gradients.
            # Applied identically to the baseline script for a fair comparison.
            all_raw = [u for g, ags in batch_raw_utilities for u in [g] + ags]
            u_mean = sum(all_raw) / len(all_raw)
            u_std = (sum((u - u_mean) ** 2 for u in all_raw) / len(all_raw)) ** 0.5

            for sample_i, (raw_global_u, raw_agent_us) in enumerate(batch_raw_utilities):
                collab_lp  = collab_log_probs[sample_i]
                role_lps   = role_lps_per_pos[sample_i]
                llm_lps    = llm_lps_per_pos[sample_i]

                norm_global_u = (raw_global_u - u_mean) / (u_std + 1e-8)
                sample_loss   = -collab_lp * norm_global_u

                for j, raw_agent_u in enumerate(raw_agent_us):
                    norm_agent_u = (raw_agent_u - u_mean) / (u_std + 1e-8)
                    sample_loss  = sample_loss - (role_lps[j] + llm_lps[j]) * norm_agent_u

                answers_loss.append(sample_loss)

            # ── Loss computation and backprop ──────────────────────────────
            answer_loss = torch.stack(answers_loss).sum() / len(answers_loss)
            vae_loss = vae_loss.mean()

            is_solved_tensor = torch.tensor(
                is_solved_list, dtype=torch.float32, device=device
            ).unsqueeze(1)
            adjust_loss = (
                (1 - is_solved_tensor) * (router.num_determiner.max_agent - agents_num)
                + 0.25 * is_solved_tensor * agents_num
            ).mean()

            loss = task_loss + answer_loss + vae_loss * 0.001
            loss.backward()
            optimizer.step()

            accuracy = total_solved / total_executed
            avg_error_penalty = (
                sum(batch_error_penalties) / len(batch_error_penalties)
                if batch_error_penalties else 0.0
            )

            logger.info(f"Batch time: {time.time() - start_ts:.3f}s")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Avg error penalty: {avg_error_penalty:.4f}")
            logger.info(f"Utilities: {utilities}")

        # ── End of epoch ───────────────────────────────────────────────────
        logger.info(f"Epoch {epoch} finished " + 80 * '-')
        logger.info(f"  Total errors detected: {epoch_errors_total}")
        logger.info(f"  Error breakdown: {epoch_error_details}")
        logger.info(f"  Robustness tracker:\n{tracker.summary()}")

        torch.save(router.state_dict(), f"mbpp_error_router_epoch{epoch}.pth")
        tracker.save(args.tracker_save_path)

    logger.info("End training...")

    # ══════════════════════════════════════════════════════════════════════════
    #  TESTING
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Start testing...")
    total_solved, total_executed = 0, 0
    test_error_count = 0
    test_error_details = {
        'm1_task_spec': 0.0, 'm2_communication': 0.0,
        'm3_plan_verify': 0.0, 'm4_reasoning': 0.0,
        'm5_infra': 0.0, 'tool_errors': 0.0, 'user_errors': 0.0,
    }

    test_loader = MbppDataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True
    )

    for i_batch, current_batch in enumerate(test_loader):
        logger.info(f"Test Batch {i_batch} " + 80 * '-')
        start_ts = time.time()

        queries = [item['task'] for item in current_batch]
        tests = [item['test_list'] for item in current_batch]
        task_labels = [2 for _ in current_batch]
        tasks_y = torch.tensor(task_labels).to(device)

        results, costs, log_probs, tasks_probs, vae_loss, agents_num, \
            traces, per_agent_info = router.forward(
                queries, tasks, llms, reasonings,
                task_labels, prompt_file=args.prompt_file,
            )

        utilities = []
        pattern = r'```python.*```'
        for query, result, test, cost, trace in zip(
            queries, results, tests, costs, traces
        ):
            match = re.search(pattern, result, re.DOTALL | re.MULTILINE)
            if match:
                answer = match.group(0).lstrip("```python\n").rstrip("\n```")
                is_solved, _, _ = PyExecutor().execute(answer, test, timeout=100)
            else:
                is_solved = 0

            total_solved += is_solved
            total_executed += 1

            # Error evaluation (LLM-based, MAST codebook)
            errors, propagation = evaluator.evaluate(
                trace=trace,
                query=query,
                ground_truth="[test cases]",
                final_answer=str(result)[:200],
                is_solved=bool(is_solved),
                task_domain='Code',
            )

            num_steps = len(trace)
            error_penalty = reward_computer.compute_penalty(
                errors, propagation, num_steps=num_steps
            )
            details = reward_computer.compute_detailed_penalty(
                errors, propagation, num_steps=num_steps
            )
            test_error_count += details['error_count']
            for cat in test_error_details:
                test_error_details[cat] += details.get(cat, 0.0)

            # ── Persist LLM responses for post-experiment diagnosis ────────
            if response_log_path:
                append_response_log(response_log_path, {
                    'phase': 'test',
                    'batch': i_batch,
                    'global_sample_idx': total_executed,
                    'query': query,
                    'is_solved': bool(is_solved),
                    'cost': cost,
                    'error_penalty': error_penalty,
                    'trace': trace,
                    'errors': errors,
                    'propagation': propagation,
                })

            utility = (is_solved
                       - args.beta * cost
                       - args.gamma * error_penalty)
            utilities.append(utility)

        accuracy = total_solved / total_executed
        logger.info(f"Batch time: {time.time() - start_ts:.3f}s")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Utilities: {utilities}")

    logger.info("=" * 80)
    logger.info(f"Final Test Accuracy: {total_solved / max(total_executed, 1):.4f}")
    logger.info(f"Total test errors: {test_error_count}")
    logger.info(f"Test error breakdown: {test_error_details}")
    logger.info("End testing...")
