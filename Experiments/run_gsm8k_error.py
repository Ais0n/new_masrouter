"""
Error-augmented MasRouter training on GSM8K.

Compared to the baseline run_gsm8k.py, this script:
  1. Uses ErrorAwareMasRouter with robustness-augmented embeddings
  2. Runs an LLM-based error evaluator (MAST codebook, 28 error types)
  3. Adds an error penalty to the REINFORCE reward:
       utility = is_solved − β·cost − γ·error_penalty
  4. Updates the RobustnessTracker after each episode
  5. Logs per-category error statistics for analysis

Usage:
    python Experiments/run_gsm8k_error.py --gamma 0.1
    python Experiments/run_gsm8k_error.py --gamma 0.0   # ablation: no error penalty
    python Experiments/run_gsm8k_error.py --gamma 0.1 --verbose --max_train_samples 4
"""

import sys
import os
import io
import argparse
import yaml
import json
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
from MAR.Tools.reader.readers import JSONLReader
from MAR.Utils.utils import fix_random_seed, split_list
from MAR.Utils.globals import Cost, PromptTokens, CompletionTokens
from MAR.Utils.log import configure_logging
from MAR.ErrorAnalysis.robustness_tracker import RobustnessTracker
from MAR.ErrorAnalysis.error_evaluator import LLMErrorEvaluator
from MAR.ErrorAnalysis.error_reward import ErrorRewardComputer
from MAR.Utils.llm_call_logger import LLMCallLogger
from Datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch * batch_size : i_batch * batch_size + batch_size]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Error-augmented MasRouter on GSM8K"
    )
    # ── Dataset / model args (same as baseline) ───────────────────────────
    parser.add_argument("--dataset_json", type=str,
                        default="datasets/gsm8k/gsm8k.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--domain', type=str, default="gsm8k")
    parser.add_argument('--decision_method', type=str, default='FinalRefer')
    parser.add_argument('--prompt_file', type=str,
                        default='MAR/Roles/FinalNode/gsm8k.json')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--cost_rate', type=float, default=200.0)
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
                        default='logs/robustness_tracker_gsm8k.json',
                        help="Path to save/load robustness tracker state")

    # ── Response logging ──────────────────────────────────────────────────
    parser.add_argument('--response_log', type=str, default=None,
                        help="Path to JSONL file for per-sample response logging. "
                             "Defaults to logs/gsm8k_error_responses_<timestamp>.jsonl")
    parser.add_argument('--no_response_log', action='store_true',
                        help="Disable per-sample response logging entirely")

    # ── Debug / quick-run args ────────────────────────────────────────────
    parser.add_argument('--verbose', action='store_true',
                        help="Print detailed per-sample traces, errors, and "
                             "penalty computations")
    parser.add_argument('--max_train_samples', type=int, default=0,
                        help="Limit training set size (0=use all). "
                             "Useful for quick sanity checks.")
    parser.add_argument('--max_test_samples', type=int, default=0,
                        help="Limit test set size (0=use all).")

    args = parser.parse_args()
    if args.beta is None:
        args.beta = args.cost_rate  # backward-compatible default
    return args


def append_response_log(path: str, record: dict):
    """Append one JSON record (one line) to the response log file.
    Safe to call after each sample — a crash loses nothing already written."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def log_trace_and_errors(sample_idx, query, true_answer, result, trace,
                         errors, propagation, details, error_penalty,
                         utility, is_solved):
    """Pretty-print trace, errors, propagation and reward for one sample."""
    sep = '┈' * 70
    logger.info(f'\n{"═" * 70}')
    logger.info(f'  SAMPLE {sample_idx} DETAILS')
    logger.info(f'{"═" * 70}')
    logger.info(f'  Query:   {query[:120]}...' if len(query) > 120 else f'  Query:   {query}')
    logger.info(f'  Answer:  {true_answer}   |  Predicted: {result[:80]}...' if len(str(result)) > 80 else f'  Answer:  {true_answer}   |  Predicted: {result}')
    logger.info(f'  Solved:  {bool(is_solved)}')

    # ── Trace ──
    logger.info(f'\n  ── Collected Trace ({len(trace)} steps) ──')
    for step in trace:
        pred_str = ', '.join(step.get('spatial_predecessors', [])) or 'none'
        resp_preview = str(step.get('response', ''))[:100].replace('\n', ' ')
        logger.info(
            f'    Step {step["step_id"]:>2d} │ round={step.get("round", "?")}, '
            f'node={step["node_id"]}, role={step.get("role", "?")}, '
            f'llm={step.get("llm_name", "?")}, predecessors=[{pred_str}]'
        )
        logger.info(f'           │ response: {resp_preview}' + ('...' if len(str(step.get('response', ''))) > 100 else ''))

    # ── Detected errors ──
    logger.info(f'\n  ── Detected Errors ({len(errors)}) ──')
    if not errors:
        logger.info(f'    (none)')
    for e in errors:
        logger.info(
            f'    {e["error_id"]:>3s} │ type={e["error_type"]}, '
            f'steps={e["error_start_step"]}-{e["error_end_step"]}, '
            f'role={e.get("agent_role", "?")}, llm={e.get("agent_llm", "?")}')
        logger.info(f'         │ {e["explanation"][:120]}')

    # ── Propagation ──
    traj = propagation.get('trajectory', [])
    logger.info(f'\n  ── Error Propagation ({len(traj)} edges) ──')
    if not traj:
        logger.info(f'    (none)')
    for edge in traj:
        logger.info(
            f'    {edge["from"]} → {edge["to"]}  │ {edge["explanation"][:100]}')

    # ── Penalty computation ──
    logger.info(f'\n  ── Reward Computation ──')
    logger.info(f'    error_count      = {details["error_count"]}')
    logger.info(f'    source_count     = {details["source_count"]}')
    logger.info(f'    propagated_count = {details["propagated_count"]}')
    logger.info(f'    standalone_count = {details["standalone_count"]}')
    logger.info(f'    penalty_total    = {details["total"]:.4f}')
    logger.info(f'      M1_task_spec   = {details["m1_task_spec"]:.4f}, '
                f'M2_comm      = {details["m2_communication"]:.4f}, '
                f'M3_plan_ver  = {details["m3_plan_verify"]:.4f}')
    logger.info(f'      M4_reasoning   = {details["m4_reasoning"]:.4f}, '
                f'M5_infra     = {details["m5_infra"]:.4f}')
    logger.info(f'      tool_errors    = {details["tool_errors"]:.4f}, '
                f'user_errors  = {details["user_errors"]:.4f}')
    logger.info(f'    error_penalty    = {error_penalty:.4f}')
    logger.info(f'    utility = {bool(is_solved)} − β·cost − γ·{error_penalty:.4f} = {utility:.4f}')
    logger.info(f'{sep}')


if __name__ == '__main__':
    args = parse_args()
    fix_random_seed(1234)

    dataset = JSONLReader().parse_file("Datasets/gsm8k/gsm8k.jsonl")
    dataset = gsm_data_process(dataset)
    train_dataset, test_dataset = split_list(dataset, 0.2)

    # Optionally limit dataset sizes for quick sanity checks
    if args.max_train_samples > 0:
        train_dataset = train_dataset[:args.max_train_samples]
    if args.max_test_samples > 0:
        test_dataset = test_dataset[:args.max_test_samples]

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    log_file = f"gsm8k_error_{current_time}.txt"
    configure_logging(log_name=log_file)

    # ── Response log file ──────────────────────────────────────────────────
    if args.no_response_log:
        response_log_path = None
    else:
        response_log_path = (
            args.response_log
            or f"logs/gsm8k_error_responses_{current_time}.jsonl"
        )
        logger.info(f"Response log: {response_log_path}")

    # ── Initialize components ──────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Robustness tracker (load existing if available)
    tracker = RobustnessTracker()
    if os.path.exists(args.tracker_save_path):
        tracker.load(args.tracker_save_path)
        logger.info(f"Loaded robustness tracker from {args.tracker_save_path}")

    # Error-aware router
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

    # LLM call logger (active in verbose mode)
    llm_logger = LLMCallLogger()
    if args.verbose:
        llm_logger.enable()
        logger.info("LLM call logging enabled (--verbose)")

    tasks = tasks_profile
    llms = llm_profile
    reasonings = reasoning_profile

    # ── Logging configuration ──────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("Error-Augmented MasRouter Training on GSM8K")
    logger.info(f"  beta={args.beta}, gamma={args.gamma}")
    logger.info(f"  eval_llm={args.eval_llm}")
    logger.info(f"  source_multiplier={args.source_multiplier}, "
                f"propagated_multiplier={args.propagated_multiplier}")
    logger.info(f"  verbose={args.verbose}")
    logger.info(f"  train_samples={len(train_dataset)}, test_samples={len(test_dataset)}")
    logger.info(f"  train_batches={len(train_dataset)//args.batch_size}, "
                f"test_batches={len(test_dataset)//args.batch_size}")
    logger.info(f"  Estimated time per sample: ~5 min (LLM API bound)")
    est_train = len(train_dataset) * args.epochs * 5 / 60
    est_test = len(test_dataset) * 5 / 60
    logger.info(f"  Estimated total: ~{est_train:.1f}h training + ~{est_test:.1f}h testing")
    logger.info("=" * 80)

    # ══════════════════════════════════════════════════════════════════════════
    #  TRAINING
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Start training...")
    train_batches = int(len(train_dataset) / args.batch_size)

    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch} " + 80 * '-')
        total_solved, total_executed = 0, 0
        epoch_errors_total = 0
        epoch_error_details = {
            'm1_task_spec': 0.0, 'm2_communication': 0.0,
            'm3_plan_verify': 0.0, 'm4_reasoning': 0.0,
            'm5_infra': 0.0, 'tool_errors': 0.0, 'user_errors': 0.0,
        }

        if epoch < args.start_epoch:
            router.load_state_dict(torch.load(
                f"gsm8k_error_router_epoch{epoch}.pth",
                map_location=device
            ))
            continue

        for i_batch in range(train_batches):
            logger.info(f"Batch {i_batch} " + 80 * '-')
            start_ts = time.time()

            current_batch = dataloader(train_dataset, args.batch_size, i_batch)
            queries = [item['task'] for item in current_batch]
            answers = [item['answer'] for item in current_batch]
            task_labels = [0 for _ in current_batch]  # 0 = Math
            tasks_y = torch.tensor(task_labels).to(device)

            optimizer.zero_grad()

            # Reset LLM call logger for this batch
            if args.verbose:
                llm_logger.reset_sample()

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

            # Print batch-level LLM call summary right after forward pass
            if args.verbose:
                llm_logger.print_sample_summary(
                    f"Train Epoch {epoch} Batch {i_batch} "
                    f"({len(current_batch)} samples)"
                )

            task_loss = F.cross_entropy(tasks_probs, tasks_y)

            # ── Per-sample reward computation ──────────────────────────────
            utilities = []
            answers_loss = []
            is_solved_list = []
            batch_error_penalties = []
            batch_raw_utilities = []  # (raw_global_u, [raw_agent_u_j, ...]) per sample

            for sample_i, (query, result, true_answer, cost, trace) in enumerate(
                zip(queries, results, answers, costs, traces)
            ):
                collab_lp = collab_log_probs[sample_i]  # [1] tensor
                role_lps = role_lps_per_pos[sample_i]    # List[Tensor]
                llm_lps = llm_lps_per_pos[sample_i]      # List[Tensor]
                role_names = role_names_pq[sample_i]      # List[str]
                # Task success
                predict_answer = gsm_get_predict(result)
                is_solved = float(predict_answer) == float(true_answer)
                total_solved += is_solved
                total_executed += 1
                is_solved_list.append(is_solved)

                # ── Error evaluation (LLM-based, MAST codebook) ────────────
                errors, propagation = evaluator.evaluate(
                    trace=trace,
                    query=query,
                    ground_truth=str(true_answer),
                    final_answer=str(result),
                    is_solved=bool(is_solved),
                    task_domain='Math',
                )

                # Compute error penalty (global, for logging / collab utility)
                num_steps = len(trace)
                error_penalty = reward_computer.compute_penalty(
                    errors, propagation, num_steps=num_steps
                )
                batch_error_penalties.append(error_penalty)

                # Per-agent penalty decomposition
                per_agent_penalties = reward_computer.compute_per_agent_penalty(
                    errors, propagation, trace
                )

                # Detailed breakdown for logging
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
                        'true_answer': str(true_answer),
                        'predicted_answer': str(result),
                        'is_solved': bool(is_solved),
                        'cost': cost,
                        'error_penalty': error_penalty,
                        'trace': trace,
                        'errors': errors,
                        'propagation': propagation,
                    })

                # ── Error-augmented reward ─────────────────────────────────
                # Global utility for collab decision (sees all errors)
                global_utility = (is_solved
                                  - args.beta * cost
                                  - args.gamma * error_penalty)
                utility = global_utility  # for logging
                utilities.append(utility)

                # ── Verbose per-sample logging ─────────────────────────────
                if args.verbose:
                    log_trace_and_errors(
                        sample_idx=total_executed,
                        query=query,
                        true_answer=true_answer,
                        result=str(result),
                        trace=trace,
                        errors=errors,
                        propagation=propagation,
                        details=details,
                        error_penalty=error_penalty,
                        utility=utility,
                        is_solved=is_solved,
                    )

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
            if args.verbose:
                batch_calls = llm_logger.get_sample_calls()
                if batch_calls:
                    avg_call_time = sum(c['elapsed'] for c in batch_calls) / len(batch_calls)
                    logger.info(
                        f"LLM calls this batch: {len(batch_calls)}, "
                        f"avg {avg_call_time:.1f}s/call, "
                        f"total {sum(c['elapsed'] for c in batch_calls):.1f}s"
                    )

        # ── End of epoch ───────────────────────────────────────────────────
        logger.info(f"Epoch {epoch} finished " + 80 * '-')
        logger.info(f"  Total errors detected: {epoch_errors_total}")
        logger.info(f"  Error breakdown: {epoch_error_details}")
        logger.info(f"  Robustness tracker:\n{tracker.summary()}")

        torch.save(router.state_dict(), f"gsm8k_error_router_epoch{epoch}.pth")
        tracker.save(args.tracker_save_path)

        if args.verbose:
            llm_logger.print_total_summary()

    logger.info("Finish training...")

    # ══════════════════════════════════════════════════════════════════════════
    #  TESTING
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("Start testing...")
    test_batches = int(len(test_dataset) / args.batch_size)
    total_solved, total_executed = 0, 0
    test_error_count = 0
    test_error_details = {
        'm1_task_spec': 0.0, 'm2_communication': 0.0,
        'm3_plan_verify': 0.0, 'm4_reasoning': 0.0,
        'm5_infra': 0.0, 'tool_errors': 0.0, 'user_errors': 0.0,
    }

    for i_batch in range(test_batches):
        logger.info(f"Test Batch {i_batch} " + 80 * '-')
        start_ts = time.time()

        current_batch = dataloader(test_dataset, args.batch_size, i_batch)
        queries = [item['task'] for item in current_batch]
        answers = [item['answer'] for item in current_batch]
        task_labels = [0 for _ in current_batch]
        tasks_y = torch.tensor(task_labels).to(device)

        if args.verbose:
            llm_logger.reset_sample()

        results, costs, log_probs, tasks_probs, vae_loss, agents_num, \
            traces, per_agent_info = router.forward(
                queries, tasks, llms, reasonings,
                task_labels, prompt_file=args.prompt_file,
            )

        if args.verbose:
            llm_logger.print_sample_summary(
                f"Test Batch {i_batch} ({len(current_batch)} samples)"
            )

        utilities = []
        for query, result, true_answer, cost, trace in zip(
            queries, results, answers, costs, traces
        ):
            predict_answer = gsm_get_predict(result)
            is_solved = float(predict_answer) == float(true_answer)
            total_solved += is_solved
            total_executed += 1

            # Error evaluation (LLM-based, MAST codebook)
            errors, propagation = evaluator.evaluate(
                trace=trace,
                query=query,
                ground_truth=str(true_answer),
                final_answer=str(result),
                is_solved=bool(is_solved),
                task_domain='Math',
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
                    'true_answer': str(true_answer),
                    'predicted_answer': str(result),
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

            if args.verbose:
                log_trace_and_errors(
                    sample_idx=total_executed,
                    query=query,
                    true_answer=str(true_answer),
                    result=str(result),
                    trace=trace,
                    errors=errors,
                    propagation=propagation,
                    details=details,
                    error_penalty=error_penalty,
                    utility=utility,
                    is_solved=is_solved,
                )

        accuracy = total_solved / total_executed
        logger.info(f"Batch time: {time.time() - start_ts:.3f}s")
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info(f"Utilities: {utilities}")
        if args.verbose:
            batch_calls = llm_logger.get_sample_calls()
            if batch_calls:
                avg_call_time = sum(c['elapsed'] for c in batch_calls) / len(batch_calls)
                logger.info(
                    f"LLM calls this batch: {len(batch_calls)}, "
                    f"avg {avg_call_time:.1f}s/call, "
                    f"total {sum(c['elapsed'] for c in batch_calls):.1f}s"
                )

    logger.info("=" * 80)
    logger.info(f"Final Test Accuracy: {total_solved / max(total_executed, 1):.4f}")
    logger.info(f"Total test errors: {test_error_count}")
    logger.info(f"Test error breakdown: {test_error_details}")
    if args.verbose:
        llm_logger.print_total_summary()
    logger.info("Finish testing...")
