#!/usr/bin/env python
"""Quick smoke test for the error analysis modules."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MAR.ErrorAnalysis.robustness_tracker import RobustnessTracker
from MAR.ErrorAnalysis.error_evaluator import LLMErrorEvaluator
from MAR.ErrorAnalysis.error_reward import ErrorRewardComputer
from MAR.ErrorAnalysis.error_taxonomy import NUM_ERROR_TYPES
import torch

def test_tracker():
    tracker = RobustnessTracker()
    trace = [
        {'step_id': 0, 'round': 0, 'node_id': 'a1', 'role': 'MathSolver', 'llm_name': 'openai/gpt-4o-mini', 'response': 'test'},
        {'step_id': 1, 'round': 0, 'node_id': 'a2', 'role': 'Analyzer', 'llm_name': 'anthropic/claude-3.5-haiku', 'response': 'test2'},
    ]
    errors = [
        {'error_id': 'E1', 'error_start_step': 0, 'error_end_step': 0, 'error_type': 'M1.1 Disobey-Task-Specification'},
        {'error_id': 'E2', 'error_start_step': 1, 'error_end_step': 1, 'error_type': 'M2.5 Ignored-Other-Agents-Input'},
    ]
    propagation = {'trajectory': [{'from': 'E1', 'to': 'E2', 'explanation': 'test'}]}
    tracker.update(trace, errors, propagation)
    rob = tracker.get_role_robustness('MathSolver')
    assert rob.shape == (NUM_ERROR_TYPES,), f"Expected shape ({NUM_ERROR_TYPES},), got {rob.shape}"
    assert rob[0] > 0, "M1.1 error rate should be > 0 for MathSolver"
    print("[PASS] RobustnessTracker")
    print(f"  MathSolver robustness: {rob}")
    print(f"  Summary:\n{tracker.summary()}")

def test_llm_evaluator():
    evaluator = LLMErrorEvaluator(llm_name='gpt-4o-mini')
    # Test instantiation and formatting (no actual LLM call)
    trace = [
        {'step_id': 1, 'round': 0, 'node_id': 'a1', 'role': 'MathSolver', 'llm_name': 'gpt-4o-mini',
         'response': 'The answer is 42.', 'spatial_predecessors': []},
        {'step_id': 2, 'round': 0, 'node_id': 'a2', 'role': 'Checker', 'llm_name': 'claude-3.5-haiku',
         'response': 'I verify the answer is 42.', 'spatial_predecessors': ['a1']},
    ]
    formatted = evaluator._format_trace(trace)
    assert 'Step 1' in formatted
    assert 'Step 2' in formatted
    assert 'MathSolver' in formatted
    # Test JSON parsing
    test_json = '{"errors": [{"error_id": "E1", "error_type": "M1.1 Disobey-Task-Specification"}], "propagation": {"trajectory": []}}'
    parsed = evaluator._parse_json_response(test_json)
    assert len(parsed['errors']) == 1
    # Test attribution
    errors = [{'error_id': 'E1', 'error_start_step': 1, 'error_end_step': 1, 'error_type': 'M1.1 Disobey-Task-Specification'}]
    evaluator._add_agent_attribution(errors, trace)
    assert errors[0]['agent_role'] == 'MathSolver'
    assert errors[0]['agent_llm'] == 'gpt-4o-mini'
    print(f"\n[PASS] LLMErrorEvaluator (format/parse/attribution, no API call)")
    print(f"  Trace formatted: {len(formatted)} chars")
    print(f"  JSON parse OK, attribution OK")

def test_reward_computer():
    computer = ErrorRewardComputer()
    trace = [
        {'step_id': 0, 'round': 0, 'node_id': 'a1', 'role': 'MathSolver', 'llm_name': 'gpt-4o-mini', 'response': 'x'},
        {'step_id': 1, 'round': 0, 'node_id': 'a2', 'role': 'Checker', 'llm_name': 'claude-3.5-haiku', 'response': 'y'},
    ]
    errors = [
        {'error_id': 'E1', 'error_start_step': 0, 'error_end_step': 0,
         'error_type': 'M1.1 Disobey-Task-Specification', 'agent_role': 'MathSolver'},
        {'error_id': 'E2', 'error_start_step': 1, 'error_end_step': 1,
         'error_type': 'M3.1 Premature-Termination', 'agent_role': 'Checker'},
    ]
    propagation = {'trajectory': [{'from': 'E1', 'to': 'E2', 'explanation': 'caused'}]}

    # Global penalty
    num_steps = len(trace)
    penalty = computer.compute_penalty(errors, propagation, num_steps=num_steps)
    details = computer.compute_detailed_penalty(errors, propagation, num_steps=num_steps)
    assert penalty > 0, "Penalty should be positive when errors exist"

    # Per-agent penalty
    per_agent = computer.compute_per_agent_penalty(errors, propagation, trace)
    assert 'MathSolver' in per_agent, "MathSolver should have a penalty"
    assert 'Checker' in per_agent, "Checker should have a penalty"
    assert per_agent['MathSolver'] > per_agent['Checker'], \
        "Source error (MathSolver) should have higher penalty than propagated (Checker)"
    assert abs(sum(per_agent.values()) - penalty) < 1e-6, \
        "Per-agent penalties should sum to global penalty"

    print(f"\n[PASS] ErrorRewardComputer")
    print(f"  Global penalty: {penalty:.4f}")
    print(f"  Per-agent penalties: {per_agent}")
    print(f"  Details: {details}")

def test_augmentor():
    from MAR.MasRouter.mas_router_error import RobustnessAugmentor
    aug = RobustnessAugmentor(embedding_dim=384, num_error_types=NUM_ERROR_TYPES)
    emb = torch.randn(5, 384)
    rob = torch.randn(5, NUM_ERROR_TYPES)
    out = aug(emb, rob)
    assert out.shape == emb.shape, f"Expected shape {emb.shape}, got {out.shape}"
    print(f"\n[PASS] RobustnessAugmentor")
    print(f"  Input shape: {emb.shape}, Output shape: {out.shape}")

if __name__ == '__main__':
    test_tracker()
    test_llm_evaluator()
    test_reward_computer()
    test_augmentor()
    print("\n" + "="*60)
    print("All tests passed!")
