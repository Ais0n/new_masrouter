"""
Error-based Reward Computer for MasRouter training.

Computes error penalties from detected errors and their propagation,
which are then integrated into the REINFORCE reward signal:

    utility = is_solved  −  β · cost  −  γ · error_penalty

Supports two modes:
  - Global penalty:     compute_penalty()  → single scalar for the whole query
  - Per-agent penalty:  compute_per_agent_penalty()  → one scalar per agent role,
                        enabling counterfactual credit assignment so each
                        role/LLM selection receives its own gradient signal.

The per-error contribution is:
    severity(type)  ×  propagation_multiplier  ×  span_factor
where:
    span_factor = 1.0 + 0.1 × (end_step − start_step) / max(num_steps − 1, 1)

span_factor is normalized by trace length so it is always in [1.0, 1.1],
making penalties comparable across traces of different sizes.
"""

from __future__ import annotations
from typing import Dict, List, Optional

from MAR.ErrorAnalysis.error_taxonomy import (
    ERROR_CODE_TO_IDX, DEFAULT_ERROR_WEIGHTS,
    M1_TASK_SPEC_ERRORS, M2_COMMUNICATION_ERRORS, M3_PLAN_VERIFY_ERRORS,
    M4_REASONING_ERRORS, M5_INFRA_ERRORS, TOOL_ERRORS, USER_ERRORS,
)


class ErrorRewardComputer:
    """Compute error-based reward penalties from error evaluation results."""

    def __init__(self,
                 error_weights: Optional[Dict[str, float]] = None,
                 source_multiplier: float = 1.0,
                 propagated_multiplier: float = 0.3):
        """
        Args:
            error_weights: Per-error-type severity weights.
                           Defaults to DEFAULT_ERROR_WEIGHTS from taxonomy.
            source_multiplier: Weight multiplier for source errors
                               (errors that cause other errors).
            propagated_multiplier: Weight multiplier for propagated errors
                                   (errors caused by other errors). Lower than
                                   source_multiplier to focus penalty on root causes.
        """
        self.error_weights = error_weights or DEFAULT_ERROR_WEIGHTS.copy()
        self.source_multiplier = source_multiplier
        self.propagated_multiplier = propagated_multiplier

    # ── Shared helper ──────────────────────────────────────────────────────

    def _error_contribution(self, error: dict,
                            source_ids: set, propagated_ids: set,
                            num_steps: int = 1) -> float:
        """Compute a single error's weighted penalty contribution.

        Args:
            error:          Error dict from the evaluator.
            source_ids:     Set of error IDs that are propagation sources.
            propagated_ids: Set of error IDs that are propagation targets.
            num_steps:      Total trace steps (for span normalisation).
        """
        error_id = error.get('error_id', '')
        error_type_str = error.get('error_type', '')
        error_code = error_type_str.split()[0] if error_type_str else ''

        base_weight = self.error_weights.get(error_code, 1.0)

        if error_id in source_ids:
            multiplier = self.source_multiplier
        elif error_id in propagated_ids:
            multiplier = self.propagated_multiplier
        else:
            multiplier = 1.0

        start = error.get('error_start_step', 0)
        end = error.get('error_end_step', start)
        span_norm = max(0, end - start) / max(num_steps - 1, 1)
        span_factor = 1.0 + 0.1 * span_norm

        return base_weight * multiplier * span_factor

    @staticmethod
    def _propagation_sets(propagation: dict):
        """Extract source / propagated error-id sets from a propagation dict."""
        source_ids: set = set()
        propagated_ids: set = set()
        if propagation and 'trajectory' in propagation:
            for edge in propagation['trajectory']:
                source_ids.add(edge['from'])
                propagated_ids.add(edge['to'])
        return source_ids, propagated_ids

    # ── Global penalty ─────────────────────────────────────────────────────

    def compute_penalty(self, errors: List[dict],
                        propagation: dict,
                        num_steps: int = 1) -> float:
        """
        Compute a scalar error penalty from detected errors and propagation.

        Args:
            errors:      List of error dicts from an error evaluator.
            propagation: Propagation dict with 'trajectory' key.
            num_steps:   Total steps in the execution trace (for span
                         normalisation).  Pass ``len(trace)``.

        Returns:
            Scalar penalty value (higher = more errors, worse performance).
        """
        if not errors:
            return 0.0

        source_ids, propagated_ids = self._propagation_sets(propagation)

        penalty = 0.0
        for error in errors:
            penalty += self._error_contribution(
                error, source_ids, propagated_ids, num_steps,
            )
        return penalty

    # ── Per-agent decomposition (counterfactual credit assignment) ─────────

    def compute_per_agent_penalty(
        self,
        errors: List[dict],
        propagation: dict,
        trace: List[dict],
    ) -> Dict[str, float]:
        """
        Compute error penalty attributed to each agent role.

        Uses the ``agent_role`` field set by the evaluator (falls back to
        trace step→role mapping).  Errors that cannot be attributed are
        spread evenly across all agents.

        Args:
            errors:      List of error dicts (with ``agent_role`` if available).
            propagation: Propagation dict with ``trajectory`` key.
            trace:       Step-level execution trace from Graph.

        Returns:
            Dict mapping **role name** → cumulative penalty for that agent.
            Only roles that appear in the trace are included.
        """
        # Collect unique roles from the trace (excludes FinalNode etc.)
        step_to_role: Dict[int, str] = {}
        all_roles: list = []          # preserves insertion order
        seen_roles: set = set()
        for step in trace:
            role = step.get('role', '')
            if role:
                step_to_role[step['step_id']] = role
                if role not in seen_roles:
                    all_roles.append(role)
                    seen_roles.add(role)

        per_agent: Dict[str, float] = {r: 0.0 for r in all_roles}

        if not errors or not all_roles:
            return per_agent

        source_ids, propagated_ids = self._propagation_sets(propagation)
        num_steps = len(trace)

        for error in errors:
            contribution = self._error_contribution(
                error, source_ids, propagated_ids, num_steps,
            )

            # Attribute to agent via evaluator annotation or step→role map
            role = (error.get('agent_role', '')
                    or step_to_role.get(error.get('error_start_step', -1), ''))
            if role and role in per_agent:
                per_agent[role] += contribution
            else:
                # Cannot attribute → spread evenly
                share = contribution / len(per_agent)
                for r in per_agent:
                    per_agent[r] += share

        return per_agent

    # ── Detailed category breakdown ────────────────────────────────────────

    def compute_detailed_penalty(self, errors: List[dict],
                                 propagation: dict,
                                 num_steps: int = 1) -> Dict[str, float]:
        """
        Like compute_penalty but returns a breakdown by error category.

        Args:
            errors:      List of error dicts from an error evaluator.
            propagation: Propagation dict with 'trajectory' key.
            num_steps:   Total steps in the execution trace (for span
                         normalisation).  Pass ``len(trace)``.

        Returns:
            Dict with keys: 'total', 'm1_task_spec', 'm2_communication',
            'm3_plan_verify', 'm4_reasoning', 'm5_infra', 'tool_errors',
            'user_errors', 'source_count', 'propagated_count',
            'standalone_count', 'error_count'.
        """
        result = {
            'total': 0.0,
            'm1_task_spec': 0.0,
            'm2_communication': 0.0,
            'm3_plan_verify': 0.0,
            'm4_reasoning': 0.0,
            'm5_infra': 0.0,
            'tool_errors': 0.0,
            'user_errors': 0.0,
            'source_count': 0,
            'propagated_count': 0,
            'standalone_count': 0,
            'error_count': len(errors),
        }
        if not errors:
            return result

        source_ids, propagated_ids = self._propagation_sets(propagation)

        # Category code sets for fast lookup
        _m1 = set(M1_TASK_SPEC_ERRORS)
        _m2 = set(M2_COMMUNICATION_ERRORS)
        _m3 = set(M3_PLAN_VERIFY_ERRORS)
        _m4 = set(M4_REASONING_ERRORS)
        _m5 = set(M5_INFRA_ERRORS)
        _tool = set(TOOL_ERRORS)
        _user = set(USER_ERRORS)

        for error in errors:
            error_id = error.get('error_id', '')
            error_type_str = error.get('error_type', '')
            error_code = error_type_str.split()[0] if error_type_str else ''

            # Count propagation role
            if error_id in source_ids:
                result['source_count'] += 1
            elif error_id in propagated_ids:
                result['propagated_count'] += 1
            else:
                result['standalone_count'] += 1

            contribution = self._error_contribution(
                error, source_ids, propagated_ids, num_steps,
            )
            result['total'] += contribution

            # Assign contribution to appropriate category
            if error_code in _m1:
                result['m1_task_spec'] += contribution
            elif error_code in _m2:
                result['m2_communication'] += contribution
            elif error_code in _m3:
                result['m3_plan_verify'] += contribution
            elif error_code in _m4:
                result['m4_reasoning'] += contribution
            elif error_code in _m5:
                result['m5_infra'] += contribution
            elif error_code in _tool:
                result['tool_errors'] += contribution
            elif error_code in _user:
                result['user_errors'] += contribution

        return result
