"""
Robustness Tracker – maintains per-role and per-LLM error statistics
from past MAS executions, and provides robustness vectors that the
ErrorAwareMasRouter can use as additional input features.

A robustness vector for an entity (role or LLM) has shape [NUM_ERROR_TYPES].
Each entry is the normalized error rate for that error type:

    robustness[i] = count_of_error_type_i / total_participations

Lower values indicate higher robustness.  A vector of all zeros means either
the entity has never been observed or it has never produced errors.
"""

from __future__ import annotations
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from MAR.ErrorAnalysis.error_taxonomy import (
    NUM_ERROR_TYPES, ERROR_CODE_TO_IDX, ERROR_TYPE_LIST
)


class RobustnessTracker:
    """Track error statistics per role and per LLM across training episodes."""

    def __init__(self, num_error_types: int = NUM_ERROR_TYPES,
                 smoothing: float = 1.0):
        """
        Args:
            num_error_types: Dimensionality of the robustness vector.
            smoothing: Laplace smoothing added to total counts to avoid
                       division-by-zero and to soften early estimates.
        """
        self.num_error_types = num_error_types
        self.smoothing = smoothing

        # role_name → {'total': int, 'errors': List[int]}
        self.role_stats: Dict[str, Dict] = defaultdict(
            lambda: {'total': 0, 'errors': [0] * num_error_types}
        )
        # llm_name → {'total': int, 'errors': List[int]}
        self.llm_stats: Dict[str, Dict] = defaultdict(
            lambda: {'total': 0, 'errors': [0] * num_error_types}
        )
        # (role_name, llm_name) → {'total': int, 'errors': List[int]}
        self.pair_stats: Dict[Tuple[str, str], Dict] = defaultdict(
            lambda: {'total': 0, 'errors': [0] * num_error_types}
        )

    # ── Query robustness vectors ───────────────────────────────────────────

    def get_role_robustness(self, role_name: str) -> torch.Tensor:
        """Return [num_error_types] tensor of error rates for *role_name*."""
        stats = self.role_stats[role_name]
        total = stats['total'] + self.smoothing
        return torch.tensor(stats['errors'], dtype=torch.float32) / total

    def get_llm_robustness(self, llm_name: str) -> torch.Tensor:
        """Return [num_error_types] tensor of error rates for *llm_name*."""
        stats = self.llm_stats[llm_name]
        total = stats['total'] + self.smoothing
        return torch.tensor(stats['errors'], dtype=torch.float32) / total

    def get_pair_robustness(self, role_name: str, llm_name: str) -> torch.Tensor:
        """Return [num_error_types] tensor of error rates for a (role, LLM) pair."""
        stats = self.pair_stats[(role_name, llm_name)]
        total = stats['total'] + self.smoothing
        return torch.tensor(stats['errors'], dtype=torch.float32) / total

    # ── Batch queries (for augmenting embedding matrices) ──────────────────

    def get_roles_robustness_matrix(self, role_names: List[str]) -> torch.Tensor:
        """Return [len(role_names), num_error_types] matrix."""
        return torch.stack([self.get_role_robustness(r) for r in role_names])

    def get_llms_robustness_matrix(self, llm_names: List[str]) -> torch.Tensor:
        """Return [len(llm_names), num_error_types] matrix."""
        return torch.stack([self.get_llm_robustness(l) for l in llm_names])

    # ── Update statistics from a completed episode ─────────────────────────

    def update(self, trace: List[dict], errors: List[dict],
               propagation: Optional[dict] = None):
        """
        Update statistics after an episode.

        Args:
            trace: List of step dicts from Graph.trace. Each dict has keys
                   'role', 'llm_name', 'step_id', etc.
            errors: List of error dicts. Each has 'error_id', 'error_type',
                    'error_start_step', 'error_end_step'.
            propagation: Optional dict with 'trajectory' (list of from→to edges).
        """
        # Build step_id → (role, llm_name) mapping
        step_agents: Dict[int, Tuple[str, str]] = {}
        for step in trace:
            role = step.get('role', '')
            llm = step.get('llm_name', '')
            step_agents[step['step_id']] = (role, llm)

        # Count participations
        for step in trace:
            role, llm = step.get('role', ''), step.get('llm_name', '')
            if role:
                self.role_stats[role]['total'] += 1
            if llm:
                self.llm_stats[llm]['total'] += 1
            if role and llm:
                self.pair_stats[(role, llm)]['total'] += 1

        # Determine which errors are "source" vs "propagated"
        propagated_ids = set()
        if propagation and 'trajectory' in propagation:
            for edge in propagation['trajectory']:
                propagated_ids.add(edge['to'])

        # Attribute errors to agents
        for error in errors:
            # Extract error code (e.g., "M1.1" from "M1.1 Disobey-Task-Specification")
            error_type_str = error.get('error_type', '')
            error_code = error_type_str.split()[0] if error_type_str else ''
            error_idx = ERROR_CODE_TO_IDX.get(error_code, -1)
            if error_idx < 0:
                continue

            # Weight: source errors count fully, propagated errors count less
            weight = 0.5 if error.get('error_id', '') in propagated_ids else 1.0

            start = error.get('error_start_step', 0)
            end = error.get('error_end_step', start)
            for sid in range(start, end + 1):
                if sid in step_agents:
                    role, llm = step_agents[sid]
                    if role:
                        self.role_stats[role]['errors'][error_idx] += weight
                    if llm:
                        self.llm_stats[llm]['errors'][error_idx] += weight
                    if role and llm:
                        self.pair_stats[(role, llm)]['errors'][error_idx] += weight

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save tracker state to a JSON file."""
        data = {
            'role_stats': dict(self.role_stats),
            'llm_stats': dict(self.llm_stats),
            'pair_stats': {f"{k[0]}||{k[1]}": v for k, v in self.pair_stats.items()},
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load tracker state from a JSON file.

        If the saved error vectors have a different length than the current
        num_error_types (e.g. taxonomy was updated), truncate or zero-pad
        automatically so the tracker remains usable.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        def _resize_errors(errors_list):
            """Truncate or pad *errors_list* to self.num_error_types."""
            if len(errors_list) == self.num_error_types:
                return errors_list
            if len(errors_list) > self.num_error_types:
                return errors_list[:self.num_error_types]
            return errors_list + [0] * (self.num_error_types - len(errors_list))

        for role, stats in data.get('role_stats', {}).items():
            stats['errors'] = _resize_errors(stats['errors'])
            self.role_stats[role] = stats
        for llm, stats in data.get('llm_stats', {}).items():
            stats['errors'] = _resize_errors(stats['errors'])
            self.llm_stats[llm] = stats
        for key, stats in data.get('pair_stats', {}).items():
            role, llm = key.split('||')
            stats['errors'] = _resize_errors(stats['errors'])
            self.pair_stats[(role, llm)] = stats

    def summary(self) -> str:
        """Human-readable summary of tracked statistics."""
        lines = ["=== Robustness Tracker Summary ==="]
        lines.append(f"\nTracked roles: {len(self.role_stats)}")
        for role, stats in sorted(self.role_stats.items()):
            total_errors = sum(stats['errors'])
            lines.append(f"  {role}: {stats['total']} uses, {total_errors:.1f} total errors")
        lines.append(f"\nTracked LLMs: {len(self.llm_stats)}")
        for llm, stats in sorted(self.llm_stats.items()):
            total_errors = sum(stats['errors'])
            lines.append(f"  {llm}: {stats['total']} uses, {total_errors:.1f} total errors")
        return "\n".join(lines)
