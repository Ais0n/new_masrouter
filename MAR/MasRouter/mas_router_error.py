"""
Error-Aware MasRouter – extends the original MasRouter with:

1. **Robustness augmentation**: Role and LLM embeddings are augmented with
   historical error-rate vectors via a gated fusion mechanism.
2. **Trace collection**: Each Graph execution collects a step-level trace.
3. **Error-augmented reward**: Training reward includes an error penalty term
   shaped by error severity and propagation structure.

Usage (drop-in replacement for MasRouter):

    from MAR.MasRouter.mas_router_error import ErrorAwareMasRouter
    from MAR.ErrorAnalysis import RobustnessTracker

    tracker = RobustnessTracker()
    router = ErrorAwareMasRouter(
        max_agent=6, device=device,
        robustness_tracker=tracker,
    ).to(device)

    # forward() returns an extra `traces` list
    results, costs, log_probs, tasks_probs, vae_loss, agents_num, traces = \\
        router.forward(queries, tasks, llms, reasonings, ...)
"""

from __future__ import annotations
from typing import List, Dict, Optional

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from MAR.MasRouter.mas_router import (
    MasRouter, TaskClassifier, CollabDeterminer, NumDeterminer,
    RoleAllocation, LLMRouter, GFusion, VAE, vae_loss_function,
)
from MAR.LLM.llm_embedding import SentenceEncoder
from MAR.Graph.graph import Graph
from MAR.Utils.utils import get_kwargs
from MAR.Utils.globals import Cost
from MAR.ErrorAnalysis.error_taxonomy import NUM_ERROR_TYPES
from MAR.ErrorAnalysis.robustness_tracker import RobustnessTracker

from loguru import logger


# ═══════════════════════════════════════════════════════════════════════════════
#  Robustness Augmentor (Gated Fusion)
# ═══════════════════════════════════════════════════════════════════════════════

class RobustnessAugmentor(nn.Module):
    """
    Augments text embeddings with robustness (error-rate) vectors via a
    learned gated fusion mechanism.

    Given:
        embeddings:  [N, embedding_dim]   – original text embeddings
        robustness:  [N, num_error_types] – per-entity error rate vectors

    Output:
        augmented:   [N, embedding_dim]   – same shape, drop-in compatible

    The gate learns *how much* robustness information to incorporate:
        rob_proj = Linear(robustness)           → [N, embedding_dim]
        gate     = σ(Linear([emb; rob_proj]))   → [N, embedding_dim]
        output   = emb + gate ⊙ rob_proj
    """

    def __init__(self, embedding_dim: int = 384,
                 num_error_types: int = NUM_ERROR_TYPES):
        super().__init__()
        self.proj = nn.Linear(num_error_types, embedding_dim)
        self.gate = nn.Linear(embedding_dim * 2, embedding_dim)

        # Initialize gate bias to negative so augmentation starts small
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, embeddings: torch.Tensor,
                robustness: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings:  [N, embedding_dim]
            robustness:  [N, num_error_types]
        Returns:
            augmented:   [N, embedding_dim]
        """
        rob_proj = self.proj(robustness)                              # [N, D]
        combined = torch.cat([embeddings, rob_proj], dim=-1)          # [N, 2D]
        gate_values = torch.sigmoid(self.gate(combined))              # [N, D]
        augmented = embeddings + gate_values * rob_proj               # [N, D]
        return augmented


# ═══════════════════════════════════════════════════════════════════════════════
#  Error-Aware MasRouter
# ═══════════════════════════════════════════════════════════════════════════════

class ErrorAwareMasRouter(MasRouter):
    """
    MasRouter with robustness-augmented embeddings and trace collection.

    Differences from the base MasRouter:
      - Role embeddings are augmented with per-role error statistics.
      - LLM embeddings are augmented with per-LLM error statistics.
      - Graph execution collects step-level traces.
      - forward() returns an additional `traces` list.

    When robustness_tracker is None or has no data, this behaves identically
    to the base MasRouter (augmentation produces zero modifications due to
    the gating mechanism).
    """

    def __init__(self, in_dim: int = 384, hidden_dim: int = 64,
                 max_agent: int = 6, temp: float = 0.5, device=None,
                 robustness_tracker: Optional[RobustnessTracker] = None,
                 num_error_types: int = NUM_ERROR_TYPES):
        super().__init__(in_dim=in_dim, hidden_dim=hidden_dim,
                         max_agent=max_agent, temp=temp, device=device)

        self.robustness_tracker = robustness_tracker
        self.num_error_types = num_error_types

        # Gated fusion modules for augmenting embeddings with robustness
        self.role_robustness_augmentor = RobustnessAugmentor(
            embedding_dim=in_dim, num_error_types=num_error_types,
        )
        self.llm_robustness_augmentor = RobustnessAugmentor(
            embedding_dim=in_dim, num_error_types=num_error_types,
        )

    def set_robustness_tracker(self, tracker: RobustnessTracker):
        """Update the robustness tracker (e.g., after loading from checkpoint)."""
        self.robustness_tracker = tracker

    def encoder_roles(self):
        """
        Override: encode role descriptions and augment with robustness vectors.
        """
        task_role_database, task_role_emb = super().encoder_roles()

        if self.robustness_tracker is not None:
            for task_name in task_role_emb:
                # Skip non-role directories (e.g. FinalNode) whose JSONs
                # don't have the standard role schema with a 'Name' key.
                roles = task_role_database[task_name]
                if not roles or 'Name' not in roles[0]:
                    continue
                role_names = [role['Name'] for role in roles]
                rob_matrix = self.robustness_tracker.get_roles_robustness_matrix(
                    role_names
                ).to(self.device)
                # Augment role embeddings with robustness information
                task_role_emb[task_name] = self.role_robustness_augmentor(
                    task_role_emb[task_name], rob_matrix
                )

        return task_role_database, task_role_emb

    def forward(self, queries: List[str], tasks: List[Dict[str, str]],
                llms: List[Dict[str, str]], collabs: List[Dict[str, str]],
                given_task: Optional[List[int]] = None,
                prompt_file: str = 'MAR/Roles/FinalNode/gsm8k.json'):
        """
        Forward pass with robustness augmentation and trace collection.

        Returns:
            final_result: List[str]          – answers for each query
            costs:        List[float]        – cost for each query
            log_probs:    Tensor [N_q, 1]    – combined log probabilities
            tasks_probs:  Tensor [N_q, N_t]  – task classification scores
            vae_loss:     Tensor              – VAE reconstruction losses
            agent_num_float: Tensor [N_q, 1] – predicted agent counts
            traces:       List[List[dict]]   – step-level execution traces
            per_agent_info: dict             – per-position log_probs for
                            counterfactual credit assignment:
                            {
                              'collab_log_probs':     Tensor [N_q, 1],
                              'role_lps_per_pos':     List[List[Tensor]],
                              'llm_lps_per_pos':      List[List[Tensor]],
                              'role_names_per_query': List[List[str]],
                              'llm_names_per_query':  List[List[str]],
                            }
        """
        # ── Preprocessing ──────────────────────────────────────────────────
        tasks_list = self._preprocess_data(tasks)
        llms_list = self._preprocess_data(llms)
        collabs_list = self._preprocess_data(collabs)

        # Role encoding (with augmentation via overridden encoder_roles)
        task_role_database, task_role_emb = self.encoder_roles()

        # ── Text embedding ─────────────────────────────────────────────────
        queries_embedding = self.text_encoder(queries)
        tasks_embedding = self.text_encoder(tasks_list)
        llms_embedding = self.text_encoder(llms_list)
        collabs_embedding = self.text_encoder(collabs_list)

        # ── LLM embedding augmentation ─────────────────────────────────────
        if self.robustness_tracker is not None:
            llm_names = [llm['Name'] for llm in llms]
            llm_rob = self.robustness_tracker.get_llms_robustness_matrix(
                llm_names
            ).to(self.device)
            llms_embedding = self.llm_robustness_augmentor(
                llms_embedding, llm_rob
            )

        # ── Task classification ────────────────────────────────────────────
        selected_tasks_idx, tasks_probs, query_context = self.task_classifier(
            queries_embedding, tasks_embedding
        )
        selected_tasks = (
            [tasks[idx] for idx in selected_tasks_idx]
            if given_task is None
            else [tasks[idx] for idx in given_task]
        )
        tasks_role_list = [
            task_role_database[task['Name']] for task in selected_tasks
        ]
        tasks_role_emb_list = [
            task_role_emb[task['Name']] for task in selected_tasks
        ]

        # ── Collaboration selection ────────────────────────────────────────
        selected_collabs_idx, collab_log_probs, collab_context, collab_vae_loss = \
            self.collab_determiner(collabs_embedding, queries_embedding)
        selected_collabs = [collabs[idx] for idx in selected_collabs_idx]

        # ── Number of agents ───────────────────────────────────────────────
        agent_num_int, agent_num_float, num_vae_loss = \
            self.num_determiner(queries_embedding)

        # ── Role allocation ────────────────────────────────────────────────
        selected_roles_idx, role_log_probs, role_context, role_vae_loss, \
            role_lps_per_pos = self.role_allocation(
                tasks_role_emb_list,
                torch.concat([query_context, collab_context], dim=-1),
                agent_num_int,
            )
        selected_roles = [
            [tasks_roles[sid.item()] for sid in selected_roles_id_list]
            for tasks_roles, selected_roles_id_list
            in zip(tasks_role_list, selected_roles_idx)
        ]

        # ── LLM routing ───────────────────────────────────────────────────
        selected_llms_idx, llm_log_probs, llm_vae_loss, \
            llm_lps_per_pos = self.llm_router(
            llms_embedding,
            torch.concat([query_context, collab_context, role_context], dim=-1),
            agent_num_int,
            agent_num_float,
        )
        selected_llms = [
            [llms[idx] for idx in selected_llms_id_list]
            for selected_llms_id_list in selected_llms_idx
        ]

        log_probs = llm_log_probs + role_log_probs + collab_log_probs
        vae_loss = collab_vae_loss + num_vae_loss + role_vae_loss + llm_vae_loss

        # ── Graph execution with trace collection ──────────────────────────
        final_result = []
        costs = []
        traces = []
        role_names_per_query: List[List[str]] = []
        llm_names_per_query: List[List[str]] = []

        for query, task, llm_selected, collab, roles in zip(
            queries, selected_tasks, selected_llms, selected_collabs, selected_roles
        ):
            previous_cost = Cost.instance().value
            kwargs = get_kwargs(collab['Name'], len(llm_selected))
            llm_names = [llm['Name'] for llm in llm_selected]
            role_names = [role['Name'] for role in roles]
            role_names_per_query.append(role_names)
            llm_names_per_query.append(llm_names)

            logger.info(f'Query: {query}')
            logger.info(f'Task: {task["Name"]}')
            logger.info(f'LLMs: {llm_names}')
            logger.info(f'Reasoning: {collab["Name"]}')
            logger.info(f'Roles: {role_names}')
            logger.info('-----------------------------------')

            g = Graph(
                domain=task['Name'],
                llm_names=llm_names,
                agent_names=role_names,
                decision_method="FinalRefer",
                prompt_file=prompt_file,
                reasoning_name=collab["Name"],
                collect_trace=True,   # ← Enable trace collection
                **kwargs,
            )
            self.g = g
            final_result.append(
                g.run(inputs={"query": query},
                      num_rounds=kwargs["num_rounds"])[0][0]
            )
            costs.append(Cost.instance().value - previous_cost)
            traces.append(g.trace)  # ← Collect trace

        per_agent_info = {
            'collab_log_probs': collab_log_probs,         # [N_q, 1]
            'role_lps_per_pos': role_lps_per_pos,         # List[List[Tensor]]
            'llm_lps_per_pos': llm_lps_per_pos,           # List[List[Tensor]]
            'role_names_per_query': role_names_per_query,  # List[List[str]]
            'llm_names_per_query': llm_names_per_query,    # List[List[str]]
        }

        return final_result, costs, log_probs, tasks_probs, vae_loss, \
               agent_num_float, traces, per_agent_info
