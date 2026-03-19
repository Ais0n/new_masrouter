"""
LLM-based Error Evaluator for MasRouter traces.

Evaluates step-level errors in MAS execution traces using the MAST codebook
taxonomy (Codebook.xlsx).  Follows the same methodology as the MAST labeling
pipeline (label_magentic_one_gaia_openrouter.py):

  1. Format the trace into numbered steps with role / LLM / response / predecessors
  2. Provide the full codebook (28 error types) as context
  3. Ask an LLM judge to label ALL step-level errors + propagation
  4. Parse the JSON response and add agent attribution from trace data

Output format (MAST-compatible):
    errors:       List[dict]  – each with error_id, error_start_step, error_end_step,
                                error_type, explanation, agent_role, agent_llm
    propagation:  dict        – {"trajectory": [{"from": ..., "to": ..., "explanation": ...}]}
"""

from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Tuple

from loguru import logger

from MAR.ErrorAnalysis.error_taxonomy import get_codebook_text, ERROR_TYPES


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM-based Error Evaluator (primary evaluator)
# ═══════════════════════════════════════════════════════════════════════════════

class LLMErrorEvaluator:
    """
    Uses an LLM judge to evaluate MasRouter traces for step-level errors,
    following the MAST codebook taxonomy.

    This is the **primary** error evaluator.  It mirrors the methodology of
    ``label_magentic_one_gaia_openrouter.py`` but operates on MasRouter's
    in-memory trace format rather than console log files.

    Usage:
        evaluator = LLMErrorEvaluator()                     # default model
        evaluator = LLMErrorEvaluator(llm_name="gpt-4o")    # stronger model
        errors, propagation = evaluator.evaluate(trace, query=..., ...)
    """

    SYSTEM_PROMPT = (
        "You are an expert qualitative coding assistant for multi-agent system traces. "
        "Use the provided hierarchical codebook to label step-level errors. "
        "Return only valid JSON with keys 'errors' and 'propagation' and no extra text."
    )

    USER_PROMPT_TEMPLATE = """
Task prompt:
{query}

Expected answer:
{ground_truth}

Final answer produced by the MAS:
{final_answer}

Task solved correctly: {is_solved}
Task domain: {task_domain}

Step definitions:
- Each step corresponds to one agent's execution within the MAS graph.
- Step indices below start at 1.

Codebook (hierarchical – use ONLY these error types):
{codebook}

Execution Trace ({num_steps} steps):
{trace_text}

Label ALL errors and their propagation. Output JSON only.

JSON schema:
{{
  "errors": [
    {{
      "error_id": "E1",
      "error_start_step": 1,
      "error_end_step": 1,
      "error_type": "M1.1 Disobey-Task-Specification",
      "explanation": "Brief description; include why this type fits and who is responsible (agent, tool, or user)."
    }}
  ],
  "propagation": {{
    "trajectory": [
      {{
        "from": "E1",
        "to": "E2",
        "explanation": "How the issue evolves or propagates."
      }}
    ]
  }}
}}

Rules:
- Use ONLY error types from the codebook (keep exact code + name).
- Use step indices from the trace above.
- Evaluate EACH step independently for potential errors.
- If no propagation links, return an empty trajectory list.
- If no errors, return empty errors list and empty trajectory list.
""".strip()

    def __init__(self,
                 llm_name: str = "gpt-4o-mini",
                 max_trace_chars: int = 30000,
                 max_step_chars: int = 2000,
                 temperature: float = 0.2):
        """
        Args:
            llm_name: Model name compatible with MasRouter's LLMRegistry
                      (e.g., "gpt-4o-mini", "gpt-4o").
            max_trace_chars: Maximum total characters for trace in prompt.
            max_step_chars: Maximum characters per step response.
            temperature: LLM temperature for evaluation.
        """
        self.llm_name = llm_name
        self.max_trace_chars = max_trace_chars
        self.max_step_chars = max_step_chars
        self.temperature = temperature

    def evaluate(self, trace: List[dict], query: str = "",
                 ground_truth: str = "", final_answer: str = "",
                 is_solved: bool = False,
                 task_domain: str = "") -> Tuple[List[dict], dict]:
        """
        Evaluate a trace using an LLM judge with the MAST codebook.

        Args:
            trace: List of step dicts from Graph.trace. Each dict has keys:
                   step_id, round, node_id, role, llm_name, response,
                   prompt, spatial_predecessors.
            query: The original task/query.
            ground_truth: Expected answer (if available).
            final_answer: The system's final answer.
            is_solved: Whether the task was solved correctly.
            task_domain: 'Math', 'Code', 'Commonsense', etc.

        Returns:
            (errors, propagation) tuple in MAST-compatible format.
        """
        from MAR.LLM.llm_registry import LLMRegistry

        if not trace:
            return [], {'trajectory': []}

        # ── Build the prompt ───────────────────────────────────────────────
        codebook = get_codebook_text()
        trace_text = self._format_trace(trace)

        user_prompt = self.USER_PROMPT_TEMPLATE.format(
            query=query.strip(),
            ground_truth=ground_truth.strip(),
            final_answer=(final_answer or "(missing)").strip(),
            is_solved=is_solved,
            task_domain=task_domain or "General",
            codebook=codebook,
            num_steps=len(trace),
            trace_text=trace_text,
        )

        messages = [
            {'role': 'system', 'content': self.SYSTEM_PROMPT},
            {'role': 'user', 'content': user_prompt},
        ]

        # ── Call the LLM ──────────────────────────────────────────────────
        try:
            llm = LLMRegistry.get(self.llm_name)
            response = llm.gen(messages)
            parsed = self._parse_json_response(response)

            errors = parsed.get('errors', [])
            propagation = self._normalize_propagation(
                parsed.get('propagation', {})
            )

            # Add agent attribution from trace data
            self._add_agent_attribution(errors, trace)

            return errors, propagation

        except Exception as e:
            logger.warning(f"LLMErrorEvaluator failed: {e}")
            return [], {'trajectory': []}

    # ── Trace formatting ───────────────────────────────────────────────────

    def _format_trace(self, trace: List[dict]) -> str:
        """
        Format trace steps for the LLM prompt.

        Mirrors the step formatting in label_magentic_one_gaia_openrouter.py:
        each step gets an index, role, LLM name, predecessors, and
        (truncated) response content.
        """
        chunks: List[str] = []
        total_chars = 0
        for step in trace:
            response_text = str(step.get('response', ''))
            if len(response_text) > self.max_step_chars:
                response_text = response_text[:self.max_step_chars] + "\n...[truncated]"

            predecessors = step.get('spatial_predecessors', [])
            pred_str = ', '.join(predecessors) if predecessors else 'none'

            step_text = (
                f"Step {step['step_id']} "
                f"(role: {step.get('role', 'unknown')}, "
                f"LLM: {step.get('llm_name', 'unknown')}, "
                f"round: {step.get('round', 0)}, "
                f"predecessors: [{pred_str}])\n"
                f"{response_text}"
            ).strip()

            if total_chars + len(step_text) + 2 > self.max_trace_chars:
                chunks.append("[... remaining steps truncated ...]")
                break

            chunks.append(step_text)
            total_chars += len(step_text) + 2

        return "\n\n".join(chunks)

    # ── Response parsing ───────────────────────────────────────────────────

    def _parse_json_response(self, text: str) -> dict:
        """
        Parse LLM response as JSON, handling common formatting issues.
        Follows the same robustness as label_magentic_one_gaia_openrouter.py.
        """
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting first JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                pass

        # Try extracting from markdown code block
        code_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass

        logger.warning(
            f"LLMErrorEvaluator: could not parse JSON from response "
            f"(length={len(text)}). Returning empty."
        )
        return {'errors': [], 'propagation': {'trajectory': []}}

    def _normalize_propagation(self, propagation) -> dict:
        """
        Normalize the propagation field to the expected format.
        The LLM sometimes returns a list instead of {"trajectory": [...]}.
        """
        if isinstance(propagation, list):
            return {'trajectory': propagation}
        if isinstance(propagation, dict):
            if 'trajectory' not in propagation:
                propagation['trajectory'] = []
            return propagation
        return {'trajectory': []}

    def _add_agent_attribution(self, errors: List[dict],
                               trace: List[dict]):
        """
        Enrich error dicts with agent_role and agent_llm from trace data.
        """
        step_info = {step['step_id']: step for step in trace}
        for error in errors:
            sid = error.get('error_start_step', 0)
            if sid in step_info:
                error.setdefault('agent_role', step_info[sid].get('role', ''))
                error.setdefault('agent_llm', step_info[sid].get('llm_name', ''))
            else:
                error.setdefault('agent_role', '')
                error.setdefault('agent_llm', '')
