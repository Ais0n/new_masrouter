#!/usr/bin/env python3
"""
LLM Router Inference — plugs the DPO-trained router into MASRouter's execution loop.

Given a query + optional error state, calls the fine-tuned Qwen router to produce
a MASRouter-format routing decision, then executes the MAS accordingly.

Usage (standalone):
  python router_inference.py \
    --adapter checkpoints/llm_router_dpo/final \
    --query "What is the capital of France?"

Usage (as a module):
  from LLMRouter.router_inference import LLMRouter
  router = LLMRouter(adapter_path="checkpoints/llm_router_dpo/final")
  decision = router.route(query, error_state=None)
  # decision: {"collaboration_mode": "Chain", "num_agents": 2, "llms": [...], "rationale": "..."}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from LLMRouter.train_dpo import SYSTEM_PROMPT
except ModuleNotFoundError:
    from train_dpo import SYSTEM_PROMPT

# ── Valid routing vocabulary (mirrors MASRouter) ────────────────────────────────
VALID_MODES = {"IO", "CoT", "Chain", "FullConnected", "Debate", "Reflection"}
VALID_LLMS  = {
    "openai/gpt-4o-mini",
    "google/gemini-2.0-flash-001",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.1-70b-instruct",
    "anthropic/claude-3.5-haiku",
}
SINGLE_AGENT_MODES = {"IO", "CoT"}

# ── Default routing decision (fallback if the model output is malformed) ────────
DEFAULT_DECISION = {
    "collaboration_mode": "Chain",
    "num_agents": 2,
    "llms": ["openai/gpt-4o-mini", "google/gemini-2.0-flash-001"],
    "rationale": "Default fallback: sequential verification chain.",
}


@dataclass
class ErrorState:
    """Structured error state passed to the router when an error is detected mid-trace."""
    coarse_type:   str           # spec-violation | missing-grounding | tool-action-error | ...
    severity:      str           # low | medium | high
    outcome:       str           # propagated | standalone
    explanation:   str
    propagates_to: list = None   # error IDs this error has already caused

    def to_prompt_block(self) -> str:
        return json.dumps({
            "coarse_type":   self.coarse_type,
            "severity":      self.severity,
            "outcome":       self.outcome,
            "propagates_to": self.propagates_to or [],
            "explanation":   self.explanation,
        }, indent=2, ensure_ascii=False)


def build_user_message(query: str, error_state: Optional[ErrorState],
                       context_steps: Optional[list] = None) -> str:
    """Reconstruct the same prompt format used during DPO training."""
    parts = [f"[Query]\n{query}"]

    if error_state is not None:
        parts.append(f"[Routing-Relevant Error State]\n{error_state.to_prompt_block()}")
    else:
        parts.append(
            "[Routing-Relevant Error State]\n"
            '{"coarse_type": "none", "severity": "low", "outcome": "standalone", '
            '"propagates_to": [], "explanation": "No error detected; initial routing."}'
        )

    if context_steps:
        lines = []
        for s in context_steps[-4:]:   # last 4 steps, matching training setting
            content = s.get("content", "").strip()
            if len(content) > 400:
                content = content[:380] + "\n...[truncated]"
            lines.append(f"Step {s.get('step_idx', '?')} ({s.get('role', '?')}): {content}")
        parts.append("[Recent Trace Context]\n" + "\n\n".join(lines))
    else:
        parts.append("[Recent Trace Context]\n(no context — initial routing)")

    parts.append(
        "[Task]\n"
        "A multi-agent system encountered the error described above. "
        "Based on the error type, severity, and observed outcome, select a routing configuration "
        "that leads to early containment (no further propagation) or fast recovery (resolved within ≤2 steps). "
        "Use the MASRouter vocabulary: collaboration_mode ∈ {IO, CoT, Chain, FullConnected, Debate, Reflection}, "
        "num_agents ∈ [1, 3], llms is a list of model names of length num_agents.\n\n"
        "Output JSON only, schema:\n"
        "{\n"
        '  "collaboration_mode": "<IO|CoT|Chain|FullConnected|Debate|Reflection>",\n'
        '  "num_agents": <int 1-3>,\n'
        '  "llms": ["<llm_name>", ...],\n'
        '  "rationale": "<one-sentence explanation>"\n'
        "}"
    )

    return "\n\n".join(parts)


def parse_routing_decision(raw: str) -> dict:
    """Extract and validate a routing decision JSON from model output."""
    # Strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)

    # Try direct parse first
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: find the first {...} block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return DEFAULT_DECISION
        try:
            d = json.loads(m.group())
        except json.JSONDecodeError:
            return DEFAULT_DECISION

    # Validate and repair
    mode = d.get("collaboration_mode", "Chain")
    if mode not in VALID_MODES:
        mode = "Chain"
        d["collaboration_mode"] = mode

    n = d.get("num_agents", 2)
    if not isinstance(n, int) or n < 1 or n > 3:
        n = 1 if mode in SINGLE_AGENT_MODES else 2
        d["num_agents"] = n

    if mode in SINGLE_AGENT_MODES and n != 1:
        d["num_agents"] = 1
        n = 1
    if mode not in SINGLE_AGENT_MODES and n < 2:
        d["num_agents"] = 2
        n = 2

    llms = d.get("llms", [])
    if not isinstance(llms, list) or len(llms) != n:
        # Repair: fill with default LLMs
        default_pool = ["openai/gpt-4o-mini", "google/gemini-2.0-flash-001",
                        "deepseek/deepseek-chat"]
        d["llms"] = default_pool[:n]

    # Validate LLM names (warn but don't hard-fail — model might use short names)
    repaired_llms = []
    for llm in d["llms"]:
        if llm in VALID_LLMS:
            repaired_llms.append(llm)
        else:
            # Try prefix matching (e.g. "gpt-4o-mini" → "openai/gpt-4o-mini")
            match = next((v for v in VALID_LLMS if llm in v), "openai/gpt-4o-mini")
            repaired_llms.append(match)

    # Deduplicate: if model repeated the same LLM, fill gaps from default pool
    if len(set(repaired_llms)) < len(repaired_llms):
        default_pool = ["openai/gpt-4o-mini", "google/gemini-2.0-flash-001",
                        "deepseek/deepseek-chat", "meta-llama/llama-3.1-70b-instruct",
                        "anthropic/claude-3.5-haiku"]
        seen, deduped = set(), []
        for llm in repaired_llms:
            if llm not in seen:
                seen.add(llm)
                deduped.append(llm)
        for llm in default_pool:
            if len(deduped) >= n:
                break
            if llm not in seen:
                seen.add(llm)
                deduped.append(llm)
        repaired_llms = deduped[:n]

    d["llms"] = repaired_llms

    if "rationale" not in d or not d["rationale"]:
        d["rationale"] = "Routing decision based on error state."

    return d


class LLMRouter:
    """
    Drop-in LLM router that replaces MASRouter's cascaded classifier.

    Loads the DPO-trained LoRA adapter on top of the base Qwen model and
    generates routing decisions as structured JSON text.
    """

    def __init__(
        self,
        adapter_path: str,
        base_model: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.1,   # low temperature for deterministic JSON output
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature

        print(f"Loading tokenizer from {adapter_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading base model {base_model}...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=device,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print(f"Loading LoRA adapter from {adapter_path}...")
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        print("Router ready.")

    @torch.inference_mode()
    def route(
        self,
        query: str,
        error_state: Optional[ErrorState] = None,
        context_steps: Optional[list] = None,
    ) -> dict:
        """
        Generate a routing decision.

        Args:
            query:         The task query string.
            error_state:   Structured error state (None for initial, error-free routing).
            context_steps: Recent trace steps as list of {step_idx, role, content} dicts.

        Returns:
            dict with keys: collaboration_mode, num_agents, llms, rationale
        """
        user_msg = build_user_message(query, error_state, context_steps)
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": user_msg},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only the newly generated tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return parse_routing_decision(raw)


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter",     required=True, help="Path to saved LoRA adapter dir.")
    parser.add_argument("--base-model",  default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--query",       required=True, help="Task query string.")
    parser.add_argument("--coarse-type", default=None,
                        help="Coarse error type (omit for initial routing).")
    parser.add_argument("--severity",    default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--outcome",     default="standalone",
                        choices=["propagated", "standalone"])
    parser.add_argument("--explanation", default="")
    args = parser.parse_args()

    router = LLMRouter(adapter_path=args.adapter, base_model=args.base_model)

    error_state = None
    if args.coarse_type:
        error_state = ErrorState(
            coarse_type=args.coarse_type,
            severity=args.severity,
            outcome=args.outcome,
            explanation=args.explanation,
        )

    decision = router.route(args.query, error_state=error_state)
    print(json.dumps(decision, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
