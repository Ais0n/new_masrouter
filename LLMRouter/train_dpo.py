#!/usr/bin/env python3
"""
DPO Fine-tuning Script for the LLM-based MAS Router.

Trains a Qwen3.5-27B-Instruct model to output MASRouter-format routing decisions
conditioned on (query, error_state, trace_context) using segment-level DPO.

The router learns to:
  - chosen:   routing decisions that lead to early containment / fast recovery
  - rejected: routing decisions that propagate errors or delay recovery

Architecture:
  Base:   Qwen/Qwen3-4B-Instruct-2507  (4B dense; ~2 GB at 4-bit, fits single RTX4090)
  Adapt:  QLoRA (4-bit NF4 + BF16 compute) via bitsandbytes
  Train:  TRL DPOTrainer with beta=0.1, sigmoid loss

Usage:
  # 4× RTX 4090 (recommended — ZeRO-2 DDP, each GPU holds full 4-bit model ~16 GB)
  torchrun --nproc_per_node=4 train_dpo.py \
    --data-path ../../MAST-Data/output/dpo_pairs/dpo_pairs_diverse.jsonl \
    --deepspeed deepspeed_zero2.json

  # Single GPU (A100 80 GB or RTX 4090 24 GB both work)
  python train_dpo.py --data-path ../../MAST-Data/output/dpo_pairs/dpo_pairs_diverse.jsonl

  # Quick smoke-test on CPU (tiny model, 4 fake pairs — no GPU required)
  python train_dpo.py --smoke-test

Requirements:
  pip install transformers trl peft bitsandbytes accelerate deepspeed datasets
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer

# ── Defaults ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_OUTPUT  = "checkpoints/llm_router_dpo"
SYSTEM_PROMPT   = (
    "You are an expert MAS routing advisor. "
    "Given a task query, a detected error state, and recent trace context, "
    "output a single routing decision in JSON format that leads to early error containment "
    "or fast recovery. "
    "Your output must be valid JSON with keys: "
    "collaboration_mode, num_agents, llms, rationale."
)

# ── LoRA target modules for Qwen3.x ─────────────────────────────────────────────
QWEN_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ── Dataset helpers ─────────────────────────────────────────────────────────────

def load_pairs(path: str) -> List[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def format_as_messages(pair: dict, tokenizer) -> dict:
    """
    Convert a (prompt, chosen, rejected) pair into the TRL DPOTrainer message format.

    TRL expects:
      prompt   -> list of system+user messages (no assistant turn)
      chosen   -> list containing the preferred assistant message
      rejected -> list containing the dispreferred assistant message
    """
    messages_prompt = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": pair["prompt"]},
    ]
    messages_chosen = [
        {"role": "assistant", "content": pair["chosen"]},
    ]
    messages_rejected = [
        {"role": "assistant", "content": pair["rejected"]},
    ]
    return {
        "prompt":   messages_prompt,
        "chosen":   messages_chosen,
        "rejected": messages_rejected,
        # Pass metadata through for later analysis (DPOTrainer ignores unknown fields)
        "meta_coarse_type":   pair.get("meta", {}).get("coarse_type", ""),
        "meta_outcome":       pair.get("meta", {}).get("outcome", ""),
        "meta_severity":      pair.get("meta", {}).get("severity_label", ""),
    }


def build_dataset(pairs: List[dict], tokenizer, val_ratio: float = 0.1, seed: int = 42):
    """Split pairs into train/val and return HuggingFace Dataset objects."""
    random.seed(seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_pairs   = pairs[:n_val]
    train_pairs = pairs[n_val:]

    def to_dataset(ps):
        records = [format_as_messages(p, tokenizer) for p in ps]
        return Dataset.from_list(records)

    return to_dataset(train_pairs), to_dataset(val_pairs)


# ── Model loading ───────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name: str, smoke_test: bool = False):
    """Load Qwen3.5-27B with QLoRA (4-bit NF4). Falls back to a tiny model for smoke tests."""
    if smoke_test:
        # Use a tiny public model so the smoke-test runs on CPU without GPU/big download
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"[smoke-test] Using {model_name} on CPU.")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        return model, tokenizer

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",      # DPO requires left-padding for batch generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map must NOT be "auto" for DDP training — "auto" enables pipeline-parallel
    # which conflicts with DPO's gradient flow through LoRA adapters.
    # With torchrun + ZeRO-2, each rank loads the model onto its own GPU.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": f"cuda:{local_rank}"} if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",  # requires flash-attn package
    )
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,   # ~30% memory saving on activations
    )
    return model, tokenizer


def apply_lora(model, smoke_test: bool = False) -> object:
    """Wrap the model with LoRA adapters."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8 if smoke_test else 16,
        lora_alpha=16 if smoke_test else 32,
        target_modules=QWEN_LORA_TARGETS,
        lora_dropout=0.05,
        bias="none",
        use_rslora=False,
    )
    model = get_peft_model(model, lora_config)
    # Gradient checkpointing is already enabled in prepare_model_for_kbit_training,
    # but call again on the PEFT model to ensure the adapter layers are covered.
    if not smoke_test:
        model.enable_input_require_grads()
    return model


# ── Training ────────────────────────────────────────────────────────────────────

def train(args):
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model, smoke_test=args.smoke_test)
    model = apply_lora(model, smoke_test=args.smoke_test)
    model.print_trainable_parameters()

    # Load data
    if args.smoke_test:
        # Manufacture 4 fake pairs so the training loop can run
        fake_pair = {
            "prompt": "Test query about MAS routing.",
            "chosen":  '{"collaboration_mode":"Chain","num_agents":2,"llms":["openai/gpt-4o-mini","google/gemini-2.0-flash-001"],"rationale":"Verification step contains error."}',
            "rejected": '{"collaboration_mode":"IO","num_agents":1,"llms":["meta-llama/llama-3.1-70b-instruct"],"rationale":"Single agent propagates error."}',
            "meta": {"coarse_type": "spec-violation", "outcome": "standalone", "severity_label": "medium"},
        }
        pairs = [fake_pair] * 4
    else:
        pairs = load_pairs(args.data_path)
        print(f"Loaded {len(pairs)} DPO pairs from {args.data_path}")

    train_dataset, val_dataset = build_dataset(pairs, tokenizer, val_ratio=0.1)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── TRL version compatibility ──────────────────────────────────────────────
    import inspect
    import trl as _trl
    print(f"TRL version: {_trl.__version__}")

    # 1. tokenizer kwarg renamed to processing_class in TRL >= 0.9
    _trainer_sig = inspect.signature(DPOTrainer.__init__).parameters
    _tokenizer_kwarg = "processing_class" if "processing_class" in _trainer_sig else "tokenizer"

    # 2. DPO-specific args (beta, max_length, etc.) live in DPOConfig in newer TRL
    #    but in DPOTrainer in older TRL.  DPOConfig uses **kwargs internally so
    #    inspect misses them — probe by actually instantiating DPOConfig.
    _dpo_specific = {
        "beta":              args.beta,
        "loss_type":         "sigmoid",
        "max_length":        args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "label_smoothing":   0.0,
    }
    _in_config, _in_trainer = {}, {}
    for _k, _v in _dpo_specific.items():
        try:
            DPOConfig(output_dir="/tmp/_probe", **{_k: _v})
            _in_config[_k] = _v
        except TypeError:
            if _k in _trainer_sig:
                _in_trainer[_k] = _v
            else:
                print(f"  [compat] '{_k}' not accepted by DPOConfig or DPOTrainer — skipping.")
    if _in_trainer:
        print(f"  [compat] passing {list(_in_trainer)} to DPOTrainer instead of DPOConfig.")

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=not args.smoke_test,
        fp16=False,
        gradient_checkpointing=not args.smoke_test,
        logging_steps=1 if args.smoke_test else 10,
        eval_strategy="steps",
        eval_steps=5 if args.smoke_test else 25,
        save_strategy="steps",
        save_steps=5 if args.smoke_test else 25,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        deepspeed=args.deepspeed,
        **_in_config,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        **{_tokenizer_kwarg: tokenizer},
        **_in_trainer,
    )

    print("Starting DPO training...")
    trainer.train()

    # Save final adapter + tokenizer
    final_dir = Path(args.output_dir) / "final"
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nSaved LoRA adapter → {final_dir}")

    # Save training metadata
    meta = {
        "base_model":     args.model,
        "data_path":      args.data_path,
        "n_train_pairs":  len(train_dataset),
        "n_val_pairs":    len(val_dataset),
        "beta":           args.beta,
        "epochs":         args.epochs,
        "lr":             args.lr,
        "lora_r":         16,
        "lora_alpha":     32,
    }
    (final_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Training metadata → {final_dir}/training_meta.json")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DPO fine-tuning for LLM-based MAS router.")

    # Data
    parser.add_argument("--data-path", default="../../MAST-Data/output/dpo_pairs/dpo_pairs_diverse.jsonl")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)

    # Model
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace model ID for the base LLM router.")

    # DPO hyperparameters
    parser.add_argument("--beta",   type=float, default=0.1,
                        help="KL penalty coefficient (standard DPO beta). Default: 0.1")
    parser.add_argument("--epochs", type=int,   default=3)
    parser.add_argument("--lr",     type=float, default=5e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps. Effective batch = per_device × grad_accum.")
    parser.add_argument("--max-length",        type=int, default=2048)
    parser.add_argument("--max-prompt-length", type=int, default=1800)

    # Hardware
    parser.add_argument("--deepspeed", default=None,
                        help="Path to DeepSpeed config JSON (e.g. deepspeed_zero2.json). "
                             "Required for multi-GPU torchrun. Omit for single GPU.")

    # Misc
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a 2-step training loop on a tiny model to verify the pipeline.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
