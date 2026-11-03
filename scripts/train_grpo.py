#!/usr/bin/env python3
"""GRPO with an execution reward, starting from the best SFT adapter.

    python scripts/train_grpo.py --config configs/grpo_qwen3_4b.yaml \
        --set model.adapter=outputs/sft-plain/adapter

Start from the SFT checkpoint, never the base model: with a base model the
execution reward is near zero on almost every rollout, every completion in the
group scores the same, the advantage is zero and nothing moves. That failure is
silent -- the loss looks fine -- so the script refuses to start without an
adapter unless you pass `--allow-base-model`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _trainer_utils import describe_trainable, filter_kwargs
from text2sql.config import dump_config, load_config
from text2sql.rewards import RewardWeights, execution_reward, format_reward
from text2sql.sql_exec import ComparisonMode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--allow-base-model", action="store_true",
                        help="train from the base model anyway (expect rewards near zero)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    adapter = cfg.get_path("model.adapter")
    if not adapter and not args.allow_base_model:
        print("[fail] model.adapter is not set. GRPO from a base model stalls at "
              "zero reward; point it at an SFT adapter, or pass --allow-base-model.")
        return 1

    import torch
    from datasets import load_dataset as hf_load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    model_name = cfg.get_path("model.name", "Qwen/Qwen3-4B")
    run_name = cfg.get_path("run_name", "grpo")
    out_dir = Path(cfg.get_path("output_dir", f"outputs/{run_name}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_file = cfg.get_path("data.train_file", "data/processed/train.jsonl")
    if not Path(train_file).exists():
        print(f"[fail] {train_file} not found; run scripts/build_dataset.py first")
        return 1

    dataset = hf_load_dataset("json", data_files={"train": train_file})["train"]
    # GRPO needs the prompt plus whatever the reward function reads.
    keep = ["prompt", "db_path", "gold_sql"]
    dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep])
    subset = cfg.get_path("data.subset_size")
    if subset:
        dataset = dataset.shuffle(seed=cfg.get_path("seed", 13)).select(range(min(subset, len(dataset))))
    print(f"[data ] {len(dataset)} prompts for rollouts")

    weights = RewardWeights(
        correct=cfg.get_path("reward.correct", 1.0),
        executable=cfg.get_path("reward.executable", 0.10),
        formatted=cfg.get_path("reward.formatted", 0.05),
        comparison=ComparisonMode(cfg.get_path("reward.comparison", "bird")),
        timeout=cfg.get_path("reward.timeout", 15.0),
        max_workers=cfg.get_path("reward.max_workers", 16),
    )

    def reward_execution(prompts=None, completions=None, **kwargs):
        return execution_reward(prompts, completions, weights=weights, **kwargs)

    reward_execution.__name__ = "execution_reward"

    reward_funcs = [reward_execution]
    if cfg.get_path("reward.include_format_reward", False):
        reward_funcs.append(format_reward)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {
        "dtype": cfg.get_path("model.dtype", "auto"),
        "device_map": cfg.get_path("model.device_map", "auto"),
        "attn_implementation": cfg.get_path("model.attn_implementation", "sdpa"),
    }
    if cfg.get_path("model.load_in_4bit", True):
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, cfg.get_path("model.compute_dtype", "bfloat16")),
        )

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    peft_config = None
    if adapter:
        from peft import PeftModel

        print(f"[model] continuing from adapter {adapter}")
        model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=cfg.get_path("lora.r", 32),
            lora_alpha=cfg.get_path("lora.alpha", 64),
            lora_dropout=cfg.get_path("lora.dropout", 0.0),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cfg.get_path(
                "lora.target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )

    grpo_kwargs = {
        "output_dir": str(out_dir),
        "run_name": run_name,
        "num_generations": cfg.get_path("grpo.num_generations", 8),
        "max_prompt_length": cfg.get_path("grpo.max_prompt_length", 3072),
        "max_completion_length": cfg.get_path("grpo.max_completion_length", 384),
        "temperature": cfg.get_path("grpo.temperature", 1.0),
        "beta": cfg.get_path("grpo.beta", 0.02),
        "num_train_epochs": cfg.get_path("grpo.num_train_epochs", 1),
        "per_device_train_batch_size": cfg.get_path("grpo.per_device_train_batch_size", 8),
        "gradient_accumulation_steps": cfg.get_path("grpo.gradient_accumulation_steps", 4),
        "learning_rate": cfg.get_path("grpo.learning_rate", 1e-6),
        "lr_scheduler_type": cfg.get_path("grpo.lr_scheduler_type", "constant_with_warmup"),
        "warmup_ratio": cfg.get_path("grpo.warmup_ratio", 0.05),
        "logging_steps": cfg.get_path("grpo.logging_steps", 1),
        "save_strategy": cfg.get_path("grpo.save_strategy", "steps"),
        "save_steps": cfg.get_path("grpo.save_steps", 50),
        "save_total_limit": cfg.get_path("grpo.save_total_limit", 3),
        "bf16": cfg.get_path("grpo.bf16", True),
        "gradient_checkpointing": cfg.get_path("grpo.gradient_checkpointing", True),
        "optim": cfg.get_path("grpo.optim", "paged_adamw_8bit"),
        "max_grad_norm": cfg.get_path("grpo.max_grad_norm", 0.5),
        "seed": cfg.get_path("seed", 13),
        "report_to": cfg.get_path("grpo.report_to", ["wandb"]),
        "log_completions": cfg.get_path("grpo.log_completions", True),
        # vLLM makes rollouts several times faster; needs a second GPU or
        # `vllm_mode: colocate` with the weights sharing one card.
        "use_vllm": cfg.get_path("grpo.use_vllm", False),
        "vllm_mode": cfg.get_path("grpo.vllm_mode", "colocate"),
        "vllm_gpu_memory_utilization": cfg.get_path("grpo.vllm_gpu_memory_utilization", 0.3),
    }
    grpo_config = GRPOConfig(**filter_kwargs(GRPOConfig, grpo_kwargs, "GRPOConfig"))

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    describe_trainable(trainer.model)

    dump_config(dict(cfg), out_dir / "run.yaml")
    result = trainer.train(resume_from_checkpoint=cfg.get_path("grpo.resume", None))
    trainer.save_model(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))
    (out_dir / "train_metrics.json").write_text(json.dumps(dict(result.metrics), indent=2), encoding="utf-8")
    print(f"[done ] adapter -> {out_dir / 'adapter'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
