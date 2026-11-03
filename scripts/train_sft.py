#!/usr/bin/env python3
"""QLoRA supervised fine-tuning on the BIRD + Spider training set.

    python scripts/train_sft.py --config configs/sft_qwen3_4b.yaml
    python scripts/train_sft.py --config configs/sft_qwen3_4b_reasoning.yaml \
        --set train.num_train_epochs=2 --set run_name=sft-r2

Everything that defines the run lives in the YAML config, which is copied into
the output directory so a checkpoint can always be traced back to its recipe.
Needs a GPU: `pip install -e ".[train]"`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _trainer_utils import describe_trainable, filter_kwargs
from text2sql.config import dump_config, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    import torch
    from datasets import load_dataset as hf_load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    model_name = cfg.get_path("model.name", "Qwen/Qwen3-4B")
    run_name = cfg.get_path("run_name", "sft")
    out_dir = Path(cfg.get_path("output_dir", f"outputs/{run_name}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_file = cfg.get_path("data.train_file", "data/processed/train.jsonl")
    val_file = cfg.get_path("data.val_file", "data/processed/val.jsonl")
    if not Path(train_file).exists():
        print(f"[fail] {train_file} not found; run scripts/build_dataset.py first")
        return 1

    data_files = {"train": train_file}
    if Path(val_file).exists():
        data_files["validation"] = val_file
    dataset = hf_load_dataset("json", data_files=data_files)
    # SFT trains on the full chat turn; the extra columns are for GRPO/eval.
    keep = ["messages"]
    dataset = dataset.remove_columns([c for c in dataset["train"].column_names if c not in keep])
    print(f"[data ] train={len(dataset['train'])}"
          + (f" val={len(dataset['validation'])}" if "validation" in dataset else ""))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if cfg.get_path("model.load_in_4bit", True):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg.get_path("model.bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=cfg.get_path("model.double_quant", True),
            bnb_4bit_compute_dtype=getattr(
                torch, cfg.get_path("model.compute_dtype", "bfloat16")
            ),
        )

    peft_config = LoraConfig(
        r=cfg.get_path("lora.r", 32),
        lora_alpha=cfg.get_path("lora.alpha", 64),
        lora_dropout=cfg.get_path("lora.dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.get_path(
            "lora.target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    sft_kwargs = {
        "output_dir": str(out_dir),
        "run_name": run_name,
        "max_length": cfg.get_path("train.max_seq_length", 4096),
        "packing": cfg.get_path("train.packing", False),
        "num_train_epochs": cfg.get_path("train.num_train_epochs", 2),
        "per_device_train_batch_size": cfg.get_path("train.per_device_train_batch_size", 2),
        "gradient_accumulation_steps": cfg.get_path("train.gradient_accumulation_steps", 8),
        "learning_rate": cfg.get_path("train.learning_rate", 1e-4),
        "lr_scheduler_type": cfg.get_path("train.lr_scheduler_type", "cosine"),
        "warmup_ratio": cfg.get_path("train.warmup_ratio", 0.03),
        "logging_steps": cfg.get_path("train.logging_steps", 10),
        "save_strategy": cfg.get_path("train.save_strategy", "epoch"),
        "save_total_limit": cfg.get_path("train.save_total_limit", 3),
        "eval_strategy": cfg.get_path("train.eval_strategy", "epoch" if "validation" in dataset else "no"),
        "bf16": cfg.get_path("train.bf16", True),
        "gradient_checkpointing": cfg.get_path("train.gradient_checkpointing", True),
        "optim": cfg.get_path("train.optim", "paged_adamw_8bit"),
        "max_grad_norm": cfg.get_path("train.max_grad_norm", 1.0),
        "seed": cfg.get_path("seed", 13),
        "report_to": cfg.get_path("train.report_to", ["wandb"]),
        "model_init_kwargs": {
            "quantization_config": quant_config,
            "dtype": cfg.get_path("model.dtype", "auto"),
            "device_map": cfg.get_path("model.device_map", "auto"),
            "attn_implementation": cfg.get_path("model.attn_implementation", "sdpa"),
        },
    }
    sft_config = SFTConfig(**filter_kwargs(SFTConfig, sft_kwargs, "SFTConfig"))

    trainer = SFTTrainer(
        model=model_name,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    describe_trainable(trainer.model)

    dump_config(dict(cfg), out_dir / "run.yaml")
    result = trainer.train(resume_from_checkpoint=cfg.get_path("train.resume", None))
    trainer.save_model(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))

    metrics = dict(result.metrics)
    (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[done ] adapter -> {out_dir / 'adapter'}")
    print(f"[next ] python scripts/generate.py --backend vllm --model {model_name} "
          f"--lora {out_dir / 'adapter'} --dataset bird --split dev --run-name {run_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
