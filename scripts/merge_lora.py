#!/usr/bin/env python3
"""Merge a LoRA adapter into the base weights for serving.

    python scripts/merge_lora.py --base Qwen/Qwen3-4B \
        --adapter outputs/grpo/adapter --out outputs/qwen3-4b-text2sql-merged

vLLM can serve an adapter directly, but a merged model is what you push to the
Hub and what quantisation tools expect. Runs on CPU, slowly, if you have RAM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.adapter.exists():
        print(f"[fail] adapter not found: {args.adapter}")
        return 1

    print(f"[load ] {args.base} ({args.dtype}, {args.device_map})")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=getattr(torch, args.dtype), device_map=args.device_map
    )
    print(f"[load ] adapter {args.adapter}")
    model = PeftModel.from_pretrained(model, str(args.adapter))
    print("[merge] merge_and_unload")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.adapter) if (args.adapter / "tokenizer_config.json").exists() else args.base
    )
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), safe_serialization=True)
    tokenizer.save_pretrained(str(args.out))
    print(f"[done ] merged model -> {args.out}")

    if args.push_to_hub:
        print(f"[push ] {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub, private=args.private)
        tokenizer.push_to_hub(args.push_to_hub, private=args.private)
    return 0


if __name__ == "__main__":
    sys.exit(main())
