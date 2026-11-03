---
license: apache-2.0
base_model: Qwen/Qwen3-4B
tags: [text-to-sql, sql, qlora, grpo, bird, spider]
datasets: [bird-bench, spider]
language: [en]
pipeline_tag: text-generation
---

# Qwen3-4B text-to-SQL (QLoRA SFT + GRPO)

<!-- Fill the bracketed values from results/RESULTS.md before publishing. -->

A 4B model fine-tuned for text-to-SQL over SQLite: QLoRA supervised fine-tuning
on BIRD + Spider, then GRPO with a reward based on **executing** the generated
query and comparing its result set with the gold one.

## Results

| Model | Method | BIRD dev EX (%) | Spider dev EX (%) |
| --- | --- | ---: | ---: |
| Qwen3-4B | Zero-shot | [ ] | [ ] |
| **This model** | SFT + GRPO | **[ ]** | **[ ]** |
| [large API model] | Prompted | [ ] | [ ] |

Execution accuracy uses the official BIRD metric (set comparison of result rows).

## Intended use

Turning natural-language questions into SQLite queries against a schema given in
the prompt. Not intended to run queries unsupervised against production
databases, and not evaluated on dialects other than SQLite.

## Prompt format

The model expects the chat format it was trained on: a system turn asking for
one SQLite query in a ```sql block, and a user turn holding the schema
(CREATE TABLE statements, with a few sample values as comments), an optional
evidence hint, and the question. See `src/text2sql/prompts.py` in the repo.

## Training

| | |
| --- | --- |
| Base | Qwen/Qwen3-4B |
| SFT | QLoRA, 4-bit NF4, LoRA r=32 α=64, lr 1e-4, [N] epochs |
| RL | GRPO, [N] generations per prompt, lr 1e-6, KL β=0.02 |
| Reward | 1.0 result match · 0.10 executes · 0.05 parseable SQL block |
| Data | BIRD train + Spider train, schemas trimmed to [N] tables |
| Hardware | [GPU], [N] GPU hours total |

## Limitations

* SQLite only; other dialects are untested.
* Accuracy drops on schemas larger than the trimming budget used in training.
* The execution reward optimises for matching result sets, so a query can be
  right for the wrong reason on a small table.
* Inherits whatever biases and gaps BIRD and Spider carry.

## Citation

Repository: [link] · Benchmarks: BIRD, Spider.
