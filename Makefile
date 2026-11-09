# One-command paths through the project. Override variables on the command line:
#   make sft CONFIG=configs/sft_qwen3_4b_reasoning.yaml

PY           ?= python
BASE_MODEL   ?= Qwen/Qwen3-4B
# The prompted baseline. `openai-api` is a hosted OpenAI model, `anthropic` a
# hosted Claude one; both get the same prompts as the fine-tuned model. See
# "Choosing the API baseline" in the README for what each one costs on BIRD dev.
API_BACKEND  ?= openai-api
API_MODEL    ?= gpt-5.4-nano
# Must match configs/data.yaml, or the baseline is answering easier prompts
# than the fine-tuned model ever sees.
MAX_COLUMNS  ?= 20
DATASET      ?= bird
SPLIT        ?= dev
# `make generalise` scores a fine-tuned adapter on Spider dev. There is
# deliberately no target for Spider *test*: it is held out of training and of
# the iteration loop (docs/MEASUREMENTS.md, D15), so spending it is a one-off
# decision taken by hand, not something a tab-complete can do by accident.
ADAPTER      ?=
RUN_NAME     ?= spider-dev
DATA_ROOT    ?= data/raw
RESULTS      ?= results
GPU          ?= a100-40gb
LIMIT        ?=
MODEL        ?= outputs/qwen3-4b-text2sql-merged
SERVED_NAME  ?= qwen3-4b-text2sql
PORT         ?= 8000
API_PORT     ?= 8080

LIMIT_FLAG := $(if $(LIMIT),--limit $(LIMIT),)

.PHONY: help install install-train install-serve data verify dataset dataset-reasoning \
        baseline-zeroshot baseline-api sft sft-reasoning grpo merge serve api demo \
        eval generalise recall compare errors latency results test lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:          ## library + dev tools (laptop: enough for tests)
	$(PY) -m pip install -e ".[dev,viz,api]"
install-train:    ## add the training stack (GPU box)
	$(PY) -m pip install -e ".[train,dev,viz,api]"
install-serve:    ## add vLLM + FastAPI + the demo (GPU box)
	$(PY) -m pip install -e ".[serve,demo,viz]"

data:             ## download BIRD and Spider (and verify)
	$(PY) scripts/download_data.py --dataset bird-dev --dataset bird-train --dataset spider
verify:           ## check questions and SQLite files line up
	$(PY) scripts/download_data.py --verify-only

dataset:          ## build data/processed/*.jsonl (plain SQL targets)
	$(PY) scripts/build_dataset.py --config configs/data.yaml
dataset-reasoning: ## same, with reasoning-then-SQL targets
	$(PY) scripts/build_dataset.py --config configs/data.yaml \
		--style reasoning --out-dir data/processed-reasoning

baseline-zeroshot: ## week 1: zero-shot base model on BIRD dev
	$(PY) scripts/generate.py --backend vllm --model $(BASE_MODEL) \
		--dataset $(DATASET) --split $(SPLIT) --data-root $(DATA_ROOT) \
		--max-columns $(MAX_COLUMNS) \
		--run-name zeroshot-base $(LIMIT_FLAG)
	$(PY) scripts/evaluate.py --run $(RESULTS)/zeroshot-base --gpu $(GPU)

# Changing the prompt recipe invalidates every baseline measured on the old
# one (assumption A5: every compared model sees identical prompts), so this
# has been re-run twice already and will be again. Name each re-run rather
# than overwriting the last, and `make compare` the pair:
#   make baseline-api API_RUN=api-baseline-v4
#   make compare BEFORE=results/api-baseline-v3 AFTER=results/api-baseline-v4
# Needs OPENAI_API_KEY in the environment; keep the key outside this repo.
API_RUN      ?= api-baseline
baseline-api:     ## week 1: the prompted API baseline, same prompts
	$(PY) scripts/generate.py --backend $(API_BACKEND) --model $(API_MODEL) \
		--dataset $(DATASET) --split $(SPLIT) --data-root $(DATA_ROOT) \
		--max-columns $(MAX_COLUMNS) \
		--run-name $(API_RUN) $(LIMIT_FLAG)
	$(PY) scripts/evaluate.py --run $(RESULTS)/$(API_RUN)

CONFIG ?= configs/sft_qwen3_4b.yaml
sft:              ## week 3: QLoRA SFT
	$(PY) scripts/train_sft.py --config $(CONFIG)
sft-reasoning:    ## week 3: the reasoning variant
	$(PY) scripts/train_sft.py --config configs/sft_qwen3_4b_reasoning.yaml \
		--set data.train_file=data/processed-reasoning/train.jsonl \
		--set data.val_file=data/processed-reasoning/val.jsonl

grpo:             ## week 4: GRPO on execution reward
	$(PY) scripts/train_grpo.py --config configs/grpo_qwen3_4b.yaml

ADAPTER ?= outputs/grpo/adapter
merge:            ## week 5: merge the adapter for serving
	$(PY) scripts/merge_lora.py --base $(BASE_MODEL) --adapter $(ADAPTER) --out $(MODEL)

serve:            ## week 5: vLLM OpenAI-compatible server
	$(PY) -m vllm.entrypoints.openai.api_server --model $(MODEL) \
		--served-model-name $(SERVED_NAME) --port $(PORT) --max-model-len 8192
api:              ## week 5: the FastAPI wrapper (needs `make serve` running)
	T2S_MODEL=$(SERVED_NAME) T2S_VLLM_URL=http://localhost:$(PORT)/v1 \
		$(PY) -m uvicorn serve.app:app --host 0.0.0.0 --port $(API_PORT)
demo:             ## week 6: the Gradio demo (needs `make api` running)
	$(PY) serve/demo_gradio.py --api http://localhost:$(API_PORT)

RUN ?= $(RESULTS)/zeroshot-base
recall:           ## what schema trimming costs: needed columns/tables lost
	$(PY) scripts/schema_recall.py --dataset $(DATASET) --split $(SPLIT) \
		--data-root $(DATA_ROOT) --max-columns $(MAX_COLUMNS) \
		--tokenizer $(BASE_MODEL) $(LIMIT_FLAG)
BEFORE ?= $(RESULTS)/api-baseline-v2
AFTER  ?= $(RESULTS)/api-baseline-v3
compare:          ## paired EX difference between two scored runs + McNemar
	$(PY) scripts/compare_runs.py $(BEFORE) $(AFTER)
eval:             ## score an existing generations file
	$(PY) scripts/evaluate.py --run $(RUN) --gpu $(GPU)
generalise:       ## the cross-benchmark check: ADAPTER on Spider dev
	$(PY) scripts/generate.py --backend vllm --model $(BASE_MODEL) \
		$(if $(ADAPTER),--lora $(ADAPTER),) \
		--dataset spider --split dev --data-root $(DATA_ROOT) \
		--max-columns $(MAX_COLUMNS) \
		--run-name $(RUN_NAME) $(LIMIT_FLAG)
	$(PY) scripts/evaluate.py --run $(RESULTS)/$(RUN_NAME) --gpu $(GPU)
errors:           ## categorise failures + chart
	$(PY) scripts/error_analysis.py --run $(RUN) --sample 100
latency:          ## p50/p95 + throughput against the running server
	$(PY) scripts/bench_latency.py --base-url http://localhost:$(PORT)/v1 \
		--model $(SERVED_NAME) --gpu $(GPU)
results:          ## collect every summary.json into results/RESULTS.md
	$(PY) scripts/make_results_table.py --results-dir $(RESULTS)

test:             ## unit tests (no GPU, no datasets needed)
	$(PY) -m pytest -q
lint:
	$(PY) -m ruff check src scripts serve tests

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
