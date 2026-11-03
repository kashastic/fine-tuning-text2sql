# One-command paths through the project. Override variables on the command line:
#   make sft CONFIG=configs/sft_qwen3_4b_reasoning.yaml

PY           ?= python
BASE_MODEL   ?= Qwen/Qwen3-4B
API_MODEL    ?= claude-opus-5-5
DATASET      ?= bird
SPLIT        ?= dev
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
        eval errors latency results test lint clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:          ## library + dev tools (laptop: enough for tests)
	$(PY) -m pip install -e ".[dev,viz,api]"
install-train:    ## add the training stack (GPU box)
	$(PY) -m pip install -e ".[train,dev,viz,api]"
install-serve:    ## add vLLM + FastAPI + the demo (GPU box)
	$(PY) -m pip install -e ".[serve,demo,viz]"

data:             ## download BIRD (and verify)
	$(PY) scripts/download_data.py --dataset bird-dev --dataset bird-train
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
		--run-name zeroshot-base $(LIMIT_FLAG)
	$(PY) scripts/evaluate.py --run $(RESULTS)/zeroshot-base --gpu $(GPU)

baseline-api:     ## week 1: the large prompted model, same prompts
	$(PY) scripts/generate.py --backend anthropic --model $(API_MODEL) \
		--dataset $(DATASET) --split $(SPLIT) --data-root $(DATA_ROOT) \
		--run-name api-baseline $(LIMIT_FLAG)
	$(PY) scripts/evaluate.py --run $(RESULTS)/api-baseline

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
eval:             ## score an existing generations file
	$(PY) scripts/evaluate.py --run $(RUN) --gpu $(GPU)
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
