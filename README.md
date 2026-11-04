# Fine-tuning a 4B model for text-to-SQL (SFT → GRPO)

Fine-tune a small open model (~4B) for text-to-SQL, first with QLoRA supervised
fine-tuning, then with reinforcement learning from **execution feedback**
(GRPO), and measure it against a large prompted model on accuracy, latency and
cost.

**Research question.** Can a 4B model trained with SFT + GRPO on BIRD close
most of the gap to a large prompted model, while staying cheap enough to
self-host?

The reward is what makes this more than a fine-tuning tutorial: a query scores
1.0 only when *running it reproduces the gold result set*. Syntax and format
earn almost nothing, so the model is optimised for being right rather than for
looking right.

## Results

Numbers land here as each run finishes — `make results` regenerates the table
from `results/*/summary.json`.

| Model | Method | BIRD dev EX (%) | Simple / Moderate / Challenging (%) | Spider dev EX (%) | p95 latency (ms) | Cost per 1k queries (EUR) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | Zero-shot | | | | | |
| Qwen3-4B | QLoRA SFT | | | | | |
| Qwen3-4B | SFT + reasoning traces | | | | | |
| Qwen3-4B | SFT + GRPO | | | | | |
| Qwen3-4B | SFT + GRPO + self-consistency (8) | | | | | |
| Large API model | Prompted | | | | | |

Execution accuracy (EX) uses the official BIRD metric: run both queries, compare
the result sets as sets.

## Quickstart

```bash
pip install -e ".[dev,viz,api]"   # laptop: enough to run the tests
make test
```

```bash
pip install -e ".[train,serve,viz,api]"   # GPU box
make data && make dataset
```

```bash
make baseline-zeroshot LIMIT=200
```

## The six weeks, as commands

| Week | What | Command |
| --- | --- | --- |
| 1 | Data on disk, evaluation harness, two baselines | `make data && make verify`, `make baseline-zeroshot`, `make baseline-api` |
| 2 | Training set: trimmed schemas, chat format, held-out val | `make dataset` |
| 3 | QLoRA SFT, two variants | `make sft`, `make sft-reasoning` |
| 4 | GRPO from the best SFT checkpoint | `make grpo` |
| 5 | Error analysis, merge, serve, measure | `make errors RUN=results/grpo`, `make merge`, `make serve`, `make api`, `make latency` |
| 6 | Package: results table, model card, demo | `make results`, `make demo` |

Every week ends with something runnable and committed, so the project is never
half-finished.

### Week 1 — baselines

```bash
make data                      # BIRD train + dev into data/raw/
make verify                    # questions and SQLite files line up
make baseline-zeroshot         # base model, BIRD dev
make baseline-api LIMIT=200    # the prompted baseline, same prompts
```

Both baselines go through `scripts/generate.py`, which builds prompts with
`text2sql.prompts.build_prompt`. One builder for every model is deliberate: an
API baseline given a different schema format would make the headline comparison
meaningless.

Generation and scoring are separate steps. Generation costs GPU time or API
tokens; scoring is free and gets re-run whenever the comparison logic changes.

```bash
python scripts/evaluate.py --run results/zeroshot-base --gpu a100-40gb
```

### Week 2 — data

```bash
make dataset                        # plain-SQL targets
make dataset-reasoning              # reasoning-then-SQL targets
```

`scripts/build_dataset.py` writes `data/processed/{train,val,dev}.jsonl` with
one chat record per question, plus `db_path` and `gold_sql` columns — GRPO's
reward function reads those straight off the dataset.

Two things it does that matter more than they look:

* **Schema trimming** works at two levels, because BIRD's cost is width as
  much as count. `data.max_tables` keeps the tables the question plausibly
  needs plus their foreign-key neighbours; `data.max_columns` then trims
  *within* a table, since `european_football_2.Match` alone has 115 columns.
  Keys are never dropped -- without them a join stops being expressible --
  and a trimmed table says in its DDL how many columns were elided, so the
  model is told the schema is partial rather than inferring a column does
  not exist. Measured on BIRD dev with the Qwen3 tokenizer:

  | `max_columns` | p50 | p90 | p99 | max | over 3072 | over 4096 | gold columns kept |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 0 (untrimmed) | 2002 | 4281 | 6651 | 6661 | 33.7% | 17.3% | 99.3% |
  | **20** (default) | 1958 | 3037 | 3615 | 3631 | 6.3% | 0.0% | 97.6% |
  | 12 | 1761 | 2585 | 3171 | 3181 | 5.7% | 0.0% | 97.2% |

  Note the p50 barely moves: trimming only bites on the wide tables, so the
  median prompt is untouched. Below 20 the savings flatten while recall keeps
  bleeding, which is why 20 is the default. With it, the longest prompt in
  BIRD dev is 3,671 tokens and the longest prompt-plus-gold 3,828, so nothing
  is truncated at either trainer's limit.
* **Gold filtering** drops training rows whose own gold SQL does not execute.
  Teaching the model to imitate a query that errors is pure noise.

`data.val_size` holds out a validation split from **train**. BIRD dev is never
touched until final evaluation.

### Week 3 — supervised fine-tuning

```bash
make sft                                      # configs/sft_qwen3_4b.yaml
make sft CONFIG=configs/sft_qwen3_4b_reasoning.yaml
```

QLoRA (4-bit NF4 base, LoRA r=32) via TRL's `SFTTrainer`. The config is copied
into the output directory, so any checkpoint can be traced back to its recipe.
Roughly 3–6 GPU hours per run on one A100 40GB.

### Week 4 — GRPO on execution reward

```bash
make grpo   # configs/grpo_qwen3_4b.yaml, starting from outputs/sft-plain/adapter
```

The reward (`src/text2sql/rewards.py`):

| Component | Weight | Why |
| --- | ---: | --- |
| Result set matches gold | 1.00 | The only thing that actually matters |
| Query executes | 0.10 | A gradient out of the "everything errors" regime |
| Parseable ```sql block | 0.05 | Keeps the output format from drifting |

Partial credit stays small on purpose. Generous syntax bonuses teach a model to
emit safe, valid, wrong SQL — the usual way this stage produces a flat reward
curve and no accuracy gain.

The script **refuses to start without an SFT adapter**. From a base model the
execution reward is near zero on nearly every rollout, every completion in a
group scores the same, the advantage is zero and nothing moves — and the loss
curve looks perfectly healthy while it happens. Pass `--allow-base-model` if you
want to see that for yourself.

When a gold query does not execute, the model keeps its executability credit
rather than being punished for our data.

### Week 5 — error analysis, serving, measurement

```bash
make errors RUN=results/grpo              # categories + chart + markdown table
make merge ADAPTER=outputs/grpo/adapter   # LoRA -> merged weights
make serve                                # vLLM, OpenAI-compatible, port 8000
make api                                  # FastAPI wrapper, port 8080
make latency                              # p50/p95/throughput + cost per 1k
```

Failures are sorted into schema linking, joins, aggregation, filter values,
ordering/limits, projection, plus the mechanical ones (nothing parsed, query
errored, timed out). A broken gold query gets its own category and is never
counted against the model.

Latency comes from a **running server under concurrency**, not from offline
batched generation, which hides queueing and would flatter p95.

```bash
curl -s localhost:8080/sql -H 'content-type: application/json' -d '{
  "question": "How many schools in Alameda county are charter schools?",
  "db_id": "california_schools", "execute": true}' | jq
```

### Week 6 — packaging

```bash
make results      # results/RESULTS.md from every summary.json
make demo         # Gradio: question in, SQL and rows out
python scripts/merge_lora.py --base Qwen/Qwen3-4B --adapter outputs/grpo/adapter \
    --out outputs/qwen3-4b-text2sql-merged --push-to-hub <user>/qwen3-4b-text2sql
```

`MODEL_CARD.md` is the template to publish with the weights.

## Layout

```
configs/            run recipes (data, SFT x2, GRPO) -- the whole run in one file
src/text2sql/
  data.py           BIRD / Spider -> one Example shape
  schema.py         DDL from sqlite_master, schema trimming, sample values
  prompts.py        the single prompt builder every model goes through
  postprocess.py    pulling SQL out of whatever the model actually said
  sql_exec.py       read-only SQLite execution with a deadline, result comparison
  metrics.py        EX, breakdowns, self-consistency voting
  rewards.py        GRPO execution reward
  backends.py       vLLM / OpenAI-compatible / Anthropic / transformers
  pricing.py        token and GPU-hour costs for the cost column
scripts/            download, build, generate, evaluate, train x2, merge,
                    error analysis, latency bench, results table
serve/              FastAPI service + Gradio demo
tests/              75 tests on a fixture benchmark; no GPU, no downloads
```

## Reproduction

```bash
git clone <repo> && cd fine-tuning-text2sql
pip install -e ".[train,serve,viz,api]"
make data dataset
make baseline-zeroshot baseline-api
make sft && make grpo
make merge && make results
```

Every run writes `run.yaml` (resolved config), `generation_meta.json`,
`predictions.jsonl` and `summary.json` into `results/<run-name>/`, so a number
in the table can always be traced back to the exact command and prompts.

## Costs

| Stage | Compute | Rough cost |
| --- | --- | --- |
| Zero-shot + SFT evaluation sweeps | ~1 GPU hour each | €0.50–1.50 |
| SFT run | 3–6 GPU hours | €2–9 |
| GRPO run | 10–25 GPU hours | €5–35 |
| API baseline, 1,534 BIRD dev questions | ~4.5M input tokens | €15–25 |

Budget 3–4 SFT runs and 2 GRPO runs: €50–100 of rented GPU in total. Run the
API baseline with `LIMIT=200` first, check the prompts look right, then spend
the rest.

## Pitfalls this repo is built to avoid

* **Training on dev by accident.** The validation split comes out of train;
  nothing reads BIRD dev except evaluation.
* **An unfair baseline.** One prompt builder, same schema and evidence for every
  model. `tests/test_prompts.py` fails if that stops being true.
* **Context overflow.** Schema trimming is a config knob applied from week 2,
  not a patch bolted on in week 4, and it is trimmed by column as well as by
  table. Untrimmed, a third of BIRD dev overruns GRPO's prompt budget -- and a
  truncated prompt hides the schema the answer needs, so the reward is ~0
  however well the model does. Those rollouts teach nothing and still cost GPU
  time, while the loss curve looks perfectly healthy.
* **GRPO from the base model.** Refused by default.
* **Hyperparameter rabbit holes.** Two or three SFT runs is enough; the story
  matters more than one extra point.
* **Runaway queries.** Every execution is read-only, single-statement and under
  a wall-clock deadline — including thousands of untrusted rollouts per GRPO
  step.

This repository is deliberately self-contained and public-safe: no employer
data, no internal code.

## Sources

* [Reasoning-aware SFT and RL for small text-to-SQL models](https://arxiv.org/html/2504.15077v5)
* [On-prem open LLMs on BIRD: size and technique frontier (2026)](https://arxiv.org/abs/2606.29733)
* [Small fine-tuning sets for text-to-SQL (Towards AI)](https://pub.towardsai.net/your-text-to-sql-model-doesnt-need-more-data-it-needs-to-meet-every-table-970b66d9d3af)
* [FINER-SQL: GRPO for small text-to-SQL models](https://huggingface.co/thanhdath/FINER-SQL-0.5B-Spider/blob/main/README.md)
* [BIRD benchmark](https://bird-bench.github.io/) · [Spider](https://yale-lily.github.io/spider)
