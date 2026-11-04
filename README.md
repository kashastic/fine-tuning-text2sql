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
| gpt-5.4-mini | Prompted (API baseline) | 60.89 | 67.35 / 51.29 / 50.34 | | 2713 | 1.884 |
| gpt-5.4-nano | Prompted (API baseline) | 56.19 | 63.03 / 45.91 / 45.52 | | 2702 | 0.462 |

Execution accuracy (EX) uses the official BIRD metric: run both queries, compare
the result sets as sets.

Both API rows are real measured runs over all 1,534 BIRD dev questions
(`results/api-baseline/`, `results/api-baseline-mini/`), on identical prompts.
nano: 96.94% executed, every output parsed, $0.77 total. mini: 98.63%
executed, $3.12 total, +5.09 EX for 4x the money — and it gains most on the
hard end (+6.89 challenging, +6.03 moderate, +4.32 simple), which is where
schema linking and multi-join reasoning live.

The two models agree more than the gap suggests: 51.4% of questions both get
right, 34.7% neither does, and an oracle picking the better of the two would
score 65.25%. That 34.7% shared-failure core is the real target — it is not
fixed by spending more on a bigger prompted model.

**Read every single-run figure here with a ±1.5 point band.** This API
refuses `temperature=0` (*"only the default (1) value is supported"*) and
rejects `top_p`, so the baseline is sampled by force: two runs over
byte-identical prompts disagreed on **10.1%** of BIRD dev. A `--seed` is
passed and is best-effort only. The 5.09-point nano/mini gap is well outside
the band; a sub-2-point difference from one run pair is not evidence of
anything. `docs/MEASUREMENTS.md` has the arithmetic. Two of BIRD's own gold queries time out at the 30s
deadline under a loaded machine and so count against every model alike, which
caps attainable EX at 99.87%; `summary.json` records that as
`gold_unexecutable`. Its p95 latency is a hosted-API round trip from this
laptop, not a like-for-like number against a self-hosted GPU — the latency
comparison that matters comes from `make latency` against the served model.

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
make data                      # BIRD train+dev and Spider into data/raw/ (~9.1 GB)
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

  What that costs, measured over all 1,534 dev prompts: 85.8% of tables and
  **65% of columns** survive, and 49.4% of questions have at least one column
  hidden. Most of that is harmless -- the hidden column was irrelevant -- but
  where a column the gold query *needs* goes missing, the API baseline scores
  ~41% against ~57.5% on complete prompts. Of the 303 questions that lost a
  table, **none** lost a table the gold needs: table selection is sound, and
  every real loss is a dropped column.

  So column selection runs in three tiers, chosen by measuring which needed
  columns were being dropped and why:

  1. **Keys** -- primary and foreign -- are never dropped.
  2. **Columns the question or evidence names verbatim** are never dropped
     either, and the budget stretches to fit them. This was the largest
     recoverable loss: 54 of 165 missing columns were *explicitly named in
     the question* and discarded anyway, because matches used to be filled in
     schema order until the budget ran out.
  3. **Everything else competes on match strength**, with schema order
     breaking ties only, and a 4-character shared prefix counts as a match so
     "schools that were opened" reaches `OpenDate`.

  That took questions missing a needed column from **165 (10.8%) to 114
  (7.4%)** while columns kept went 64.9% -> 65.0% -- a 31% cut in the recall
  loss for 0.1% more prompt. The longest prompt grew 3,671 -> 3,823 tokens,
  still clear of 4,096.

  Re-running the baseline on the new prompts moved EX 55.80% -> 56.19%, which
  is **inside the noise band and is not claimed as a gain**: 76 answers
  improved and 70 regressed (McNemar p = 0.68). The coverage improvement is
  measured; its accuracy effect is positive in direction only. It is kept
  because the same prompts are the training data, where a missing column is
  not noise but a wrong label.

  Raising `max_columns` instead would be the wrong fix: 30 puts p99 at 4,134,
  past GRPO's 4,096 limit, and raising it for the baseline alone would mean
  the baseline and the fine-tuned model no longer see the same prompts. All of
  this matters more for training than for scoring -- these same prompts become
  the SFT and GRPO data, where a schema missing the needed column teaches the
  model to invent column names and earns ~0 GRPO reward whatever it does.
  The remaining 114 need **value** linking ("directly funded" is a value of
  `Charter`, not a column name); see `docs/MEASUREMENTS.md`.
* **One source of truth for prompt shape.** `serve/app.py` reads
  `max_tables`, `max_columns`, `sample_values` and `style` from
  `configs/data.yaml` -- the same file the dataset build reads -- and reports
  them on `/health`, so a deployed model can be checked against the recipe its
  weights were trained with. `T2S_*` env vars override, for serving a model
  trained on a different recipe. Training on 20-column schemas and serving
  80-column ones does not error; it just quietly gets worse.
* **Best checkpoint, not last.** The SFT configs evaluate and save every 100
  steps (~12 validation points over two epochs, not 2), keep the best by
  `eval_loss`, and stop after 3 evals without improvement. Keeping the last
  checkpoint means shipping an overfitted second epoch by default, and two
  eval points cannot show a loss curve turning.
* **Gold filtering** drops training rows whose own gold SQL does not execute.
  Teaching the model to imitate a query that errors is pure noise. It removes
  431 of 16,428 rows (2.6%), and the split tells you something about the two
  benchmarks: **4.5% of BIRD's golds are unexecutable against 0.04% of
  Spider's.** Sampling 1,200 BIRD rows to classify the failures put 83% on
  genuine annotation errors (`no such column: T2.CompanyName`), 15% on
  statements our read-only sandbox refuses, and 2% on the timeout. Spider's
  three are all annotation errors too -- a phantom table, and twice an
  `ORDER BY` before an `INTERSECT` that SQLite will not parse. So the
  deadline and the sandbox policy are not what is throwing data away; BIRD's
  annotation quality is.

### What is held out, and from what

The project trains on two benchmarks and reports on three splits, so "held
out" needs to mean something more precise than "not in train":

| split | rows | role |
| --- | ---: | --- |
| train | 15,697 | BIRD train + Spider train, minus val and bad gold |
| val | 300 | held out of **train**, for checkpoint selection |
| BIRD dev | 1,534 | the headline number |
| Spider dev | 1,034 | held-out **databases**, the generalisation check |
| Spider test | 2,147 | held out of training *and* of the iteration loop |

Every pair of these is database-disjoint -- verified, not assumed: BIRD train
and dev share 0 databases, Spider train and dev share 0, and BIRD and Spider
share 0 with each other. A dev answer cannot be memorised from train because
the dev schemas are not in train at all.

Two consequences worth being blunt about:

* **Spider dev measures held-out databases, not cross-benchmark transfer.**
  Putting Spider train into SFT buys 7,000 examples over 140 extra schemas,
  which is the cheapest defence against learning BIRD's conventions instead
  of SQL. The cost is that Spider dev is no longer a foreign benchmark. One
  SFT run gives you one or the other; measuring true transfer needs a
  second BIRD-only run, which does not fit the compute budget.
* **Spider test is the only number nothing was tuned against.** It is
  excluded from training *and* from every ablation, checkpoint choice and
  prompt tweak, then reported once on the final model. That is why there is
  deliberately no `make` target for it -- `make generalise` scores Spider dev,
  and spending the test set is a decision taken by hand:

  ```bash
  python scripts/generate.py --backend vllm --model Qwen/Qwen3-4B \
      --lora outputs/grpo/adapter --dataset spider --split test \
      --max-columns 20 --run-name final-spider-test
  ```

  The discipline costs nothing. Breaking it costs the number its meaning,
  quietly and without any error.

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
docs/
  MEASUREMENTS.md   every number this repo quotes, what it assumes, and the
                    decisions behind it -- read this before trusting a figure
src/text2sql/
  data.py           BIRD / Spider -> one Example shape
  schema.py         DDL from sqlite_master, schema trimming, sample values
  prompts.py        the single prompt builder every model goes through
  postprocess.py    pulling SQL out of whatever the model actually said
  sql_exec.py       read-only SQLite execution with a deadline, result comparison
  metrics.py        EX, breakdowns, self-consistency voting
  rewards.py        GRPO execution reward
  backends.py       vLLM / OpenAI-compatible / hosted OpenAI / Anthropic / transformers
  pricing.py        token and GPU-hour costs for the cost column
scripts/            download, build, generate, evaluate, train x2, merge,
                    error analysis, latency bench, results table
serve/              FastAPI service + Gradio demo
tests/              95 tests on a fixture benchmark; no GPU, no downloads
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
| API baseline, 1,534 BIRD dev questions | 2.63M in / 0.19M out (measured) | $0.77 on gpt-5.4-nano |

Budget 3–4 SFT runs and 2 GRPO runs: €50–100 of rented GPU in total. Run the
API baseline with `LIMIT=200` first, check the prompts look right, then spend
the rest.

### Choosing the API baseline

The prompted baseline is whichever hosted model you point it at -- `make
baseline-api API_BACKEND=anthropic API_MODEL=claude-haiku-5-5`, or
`API_BACKEND=openai-api API_MODEL=gpt-5.4-nano`. Cost over the full 1,534
dev questions, priced at the volumes the gpt-5.4-nano run actually billed
(2.63M input, 0.19M output — 1,714 in / 126 out per question):

| Model | Backend | Cost | What the comparison then claims |
| --- | --- | ---: | --- |
| `claude-opus-5-5` | `anthropic` | $14.39 | gap to a frontier model |
| `claude-sonnet-5-5` | `anthropic` | $7.20 | gap to a strong mid-tier model |
| `gpt-5.4-mini` | `openai-api` | **$3.12 (measured)** | gap to a strong cheap model |
| `gpt-5.4-nano` | `openai-api` | **$0.77 (measured)** | gap to a cheap hosted model |
| `claude-haiku-5-5` | `anthropic` | $0.36 | gap to a cheap hosted model |

The nano and mini rows are measured; the others scale nano's token volumes by
list price, so they are estimates in one specific way — a chattier or
reasoning-heavier model emits more output tokens, and output is the expensive
side. mini is the worked example: estimated at $2.84 from nano's 126 output
tokens per answer, it actually billed 166, so it came in at $3.12. Treat the
unmeasured rows as lower bounds.

Picking a cheap baseline changes the headline. "Three points behind a
frontier model" is a result; "beats a small hosted model" is a much weaker
one, because the fine-tuned 4B is itself a small cheap model. The honest
framing for a cheap baseline is cost-per-query at matched accuracy, not a
gap to the state of the art.

Two cost caveats. On hosted reasoning models the billed output includes
reasoning tokens, so the real output volume can be several times the
visible tokens per answer. `reasoning_effort` is the lever, and the
accepted values are model-specific -- gpt-5.4-nano takes
`none|low|medium|high|xhigh` and returns a 400 for the `minimal` that
older GPT-5 models accepted. The default here is `low`, measured at zero
reasoning tokens on text-to-SQL prompts, so the baseline costs what its
visible answers cost while still being free to reason on a hard question;
`medium` starts spending them. And the token count comes from
`data/processed/dev.jsonl` at `max_columns: 20`, so it moves with the
schema budget -- the baseline is run at that same budget, or it is
answering easier prompts than the fine-tuned model ever sees.

Keys: `ANTHROPIC_API_KEY` from console.anthropic.com or `OPENAI_API_KEY`
from platform.openai.com. API billing is separate from a Claude or ChatGPT
subscription, and the prompted baseline is the only part of this project
that needs a key at all.

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
