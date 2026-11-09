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
| gpt-5.4-mini | Prompted (API baseline) | 61.73 | 68.43 / 52.37 / 48.97 | | 2398 | 1.882 |
| gpt-5.4-nano | Prompted (API baseline) | 56.26 | 63.57 / 45.91 / 42.76 | | 3137 | 0.470 |

Execution accuracy (EX) uses the official BIRD metric: run both queries, compare
the result sets as sets.

> **Both API rows were measured on the prompts the fine-tuned model will
> see** (`results/api-baseline-v3/`, `results/api-baseline-mini-v3/`). The
> assumption this whole table rests on is that every model compared sees the
> same prompt, and the schema-linking fixes below changed the prompts twice,
> so both baselines were re-run — $3.90 in total. Four earlier runs on older
> prompts are kept in `docs/MEASUREMENTS.md` rather than deleted: comparing
> them is how this project measured its own noise floor.

Both API rows are real measured runs over all 1,534 BIRD dev questions.
nano: 97.39% executed, $0.78. mini: 99.35% executed, $3.12, **+5.48 EX for
~4x the money** — and it gains most on the hard end, which is where schema
linking and multi-join reasoning live.

**Re-running these across prompt changes is how the project knows what a real
difference looks like.** Six prompted runs now exist across three prompt
recipes, and none of the prompt-to-prompt steps is statistically significant
— while the questions whose prompts did *not* change flipped by 7–10% anyway,
because this API refuses greedy decoding and samples at temperature 1. Two
things came out of that which are worth more than the re-baselining:

- **The noise floor is model-specific.** nano churns 9.9% on byte-identical
  prompts, mini 7.2% — bands of ±1.6 and ±1.4 EX points. A single-run
  difference smaller than that is not evidence.
- **A prompt missing a column the answer needs costs nano 33 EX points and
  mini 24** (57.17% → 23.81%, 62.40% → 38.10%). Paying 4x for a bigger model
  buys back only part of it, so schema linking is an *information* problem
  before it is a capability problem. This is the measurement that justifies
  the schema work below.

The two models agree more than the gap suggests: 51.11% of questions both get
right, 33.12% neither does, and an oracle picking the better of the two would
score 66.88%. That 33.12% shared-failure core is the real target — it is not
fixed by spending more on a bigger prompted model. Both rows sit on the same
prompts, so this comparison is exact rather than indicative.

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

  What that costs has to be measured against the columns the selector
  actually chose, not by searching the rendered prompt for the gold's column
  names. The repo did the latter for two sessions, and it was wrong in both
  directions: it reported 114 questions missing a needed column where the
  true number was 125, and **zero** questions missing a needed *table* where
  the true number was 24. A dropped table's name still appears in the prompt
  inside other tables' foreign-key clauses, so the proxy could not see the
  worse of the two failures. `scripts/schema_recall.py` now measures it
  exactly; the earlier figure is kept in `docs/MEASUREMENTS.md` with the
  correction, because the mistake is more instructive than the number.

  Column selection then runs in tiers, each one chosen by looking at which
  needed columns were being dropped and why:

  1. **Keys are never dropped** -- primary, outbound foreign, *and inbound*:
     columns other tables point at. A join has two ends and only one declares
     the constraint, so `legalities.uuid REFERENCES cards(uuid)` makes `uuid`
     a key of `cards`, which declares no foreign key and whose primary key is
     `id`. Missing that, `cards.uuid` competed as an ordinary column against
     a question about card rarity and lost -- and the prompt then showed both
     tables with no way to join them. **59 questions in BIRD dev were that
     one column.**
  2. **Columns the question names, by name or by content.** A verbatim name
     is never dropped and the budget stretches to fit it. Neither is a column
     whose *values* the question quotes: "the school run by Kacey Gibson"
     needs `AdmFName1`, and nothing about that question resembles that
     column's name. See value linking below.
  3. **Everything else competes on match strength**, with schema order
     breaking ties only, and a 4-character shared prefix counting as a match
     so "schools that were opened" reaches `OpenDate`. The same prefix rule
     applies to **table** names, which it originally did not: `race != races`
     dropped the `races` table outright on "the race No. 901", and a lost
     table is strictly worse than a lost column, since no column tier can
     recover it.

  Measured on full BIRD dev, one fix at a time:

  | | missing a column | missing a table |
  | --- | ---: | ---: |
  | before | 125 (8.15%) | 24 (1.56%) |
  | + inflected table names | 116 (7.56%) | 15 (0.98%) |
  | + value linking | 97 (6.32%) | 6 (0.39%) |
  | **+ inbound foreign keys** | **42 (2.74%)** | **6 (0.39%)** |

  **A 66% cut in column loss and a 75% cut in table loss, for five median
  tokens.** The tail does not move at all: p90, p99, the longest prompt
  (3,810) and the longest prompt-plus-gold (3,878) are unchanged, and 0% of
  prompts exceed 4,096 either way. The gain is not bought with width -- a
  protected key or a value-matched column *displaces* a weak name match
  inside the same budget rather than adding to it.

  Raising `max_columns` instead would be the wrong fix: 30 puts p99 at 4,134,
  past GRPO's 4,096 limit, and raising it for the baseline alone would mean
  the baseline and the fine-tuned model no longer see the same prompts. All
  of this matters more for training than for scoring -- these same prompts
  become the SFT and GRPO data, where a schema missing the needed column
  teaches the model to invent column names and earns ~0 GRPO reward whatever
  it does. The 42 that remain need meaning rather than matching ("under
  whose administration" -> `AdmFName1`), which is an embedding or LLM linker.
* **Value linking.** Schema trimming scores columns on how much their *name*
  overlaps the question, which cannot see the largest class of failure left:
  the question naming a column's **contents**. "Continuation schools" needs
  `frpm.Educational Option Type`, whose value is "Continuation School";
  "Lucas di Grassi's Q1 result" needs the `drivers` table, which no
  name-based score reaches at all. So `text2sql.values` indexes each
  database's distinct short text values once and looks the question's word
  n-grams up in that index -- a hit means the question is quoting the
  database, which is a far stronger signal than any name overlap.

  It stays affordable on BIRD's 30 GB of training databases by indexing only
  TEXT columns and only short values, capping distinct values per column, and
  caching to disk keyed by file size and mtime: **7.7 seconds for all 11
  BIRD dev databases**, 0.5–19.3s each for the four largest training ones,
  and nothing on a rebuild. Two details matter more than they look --
  singularising both sides, or "continuation schools" never meets
  "Continuation School"; and resolving overlapping matches longest-first,
  because "Los Angeles" identifies `schools.City` while "Angeles" alone is
  somebody's surname. It fails open: an unreadable database costs the hits,
  not the prompt.
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
  values.py         value linking: which columns' *contents* the question names
  prompts.py        the single prompt builder every model goes through
  postprocess.py    pulling SQL out of whatever the model actually said
  sql_exec.py       read-only SQLite execution with a deadline, result comparison
  metrics.py        EX, breakdowns, self-consistency voting
  rewards.py        GRPO execution reward
  backends.py       vLLM / OpenAI-compatible / hosted OpenAI / Anthropic / transformers
  pricing.py        token and GPU-hour costs for the cost column
scripts/            download, build, generate, evaluate, train x2, merge,
                    error analysis, latency bench, results table,
                    schema_recall.py (what trimming costs and gains),
                    compare_runs.py (is a difference real, or this API's noise?)
serve/              FastAPI service + Gradio demo
tests/              117 tests on a fixture benchmark; no GPU, no downloads.
                    Six of them run train_sft.py and train_grpo.py end to end
                    on a ~10M-param random Qwen3, on CPU, in ~25s -- a wiring
                    check, not a quality one. They skip without the `train`
                    extra installed.
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
