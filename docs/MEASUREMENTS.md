# Measurements, assumptions and decisions

Every number this project quotes, where it came from, and what it assumes.
Kept because most of these figures look like facts and are actually
measurements taken under specific conditions -- prices drift, timeouts depend
on machine load, and a proxy metric is not the thing it proxies.

Dates are absolute. "Measured" means a real run whose artifacts are in
`results/`; "estimated" means arithmetic over someone else's numbers.

Last updated 2026-11-04.

---

## 1. Standing assumptions

These apply to every number below.

| # | Assumption | Why it matters | How to check it |
| --- | --- | --- | --- |
| A1 | **API list prices as of 2026-11-04**, USD per million tokens, from `src/text2sql/pricing.py`. | Every cost figure scales linearly with these. A stale price silently skews the headline cost comparison. | Re-check the providers' pricing pages; the table carries its own "Checked" date. |
| A2 | **USD_PER_EUR = 1.08.** The plan's budget is in EUR, the prices are in USD. | Every EUR figure, including `results/RESULTS.md`'s cost column. | `pricing.USD_PER_EUR`; override from config. |
| A3 | **EX = BIRD's official metric**: execute both queries, compare result sets as sets. Not string match, not ordered comparison. | A different comparison mode gives a different number and makes us incomparable to published work. | `ComparisonMode.BIRD` in `src/text2sql/sql_exec.py`. |
| A4 | **30-second execution timeout**, applied to both gold and prediction. | Two BIRD gold queries exceed it under load (see M8). A different deadline changes EX slightly. | `--timeout` on `scripts/evaluate.py`; default 30.0. |
| A5 | **Prompts are identical across every model compared.** One builder, `text2sql.prompts.build_prompt`, same schema, evidence and sample values. | This is the whole validity of the comparison. A model given fuller schemas would win on the prompt, not the task. | `tests/test_prompts.py` fails if the builders diverge. |
| A6 | **`max_tables: 8`, `max_columns: 20`, `sample_values: 3`** for every run quoted here. | Changing any of them changes both prompt length and recall, so numbers are not comparable across settings. | `configs/data.yaml`; `results/<run>/run.yaml` records what a run used. |
| A7 | Latency figures are **hosted-API round trips from a laptop at 8-way concurrency**. | They are *not* comparable to a self-hosted GPU's latency, and they move with network conditions and concurrency. | `--max-workers`; the real comparison comes from `make latency` against the served model. |
| A12 | **The prompted baseline is sampled, not greedy**, so a single-run EX figure carries a **+/-1.5 point** noise band and sub-2-point differences mean nothing (M12). | Any ablation smaller than that needs k runs, not one. | Re-run and compare; the API rejects `temperature=0`. |
| A8 | BIRD dev is **never trained on**. The validation split comes out of train. | Contamination would invalidate every EX number in this repo. | `data.val_size` in `configs/data.yaml`. |

---

## 2. Measurements

### M1 — Dataset build (measured)

1,534 BIRD dev rows built, **zero dropped**, every gold query executes at
build time. Earlier builds silently dropped 106 rows; see D1.

### M2 — Prompt length vs `max_columns` (measured, with a caveat)

Real Qwen3 tokenizer. **Caveat: this sweep used a 300-example sample**, not
the full dev set, so the percentages carry sampling error of a few points.

| `max_columns` | p50 | p90 | p99 | max | over 3072 | over 4096 | gold cols kept |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (untrimmed) | 2002 | 4281 | 6651 | 6661 | 33.7% | 17.3% | 99.3% |
| 30 | 1967 | 3091 | 4134 | 4161 | 17.0% | 3.0% | 98.5% |
| 24 | 1967 | 3091 | 3832 | 3846 | 17.0% | 0.0% | 97.6% |
| **20 (default)** | 1958 | 3037 | 3615 | 3631 | 6.3% | 0.0% | 97.6% |
| 16 | 1932 | 2780 | 3397 | 3407 | 6.0% | 0.0% | 97.5% |
| 12 | 1761 | 2585 | 3171 | 3181 | 5.7% | 0.0% | 97.2% |

**Assumption A9: "gold columns kept" is a substring proxy**, not a
schema-linking metric. It asks whether the gold SQL's column names appear in
the rendered prompt text. It over-counts: a short column name can appear as a
substring of an unrelated one. Do not publish it as a recall figure.

On the **full** 1,534 at `max_columns: 20`: longest prompt **3,671** tokens,
longest prompt-plus-gold **3,828**. Nothing truncates at either trainer limit.

### M3 — What schema trimming actually removes (measured, full dev)

All 1,534 prompts, `max_tables: 8`, `max_columns: 20`:

| | kept | questions losing something |
| --- | ---: | ---: |
| Tables | 9,320 / 10,857 = **85.8%** | 303 (19.8%) |
| Columns (whole database) | 75,235 / 115,903 = **64.9%** | 758 (49.4%) |

Per-database: median 7 tables (max 13), median 71 columns (max 199). The
widest are `european_football_2` (199 columns, 129 questions), `card_games`
(115, 191 q), `formula_1` (94, 174 q).

### M4 — What that costs in recall (measured, full dev, *before* the M11 fix)

| | n | share |
| --- | ---: | ---: |
| Prompt contained every column the gold needs | 1,369 | 89.2% |
| **Prompt missing a needed column** | **165** | **10.8%** |
| Prompt missing a needed *table* | **0** | 0% |

EX on complete prompts **57.56%**; on incomplete prompts **41.21%**. Trimming
therefore costs roughly `0.108 x (57.56 - 41.21)` = **~1.8 EX points**.

**Assumption A10:** needed-column detection is a regex identifier match
(case-insensitive, quoted and bare forms) against the database's real column
names. It can miss a column referenced only through an alias, and can
over-match a short name. Treat 165 as approximate, and the direction as solid.

**Discarded measurement:** the same script bucketed EX by number of tables in
the gold query and put 1,469 of 1,534 questions in a "0 tables" bucket. The
table-name regex failed. The number was wrong, so it is not reported; EX by
JOIN count (M7) replaces it.

### M5 — gpt-5.4-nano prompted baseline (measured)

`results/api-baseline/`, full 1,534 dev, 2026-11-04.

| | |
| --- | ---: |
| EX | **55.80%** (856/1534) |
| simple / moderate / challenging | 63.03 / 45.26 / 43.45 |
| valid SQL | 96.94% |
| outputs that parsed | 100% |
| API errors | 0 |
| tokens | 2,629,527 in / 193,642 out (1,714 / 126 per question) |
| total spend | **$0.77** |
| cost per 1,000 queries | $0.5006 / EUR 0.4635 |
| latency | p50 1,161 ms, p95 2,938 ms |

### M6 — gpt-5.4-mini prompted baseline (measured)

`results/api-baseline-mini/`, same prompts, same day.

| | nano | mini | delta |
| --- | ---: | ---: | ---: |
| EX | 55.80% | **60.89%** | +5.09 |
| simple | 63.03 | 67.35 | +4.32 |
| moderate | 45.26 | 51.29 | +6.03 |
| challenging | 43.45 | 50.34 | +6.89 |
| valid SQL | 96.94% | 98.63% | +1.69 |
| failed to execute | 47 | 21 | -26 |
| output tokens per answer | 126 | 166 | +40 |
| total spend | $0.77 | **$3.12** | 4.1x |
| cost per 1,000 queries | EUR 0.4635 | EUR 1.8838 | |

Model agreement over the same 1,534 questions:

| | n | share |
| --- | ---: | ---: |
| both correct | 789 | 51.4% |
| only nano correct | 67 | 4.4% |
| only mini correct | 145 | 9.5% |
| **neither correct** | **533** | **34.7%** |
| oracle picking the better of the two | 1,001 | 65.25% |

### M7 — Where the failures are (nano, measured except where noted)

Failure composition over all 678 failures: **631 executed and returned wrong
rows**, 47 failed to execute. This is a semantics problem, not a syntax one.

Hand-categorised on a **random sample of 100 failures** (so +/- ~10 points
each): wrong schema linking 52%, wrong join 33%, wrong projection 3%, wrong
filter value 2%, other wrong 2%, SQL error other 2%, timeout 2%, unknown
column 2%, wrong ordering/limit 1%, gold unexecutable 1%.

EX by JOIN count in the gold query (full dev):

| JOINs | EX | n |
| ---: | ---: | ---: |
| 0 | 58.1% | 394 |
| 1 | 57.2% | 898 |
| 2 | 48.8% | 205 |
| 3+ | **35.1%** | 37 |

EX by construct present in the gold query (full dev, buckets over 25):

| construct | EX | n |
| --- | ---: | ---: |
| IIF | 33.3% | 36 |
| GROUP BY | 37.5% | 120 |
| CASE | 41.2% | 114 |
| nested SELECT | 43.5% | 124 |
| CAST | 44.7% | 152 |
| DISTINCT | 56.6% | 244 |

### M8 — Two unattainable questions (measured, load-dependent)

Question 518 (`card_games`) and 701 (`codebase_community`) have **gold**
queries that hit the 30s timeout when the evaluator's thread pool loads the
machine. They pass unloaded, which is why the dataset build (M1) saw all
1,534 execute. They count against every model equally, so comparisons stay
fair, but **attainable EX is capped at 99.87%** and the figure can move ~0.1
point with machine load. Recorded per run as `gold_unexecutable` in
`summary.json`.

### M9 — Reasoning-token spend (weakly measured -- treat with suspicion)

`reasoning_effort` on gpt-5.4-nano: `none` and `low` both returned **0
reasoning tokens**, `medium` returned 35, on **three probe calls against one
short synthetic prompt**. That is thin evidence and was not verified across
the real runs, because `generations.jsonl` does not record
`completion_tokens_details`. The indirect support is that nano's 126 output
tokens per answer is about what the visible SQL costs, leaving little room for
hidden reasoning. **If reasoning-token cost ever matters, store
`completion_tokens_details` and measure it properly.**

### M10 — Published context for the 55.80% (external, not ours)

The BIRD paper reports **GPT-4 at 54.89% EX** and **human performance at
92.96%**. A 2026 paper finds BIRD dev contains enough annotation errors that
method rankings change when labels are corrected, so the true ceiling is
unknown and some of the remaining gap is unwinnable.

**Assumption A11:** our 55.80% is **not** apples-to-apples with GPT-4's
54.89%. Different model, and our prompts are schema-trimmed with evidence and
3 sample values per column. The agreement is partly coincidence. It is
evidence the harness is sane, not a quality claim. Phrase it as "in the
published GPT-4 range", never "matches GPT-4".

---

### M11 — Why those columns were dropped, and the fix (measured, full dev)

Diagnosed causes of the 165 losses in M4:

| cause | n |
| --- | ---: |
| no lexical signal at all | 143 |
| **named verbatim in the question/evidence, lost the budget race** | **54** |
| token match, lost the budget race | 3 |

(The tiers overlap: a column can be both named and token-matched, so these
sum past 165.) The 54 were a straight defect -- the question named the column
and `select_relevant_columns` filled matches in *schema order* until the
budget ran out, so a weak match near the top of a table displaced an
explicitly named one further down.

Fix: three tiers. Keys and verbatim-named columns are never dropped and the
budget stretches to fit them; everything else competes on match strength with
schema order breaking ties only; and token matching accepts a 4-character
shared prefix, so "opened" reaches `OpenDate`.

Result, same measurement as M4:

| | before | after |
| --- | ---: | ---: |
| **Questions missing a needed column** | 165 (10.8%) | **114 (7.4%)** |
| Columns kept across all prompts | 64.9% | 65.0% |
| Median columns shown | 51 | 51 |
| Max columns shown | 84 | 89 |
| Longest prompt (Qwen3 tokens, full dev) | 3,671 | 3,823 |
| Longest prompt + gold | 3,828 | 3,891 |
| Prompts over 4,096 | 0 | **0** |

So a **31% cut in the recall loss for ~0.1% more columns**, and nothing
truncates at either trainer limit.

**EX effect: measured, and it is not significant.** Re-ran nano on the new
prompts (`results/api-baseline-v2/`, $0.77): EX **56.19%** against 55.80%,
so **+0.39 points** against an estimate of +0.8. Paired over the same 1,534
questions: 76 answers went wrong -> right, **70 went right -> wrong**, net +6
questions, exact McNemar **p = 0.679**. That is indistinguishable from noise.

Splitting the churn by whether the prompt actually changed (the old selector
was recovered from git to diff the column sets) is what explains it:

| | n | wrong->right | right->wrong | net | churn |
| --- | ---: | ---: | ---: | ---: | ---: |
| prompt byte-identical | 1,283 | 65 | 64 | +1 | **10.1%** |
| prompt changed | 251 | 11 | 6 | +5 | 6.8% |

So the fix contributed net +5 questions on the 251 prompts it touched
(directionally right, still not significant on its own), and the other 129
flips are the API disagreeing with itself. **Honest statement: coverage
improved measurably (M4 -> M11), its accuracy effect is positive in direction
and below this setup's noise floor.** Do not quote +0.39 as a gain.

Note on comparability: M2's "6.3% over 3072" came from a 300-example sample;
the post-fix full-dev figure is 8.7%. Those are different populations, not a
regression -- 0% exceed 4,096 either way.

### M12 — The noise floor: this baseline cannot be run twice (measured)

The most consequential measurement here, because it bounds what every other
comparison in this project can claim.

**gpt-5.4-nano disagrees with itself on 10.1% of BIRD dev given
byte-identical prompts** (129 of 1,283 answers flipped between two runs,
net +1). The cause is not subtle: the API refuses greedy decoding.

```
temperature=0 -> 400  "Unsupported value: 'temperature' does not support 0
                       with this model. Only the default (1) value is supported."
top_p=0       -> 400  "Unsupported parameter: 'top_p' is not supported with this model."
seed=13       -> 200  (accepted; documented as best-effort only)
```

Eight identical calls to a trivial prompt returned 2 distinct answers. So the
prompted baseline is **sampled at temperature 1 by force**, not greedy.

Consequence, as a usable rule: with ~129 discordant pairs split near evenly,
the standard error on a net difference is about `sqrt(129)/1534` ~ **0.74 EX
points, so a 2-sigma band of roughly +/-1.5 points.** On this backend, **a
single-run EX difference under about 2 points is not evidence of anything.**

Mitigations, in order of honesty: pass `seed` (now done, best-effort);
report the band; for a number that has to be defended, average over k runs
(the standard error falls as `1/sqrt(k)`) or use self-consistency, which
turns the sampling into a feature.

This does not threaten the project's headline claim -- "+10 EX over the base
model" is far outside the band -- but it does mean small ablations
(reasoning-vs-plain SFT, a schema-budget tweak) need more than one run each.

## 3. Decisions

**D1 — Always quote identifiers.** Quoting only unusual-looking names made
`PRAGMA table_info(order)` a syntax error, silently dropping 106 BIRD dev
rows (7%) while the build reported success. There is no cheap keyword test
(~140 of them), and over-quoting is harmless.

**D2 — Validate a column against the schema before sampling values.** SQLite
resolves an unknown double-quoted name as a *string literal* rather than
raising, so `SELECT DISTINCT "nope"` returns "nope" per row. The old code
relied on that query failing, and would have injected non-existent values
into prompts as real sample values.

**D3 — `max_columns: 20`, chosen from the M2 sweep, not guessed.** Below 20
the length savings flatten while recall keeps bleeding; above 24 the p99
crosses GRPO's prompt limit.

**D14 — Quote the noise band, and never a sub-2-point single-run gain.** The
temptation after the M11 fix was to report "+0.39 EX". It is not a gain; it
is inside the band (M12). Pass `seed` for best-effort reproducibility, say
that it is best-effort, and average over runs when a number has to hold up.

**D13 — Naming outranks the budget; match strength outranks schema order.**
The two halves of the M11 defect. A name the user typed is the strongest
signal available, so it should not compete for space with weak matches, and
schema position is a tie-breaker, never a ranking.

**D4 — Do not raise `max_columns` to fix the M4 recall loss.** 30 puts p99 at
4,134, past GRPO's `max_prompt_length: 4096`, and a truncated prompt hides the
schema the answer needs, so its reward is ~0 regardless of the model --
training looks healthy while learning nothing. Raising it for the baseline
alone would break A5. **Better column selection at the same budget is the
fix.** This matters more for training than for scoring: these same prompts
become the SFT and GRPO data, where a missing needed column teaches invented
column names.

**D5 — A cheap baseline changes the claim.** "N points behind a frontier
model" needs a frontier model. With nano or mini the honest framing is
**cost per query at matched accuracy** -- which is what the latency and cost
columns exist to support.

**D6 — Not using `gpt-5-mini` / `gpt-5-nano`.** Cheaper, but OpenAI's own
model pages mark both deprecated; a resume line should not rest on a
withdrawn model.

**D7 — `reasoning_effort: "low"`, not `"minimal"`.** `minimal` is a 400 on
gpt-5.4-nano, which accepts `none|low|medium|high|xhigh`. `low` leaves the
model free to reason on a hard question at no measured cost (M9).

**D8 — Generation and scoring stay separate steps.** Generating costs money;
scoring is free and gets re-run whenever the comparison logic changes.

**D9 — Retry 429 and 5xx, never 400.** An un-retried rate limit does not read
as an error in the results -- it reads as a *wrong answer*, because an empty
generation scores zero. A retried 400 just takes five times as long to tell
you the same thing.

**D10 — Carry the HTTP response body in the error.** A 400 names the parameter
it rejected; discarding that text cost a round trip to rediscover.

**D11 — Price every token-billed backend, not just Anthropic's.** The cost
column was gated on `backend == "anthropic"`, so OpenAI runs fell through to
GPU-hour pricing and were never costed, leaving one of the three headline
columns permanently empty.

**D12 — The baseline must run at the same `--max-columns` as training.**
`make baseline-api` omitted the flag, which would have given the baseline full
untrimmed schemas and *inflated* its score -- the exact unfair-baseline
failure this repo claims to prevent, and silent.

---

## 4. Overfitting and generalisation

Asked directly: does fine-tuning on this data just teach the model to
memorise it? Recording the answer because the safeguards are easy to assume
and three of them are structural rather than chosen.

**What protects against memorisation**

| | Protection | Strength |
| --- | --- | --- |
| G1 | **Train and dev use disjoint databases.** BIRD's design: dev has 11 databases, train ~69 different ones. A dev answer cannot be memorised from train, because the dev schemas are not in train. | Structural, the strongest of these. **Unverified** -- confirm once train is downloaded. |
| G2 | `data.val_size: 300` is held out of **train**, never from dev. | Chosen; see A8. |
| G3 | **QLoRA is itself a regulariser.** Rank 32, dropout 0.05, 7 projection modules: well under 1% of weights are trainable, with far less capacity to memorise than full fine-tuning. Two epochs. | Chosen. |
| G4 | **GRPO does not imitate labels at all.** The reward is whether the model's *own* query returns the gold result rows, with `beta: 0.02` KL to the reference policy. It learns from outcomes, not from copying strings. | Structural to the method. |
| G5 | **EX is execution-based.** A memorised query that does not return the right rows scores zero, so memorisation that does not generalise does not pay. | Structural to the metric. |

**What is actually at risk** -- none of these is classic memorisation, and two
are open right now:

| | Risk | Status |
| --- | --- | --- |
| R1 | **Benchmark-style overfitting.** The model learns BIRD's conventions (how evidence is phrased, its habits with `CAST` and `IIF`) and generalises worse elsewhere. The defence is Spider: a held-out *benchmark*, not a held-out split. | **OPEN -- Spider is not downloaded, so there is currently no generalisation check at all.** |
| R2 | **Train/serve skew on trimmed prompts.** Training on 20-column schemas teaches the model to expect them; serving it a full 80-column schema changes the shape of what it learned to read. | **CLOSED.** `serve/app.py` now takes its prompt shape from `configs/data.yaml` -- the same file the dataset build reads -- with `T2S_*` as an override, reports it on `/health`, and is covered by a test. The old default was `T2S_MAX_COLUMNS=0` (untrimmed) against training's 20, so the skew shipped by default rather than merely being possible. |
| R3 | **No best-checkpoint selection.** `eval_strategy: epoch` over 2 epochs gave **two** validation points and no `load_best_model_at_end`, so an overfitting epoch 2 was kept by default. | **CLOSED.** Both SFT configs now evaluate and save every 100 steps (~12 points instead of 2), keep the best checkpoint by `eval_loss`, and stop after 3 evals without improvement. `train_sft.py` validates the strategy/step agreement up front with a message naming the key to change, rather than letting transformers raise it after the model is loaded. |
| R4 | **Pretraining contamination.** Qwen3-4B may have seen BIRD. Unmeasurable from here, true of every published number on any public benchmark. | Accepted; belongs in `MODEL_CARD.md` as a stated limitation. |
| R5 | Val-loss signal is noisy at 300 examples, and EX ablations under ~2 points are unresolvable in one run (M12). | Accepted; needs k runs. |

## 5. Runs on disk

| Run | Model | Prompts | EX | Spend | Note |
| --- | --- | --- | ---: | ---: | --- |
| `api-baseline` | gpt-5.4-nano | pre-M11 selection | 55.80% | $0.77 | the first measured baseline |
| `api-baseline-v2` | gpt-5.4-nano | post-M11 selection | 56.19% | $0.77 | +0.39 is **inside** the noise band (M12) |
| `api-baseline-mini` | gpt-5.4-mini | pre-M11 selection | 60.89% | $3.12 | the stronger prompted baseline |

Total API spend to date: **$4.66**. Every GPU-side row (zero-shot base, SFT
x2, GRPO, self-consistency) is unrun.

**No training data exists yet.** `make dataset` writes only `dev.jsonl`
(1,534 evaluation prompts); BIRD train is not downloaded, so the build logs
`[warn] train: nothing written`. The M11 selection fix is in the schema code,
so it applies to train whenever that is built -- but nothing training-side
has been produced.

## 6. Known-open items

Ordered by what would change a conclusion, not by effort.

1. **No generalisation check exists** (R1). Spider needs a manual Google
   Drive download, so every Spider column is empty and nothing currently
   tests whether the model learned text-to-SQL or learned BIRD. This is the
   largest remaining hole in the evaluation.
2. **The zero-shot Qwen3-4B row is unmeasured.** It needs a GPU, and it is
   the number the "+10 EX" success criterion is measured *against* -- the
   project cannot state its headline result without it.
3. **Every future ablation needs k runs, not one** (M12): the single-run band
   is +/-1.5 EX points and nothing averages over runs. `--seed` is wired
   through but best-effort.
4. **114 questions still missing a needed column** (M11, down from 165). The
   remaining cause is value linking -- "directly funded" is a *value* of
   `Charter`, not a column name, and "continuation schools" a value of
   `Educational Option Type`. The standard fix is a cached per-database index
   of distinct values for low-cardinality text columns, matched against the
   question's literals. Not built: it is a real design change, and the index
   must be cached per database or it will dominate dataset build time.
5. **Reasoning-token accounting is unverified** (M9). Needs
   `completion_tokens_details` stored in `generations.jsonl`.
6. `results/RESULTS.md`'s **latency column mixes cost models** -- hosted API
   round trips against (eventually) a self-hosted server. A7 applies.
7. **G1 is unverified**: that BIRD's train and dev databases are disjoint is
   its documented design, not something this repo has checked. Confirm once
   train is built.
