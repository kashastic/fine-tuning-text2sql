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
| A13 | **Spider train means `train_spider.json` (7,000) only.** `train_others.json` (1,659 questions over 6 databases) is left out: it is re-annotated data from Restaurants, GeoQuery, Scholar, Academic, IMDB and Yelp, a different distribution wearing Spider's file format. | Our Spider train count is 7,000, not the 8,659 some papers quote, so "trained on Spider" is not directly comparable to theirs. | `load_spider` in `src/text2sql/data.py` reads `train_spider.json`. |
| A14 | **Spider test is held out of everything** (D15) -- not trained on, and not evaluated against while iterating. | It is the only number in this project that no decision was tuned against. Spending it early destroys what makes it worth having. | Nothing writes `data/processed/test.jsonl`; `data.splits` is `[train, dev]`. |

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

**Assumption A9 -- RETIRED, see M16.** The "gold columns kept" column above
is a substring proxy, not a schema-linking metric: it asks whether the gold
SQL's column names appear anywhere in the rendered prompt text. A9 warned it
over-counts; M16 measured by how much. It under-reported column loss (114
against a true 125) and reported **zero** table loss where there were **24**,
which sent the whole M11 effort at columns and left a worse bug untouched for
two sessions. Recall is now measured against the selected column set by
`scripts/schema_recall.py`. The prompt-length columns in this table are
unaffected and still stand; the last column should not be quoted.

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
**Superseded by M16**, which tightened exactly this: bare SQL keywords are no
longer counted as column references (that alone was worth 248 phantom
losses), and only tables the gold actually names are audited.

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

**Scope correction (M19): this was measured on nano only, and the floor turns
out to be model-specific.** Measured on byte-identical prompts, nano churns
9.93% and mini 7.16%, for bands of +/-1.61 and +/-1.37 points respectively.
The +/-1.5 quoted above is a nano number that happens to fall between the
two; M19 has the per-model figures and they are the ones to use. Nothing
here is known about a self-hosted model's floor -- vLLM at `temperature=0`
is genuinely greedy and should be far tighter, which is to be measured, not
assumed.

Mitigations, in order of honesty: pass `seed` (now done, best-effort);
report the band; for a number that has to be defended, average over k runs
(the standard error falls as `1/sqrt(k)`) or use self-consistency, which
turns the sampling into a feature.

This does not threaten the project's headline claim -- "+10 EX over the base
model" is far outside the band -- but it does mean small ablations
(reasoning-vs-plain SFT, a schema-budget tweak) need more than one run each.

### M13 — Train/dev separation (measured, was G1)

Checked directly rather than trusted as BIRD's documented design:

| | questions | databases |
| --- | ---: | ---: |
| BIRD train | 9,428 | 69 |
| BIRD dev | 1,534 | 11 |

**Databases appearing in both: 0.** Identical question strings in both: **0**.
So a dev answer cannot be memorised from train -- the dev schemas are not in
train at all. This is the strongest of the anti-memorisation protections and
it is now verified, not assumed.

### M14 — The training splits, and what gold filtering removes (measured)

Built from BIRD train + Spider train at `max_columns: 20`:

| split | rows | source |
| --- | ---: | --- |
| train | **15,697** | BIRD train (9,428) + Spider train (7,000), minus the validation holdout and bad gold |
| val | **300** | held out of **train** (A8), never from dev |
| dev | **2,568** | BIRD dev (1,534) + Spider dev (1,034); nothing dropped from either |
| *(no test file)* | -- | Spider test is deliberately not built (D15) |

Arithmetic that should tie out, and does: 9,428 + 7,000 = 16,428, less 431
unexecutable gold and 300 held out for validation, leaves 15,697. Every row
carries its `source`, so a mixed `dev.jsonl` can still be reported as two
benchmarks rather than one average that means nothing.

**431 train rows (2.6%) were dropped as `bad_gold`** -- 428 from BIRD (4.5%
of it) and 3 from Spider (0.04%, M15). The BIRD 428 were classified on a
random 1,200-row sample (46 failures, 3.8%), which is why they fail:

| cause | share of failures | example |
| --- | ---: | --- |
| **genuine annotation error** | **83%** | `no such column: T2.CompanyName`, `no such table: Customer and Suppliers by City` |
| rejected by our sandbox | 15% | `non-SELECT or stacked statement` |
| timeout | **2%** | one row |

So the dropped rows are overwhelmingly BIRD's own broken annotations, not
artifacts of our deadline -- scaled up, roughly 9 of the 428 are recoverable
timeouts. Dropping them is correct: training a model to imitate a query that
errors is pure noise. This also corroborates M10's annotation-error finding
from the data rather than from the paper.

One honest caveat: the 15% rejected for being non-SELECT or stacked (~64
rows) are excluded by **our policy**, not by being wrong. The read-only
single-statement sandbox is a deliberate safety choice; those golds are
simply outside what this project will execute.

### M15 — Spider on disk, and every split's separation (measured)

Spider turned out to be fetchable after all: Google Drive's `usercontent`
host serves the official zip to a script when `confirm=t` stands in for the
virus-scan interstitial, so R1 did not need a manual download.

| split | questions | databases | unexecutable gold |
| --- | ---: | ---: | ---: |
| BIRD train | 9,428 | 69 | 428 (4.5%) |
| BIRD dev | 1,534 | 11 | 0 |
| Spider train (`train_spider.json`) | 7,000 | 140 | **3 (0.04%)** |
| Spider dev | 1,034 | 20 | **0** |
| Spider test | 2,147 | 40 | **0** |
| Spider `train_others.json` (unused, A13) | 1,659 | 6 | not checked |

Spider dev at 1,034 matches the published count exactly. Every referenced
SQLite file is present: 0 missing databases across all five splits.

The Spider gold columns are **full-split counts, not samples** -- every one
of the 10,181 golds was executed at a 30-second deadline. A first pass
sampling 1,200 train rows found zero failures and would have let "Spider's
gold is clean" be stated as 0%; the full pass found 3. The rate is 0.04%
either way, but the honest figure is 3, not 0, and a 1,200-row sample of a
0.04% population was never going to see it. All three are annotation errors
of exactly BIRD's kind, and none is ours:

| database | failure |
| --- | --- |
| `assets_maintenance` | `no such table: Ref_Company_Types` |
| `document_management` (x2) | `ORDER BY clause should come after INTERSECT not before` -- SQL SQLite refuses to parse |

**Database overlap, every pair that matters -- all zero:**

| pair | shared databases |
| --- | ---: |
| BIRD train vs BIRD dev | 0 |
| Spider train vs Spider dev | 0 |
| Spider train vs Spider test | 0 |
| Spider dev vs Spider test | 0 |
| BIRD train vs Spider dev | 0 |
| BIRD dev vs Spider dev | 0 |
| BIRD train vs Spider train | 0 |

BIRD and Spider share **no databases at all**, so the two benchmarks are
genuinely different schemas, not the same domains relabelled.

**Spider's gold SQL is clean where BIRD's is not: 0.04% unexecutable against
BIRD's 4.5% -- a factor of 100.** This settles M14's open question from the
other side, and it is the measurement that makes the claim rather than the
argument. Our read-only, 30-second, single-statement sandbox executes 10,178
of 10,181 Spider golds. Zero failures came from the deadline and zero from
the single-statement policy. So BIRD's 428 drops are BIRD's annotation
quality; the sandbox is not a confound. M14 inferred this from error strings,
which was suggestive. This is a control.

### M16 — The recall proxy was wrong, and so was one of M4's conclusions

M4 and M11 measured "does the prompt still contain the columns the gold
needs" by searching the **rendered prompt text** for the gold's column names
(A9, which flagged it as a proxy that over-counts). It is now measured
against the **selected column set**: a gold column counts as present only
when that exact column of that exact table was chosen. The script is
`scripts/schema_recall.py`, so it is reproducible rather than re-derived by
hand a fourth time.

Same code, same 1,534 questions, same settings -- only the metric changed:

| | proxy (A9) | exact |
| --- | ---: | ---: |
| Questions missing a needed column | 114 (7.4%) | **125 (8.15%)** |
| Questions missing a needed **table** | **0** | **24 (1.56%)** |

Two corrections follow, and the second is the one that mattered.

**M11's 114 was optimistic.** A short column name appears as a substring of
unrelated text, so the proxy counted columns as present that were never
selected. The true figure was 125.

**M4's "of the 303 questions that lost a table, *none* lost a table the gold
needs" was wrong.** 24 did. The proxy could not see it: a dropped table's
name still appears in the prompt inside the foreign-key clauses of other
tables. "Table selection is sound, and every real loss is a dropped column"
was a wrong conclusion drawn from a blunt instrument, and it aimed all of
M11's work at columns while a worse bug sat untouched.

Getting the metric right needed care in the other direction too. A first
version counted every bare word of the gold as a possible column reference
and reported **373** losses: SQL keywords collide with real BIRD column names
(`count`, `order`, `date`, `year`, `name`, `max`), so `SELECT COUNT(*)`
registered as a reference to a column called `count`. Filtering bare keywords
-- quoted names are kept, since the quotes are what prove the author meant a
column -- and auditing only tables the gold actually names brought it to 125.
**248 of those 373 were the measurement's own bug**, which is a fair warning
about how easily this particular number can be made to say anything.

### M17 — Value linking, inbound keys, inflected table names (measured, full dev)

Three fixes, measured one at a time on BIRD dev at `max_tables: 8`,
`max_columns: 20`. The baseline row is the committed code, re-measured in a
detached worktree at `HEAD` rather than quoted from M11:

| | missing a column | missing a table | columns shown |
| --- | ---: | ---: | ---: |
| baseline (committed) | 125 (8.15%) | 24 (1.56%) | 65.02% |
| + inflected table names | 116 (7.56%) | 15 (0.98%) | 66.31% |
| + value linking | 97 (6.32%) | 6 (0.39%) | 66.28% |
| **+ inbound foreign keys** | **42 (2.74%)** | **6 (0.39%)** | **66.28%** |

**A 66% cut in column loss and a 75% cut in table loss.**

**Inbound foreign keys -- the largest single win (97 -> 42).** A join has two
ends and only one of them declares the constraint. `legalities.uuid
REFERENCES cards(uuid)` makes `uuid` a key column of `cards`, but `cards`
declares no foreign key and its primary key is `id`, so `cards.uuid` was
scored as an ordinary column -- and lost, because nothing in a question about
rarity resembles "uuid". **59 of the 97 remaining losses were that one
column.** The prompt showed both tables and no way to join them. Tier 1
already claimed to prevent exactly this ("without them a join stops being
expressible"); it protected only the outbound half.

**Inflected table names (24 -> 15 lost tables).** Column scoring took a
4-character shared prefix from M11; table scoring still used exact token
equality. So `race != races` and "the race No. 901" dropped the `races` table
outright. A lost table is strictly worse than a lost column: no column tier
can recover a table that is not in the prompt.

**Value linking (the remaining 9 lost tables, and 19 columns).** The question
often names a column's *contents* rather than its name. "The school run by
Kacey Gibson" needs `schools.AdmFName1`; "continuation schools" needs
`frpm.Educational Option Type`, whose value is "Continuation School"; "Lucas
di Grassi's Q1 result" needs the `drivers` table, which no name-based score
can reach at all. `text2sql.values` indexes each database's distinct short
text values once, then looks the question's word n-grams up in that index.

Two details earn their keep, both found by reading wrong answers:
singularising both sides, or "continuation schools" never meets "Continuation
School"; and resolving overlapping matches longest-first, because "Los
Angeles" identifies `schools.City` while "Angeles" alone is somebody's
surname in `AdmLName1`.

**Cost: essentially nothing.** Real Qwen3 tokenizer, full dev, the same
prompts the model would see:

| | baseline | after |
| --- | ---: | ---: |
| p50 prompt tokens | 1,945 | 1,950 |
| p90 / p99 | 3,033 / 3,680 | 3,033 / 3,680 |
| longest prompt | 3,810 | 3,810 |
| longest prompt + gold | 3,878 | 3,878 |
| over 4,096 | 0% | **0%** |

**Five median tokens, and the tail does not move.** The gain is not bought
with width: a protected key or a value-matched column *displaces* a weak name
match inside the same budget instead of adding to it. That is why "columns
shown" moved only 65.02% -> 66.28% and the longest prompt is unchanged.

(These absolute token figures are a few tokens below M2's and M11's for the
same revision, because this script joins the prompt's messages slightly
differently. The before/after pair above is internally consistent, which is
what the comparison needs; do not mix a number from here with one from M2.)

Index build cost, measured: **7.7s for all 11 BIRD dev databases** (26 MB of
cache), and 0.5-19.3s each for BIRD train's four largest (4.3-5.2 GB). Cached
on disk keyed by file size and mtime, so a rebuild pays nothing.

Full dataset rebuild with value linking on, cold cache: **53 minutes** for
all three splits, indexing **242 databases** into **148 MB** of cache under
`data/cache/` (gitignored). Row counts and drop counts came out identical to
the pre-M17 build -- 15,697 / 300 / 2,568 and 431 `bad_gold` -- which is the
expected result: value linking changes what a prompt *contains*, not which
rows survive. Verified positively rather than assumed: the dev question
naming "Kacey Gibson" now carries `AdmFName1` in its prompt, which it did
not before.

**What is left is a semantic long tail.** The 42 survivors have no dominant
cause -- the most frequent column appears 5 times -- and they need meaning
rather than matching: "Under whose administration is the school..." needs
`AdmFName1`, "the type of education offered" needs `EdOpsName`. That is an
embedding or an LLM linker, not another lexical tier.

### M18 — What M17 bought in EX: nothing measurable, and why (measured, full dev)

A5 requires every compared model to see identical prompts, and M17 changed
the prompts, so the prompted baseline was re-run on them
(`results/api-baseline-v3/`, $0.78, same settings as v2: `max_columns=20`,
`reasoning_effort=low`, `seed=13`).

**EX 56.26% against v2's 56.19%. Net +1 question in 1,534, exact McNemar
p = 1.0000.** The estimate recorded before the run was about +1 point. The
error in it is instructive: the treatment arm below did move +0.99 points,
so the estimate was about right *for the prompts that changed* -- it simply
failed to dilute by the two thirds of the split whose prompts did not. Even
taking the treatment effect at face value, the whole-split prediction should
have been +0.33 points, which is itself well inside the noise band.

Measured with `scripts/compare_runs.py`, which reproduces M11's published
comparison exactly (856/862, 76/70, p = 0.6792) -- those figures were derived
by hand and the code had not survived, so this now lives in the repo.

Prompt hashes were taken on both sides of 3b04ec7 to split the flips. **507
of 1,534 BIRD dev prompts changed (33.1%); 1,027 are byte-identical.** The
identical ones are a control arm measured in the same API run: nothing about
them changed, so whatever they flip by is noise.

| arm | n | EX before | EX after | wrong->right | right->wrong | net | churn | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 1,534 | 56.19% | 56.26% | 81 | 80 | +1 | 10.50% | 1.0000 |
| prompt byte-identical (control) | 1,027 | 56.48% | 56.09% | 49 | 53 | **-4** | 9.93% | 0.7666 |
| prompt changed (treatment) | 507 | 55.62% | 56.61% | 32 | 27 | **+5** | 11.64% | 0.6029 |

The control arm's 9.93% churn is **M12's noise floor reproduced a third time,
independently** -- the number to quote for this backend is ~10%, and it is
stable. The treatment arm is net +5 on the prompts the fix touched, which is
the same net +5 M11's treatment arm produced. Neither clears its own noise.

**The sharper test: did EX improve on the questions whose schema was actually
repaired?** Pairing the recall audit with both runs partitions dev by whether
the prompt was missing something the gold needs. (125 lost something before,
42 after; every table loss also loses that table's columns, so the two
partitions tie out: 84 + 41 = 125 and 41 + 1 = 42.)

| group | n | EX v2 | EX v3 | w->r | r->w | net | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **restored** (lost before, complete now) | 84 | 38.10% | 42.86% | 9 | 5 | **+4** | 0.4240 |
| still incomplete (both runs) | 41 | 19.51% | 24.39% | 4 | 2 | +2 | 0.6875 |
| never incomplete (control) | 1,408 | 58.31% | 58.03% | 68 | 72 | -4 | 0.8000 |
| newly incomplete | 1 | 100.00% | 0.00% | 0 | 1 | -1 | 1.0000 |

So the gain is where the mechanism says it should be -- +4 and +2 on the two
repaired groups, -4 on the 1,408 untouched ones -- and none of it is
significant at these group sizes.

**The finding worth keeping is the dose-response, not the delta.** Ranking
v3's EX by how complete the prompt was:

| prompt | n | EX |
| --- | ---: | ---: |
| complete schema | 1,408 | **58.03%** |
| schema repaired by M17 | 84 | 42.86% |
| schema still missing something | 41 | **24.39%** |

A question whose prompt is missing a needed column is **less than half as
likely** to be answered correctly. That is the strongest evidence in this
project that schema recall is worth the work -- and it is also why the EX
delta is small: the repaired 84 moved to 42.86%, not to the 58.03% that
complete prompts enjoy. **Restoring the column captured about a quarter of
the available headroom** (84 x (58.03 - 38.10) = 16.7 questions on offer, +4
delivered).

Caveat, and it cuts against reading the table as causal: these three groups
are not randomised. Questions whose columns went missing are the ones with
obscurely-named columns, and obscure naming travels with harder queries, so
part of the 58% -> 43% -> 24% gradient is difficulty rather than prompt
content. The *within-group* paired numbers (+4, +2, -4) are the clean
measurement; the gradient is an association.

**Honest statement.** Schema recall improved 3x (M17) and EX did not move
measurably. Both are true, and the second does not retract the first: a
complete schema is **necessary but far from sufficient**, the 42 survivors
are the hardest questions in the split, and a 4B model fine-tuned on these
prompts has more to gain from a complete schema than a hosted model
prompted zero-shot does. Do not quote +0.07 as anything.

**One regression, and the mechanism matters.** `formula_1/852` ("races in
2010 held on grand prixs outside Asia and Europe") had a complete schema
before and lost `circuits` entirely. Value linking raised other tables'
scores past it, and the hard `max_tables=8` cap then evicted it -- columns
shown went 19 -> 51, so the budget filled with better-scoring tables. Table
losses still fell 24 -> 6 overall, so the trade is overwhelmingly positive,
but **value linking changes table *ranking*, not just column selection**, and
that is the failure mode to watch if `max_tables` is ever lowered.

**Prompt length, on the OpenAI tokenizer this time** (M17's figures are
Qwen3): mean input 1,717 -> 1,752 tokens, **+2.05%**. The p50 moved five
Qwen3 tokens, but the mean moved 35, because the added columns land on
exactly the 507 prompts that needed them. Cost per 1k queries EUR 0.462 ->
0.470.

### M19 — The mini baseline re-run, and the noise floor is model-specific (measured, full dev)

Mini was two prompt recipes behind, so its re-run
(`results/api-baseline-mini-v3/`, $3.12) spans **both** M11 and M17 rather
than either alone: 654 of 1,534 prompts changed (42.6%), 880 are
byte-identical. Same settings as the original.

**EX 61.73% against 60.89%, net +13 questions, exact McNemar p = 0.2459.**

| arm | n | EX before | EX after | wrong->right | right->wrong | net | churn | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 1,534 | 60.89% | 61.73% | 60 | 47 | +13 | 6.98% | 0.2459 |
| prompt byte-identical (control) | 880 | 61.70% | 62.05% | 33 | 30 | +3 | 7.16% | 0.8013 |
| prompt changed (treatment) | 654 | 59.79% | 61.31% | 27 | 17 | **+10** | 6.73% | 0.1742 |

The treatment arm is the largest and cleanest signal in the project so far --
net +10 on 654 prompts, +1.52 points -- and it still does not reach
significance. Two prompt fixes, measured on the stronger model, over the
biggest arm available: p = 0.17.

**The finding that corrects an earlier one: the noise floor is
model-specific, and M12 measured only nano.** Mini's control arm churns
**7.16%** against nano's 9.93%, on the same questions in the same split.
Rescaling each control arm to full dev:

| model | control churn | discordant / 1,534 | SE | 2-sigma band |
| --- | ---: | ---: | ---: | ---: |
| gpt-5.4-nano | 9.93% | ~152 | 0.80 | **+/-1.61 EX** |
| gpt-5.4-mini | 7.16% | ~110 | 0.68 | **+/-1.37 EX** |

So M12's "+/-1.5 points" is a nano figure that happens to sit between the
two. Use +/-1.6 for nano and +/-1.4 for mini, and do not assume either
transfers to a self-hosted model: vLLM at `temperature=0` is greedy and
should be far tighter, which is a thing to measure rather than assume.
Mini's +0.84 is inside its own band, consistent with p = 0.2459.

**A missing column hurts both models, and the bigger one only partly
compensates.** Grouping by whether the current prompt is complete (1,492
complete, 42 still missing something the gold needs):

| model | complete prompt | incomplete prompt | gap | incomplete as a fraction of its own rate |
| --- | ---: | ---: | ---: | ---: |
| gpt-5.4-nano | 57.17% | 23.81% | -33.36 | 0.42 |
| gpt-5.4-mini | 62.40% | 38.10% | -24.30 | 0.61 |

(The 57.17% is the same measurement as M18's 58.03%, over a different
grouping: here the 84 M17 repaired are counted as complete, which is what
they now are.)

**This is the argument for the schema-linking work, stated as a measurement.**
Spending 4x on a bigger prompted model recovers part of the penalty for a
missing column -- mini retains 61% of its own accuracy where nano retains 42%
-- but **most of the penalty survives the upgrade**. It is an information
problem before it is a capability problem. The shared-failure rate says the
same thing from the other side: both models are wrong on **57%** of the 42
incomplete prompts against **32%** of the 1,492 complete ones.

**Both prompted rows are now on the current prompts**, so A5 holds across
them and the two-model comparison is finally legitimate rather than
indicative: mini is **+5.48 EX** over nano for ~4x the cost, 51.11% of
questions both get right, 33.12% neither does, and an oracle choosing the
better of the two would score **66.88%**. That 33.12% shared-failure core is
the target for fine-tuning, and it is not addressable by spending more on a
prompted model.

### M20 — Running the trainers found two bugs that `--help` could not (measured)

Both training scripts had never been executed. The suite checked that their
`--help` worked, which is not evidence that a config loads, that LoRA attaches
to anything, or that an adapter gets written. They are now run end to end on a
~10M-parameter randomly-initialised Qwen3 over the fixture benchmark, on CPU,
in about 25 seconds (`tests/test_training.py`, 6 tests).

It is a **wiring** test and claims nothing else. The model is random, so it
says nothing about quality; CPU says nothing about memory or throughput; and
`bitsandbytes` is CUDA-only, so **`load_in_4bit` and `paged_adamw_8bit` are
switched off and remain untested** -- which is exactly why the first real GPU
run should use a tiny subset.

**Bug 1: both trainers would have crashed on the declared minimum
transformers.** They passed `dtype=` to `from_pretrained`; transformers
renamed `torch_dtype` to `dtype` only in its 5.x line, and `pyproject.toml`
declares `transformers>=4.52`. On 4.52 the kwarg reaches
`Qwen3ForCausalLM.__init__` as an unexpected keyword and raises -- *inside*
`SFTTrainer` construction, after the dataset has loaded, so the failure
arrives a minute into a billed GPU hour. Fixed with `dtype_kwargs()` in
`scripts/_trainer_utils.py`, which probes `PretrainedConfig` for whichever
spelling the installed version carries. Signature inspection cannot find this
one: the kwarg is forwarded into the model constructor rather than consumed by
`from_pretrained`.

**Bug 2: `make grpo` before `make sft` failed in the wrong place.**
`configs/grpo_qwen3_4b.yaml` ships `adapter: outputs/sft-plain/adapter`, a
path that does not exist until SFT has run. The script carefully refused to
start when `model.adapter` was *unset* -- the documented silent-failure
guard -- but never checked the directory existed, so the likely mistake
(wrong order) loaded the 4B model first and then raised inside peft. Now
checked before the imports, with a message naming `make sft`. A directory
that exists but holds no `adapter_config.json` is caught separately, since
`outputs/sft-plain` and `outputs/sft-plain/adapter` are easy to confuse.

Note on semantics settled while fixing it: `--allow-base-model` clears the
refusal to train without an adapter, but a *configured yet missing* adapter
is still an error. Training from the base model requires both the flag and
`model.adapter=null`, because a missing configured adapter is an ordering
mistake rather than an intention.

**Also corrected:** both SFT configs justified `eval_steps: 100` with "~9.4k
train rows ... ~590 steps/epoch, ~12 eval points". That was BIRD alone,
before Spider's train split joined the mix (M14). It is 15,697 rows, ~980
steps/epoch and ~20 eval points over two epochs. The setting is still right;
its stated reason was stale.

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

**D15 — Spider test is spent once, at the end, or not at all.** With Spider
train in the SFT mix (D16), *every* evaluation set this project reports is a
held-out-database split of something the model trained on. That is a real
protection -- M15 shows the databases are disjoint -- but it is not the same
as a set that never influenced a decision. Spider test (2,147 questions, 40
databases, disjoint from all four other splits) is therefore excluded from
training *and* from the iteration loop: no ablation, no checkpoint choice, no
prompt tweak gets to see it. It is reported once, on the final model, as the
one number nothing was tuned against. The cost of this discipline is zero;
the cost of breaking it is that the number stops meaning anything, quietly
and irreversibly.

**D16 — Spider train goes into SFT, and that changes what Spider dev
measures.** 7,000 extra examples over 140 further databases is the cheapest
available defence against learning BIRD's conventions instead of SQL, and
Spider's schemas, phrasing and absence of an evidence field are genuinely
different. The honest consequence has to be stated rather than glossed:
**Spider dev is then a held-out-*database* check, not cross-benchmark
transfer.** The project can have one or the other from a single SFT run, not
both. Measuring true transfer would mean a BIRD-only run as well, and a
second 3-5 GPU-hour SFT is not in a 50-100 EUR budget. D15 is what buys back
the missing guarantee.

**D17 — Measure schema recall against the selected set, and keep the
script.** The substring proxy was cheap and wrong in a way that hid a whole
class of bug for two sessions (M16). Recall is now computed from the columns
the selector actually chose, and lives in `scripts/schema_recall.py` rather
than being rewritten from scratch each time the question comes up -- it has
been needed three times, and the third time the previous version was gone.
The script also reports prompt-token percentiles, because a recall gain that
pushes prompts past the trainer's limit is not a gain.

**D18 — Value linking indexes whole values, not value tokens.** Indexing
tokens would make "school" a hit on half of `california_schools`. Whole
values matched against question n-grams keeps precision high, at the cost of
missing a value the question paraphrases. Bounded deliberately: TEXT-affinity
columns only, values of at most 48 characters, at most 20,000 distinct values
per column, and a value found in more than 8 columns is dropped as
non-identifying. Those bounds are what keeps a 30 GB training set tractable
(M17), and they are the reason value linking cannot find a value in a column
of free text -- which is correct, since a forum post is never the literal a
question spells out in an equality.

**D19 — Value linking stays on by default, and fails open.** `link_values`
defaults to true in `schema_for_example`, and any exception while indexing
is swallowed back to "no value hits". An unreadable or exotic database then
builds the same prompt it built before, slightly worse, instead of failing a
16,000-row dataset build at row 9,000.

---

## 4. Overfitting and generalisation

Asked directly: does fine-tuning on this data just teach the model to
memorise it? Recording the answer because the safeguards are easy to assume
and three of them are structural rather than chosen.

**What protects against memorisation**

| | Protection | Strength |
| --- | --- | --- |
| G1 | **Train and dev use disjoint databases.** 9,428 train questions over 69 databases, 1,534 dev over 11, **zero overlap**, and zero identical question strings. A dev answer cannot be memorised from train, because the dev schemas are not in train at all. | Structural, the strongest of these. **Verified -- see M13**, and M15 extends it to Spider: every pair of the five splits shares 0 databases, BIRD and Spider included. |
| G2 | `data.val_size: 300` is held out of **train**, never from dev. | Chosen; see A8. |
| G3 | **QLoRA is itself a regulariser.** Rank 32, dropout 0.05, 7 projection modules: well under 1% of weights are trainable, with far less capacity to memorise than full fine-tuning. Two epochs. | Chosen. |
| G4 | **GRPO does not imitate labels at all.** The reward is whether the model's *own* query returns the gold result rows, with `beta: 0.02` KL to the reference policy. It learns from outcomes, not from copying strings. | Structural to the method. |
| G5 | **EX is execution-based.** A memorised query that does not return the right rows scores zero, so memorisation that does not generalise does not pay. | Structural to the metric. |

**What is actually at risk** -- none of these is classic memorisation, and two
are open right now:

| | Risk | Status |
| --- | --- | --- |
| R1 | **Benchmark-style overfitting.** The model learns BIRD's conventions (how evidence is phrased, its habits with `CAST` and `IIF`) and generalises worse elsewhere. The defence is Spider: a held-out *benchmark*, not a held-out split. | **CLOSED for the data; the measurement waits on a GPU.** Spider is on disk and verified (M15): 7,000 train / 1,034 dev / 2,147 test, 0 missing databases, 0 unexecutable golds, and 0 databases shared with BIRD. Spider train enters SFT as schema diversity (D16) and Spider test is held out of everything (D15). What is still unmeasured is the model's Spider EX, because no fine-tuned model exists yet -- that is R1's *number*, not R1's *mechanism*. |
| R2 | **Train/serve skew on trimmed prompts.** Training on 20-column schemas teaches the model to expect them; serving it a full 80-column schema changes the shape of what it learned to read. | **CLOSED.** `serve/app.py` now takes its prompt shape from `configs/data.yaml` -- the same file the dataset build reads -- with `T2S_*` as an override, reports it on `/health`, and is covered by a test. The old default was `T2S_MAX_COLUMNS=0` (untrimmed) against training's 20, so the skew shipped by default rather than merely being possible. |
| R3 | **No best-checkpoint selection.** `eval_strategy: epoch` over 2 epochs gave **two** validation points and no `load_best_model_at_end`, so an overfitting epoch 2 was kept by default. | **CLOSED.** Both SFT configs now evaluate and save every 100 steps (~12 points instead of 2), keep the best checkpoint by `eval_loss`, and stop after 3 evals without improvement. `train_sft.py` validates the strategy/step agreement up front with a message naming the key to change, rather than letting transformers raise it after the model is loaded. |
| R4 | **Pretraining contamination.** Qwen3-4B may have seen BIRD. Unmeasurable from here, true of every published number on any public benchmark. | Accepted; belongs in `MODEL_CARD.md` as a stated limitation. |
| R5 | Val-loss signal is noisy at 300 examples, and EX ablations under ~2 points are unresolvable in one run (M12). | Accepted; needs k runs. |

## 5. Runs on disk

| Run | Model | Prompts | EX | Spend | Note |
| --- | --- | --- | ---: | ---: | --- |
| `api-baseline` | gpt-5.4-nano | pre-M11 selection | 55.80% | $0.77 | the first measured baseline |
| `api-baseline-v2` | gpt-5.4-nano | post-M11 selection | 56.19% | $0.77 | +0.39 is **inside** the noise band (M12) |
| `api-baseline-v3` | gpt-5.4-nano | **current (post-M17)** | 56.26% | $0.78 | **the comparable nano row** (M18) |
| `api-baseline-mini` | gpt-5.4-mini | pre-M11 selection | 60.89% | $3.12 | superseded by `-mini-v3` |
| `api-baseline-mini-v3` | gpt-5.4-mini | **current (post-M17)** | 61.73% | $3.12 | **the comparable mini row** (M19) |

Total API spend to date: **$8.56**. Every GPU-side row (zero-shot base, SFT
x2, GRPO, self-consistency) is unrun.

**`api-baseline-v3` is the only row on the current prompts, and it is the
only one a fine-tuned number may be set beside.** A5 -- every compared model
sees identical prompts -- is the assumption the whole headline comparison
rests on, and M11 and M17 each broke it for everything measured earlier. The
older nano rows are kept as honest records of what was measured when, and
M18 is the reason they are not simply deleted: comparing them is how the
noise floor got measured a third time.

**Both prompted rows are now on the current prompts** (`api-baseline-v3` and
`api-baseline-mini-v3`), so A5 holds across them and across anything built
from the current dataset. The four superseded rows are kept deliberately:
comparing them is how the noise floor was measured three times and how M19
established that the floor differs by model. Every prompt change from here
costs $3.90 to re-baseline, which is the price of A5 and is cheap against
one GPU hour.

**The data pipeline is complete; nothing model-side has been run.** Both
benchmarks are on disk (~9.1 GB) and `make dataset` writes all three splits:
15,697 train / 300 val / 2,568 dev, at `max_columns: 20`, so the M11
selection fix is in the training data and not only in the eval prompts. What
does not exist is any trained model -- so every Spider EX cell, and the
zero-shot base row the "+10 EX" criterion is measured against, is still
empty for want of a GPU rather than for want of data.

## 6. Known-open items

Ordered by what would change a conclusion, not by effort.

0. **DONE (M18, M19).** Both prompted baselines are re-run on the current
   prompts: nano 56.26%, mini 61.73%. A5 holds across them, so a fine-tuned
   EX now has valid numbers to sit beside. The by-product is worth more than
   the re-baselining: the noise floor is model-specific (+/-1.61 nano,
   +/-1.37 mini), and a prompt missing a needed column costs nano 33 EX
   points and mini 24 -- an information problem that a 4x spend only partly
   buys back.
1. **The generalisation *number* is unmeasured** (R1). Spider itself is now
   on disk and verified (M15), and its train split is in the SFT mix, but no
   fine-tuned model exists yet, so every Spider EX column is still empty.
   What changed is that this is now blocked only on a GPU, not on a download.
2. **The zero-shot Qwen3-4B row is unmeasured.** It needs a GPU, and it is
   the number the "+10 EX" success criterion is measured *against* -- the
   project cannot state its headline result without it.
2b. **4-bit quantisation and `paged_adamw_8bit` are untested** (M20).
   `bitsandbytes` is CUDA-only, so the CPU smoke tests switch both off; they
   are the two settings a real run exercises first. Run the first SFT on a
   small subset for this reason, not for cost.
3. **Every future ablation needs k runs, not one** (M12): the single-run band
   is +/-1.5 EX points and nothing averages over runs. `--seed` is wired
   through but best-effort.
4. **DONE (M17).** Value linking is built, along with two bugs it uncovered;
   the figure is now **42 questions (2.74%)**, down from a true 125. What
   remains is a semantic long tail needing an embedding or LLM linker, which
   is a different project. One note for whoever picks it up: the index covers
   TEXT columns, so a value held in an INTEGER flag (`Charter = 1`) is still
   invisible to it.
5. **Reasoning-token accounting is unverified** (M9). Needs
   `completion_tokens_details` stored in `generations.jsonl`.
6. `results/RESULTS.md`'s **latency column mixes cost models** -- hosted API
   round trips against (eventually) a self-hosted server. A7 applies.
