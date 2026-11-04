# Results

Execution accuracy on BIRD dev, scored with the official set-comparison metric.

| Run               | Model        | Method   | BIRD dev EX (%) | Simple / Moderate / Challenging (%) | Spider dev EX (%) | p95 latency (ms) | Cost per 1k queries (EUR) |
| ----------------- | ------------ | -------- | --------------- | ----------------------------------- | ----------------- | ---------------- | ------------------------- |
| api-baseline-mini | gpt-5.4-mini | Prompted | 60.89           | 67.35 / 51.29 / 50.34               | --                | 2713             | 1.884                     |
| api-baseline      | gpt-5.4-nano | Prompted | 55.80           | 63.03 / 45.26 / 43.45               | --                | 2938             | 0.464                     |
