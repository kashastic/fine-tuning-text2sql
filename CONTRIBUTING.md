# Working on this repo

```bash
pip install -e ".[dev,viz,api]"
make test      # 75 tests, no GPU and no benchmark download needed
make lint
```

The test suite builds a miniature BIRD-shaped benchmark on disk
(`tests/conftest.py`): two small SQLite databases and questions in BIRD's and
Spider's own JSON shapes. Anything that touches data, prompts, scoring or
rewards can be tested against it on a laptop.

Two invariants worth keeping:

1. **One prompt builder.** Every model — fine-tuned or API baseline — goes
   through `text2sql.prompts.build_prompt`. A backend-specific prompt would
   invalidate the comparison the whole project is built on.
2. **Generation and scoring stay separate.** `scripts/generate.py` writes raw
   model output; `scripts/evaluate.py` turns it into numbers. Re-scoring must
   never require re-generating.
