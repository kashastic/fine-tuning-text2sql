"""Text-to-SQL fine-tuning: shared library.

Layout:
    config      loading YAML run configs
    data        BIRD / Spider loaders -> Example
    schema      SQLite DDL extraction, schema trimming, sample values
    prompts     prompt + chat-template construction
    postprocess extracting SQL from raw model output
    sql_exec    sandboxed SQLite execution and result comparison
    metrics     execution accuracy and friends
    rewards     GRPO reward functions built on sql_exec
"""

__version__ = "0.1.0"

from text2sql.data import Example  # noqa: E402,F401
