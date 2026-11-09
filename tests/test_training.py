"""Do the training scripts actually run? Everything but the GPU part.

These two scripts were the only ones in the repo never executed. The suite
checked that their `--help` works, which proves nothing about the parts that
fail in the first minute of a rented GPU: a config key that no longer exists,
LoRA `target_modules` that match no module in the architecture, a dataset
column the trainer cannot find, an optimiser that needs a library that is not
installed, or an adapter that is never written.

So: run them end to end on a ~10M-parameter random Qwen3 over the fixture
benchmark, on CPU, for a couple of steps.

What this CANNOT tell you, and must not be read as telling you:

  * nothing about quality -- the model is randomly initialised,
  * nothing about memory, throughput or whether 4B fits on a 40GB card,
  * nothing about 4-bit quantisation. `bitsandbytes` is CUDA-only, so
    `load_in_4bit` and `paged_adamw_8bit` are switched off here and are the
    two settings a GPU run exercises first. Keep `--limit`/a tiny subset on
    the first real run for that reason.

It is a wiring test. That is all, and it is the part that was missing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

pytest.importorskip("trl", reason="training extra not installed: pip install -e '.[train]'")
pytest.importorskip("peft", reason="training extra not installed: pip install -e '.[train]'")


def run_script(name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / name), *args],
        capture_output=True, text=True, cwd=REPO, check=False,
    )


@pytest.fixture(scope="module")
def processed(tmp_path_factory: pytest.TempPathFactory, data_root: Path) -> Path:
    """The fixture benchmark, built into the jsonl the trainers consume."""
    out = tmp_path_factory.mktemp("processed")
    built = run_script(
        "build_dataset.py",
        # `--splits` is nargs="+", so the splits go in one flag; repeating it
        # silently keeps only the last.
        "--datasets", "bird", "--splits", "train", "dev",
        "--data-root", str(data_root), "--out-dir", str(out),
        "--max-tables", "4", "--max-columns", "12", "--sample-values", "2",
        "--val-size", "1",
    )
    assert built.returncode == 0, built.stderr
    assert (out / "train.jsonl").exists(), built.stdout
    assert (out / "val.jsonl").exists(), built.stdout
    return out


# Settings that only exist because this runs on a CPU. Each one is a thing the
# GPU run does differently, which is why they are listed in one place rather
# than scattered through the tests.
def cpu_overrides(tiny_model: Path) -> list[str]:
    flags = {
        "model.name": str(tiny_model),
        "model.load_in_4bit": "false",      # bitsandbytes is CUDA-only
        "model.device_map": "cpu",
        "model.dtype": "float32",           # no bf16 on this CPU
        "model.attn_implementation": "eager",
    }
    return [f"--set={k}={v}" for k, v in flags.items()]


def test_sft_trains_and_writes_an_adapter(tmp_path: Path, tiny_model: Path,
                                          processed: Path):
    out_dir = tmp_path / "sft"
    result = run_script(
        "train_sft.py",
        "--config", "configs/sft_qwen3_4b.yaml",
        *cpu_overrides(tiny_model),
        f"--set=output_dir={out_dir}",
        f"--set=data.train_file={processed / 'train.jsonl'}",
        f"--set=data.val_file={processed / 'val.jsonl'}",
        "--set=train.max_seq_length=256",
        "--set=train.num_train_epochs=1",
        "--set=train.per_device_train_batch_size=1",
        "--set=train.gradient_accumulation_steps=1",
        "--set=train.bf16=false",
        "--set=train.gradient_checkpointing=false",
        "--set=train.optim=adamw_torch",    # paged_adamw_8bit needs bitsandbytes
        "--set=train.eval_strategy=no",
        "--set=train.save_strategy=no",
        "--set=train.load_best_model_at_end=false",
        "--set=train.early_stopping_patience=0",
        "--set=train.report_to=[]",
        "--set=train.logging_steps=1",
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    adapter = out_dir / "adapter"
    assert (adapter / "adapter_config.json").exists(), "no adapter written"
    assert any(adapter.glob("adapter_model.*")), "adapter has no weights"
    # The recipe must land beside the checkpoint, or a checkpoint cannot be
    # traced back to the config that produced it.
    assert (out_dir / "run.yaml").exists()
    metrics = json.loads((out_dir / "train_metrics.json").read_text())
    assert "train_loss" in metrics

    # LoRA must have attached to something. If target_modules matched no
    # module, peft trains nothing and the run still "succeeds".
    targets = json.loads((adapter / "adapter_config.json").read_text())["target_modules"]
    assert set(targets) >= {"q_proj", "v_proj"}, targets
    assert "[model] trainable" in result.stdout
    trainable = float(result.stdout.split("[model] trainable ")[1].split("M")[0])
    assert trainable > 0, "LoRA attached to nothing"


def test_sft_refuses_mismatched_save_and_eval_cadence(tmp_path: Path,
                                                      tiny_model: Path,
                                                      processed: Path):
    """The guard that fires before the model loads, so it costs no GPU time."""
    result = run_script(
        "train_sft.py",
        "--config", "configs/sft_qwen3_4b.yaml",
        *cpu_overrides(tiny_model),
        f"--set=output_dir={tmp_path / 'bad'}",
        f"--set=data.train_file={processed / 'train.jsonl'}",
        f"--set=data.val_file={processed / 'val.jsonl'}",
        "--set=train.load_best_model_at_end=true",
        "--set=train.save_strategy=epoch",
        "--set=train.eval_strategy=steps",
    )
    assert result.returncode == 1
    assert "load_best_model_at_end needs" in result.stdout


def test_grpo_refuses_to_start_from_a_base_model(tmp_path: Path, processed: Path):
    """Documented as a silent failure, so it has to be a loud one.

    GRPO from a base model scores ~0 on every rollout, every completion in the
    group ties, the advantage is zero and nothing moves -- while the loss
    looks perfectly healthy. This check costs nothing and must not regress.
    """
    result = run_script(
        "train_grpo.py",
        "--config", "configs/grpo_qwen3_4b.yaml",
        f"--set=output_dir={tmp_path / 'grpo'}",
        f"--set=data.train_file={processed / 'train.jsonl'}",
        "--set=model.adapter=null",
    )
    assert result.returncode == 1
    assert "model.adapter is not set" in result.stdout


def test_grpo_refuses_an_adapter_path_that_does_not_exist(tmp_path: Path,
                                                          processed: Path):
    """`make grpo` before `make sft` is the likely ordering mistake.

    The shipped config names `outputs/sft-plain/adapter`, which exists only
    after SFT has run, so this must fail before the base model loads rather
    than inside peft several GPU-minutes later.
    """
    result = run_script(
        "train_grpo.py",
        "--config", "configs/grpo_qwen3_4b.yaml",
        f"--set=output_dir={tmp_path / 'grpo'}",
        f"--set=data.train_file={processed / 'train.jsonl'}",
        f"--set=model.adapter={tmp_path / 'nope'}",
    )
    assert result.returncode == 1
    assert "does not exist" in result.stdout
    assert "make sft" in result.stdout


def test_grpo_refuses_a_directory_that_holds_no_adapter(tmp_path: Path,
                                                        processed: Path):
    """An output dir is not an adapter dir; `outputs/sft-plain` is the trap."""
    empty = tmp_path / "looks-plausible"
    empty.mkdir()
    result = run_script(
        "train_grpo.py",
        "--config", "configs/grpo_qwen3_4b.yaml",
        f"--set=output_dir={tmp_path / 'grpo'}",
        f"--set=data.train_file={processed / 'train.jsonl'}",
        f"--set=model.adapter={empty}",
    )
    assert result.returncode == 1
    assert "holds no adapter" in result.stdout


def test_grpo_trains_against_the_execution_reward(tmp_path: Path, tiny_model: Path,
                                                  processed: Path):
    """The whole GRPO loop: rollouts, real SQLite execution, a gradient step.

    The reward path is the interesting part -- it executes generated SQL
    against the fixture databases, so this covers the handoff from the
    trainer's completions to `text2sql.rewards` and back, which no other test
    reaches.
    """
    out_dir = tmp_path / "grpo"
    result = run_script(
        "train_grpo.py",
        "--config", "configs/grpo_qwen3_4b.yaml",
        # Training from the base model takes both: clearing the configured
        # adapter and saying so. A configured-but-missing adapter stays an
        # error, since that is the ordering mistake, not an intention.
        "--allow-base-model",
        "--set=model.adapter=null",
        *cpu_overrides(tiny_model),
        f"--set=output_dir={out_dir}",
        f"--set=data.train_file={processed / 'train.jsonl'}",
        "--set=grpo.num_generations=2",
        "--set=grpo.per_device_train_batch_size=2",
        "--set=grpo.gradient_accumulation_steps=1",
        "--set=grpo.max_prompt_length=192",
        "--set=grpo.max_completion_length=8",
        "--set=grpo.num_train_epochs=1",
        "--set=grpo.bf16=false",
        "--set=grpo.gradient_checkpointing=false",
        "--set=grpo.optim=adamw_torch",
        "--set=grpo.save_strategy=no",
        "--set=grpo.report_to=[]",
        "--set=grpo.log_completions=false",
        "--set=grpo.use_vllm=false",
        "--set=reward.max_workers=2",
        "--set=reward.timeout=5.0",
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    adapter = out_dir / "adapter"
    assert (adapter / "adapter_config.json").exists(), "no adapter written"
    assert (out_dir / "run.yaml").exists()
    assert (out_dir / "train_metrics.json").exists()
