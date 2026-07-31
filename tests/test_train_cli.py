import re

from click.testing import CliRunner

from scripts.tools.train import _merge_yaml_into_kwargs, train_command


def test_yaml_overrides_click_defaults_but_not_explicit_cli(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "training:\n"
        "  optimizer: nora_nadamw\n"
        "  max_lr: 0.0002\n"
        "  nora_lr: 0.004\n"
        "  batch_per_device: 8\n",
        encoding="utf-8",
    )
    click_values = {
        "optimizer": "nora_nadamw",
        "max_lr": 3e-4,
        "nora_lr": 5e-3,
        "batch_per_device": 16,
    }

    merged = _merge_yaml_into_kwargs(
        str(config_path), click_values, explicit_keys={"batch_per_device"}
    )

    assert merged["max_lr"] == 2e-4
    assert merged["nora_lr"] == 4e-3
    assert merged["batch_per_device"] == 16


def test_train_dry_run_uses_yaml_then_explicit_cli(tmp_path):
    data_path = tmp_path / "data"
    model_path = tmp_path / "model"
    data_path.mkdir()
    model_path.mkdir()
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "data:\n"
        f"  data_root_path: {data_path}\n"
        "model:\n"
        f"  param_path: {model_path}\n"
        "training:\n"
        "  train_type: seq\n"
        "  optimizer: nora_nadamw\n"
        "  max_lr: 0.0002\n"
        "  nora_lr: 0.004\n"
        "  batch_per_device: 8\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        train_command,
        ["--config", str(config_path), "--dry-run", "--batch_per_device", "16"],
    )

    assert result.exit_code == 0, result.output
    assert re.search(r"Optimizer\s+: nora_nadamw", result.output)
    assert re.search(r"Batch/device\s+: 16", result.output)
    assert re.search(r"Max LR\s+: 0.0002", result.output)
