"""Experiment records that exclude infrastructure identity and paths."""

import hashlib
import importlib.metadata
from dataclasses import asdict

PRIVATE_RECIPE_FIELDS = {
    "model_path",
    "train_file",
    "dev_file",
    "test_file",
    "output_dir",
}
DEPENDENCIES = (
    "torch",
    "numpy",
    "tokenizers",
    "safetensors",
    "huggingface-hub",
    "pydantic",
    "pyyaml",
    "transformers",
    "flash-attn",
    "pytest",
    "ruff",
)


def public_recipe(recipe):
    return {
        key: ("<private>" if value is not None else None)
        if key in PRIVATE_RECIPE_FIELDS
        else value
        for key, value in asdict(recipe).items()
    }


def public_pretrained(metadata):
    # Unknown metadata is private by default. Source directories and model
    # config _name_or_path can identify the staging environment.
    output = {}
    for key in ("loaded_tensor_count", "allow_partial_pretrained"):
        value = metadata.get(key)
        if type(value) in (int, bool):
            output[key] = value
    for key in ("mapping_sha256",):
        value = metadata.get(key)
        if (
            isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value)
        ):
            output[key] = value
    return output


def dependency_versions():
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def content_digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
