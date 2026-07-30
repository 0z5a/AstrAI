"""Pipeline configuration for JSONL preprocessing.

Supports single-sequence (SFT/pretrain) and multi-output (DPO/GRPO)
modes, both driven declaratively through ``input.sections`` or
``input.sources``.
"""

from dataclasses import field
from typing import Dict, List, Optional

from pydantic.dataclasses import dataclass

from astrai.config.base import BaseConfig


@dataclass
class InputConfig(BaseConfig):
    """Declarative input mapping.

    Single-output mode (backward-compatible)::

        {"input": {"sections": [{"field": "messages", ...}]}}

    Multi-output mode (DPO / GRPO)::

        {"input": {"sources": {
            "chosen": {"sections": [{"field": "chosen", ...}]},
            "rejected": {"sections": [{"field": "rejected", ...}]},
        }}}

    Args:
        sections (Optional[List[Dict]]): Section list for single-output mode. Defaults to None.
        sources (Optional[Dict[str, Dict]]): Source map for multi-output mode, DPO/GRPO. Defaults to None.
    """

    sections: Optional[List[Dict]] = None
    sources: Optional[Dict[str, Dict]] = None


@dataclass
class ProcessingConfig(BaseConfig):
    """Processing configuration for tokenization and packing.

    Args:
        max_seq_len (int): Maximum sequence length. Defaults to 2048.
        min_chars (int): Minimum number of characters to keep. Defaults to 50.
        max_chars (int): Maximum number of characters to keep. Defaults to 2_000_000.
        max_items (Optional[int]): Maximum number of items to process, None=unlimited. Defaults to None.
        batch_size (int): Number of records tokenized together. Defaults to 256.
        packing_strategy (str): How to pack sequences: 'simple', 'bfd', or 'bfd_split'. Defaults to "simple".
        max_packed_len (int): Maximum length of a packed bin. Defaults to 8192.
        truncation_mode (str): How to truncate over-length sequences: 'keep_start' or 'keep_end'. Defaults to "keep_start".
    """

    max_seq_len: int = 2048
    min_chars: int = 50
    max_chars: int = 2_000_000
    max_items: Optional[int] = None
    batch_size: int = 256
    packing_strategy: str = "simple"
    max_packed_len: int = 8192
    truncation_mode: str = "keep_start"


@dataclass
class OutputConfig(BaseConfig):
    """Output configuration for storage.

    Args:
        domain_key (Optional[str]): Domain key for the output store. Defaults to None.
        storage_format (str): Storage format: 'bin' or 'jsonl'. Defaults to "bin".
        max_tokens_per_shard (int): Maximum tokens per shard before splitting. Defaults to 100_000_000.
        dtype (Dict[str, str]): Per-key dtype overrides, e.g. {"input_ids": "int32"}. Defaults to {}.
        position_ids_mode (str): Position ids mode: 'none', 'doc_reset', or 'continuous'. Defaults to "doc_reset".
    """

    domain_key: Optional[str] = None
    storage_format: str = "bin"
    max_tokens_per_shard: int = 100_000_000
    dtype: Dict[str, str] = field(default_factory=dict)
    position_ids_mode: str = "doc_reset"


@dataclass
class PipelineConfig(BaseConfig):
    """Top-level preprocessing pipeline config.

    Args:
        version (int): Config schema version. Defaults to 1.
        input (InputConfig): Input mapping config.
        mask (Dict[str, str]): Per-field mask labels, e.g. {"system": "mask", "assistant": "train"}. Defaults to {}.
        mask_default (str): Default mask label for unlisted fields. Defaults to "mask".
        preprocessing (ProcessingConfig): Processing config.
        output (OutputConfig): Output config.
    """

    version: int = 1
    input: InputConfig = field(default_factory=InputConfig)
    mask: Dict[str, str] = field(default_factory=dict)
    mask_default: str = "mask"
    preprocessing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
