import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Self, Union

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(use_attribute_docstrings=True))
class BaseConfig:
    def to_dict(self) -> Dict[str, Any]:
        result = {}
        for k, v in asdict(self).items():
            if isinstance(v, tuple):
                v = list(v)
            try:
                json.dumps(v)
                result[k] = v
            except (TypeError, ValueError):
                # Skip non-serializable runtime objects (e.g. model_fn, dataset).
                # TrainConfig mixes hyperparams with callables/datasets; only the
                # JSON-serializable subset is written to checkpoint meta.
                pass
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Self:
        return cls(**d)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> Self:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_file(self, path: Union[str, Path]):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
