import logging
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor

from astrai.inference.core.cache import PagePool
from astrai.inference.core.task import Task
from astrai.inference.core.workspace import InferenceWorkspace
from astrai.inference.sample import sample
from astrai.model.automodel import AutoModel
from astrai.tokenize.tokenizer import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class SamplingBatchInfo:
    """Per-batch sampling parameters, cached across decode steps.

    Sampling params are constant for a given ordered task set, so they are
    built once (pinned-memory async H2D) and reused until the task set
    changes.  ``top_ks`` is int32 to match the native consumers.
    """

    temperatures: Tensor  # float32 [B]
    top_ks: Tensor  # int32  [B]
    top_ps: Tensor  # float32 [B]
    freq_penalties: Tensor  # float32 [B]
    has_freq: bool  # any frequency_penalty != 0 (avoids per-step GPU .any())


def _build_sampling_batch_info(tasks: List[Task], device) -> SamplingBatchInfo:
    pin = str(device).startswith("cuda")
    freq_penalties = torch.tensor(
        [t.frequency_penalty for t in tasks], dtype=torch.float32, pin_memory=pin
    ).to(device, non_blocking=True)
    return SamplingBatchInfo(
        temperatures=torch.tensor(
            [t.temperature for t in tasks], dtype=torch.float32, pin_memory=pin
        ).to(device, non_blocking=True),
        top_ks=torch.tensor(
            [t.top_k for t in tasks], dtype=torch.int32, pin_memory=pin
        ).to(device, non_blocking=True),
        top_ps=torch.tensor(
            [t.top_p for t in tasks], dtype=torch.float32, pin_memory=pin
        ).to(device, non_blocking=True),
        freq_penalties=freq_penalties,
        has_freq=bool((freq_penalties != 0).any()),
    )


class Executor:
    """Model forward passes for prefill and decode phases."""

    def __init__(
        self,
        model: AutoModel,
        tokenizer: AutoTokenizer,
        kv_cache: PagePool,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache = kv_cache
        self.device = device or next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype

        # Per-step decode cache for the steady-state case where the same
        # ordered task set decodes one token per step.  Sampling params are
        # constant across steps; position_ids grows by exactly 1.  Single-slot:
        # any task-set change is a cache miss.
        self._decode_cache: Optional[tuple] = None

        # Pre-allocated fixed-shape buffers for the decode hot path
        # (input_ids, decode mask, KV bind metadata).  Eagerly sized at init
        # so the workspace is CUDA-graph-capture friendly — no allocation
        # during capture.
        self._workspace = InferenceWorkspace(
            max_batch_size=kv_cache.max_batch_size,
            max_seq_len=kv_cache.max_seq_len,
            device=self.device,
            dtype=self.dtype,
        )

    def execute_prefill(self, tasks: List[Task], prompt_len: int, start_pos: int = 0):
        if start_pos >= prompt_len:
            return

        tasks = sorted(tasks, key=lambda t: t.task_id)
        batch_sz = len(tasks)

        input_ids = torch.tensor(
            [t.prompt_ids[start_pos:prompt_len] for t in tasks],
            dtype=torch.long,
            device=self.device,
        )

        task_ids = [t.task_id for t in tasks]
        position_ids = (
            torch.arange(start_pos, prompt_len, dtype=torch.long, device=self.device)
            .unsqueeze(0)
            .expand(batch_sz, -1)
        )
        input_mask = position_ids.unsqueeze(-1) >= torch.arange(
            prompt_len, device=self.device
        )

        with torch.inference_mode():
            self.model(
                input_ids,
                input_mask=input_mask,
                position_ids=position_ids,
                kv_cache=self.kv_cache.bind_tasks(
                    task_ids,
                    self._workspace,
                    start_pos=start_pos,
                ),
            )

    def execute_decode(
        self, tasks: List[Task], return_logprobs: bool = False
    ) -> List[int]:
        """Decode next token for each task.

        Args:
            return_logprobs: When ``True``, also record (and return)
                the log-probability of each sampled token under the
                post-strategy sampling distribution.  The logprob is
                appended to ``task.output_logprobs`` and the return
                list becomes ``List[Tuple[int, float]]``.

        Returns:
            ``List[int]`` of sampled token IDs, or
            ``List[Tuple[int, float]]`` of ``(token_id, logprob)`` when
            ``return_logprobs`` is ``True``.
        """
        if not tasks:
            return []

        input_ids = self._workspace.fill_input_ids(
            [t.output_ids[-1] if t.output_ids else t.prompt_ids[-1] for t in tasks]
        ).unsqueeze(1)

        task_ids = [t.task_id for t in tasks]

        sig = tuple(task_ids)
        cur_positions = [t.next_pos for t in tasks]
        cached = self._decode_cache
        if (
            cached is not None
            and cached[0] == sig
            and cur_positions == [p + 1 for p in cached[1]]
        ):
            _, _, info, position_ids = cached
            position_ids += 1
            self._decode_cache = (sig, cur_positions, info, position_ids)
        else:
            info = _build_sampling_batch_info(tasks, self.device)
            position_ids = torch.tensor(
                cur_positions, dtype=torch.long, device=self.device
            )
            self._decode_cache = (sig, cur_positions, info, position_ids)

        total_len = max(t.next_pos for t in tasks) + 1
        input_mask = self._workspace.decode_mask(position_ids, total_len)

        has_freq = info.has_freq
        if has_freq:
            history_lists = []
            history_lens = []
            for t in tasks:
                window = t.rep_window
                prompt_part = t.prompt_ids[-window:]
                ids = prompt_part + t.output_ids
                history_lists.append(ids)
                history_lens.append(len(ids))

            max_len = max(history_lens) if history_lens else 0
            padded_ids = torch.zeros(
                len(tasks), max_len, dtype=torch.long, device=self.device
            )
            padded_mask = torch.zeros(
                len(tasks), max_len, dtype=torch.bool, device=self.device
            )
            for i, h in enumerate(history_lists):
                L = history_lens[i]
                padded_ids[i, :L] = torch.as_tensor(
                    h, dtype=torch.long, device=self.device
                )
                padded_mask[i, :L] = True
        else:
            padded_ids = None
            padded_mask = None

        with torch.inference_mode():
            outputs = self.model(
                input_ids,
                input_mask=input_mask,
                kv_cache=self.kv_cache.bind_tasks(
                    task_ids,
                    self._workspace,
                ),
                position_ids=position_ids.unsqueeze(1),
            )
            logits = outputs["logits"][:, -1, :]

        if return_logprobs:
            tokens, logprobs = sample(
                logits,
                temperature=info.temperatures,
                top_k=info.top_ks,
                top_p=info.top_ps,
                frequency_penalty=info.freq_penalties,
                input_ids=padded_ids,
                input_mask=padded_mask,
                return_logprobs=True,
            )
            tokens_list = tokens.tolist()
            logprobs_list = logprobs.tolist()
            for t, lp in zip(tasks, logprobs_list):
                t.output_logprobs.append(float(lp))
            return list(zip(tokens_list, logprobs_list))

        return sample(
            logits,
            temperature=info.temperatures,
            top_k=info.top_ks,
            top_p=info.top_ps,
            frequency_penalty=info.freq_penalties,
            input_ids=padded_ids,
            input_mask=padded_mask,
        ).tolist()
