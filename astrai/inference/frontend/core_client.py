"""In-process request and event boundary for the inference frontend.

The Scheduler facade delegates execution and lifecycle operations to its
EngineCore. Both remain in the training model's process; request submission
and cancellation share the core's operation lock with model execution.
"""

from typing import List

from astrai.inference.core.scheduler import Scheduler


class EngineCoreClient:
    """Interface between the frontend and the engine core."""

    def send_request(self, **kwargs) -> str:
        """Submit one generation request; returns its request id."""
        raise NotImplementedError

    def send_requests(self, prompts: List[str], **kwargs) -> List[str]:
        """Submit a batch; returns the request ids in order."""
        raise NotImplementedError

    def abort_request(self, request_id: str) -> bool:
        """Cancel one request; False when it was already gone."""
        raise NotImplementedError

    def stats(self) -> dict:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


class InprocClient(EngineCoreClient):
    """Direct client of the Scheduler facade and its in-process EngineCore."""

    def __init__(self, scheduler: Scheduler):
        self._scheduler = scheduler

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    def send_request(self, **kwargs) -> str:
        return self._scheduler.add_request(**kwargs)

    def send_requests(self, prompts: List[str], **kwargs) -> List[str]:
        return self._scheduler.add_requests(prompts, **kwargs)

    def abort_request(self, request_id: str) -> bool:
        return self._scheduler.cancel_request(request_id)

    def stats(self) -> dict:
        return self._scheduler.get_stats()

    def shutdown(self) -> None:
        self._scheduler.stop()
