import logging
import os

from astrai.parallel.setup import get_rank, get_world_size


class _DistributedContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = str(get_rank())
        record.world_size = str(get_world_size())
        return True


def setup_logging(level: str = "INFO"):
    """Attach a StreamHandler to the ``astrai`` logger (idempotent).

    Call once per process at the top of CLI scripts.
    Set ``ASTR_LOG_LEVEL`` env var to override the default level.

    Level names: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
    ``DEBUG`` enables per-step prefill/decode timing logs
    (:func:`astrai.inference.runtime.executor.timed`).
    """
    logger = logging.getLogger("astrai")
    if logger.handlers:
        return
    level_name = os.environ.get("ASTR_LOG_LEVEL", level).upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    handler = logging.StreamHandler()
    handler.addFilter(_DistributedContextFilter())
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | rank=%(rank)2s/%(world_size)-2s | %(name)-32s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
