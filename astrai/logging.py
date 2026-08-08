import logging
import os


def setup_logging(level: str = "INFO"):
    """Attach a StreamHandler to the ``astrai`` logger (idempotent).

    Call once per process at the top of CLI scripts.
    Set ``ASTR_LOG_LEVEL`` env var to override the default level.

    Level names: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
    ``DEBUG`` enables per-step prefill/decode timing logs
    (:func:`astrai.inference.core.executor.timed`).
    """
    logger = logging.getLogger("astrai")
    if logger.handlers:
        return
    level_name = os.environ.get("ASTR_LOG_LEVEL", level).upper()
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
