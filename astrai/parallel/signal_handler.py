import logging
import os
import signal
import threading

logger = logging.getLogger(__name__)

_early_stop = threading.Event()
_active_context = None


def _early_handler(signum: int, frame):
    sig = signal.Signals(signum)
    logger.warning(
        "Received %s (pid=%d), requesting graceful training stop...",
        sig.name,
        os.getpid(),
    )
    _early_stop.set()
    if _active_context is not None:
        _active_context.request_stop()


def install_early_signal_handlers():
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _early_handler)


def register_signal_handlers(context):
    global _active_context
    _active_context = context
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _early_handler)
    if _early_stop.is_set():
        context.request_stop()
        logger.warning("Signal was received during initialization, stopping...")


def unregister_signal_handlers():
    global _active_context
    _active_context = None
    _early_stop.clear()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
