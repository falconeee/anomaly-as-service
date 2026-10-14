"""Project-wide logger.

Phase 0 keeps this deliberately small: a single named logger configured once,
with the level taken from the LOG_LEVEL environment variable. Structured (JSON)
logs with request/job ids arrive in Phase 8.
"""

import logging
import os

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str = "anomaly_service") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        log.addHandler(handler)
        log.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
        # Do not duplicate records through the root logger (e.g. under uvicorn/celery).
        log.propagate = False
    return log


logger = get_logger()
