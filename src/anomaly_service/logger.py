"""Project-wide logger.

- Plain text by default; JSON (one object per line) with LOG_JSON=true, for log shippers.
- `request_id` (API) and `job_id` (worker) live in context variables, so every log line of
  a request/job carries them without passing them around explicitly.
"""

from __future__ import annotations

import json
import logging
import os
from contextvars import ContextVar
from datetime import UTC, datetime

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
job_id_var: ContextVar[str | None] = ContextVar("job_id", default=None)

_TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.job_id = job_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key in ("request_id", "job_id"):
            if value := getattr(record, key, None):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "anomaly_service") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))
        handler.addFilter(_ContextFilter())
        log.addHandler(handler)
        log.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
        # Do not duplicate records through the root logger (e.g. under uvicorn/celery).
        log.propagate = False
    return log


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    log = get_logger()
    log.setLevel(level.upper())
    formatter = JsonFormatter() if json_logs else logging.Formatter(_TEXT_FORMAT)
    for handler in log.handlers:
        handler.setFormatter(formatter)


logger = get_logger()
