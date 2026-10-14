"""Job registry in Redis: which job ids exist, when they were submitted, idempotency keys.

Celery's result backend answers "what is the state/result of task X", but it reports PENDING
for ANY unknown id, so it cannot tell "queued" from "never existed". This small registry
can, and also implements Idempotency-Key: the same key always maps to the same job.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from redis import Redis

_JOB_PREFIX = "anomaly:job:"
_IDEMPOTENCY_PREFIX = "anomaly:idempotency:"


class JobStore:
    def __init__(self, redis: Redis, ttl_s: int):
        self.redis = redis
        self.ttl_s = ttl_s

    def create(self, job_id: str, n_points: int) -> None:
        key = _JOB_PREFIX + job_id
        self.redis.hset(
            key, mapping={"submitted_at": datetime.now(UTC).isoformat(), "n_points": str(n_points)}
        )
        self.redis.expire(key, self.ttl_s)

    def get(self, job_id: str) -> dict[str, str] | None:
        data = self.redis.hgetall(_JOB_PREFIX + job_id)
        if not data:
            return None
        return {_text(k): _text(v) for k, v in data.items()}  # type: ignore[union-attr]

    def claim(self, idempotency_key: str, job_id: str) -> str | None:
        """
        Atomically binds `idempotency_key` to `job_id` (SET NX). Returns None when the key is
        new (the caller should enqueue `job_id`), or the job id it was already bound to.
        """
        key = _IDEMPOTENCY_PREFIX + idempotency_key
        if self.redis.set(key, job_id, nx=True, ex=self.ttl_s):
            return None
        existing = self.redis.get(key)
        return _text(existing) if existing is not None else None


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else value
