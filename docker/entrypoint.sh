#!/bin/sh
# Picks the process this container runs. Everything after the role is passed through.
set -e

role="${1:-api}"
[ "$#" -gt 0 ] && shift

case "$role" in
  api)
    # One uvicorn process per container: scale with more containers, not more workers
    # (each process would hold its own copy of the model).
    exec uvicorn anomaly_service.api.main:create_app --factory \
      --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers "$@"
    ;;
  worker)
    # solo pool: one job at a time in the main process (no fork after PyTorch init);
    # scale with `docker compose up --scale worker=N`.
    exec celery --app anomaly_service.worker.celery_app:celery_app worker \
      --pool solo --queues "${DETECTION_QUEUE:-detections}" \
      --loglevel "${LOG_LEVEL:-INFO}" --hostname "worker@%h" "$@"
    ;;
  train)
    exec python -m anomaly_service.training.train "$@"
    ;;
  promote)
    exec python -m anomaly_service.training.promote "$@"
    ;;
  *)
    exec "$role" "$@"
    ;;
esac
