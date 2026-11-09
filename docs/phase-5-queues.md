# Phase 5: Background processing (queues)

Very large series take a while to analyze. Instead of keeping the HTTP connection open for the
whole computation, the API records the request in a **queue** and responds right away with an identifier.
Separate processes (**workers**) consume the queue, and the client fetches the result later.

## Main files

| File | Role |
|---|---|
| `src/anomaly_service/worker/celery_app.py` | Celery configuration (queue and message delivery) |
| `src/anomaly_service/worker/tasks.py` | Tasks: detection and webhook delivery |
| `src/anomaly_service/jobs.py` | Job registry and duplicate control in Redis |
| `src/anomaly_service/api/routes.py` | `/v1/jobs` endpoints |

## Flow

```
client ──POST /v1/jobs──▶ API ──validates──▶ Redis ("detections" queue) ──▶ worker
   ▲                        │                                                │
   └──GET /v1/jobs/{id}─────┴──────── Redis (results, kept for 24 h) ◀───────┘
                                                                             └─▶ webhook (optional)
```

| Endpoint | Response |
|---|---|
| `POST /v1/jobs` | **202** with `job_id`, `status` and `status_url` (also in the `Location` header) |
| `GET /v1/jobs/{id}` | `status`: `queued`, `running`, `succeeded` (with `result`) or `failed` (with `error`) |

`result` has exactly the same format as the `/v1/detect` response.

## Advantages of the queue

- The API only validates and enqueues, so it responds in milliseconds.
- The workers process at whatever pace they can; demand spikes wait in the queue instead of
  bringing the service down.
- The API and the workers can be scaled separately.
- If a worker dies in the middle of a job, the job goes back to the queue.

## Delivery guarantee

The configuration guarantees that each job runs **at least once**:

| Setting | Effect |
|---|---|
| `task_acks_late=True` | The message is only removed from the queue **after** the job finishes |
| `task_reject_on_worker_lost=True` | If the worker process dies, the message goes back to the queue |
| `worker_prefetch_multiplier=1` | Each worker takes one job at a time, without reserving others while busy |
| `visibility_timeout` = 2 × time limit | Maximum time before Redis considers a message lost and redelivers it; it must be longer than the longest job |

"At least once" means that, in case of failure, a job may run twice.
That is not a problem here: detection always produces the same result for the same input.

## Avoiding duplicate jobs: `Idempotency-Key`

If the client resends the same request (for example, after a network timeout), a second
job must not be created. With the `Idempotency-Key: <key>` header, the first request associates the
key with the `job_id`; subsequent ones with the same key get **200** with the same `job_id`.

## Retries

| Situation | Behavior |
|---|---|
| Model unavailable (MLflow down, no promoted model) | Retry with increasing backoff, up to 5 times |
| Wrong variables in the series | Immediate failure, no retry (the API already refuses it on submission when it knows the model) |
| Webhook delivery failure | Retry, up to 6 times, without repeating the detection |

## Webhook notification

If the request includes `"callback_url": "https://..."`, when it finishes the worker sends a POST to
that address with `{"job_id", "status", "result"}`.

## One job per worker (`--pool solo`)

Celery normally creates subprocesses by forking the main process. Forking after
PyTorch has started its threads can hang the process, and each subprocess would have its own
copy of the model. With `--pool solo`, each worker container runs one job at a time; to
process more jobs in parallel, you start more containers
(`docker compose up -d --scale worker=3`).

## Telling "queued" apart from "does not exist"

Celery reports `PENDING` for **any** unknown identifier. That is why the API records
each job in Redis at submission time and responds **404** for identifiers that never
existed or have already expired.

## Design choices

- Celery with Redis: Redis serves at the same time as the queue, the result store and the
  job registry. RabbitMQ would need an extra service for the results; Kafka is aimed at
  event streams, not at job queues with individual results.
- Same image for API and worker (Phase 6): immediate and queued detection run exactly the
  same code.
- In the tests, the jobs run inside the test process itself
  (`CELERY_TASK_ALWAYS_EAGER=true`), with the queue and results in memory and a simulated Redis
  (`fakeredis`); the code that runs is the same as in production.
- For extremely large series, a possible evolution is to store the series in a file
  store and send only the reference to the queue.

## Step-by-step reproduction

Requires the Docker stack (Phase 6), which includes Redis and the worker.

1. Start the stack and train a model:
   ```bash
   make up && make train-docker
   ```
2. Submit a job:
   ```bash
   curl -s -X POST localhost:8000/v1/jobs -H 'X-API-Key: change-me' \
     -H 'Content-Type: application/json' -H 'Idempotency-Key: test-1' \
     -d @examples/request_synthetic.json
   ```
   Expected result: `{"job_id": "...", "status": "queued", "status_url": "/v1/jobs/..."}`.
3. Fetch the result (replacing the identifier):
   ```bash
   curl -s localhost:8000/v1/jobs/<job_id> -H 'X-API-Key: change-me' | python -m json.tool | head
   ```
   Expected result: `"status": "succeeded"` and the `result` field.
4. Repeat step 2 with the same `Idempotency-Key`: the same `job_id` is returned.
5. Follow the worker logs (each line carries the `job_id`):
   ```bash
   docker compose logs -f worker
   ```
