# Phase 8: Observability and model monitoring

This phase makes visible what happens to the running service: volume and response time,
errors, queue size and, above all, how the model behaves on the incoming data.

## Main files

| File | Role |
|---|---|
| `src/anomaly_service/observability.py` | Definition of the model and queue metrics |
| `src/anomaly_service/logger.py` | Text or JSON logs, with request and job identifiers |
| `monitoring/prometheus.yml` | Where Prometheus scrapes the metrics |
| `monitoring/grafana/` | Data source and dashboard loaded automatically by Grafana |

## Where the metrics are exposed

- API: `http://localhost:8000/metrics`
- Each worker: port 9100 inside the Docker network

Prometheus scrapes these addresses every 15 s, and Grafana shows the
**Anomaly-as-a-Service** dashboard.

## Service metrics

They follow the four "golden signals" used in operations:

| Signal | Metric | Panel |
|---|---|---|
| Latency | `http_request_duration_seconds`, `anomaly_inference_seconds{mode}` | Response time (p95) per route; immediate vs. queued inference time |
| Traffic | `http_requests_total` | Requests per second per route |
| Errors | `http_requests_total{status="5xx"}`, `anomaly_detections_total{outcome="error"}` | Error rate |
| Saturation | `anomaly_queue_depth` | Jobs waiting in the queue |

A queue that only grows means more workers are needed.

## Model metrics

A service can respond fast and without errors and still give bad answers: anomaly
models fail silently when the data changes. These metrics follow the model's
behavior:

| Metric | What it reveals |
|---|---|
| `anomaly_feature_drift_z{feature}` | **Data drift**: distance between the last request's mean and the training mean, in training standard deviations, per variable |
| `anomaly_max_score_ratio` | **Score drift**: distribution of "highest score ÷ threshold" per request. If it rises under normal traffic, the model no longer represents the system well |
| `anomaly_events_total{severity}` | Number of anomalies per severity. A jump can be a real incident or an outdated model |
| `anomaly_model_info{name,version,alias}` | Model version in use in each process. It changes when `@champion` is switched, which makes it possible to relate behavior changes to model swaps |

It was this kind of data change that caused excessive alarms on SMD (Phase 2). In production, these
first two metrics would signal the problem before complaints about the alerts.

## Several workers

Each worker container exposes its own metrics. Prometheus discovers every container of the
`worker` service through Docker's internal DNS (`dns_sd_configs`), so increasing the number of workers
requires no configuration change. The model metrics show up both in the API (`mode="sync"`) and
in the workers (`mode="async"`), and the dashboard adds up both sources.

## Structured logs

With `LOG_JSON=true` (the default in Docker), each log line is a JSON object:

```json
{"ts": "2026-10-14T09:58:46.845+00:00", "level": "INFO", "logger": "anomaly_service", "msg": "Job done: 1500 points, 3 anomalies", "job_id": "ef4fbc65..."}
```

- In the API, each request gets a `request_id` (reusing the `X-Request-ID` header
  sent by the client, or generating a new one), which shows up in every log line of the request and is
  returned in the response header.
- In the worker, each line carries the `job_id` of the running job.

This makes it possible to filter all the logs of a specific request or job.

## Design choices

- Data drift is measured per request, compared with training. Aggregated analyses (windows
  of hours, statistical tests such as PSI or KS) would be an evolution, for example with periodic
  reports on samples of the requests.
- The dashboard is a JSON file versioned in the repository and loaded automatically when
  Grafana starts.

## Step-by-step reproduction

1. Start the stack, train a model and generate some traffic:
   ```bash
   make up && make e2e
   ```
2. Start Prometheus and Grafana:
   ```bash
   make monitoring
   ```
3. Open Grafana at http://localhost:3000 (anonymous read access; administrator:
   `admin`/`admin`) and the **Anomaly-as-a-Service** dashboard.
4. Check at http://localhost:9090/targets that `api` and `worker` are `UP`.
5. Look at the model metrics directly:
   ```bash
   curl -s localhost:8000/metrics | grep ^anomaly_
   ```
6. Look at the logs with identifiers:
   ```bash
   docker compose logs api | grep request_id
   ```
