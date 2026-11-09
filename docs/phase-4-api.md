# Phase 4: API

The API receives a time series as JSON and returns the anomalies found, each with a
severity, causal variables and a text explanation. The model served is always the
`@champion` from the MLflow catalog, reloaded automatically when it changes.

## Main files

| File | Role |
|---|---|
| `src/anomaly_service/api/main.py` | Application creation, startup and model reloading |
| `src/anomaly_service/api/routes.py` | Endpoints |
| `src/anomaly_service/api/deps.py` | Authentication and access to the model/settings |
| `src/anomaly_service/schemas.py` | Request and response formats |
| `src/anomaly_service/inference/service.py` | Detection (`InferenceService`) and model management (`ModelManager`) |
| `src/anomaly_service/inference/events.py` | Grouping windows into events and severity |
| `src/anomaly_service/inference/explain.py` | Causes and explanation text |

## Endpoints

| Method | Route | Purpose |
|---|---|---|
| POST | `/v1/detect` | Immediate detection, for series up to `SYNC_MAX_POINTS` points (default 20,000) |
| POST | `/v1/jobs` | Background detection, for large series (Phase 5) |
| GET | `/v1/jobs/{id}` | Status and result of a background job |
| GET | `/v1/models/current` | Model in use: version, threshold, expected variables, windows |
| POST | `/admin/reload` | Reloads the model immediately |
| GET | `/health/live` | Whether the process is up |
| GET | `/health/ready` | Whether a model is loaded (and whether the queue is reachable) |
| GET | `/docs` | Interactive documentation (Swagger) |

## From series to response

1. **Validation.** The request is checked: all series with the same length, the same number of
   timestamps, options within limits. `null` values are accepted as missing data.
2. **Columns.** The variables are checked against the model's: extra columns are ignored
   (listed in `ignored_features`) and the order does not matter. If any is missing, the response is **422**
   with `missing` and `extra`.
3. **Score.** The model computes a score per window.
4. **Events.** Windows above the threshold become events. Anomalous stretches separated by up to
   `max_gap` normal windows are merged into a single event, so as not to raise several alerts for the
   same incident.
5. **Severity.** Depends on how many times the event's highest score exceeds the threshold: less than 1.5
   times is `low`, less than 3 times is `medium`, 3 times or more is `high`.
6. **Causes.** Each variable's deviation is summed over the event. A variable is a cause if
   it deviated from its own normal above `z_thresh` **and** stood out relative to the others. At most
   `top_k` causes are returned.
7. **Cause type.**
   - `magnitude`: the variable's value became unusual;
   - `correlation`: the value looks normal, but the variable stopped following the others as it
     used to.
8. **Observed and expected.** For each cause, the value received and the value the model considers
   normal, at the instant when that variable deviated the most.
9. **Text.** A sentence template (Portuguese or English) builds the explanation from those numbers.
   No language model is involved: the same input always produces the same text, at no
   extra cost and with no risk of the text citing causes the numbers do not support.

### Warm-up

The first `max(window_sizes) − 1` points do not have enough history (the model fills the
windows by repeating the first value). By default, they do not produce events; the response reports
how many there are in `warmup_points`. To analyze them anyway: `"include_warmup": true`.

### End of events

Each window looks backwards. That is why, right after an anomaly ends, the next windows
still contain anomalous points, and the reported end of the event tends to be a little later than the real
end.

## Automatic model reloading

Every `MODEL_REFRESH_INTERVAL_S` seconds, the API checks which version `@champion` points to.
If it changed, it loads the new version in the background and starts using it; in-flight requests
finish with the previous version. You can also force a reload with `POST /admin/reload`.

## Starting without a model

On a fresh installation there is no promoted model yet. In that case the API starts anyway:
`/health/live` responds 200, `/health/ready` responds 503 and `/v1/detect` responds 503 with
instructions. As soon as a model is promoted, the automatic reload loads it.

## Heavy processing without blocking the API

The model computation keeps the CPU busy. If it ran directly on the API's (asynchronous) main
loop, no other request would be served while it runs, not even the health checks.
That is why detection runs in a separate thread (`run_in_threadpool`). For large series, the API
responds **413** and points to `/v1/jobs`.

## Authentication

Sent in the `X-API-Key` header. The comparison runs in constant time, so that the key cannot
be discovered by measuring the response time. Without the `API_KEY` variable set,
authentication is off (for local development only).

## Design choices

- The request and response formats live in `schemas.py`, outside the `api/` folder, because they are also
  used by the worker and the inference layer.
- One API process per container: each process would have its own copy of the model in memory;
  to handle more load, you start more containers.
- Stateless processing between requests: each request carries the full series. Any
  API replica serves any request, and repeating a request gives the same result. On the
  other hand, the threshold does not adapt over time and the client has to resend the history.

## Step-by-step reproduction

1. Train and promote a model (if there is none yet):
   ```bash
   make train
   ```
2. Start the local API (without Docker):
   ```bash
   make serve            # http://localhost:8000 ; use PORT=8001 if the port is taken
   ```
3. Send the sample series:
   ```bash
   curl -s -X POST localhost:8000/v1/detect -H 'Content-Type: application/json' \
     -d @examples/request_synthetic.json | python -m json.tool | head -60
   ```
   Expected result: among the anomalies, events that start near rows 314, 805 and 1368,
   with causes `f04`, `f01` and `f07` (the variables where the anomalies were injected). Short
   `low`-severity events may also show up.
4. Explore the interactive documentation at http://localhost:8000/docs.
5. Run the API tests:
   ```bash
   uv run pytest tests/api tests/unit/test_inference.py -v
   ```
