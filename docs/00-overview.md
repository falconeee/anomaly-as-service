# Overview

This folder documents, phase by phase, how the service was built and how to reproduce each
phase. The guides can be read in order (from the development environment to monitoring) or
independently, looking up only the topic of interest.

## The problem

A monitored system (a server, an industrial machine, an application) produces several metrics
at the same time: CPU, memory, temperature, latency, etc. That is a **multivariate time
series**: a table with one row per instant and one column per variable.

The goal is to answer three questions:

1. **When** did the system behave abnormally?
2. **How severe** was it?
3. **Why**: which variables caused the anomaly, and was the problem an unusual value
   (*magnitude*) or a change in how the variables relate to each other (*correlation*)?

## The model (MSCVAE) in plain language

1. **Correlation matrices at several scales.** For each instant, the model looks at the last 10,
   30 and 60 measurements and computes how much each pair of variables moves together. The result is a
   "snapshot" of the relationships between the variables at that moment.
2. **Learning normal.** A neural network (a *variational autoencoder*) is trained on normal
   data only to compress and reconstruct those snapshots and the raw values.
3. **Measuring surprise.** On new data, the reconstruction error becomes the anomaly **score**:
   the worse the model can reconstruct it, the more unusual the moment.
4. **Deciding what is anomalous.** The **threshold** is computed with Extreme Value
   Theory (the SPOT algorithm): it models the highest errors observed on normal data and picks
   a value that normal data would only exceed with a probability of 1 in 10,000.
5. **Explaining.** Each variable's error is compared with its own typical error. The variables
   that deviated most from their own normal are reported as causes.

## Service architecture

```
                 ┌──────────────────────── docker compose ───────────────────────────┐
 client ──HTTP───▶ api (FastAPI) ──enqueues──▶ redis ◀──consumes── worker (Celery)   │
                 │   │ POST /v1/detect          (queue + results)        │            │
                 │   └──▶ InferenceService ◀── models:/<name>@champion ──┘            │
                 │                                   ▲  (automatic reload)            │
                 │ trainer (one-off job) ──registers and promotes──▶ mlflow ─▶ postgres│
                 │                                                  └──▶ volume       │
                 │ [monitoring profile] prometheus ──▶ grafana      (artifacts)       │
                 └────────────────────────────────────────────────────────────────────┘
```

- **api**: receives the series and responds with the anomalies. For large series, it sends the job
  to the queue.
- **worker**: processes the queued jobs with the same code as the API.
- **mlflow**: keeps the training history and the model versions.
- **trainer**: trains a model, registers it in MLflow and decides whether it replaces the current one.

## Phase roadmap

| Phase | Guide | Topic |
|---|---|---|
| 0 | [Foundation](phase-0-foundation.md) | Python environment with uv, code organization, quality tools |
| 1 | [Production-ready model](phase-1-model.md) | Saving and loading the model safely, reproducibility, tests |
| 2 | [Training and evaluation](phase-2-training-mlflow.md) | Training pipeline, metrics, experiment tracking in MLflow, your own data |
| 3 | [Versioning](phase-3-registry.md) | Model versions, automatic promotion, rollback |
| 4 | [API](phase-4-api.md) | Endpoints, validation, events and explanations |
| 5 | [Queues](phase-5-queues.md) | Asynchronous processing with Celery and Redis |
| 6 | [Containers](phase-6-containers.md) | Docker images and the full stack with Compose |
| 7 | [CI/CD](phase-7-cicd.md) | Automated tests and image publishing on GitHub |
| 8 | [Observability](phase-8-observability.md) | Metrics, dashboard and logs |

## Glossary

| Term | Meaning |
|---|---|
| Window | Stretch with the last N measurements, used to compute a "snapshot" of the system |
| Score (φ) | Number that measures how unusual an instant is; higher means more anomalous |
| Threshold | Score value above which an instant is considered anomalous |
| SPOT / EVT | Statistical method that sets the threshold from the extreme values of the normal data |
| Event | Continuous period of anomalous instants, reported as a single anomaly |
| RCA (*root cause analysis*) | Which variables explain the anomaly |
| Warm-up | The first points of a series, without enough history for a reliable evaluation |
| MLflow | Tool that tracks training runs (parameters, metrics, files) and versions models |
| Model Registry | MLflow's model catalog, with numbered versions (1, 2, 3...) |
| Alias | A nickname that points to a version: `@champion` (in production) and `@challenger` (candidate) |
| Quality gate | Rule that decides whether the candidate replaces the model in production |
| Queue / broker | List of pending jobs (here, in Redis), consumed by the workers |
| Worker | Process that runs the queued jobs |
| Hot reload | Switching the model in use without restarting the service |
