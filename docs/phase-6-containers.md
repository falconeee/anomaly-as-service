# Phase 6: Containers and Docker Compose

This phase packages the service into Docker images and describes the full infrastructure (API, worker,
MLflow, database, queue and monitoring) in a single file, `docker-compose.yml`. With it,
the whole environment comes up with one command, the same on any machine.

## Services

| Service | Image | Role | Host port |
|---|---|---|---|
| `api` | `docker/Dockerfile` | FastAPI API | 8000 |
| `worker` | same image | Processes the queue | — |
| `trainer` | same image + `train` group | Trains and registers models (run on demand) | — |
| `mlflow` | `docker/mlflow.Dockerfile` | Training history and model catalog | 5050 |
| `postgres` | `postgres:16-alpine` | MLflow's database | — |
| `redis` | `redis:7.4-alpine` | Queue, results and job registry | — |
| `prometheus` | `prom/prometheus` | Metric collection (`monitoring` profile) | 9090 |
| `grafana` | `grafana/grafana` | Dashboard (`monitoring` profile) | 3000 |

Services with a profile only start when requested: `trainer` with `--profile train` (or
`docker compose run trainer`) and Prometheus/Grafana with `--profile monitoring`.

## How the image is built

`docker/Dockerfile` has two stages:

```
build:   Python + uv ──▶ installs dependencies ──▶ installs the package   (result: /app/.venv)
runtime: Python ◀── copies only /app/.venv  (no uv, no compilers, unprivileged user)
```

Relevant details:

- **Layer order.** First only the dependencies are installed, from `pyproject.toml` and
  `uv.lock`. That layer (large, because of PyTorch) is only rebuilt when the dependencies change;
  changing the code only rebuilds the small package layer.
- **uv cache.** Downloads are cached between builds (`--mount=type=cache`).
- **`--frozen`.** Installs exactly `uv.lock` and fails if it is out of date.
- **PyTorch without GPU.** The image is about 1.9 GB, of which 650 MB is PyTorch; with the GPU
  version it would go above 5 GB.
- **Unprivileged user** (uid 10001): limits the damage if the process is compromised.

## One image, several roles

The `docker/entrypoint.sh` script decides what the container runs:

| Command | Runs |
|---|---|
| `api` | API server (uvicorn) |
| `worker` | Celery worker |
| `train ...` | Training pipeline (same arguments as `python -m anomaly_service.training.train`) |
| `promote ...` | Version commands: `challenger`, `rollback`, `status` |

Since the API and the worker use the same image, they run exactly the same code.

## MLflow: separate database and files

- **Postgres** stores small, queryable information: runs, parameters, metrics, versions,
  aliases and tags.
- A **volume** (`/mlartifacts`) stores the large files: models, plots and model cards.
- With `--serve-artifacts`, the MLflow server itself receives and serves those files. The API, the worker and
  the trainer only need the MLflow address (`MLFLOW_TRACKING_URI`); none of them needs
  storage credentials.

In the cloud, you just switch the file destination to a bucket (for example
`--artifacts-destination s3://bucket/path`); the required library (boto3) is already in the image.

## Startup order

`depends_on` with `condition: service_healthy` makes each service wait for the previous one to be ready:
Postgres → MLflow → API and worker. The API health check uses `/health/live` (process up)
and not `/health/ready` (model loaded), because on a fresh installation there is no model yet and the
API needs to start anyway.

## Design choices

- **Volume instead of MinIO.** The initial idea was to use MinIO as S3-compatible
  storage, but the project stopped publishing public images (the `minio/minio` repository no longer
  responds on Docker Hub). The volume with the MLflow proxy keeps the database and the files separate.
- **Port 5050 for MLflow.** On macOS, port 5000 is taken by the AirPlay Receiver. The port is
  configurable (`MLFLOW_PORT`) and is also included in `--allowed-hosts`, the list of addresses
  MLflow accepts (protection against DNS rebinding attacks).
- **`postgresql+psycopg://` driver.** The current SQLAlchemy version uses the psycopg 3 driver by
  default; the driver is made explicit so as not to depend on that.
- **`OMP_NUM_THREADS=2`.** Without it, each PyTorch process would create one thread per machine core.
- **`GIT_PYTHON_REFRESH=quiet`.** The image has no git; the commit is passed in when the
  image is built (`GIT_SHA`).

## Step-by-step reproduction

Prerequisites: Docker Desktop (or Docker Engine) with Compose and `make`. If the `docker` command is not
found on macOS, add `/Applications/Docker.app/Contents/Resources/bin` to `PATH`.

1. Build and start the stack:
   ```bash
   make up
   ```
   Expected result: `Container ... Healthy` for `postgres`, `redis`, `mlflow`, `api` and `worker`.
2. Check the state:
   ```bash
   docker compose ps
   ```
3. Train and promote a model inside Docker:
   ```bash
   make train-docker
   ```
   Expected result at the end: `promoted=True`.
4. Run the end-to-end test:
   ```bash
   make e2e
   ```
   Expected result at the end: `E2E OK`.
5. Open the interfaces: http://localhost:5050 (MLflow) and http://localhost:8000/docs (API).
6. Stop the stack:
   ```bash
   make down           # keeps the data (models, history)
   make clean-docker   # also deletes the volumes
   ```
