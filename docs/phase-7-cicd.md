# Phase 7: Continuous integration and delivery (CI/CD)

The GitHub Actions workflows run every check on each push and pull request, including
an end-to-end test with the Docker stack, and publish the images when a version is tagged.

## Main files

| File | When it runs | What it does |
|---|---|---|
| `.github/workflows/ci.yml` (`test` job) | Push to `main` and pull requests | Installs dependencies from `uv.lock`, ruff, mypy, pytest with coverage |
| `.github/workflows/ci.yml` (`e2e` job) | After `test` | Builds the images, starts the stack, runs `scripts/e2e.sh`, shows the logs on failure |
| `.github/workflows/release.yml` | Push of a `v*` tag (e.g. `v0.1.0`) | Publishes the images to GitHub Container Registry (GHCR) |
| `.github/dependabot.yml` | Weekly | Opens pull requests with dependency updates |
| `scripts/e2e.sh` | CI and `make e2e` | End-to-end test |

## Test layers

```
        end to end (CI, Docker)       real stack: MLflow + Postgres + Redis + worker
      ─────────────────────────────
     API and integration              in-memory API + simulated Redis; MLflow on SQLite
   ───────────────────────────────
  unit                                model, metrics, events, explanation, data loading
```

- The unit and integration tests are fast (~20 s) and run on every commit.
- The end-to-end test is slower, but it is the only one that catches configuration errors: a port
  in use, a missing database driver, permissions, the address MLflow accepts. Several errors of this kind
  only showed up on this project's first real Docker Compose run.

## What the end-to-end test checks

1. Trains a model on synthetic data inside the `trainer` container, registers and promotes it.
2. Asks the API to reload the model and waits for `/health/ready` to respond 200.
3. Sends `examples/request_synthetic.json` to `/v1/detect` and checks that there are anomalies.
4. Sends the same series to `/v1/jobs` and follows it until `succeeded`.
5. Checks that `/metrics` exposes the service metrics.

The test uses a really trained model, not a mock, to follow the same path as
production.

## Cache

- `astral-sh/setup-uv` keeps the downloaded packages between runs; without a change to `uv.lock`, nothing is
  downloaded again.
- In the release, the Docker layers are reused (`cache-from/to: type=gha`).

## Publishing the images

When a `v1.2.3` tag is created, the workflow publishes two images, each for Intel/AMD
(`amd64`) and ARM (`arm64`, such as Apple Silicon Macs) processors:

- `ghcr.io/<user>/anomaly-service` with the tags `1.2.3`, `1.2` and `sha-<commit>`;
- `ghcr.io/<user>/anomaly-service-mlflow` with the same tags.

Authentication uses the automatic GitHub Actions token (`GITHUB_TOKEN`); no additional secret
is needed.

## Permissions

The workflows only ask for what they need: read access to the repository (`contents: read`); the release also
asks for `packages: write` to publish images. Old runs on the same branch are cancelled
when a new one starts (`concurrency`).

## Step-by-step reproduction

1. Run the same checks as the `test` job locally:
   ```bash
   make check
   ```
2. Run the equivalent of the `e2e` job locally:
   ```bash
   make up && make e2e
   ```
3. Trigger CI: push a commit to GitHub and follow it in the *Actions* tab (`test` and
   `e2e` jobs).
4. Publish images (when desired):
   ```bash
   git tag v0.1.0 && git push origin v0.1.0
   ```
   The images show up under *Packages* on the GitHub profile.
