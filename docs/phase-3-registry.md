# Phase 3: Model versioning and promotion

Each registered training becomes a new model **version** in the MLflow catalog (Model Registry).
An automatic rule decides whether the new version replaces the one in production. Promoting or rolling
back just means changing an alias, without rebuilding or restarting the service.

## Main files

| File | Role |
|---|---|
| `src/anomaly_service/training/promote.py` | Promotion rule, rollback and version listing |
| `src/anomaly_service/training/train.py` | Registers the new version (`--register`) and calls promotion (`--promote`) |

## Versions and aliases

- **Version**: each registered model gets a number (1, 2, 3...) that never changes.
- **Alias**: a nickname that points to a version and can be moved:
  - `@challenger`: the most recent version, the candidate;
  - `@champion`: the version in production, used by the API and the workers.

The API always loads `models:/<name>@champion`. When the alias changes, the API notices and switches
models by itself (Phase 4).

MLflow had an earlier mechanism, *stages* (`Staging`, `Production`), which was deprecated
in favor of aliases.

## Promotion rule (quality gate)

```
training ──registers──▶ version N with @challenger ──rule──▶ approved: @champion moves to N
                                                      └──▶ rejected: @champion stays the same
```

The rule loads both versions (candidate and current) **from the catalog itself**, the same way the
API would, and evaluates both on the **same test data**. The candidate is approved if:

1. its latency is within the limit (`max_latency_ms`); and
2. there is no model in production, **or** the chosen metric (`f1` by default) is greater than or equal to the
   current model's minus a tolerance (`tolerance`, 0.01 by default).

Reasons for evaluating both versions again, instead of comparing old numbers:

- the comparison is fair (same data, same evaluation code);
- if the registered model does not load (missing dependency, corrupted file), the failure shows up
  before it goes into production.

The tolerance avoids unnecessary swaps: two equivalent trainings vary a little by chance, and
without a margin a model equal to the current one could be rejected or approved at random.

## Recording the decision

The decision is recorded as *tags* on the version itself, visible in the MLflow UI:

| Tag | Example |
|---|---|
| `promotion.decision` | `promoted`, `rejected` or `rollback` |
| `promotion.reason` | `f1 0.3694 < champion 0.3816 - tolerance 0.01` |
| `promotion.compared_with` | version compared against (`none` if there was none) |
| `promotion.dataset_digest` | hash of the data used in the comparison |
| `replaced_by` | on the old version: which version replaced it |

The version also stores the code commit (`git_sha`), the data hash (`data.digest`) and the
*model card* as its description. With this, from the model in production you can get to the code,
the data and the configuration that produced it.

### Real example

On SMD, version 2 was rejected with the reason `f1 0.3694 < champion 0.3816 - tolerance 0.01`, and
version 1 stayed in production.

## Design choices

- Aliases instead of pinning the version number in the API configuration: promoting or rolling back does not
  require a new deployment.
- The default metric is F1 at the real threshold, because that is the behavior the service delivers. It can
  be changed in `promotion.metric` in the configuration.

## Step-by-step reproduction

1. Train twice with the same configuration:
   ```bash
   make train      # version 1 becomes @champion ("no champion yet")
   make train      # version 2 is compared with 1; same result, so it is promoted
   ```
2. List the versions:
   ```bash
   make status
   ```
   Expected result: a table with `version`, `aliases`, `f1`, `pa_f1`, `auc_pr`, `decision`
   and `git_sha`; version 2 with `challenger,champion`.
3. Roll back to version 1 and check:
   ```bash
   uv run python -m anomaly_service.training.promote rollback --model mscvae-synthetic --version 1
   make status
   ```
   Expected result: `champion` on version 1.
4. Run the full-cycle tests (registration, promotion, rejection and rollback):
   ```bash
   uv run pytest tests/integration/test_training_pipeline.py -v
   ```
