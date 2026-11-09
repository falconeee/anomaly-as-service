# Phase 2: Training, evaluation and MLflow tracking

This phase turns training into a repeatable process: a configuration file describes
data, model and training; a single command trains, evaluates, logs everything to MLflow and, if requested,
publishes the model to the version catalog.

## Main files

| File | Role |
|---|---|
| `configs/train_synthetic.yaml` | Quick training on synthetic data (~15 s on CPU) |
| `configs/train_smd.yaml` | Training on the public SMD dataset |
| `configs/train_csv_example.yaml` | Configuration template for your own data (CSV) |
| `src/anomaly_service/training/config.py` | Configuration validation and data loading |
| `src/anomaly_service/training/train.py` | Pipeline: data → training → evaluation → MLflow → registry |
| `src/anomaly_service/training/evaluate.py` | Detection and root-cause metrics |
| `src/anomaly_service/data/smd.py` | Downloading and reading SMD |
| `src/anomaly_service/data/csv_source.py` | Reading your own CSV data |
| `src/anomaly_service/model/pyfunc.py` | Adapter from the model to MLflow's standard format |

## The configuration file

```yaml
experiment: mscvae-synthetic        # experiment name in MLflow
registered_model: mscvae-synthetic  # model name in the catalog (used by the API)
data:
  source: synthetic                 # synthetic | smd | csv
model:
  window_sizes: [5, 10, 20]         # window sizes, in rows
  seq_len: 3                        # how many previous windows the model considers
training:
  epochs: 10
  calibration_fraction: 0.2         # final fraction of training used to calibrate the threshold
promotion:
  metric: f1                        # metric used to decide promotion (Phase 3)
```

Unknown fields raise an error, which prevents a mistyped name from being silently ignored.

## Data sources

| `source` | Data | Anomaly labels |
|---|---|---|
| `synthetic` | Generated with anomalies at known positions | Yes (detection and cause) |
| `smd` | Server Machine Dataset, 38 server metrics per minute | Yes (detection and cause) |
| `csv` | Your own file | Optional (`label` column) |

### Your own data (CSV)

Expected format: one row per instant, one column per variable.

```
timestamp,cpu,memoria,latencia_ms,label
2024-01-01T00:00:00,41.2,63.0,120.5,0
2024-01-01T00:01:00,40.8,63.4,118.9,0
```

- `timestamp` (optional): used to sort the rows and to show dates in the responses.
- `label` (optional): `1` on known anomalies and `0` elsewhere. It is used **only to evaluate**
  the model; training does not use labels.
- The other numeric columns are the variables (or only those listed in `features`).
- The file is split in time: the first rows (70% by default) train the model and the
  last ones (30%) evaluate it. The training stretch must represent **normal** operation.
- Without the `label` column, the detection metrics are zero and promotion (Phase 3) only checks
  latency.

### Constant variables

A variable that does not change during training has a standard deviation close to zero. In normalization
(subtracting the mean and dividing by the standard deviation), any later variation would become a huge number.
That is why these variables are removed (`drop_constant: true`). In SMD, 8 of the 38 metrics are
constant during training.

## What gets logged to MLflow

Each training is an MLflow *run* with:

| Item | What it is for |
|---|---|
| Parameters (the whole configuration) and `config.yaml` | Knowing exactly how the model was trained |
| Loss per epoch (`train.loss`) | Following training convergence |
| Dataset with a content hash (`data.digest`) | Knowing which data produced the model |
| Code commit (`git_sha`, with `-dirty` if there were uncommitted changes) | Reproducing the training |
| Evaluation metrics | Comparing trainings and versions |
| `plots/test_scores.png` | Plot of the score over the test set, with the threshold and the real anomalies |
| `model_card.md` | Model summary: data, configuration, metrics, intended use, limitations |
| `model/` | The model in MLflow's standard format (*pyfunc*) |

The model does not depend on MLflow: training receives a function (`callback`) called at every epoch, and it is
the pipeline that sends those values to MLflow.

### Model format in MLflow

MLflow has a generic format called *pyfunc*: any model gets a
`predict(table)` method. The model is logged in *models from code* mode, in which MLflow stores a
small script (`pyfunc_model.py`) instead of an object serialized with `pickle`; the weights come from the
safe folder from Phase 1. The API does not use the generic `predict`: it extracts the full MSCVAE, which
also computes the causes.

## Evaluation metrics

| Metric | What it measures |
|---|---|
| `precision` | Of the points flagged as anomalous, how many really were |
| `recall` | Of the real anomalous points, how many were flagged |
| `f1` | Balance between precision and recall, **at the threshold used in production** |
| `pa_f1` | F1 with *point-adjust*: hitting one point of an anomalous stretch counts the whole stretch as a hit |
| `auc_pr` | Score quality regardless of the threshold |
| `best_f1` | F1 at the best possible threshold, chosen by looking at the test set itself (theoretical reference) |
| `false_alarm_rate` | Fraction of normal points flagged as anomalous |
| `hitrate_100`, `hitrate_150` | Fraction of the real causal variables that appear among the first in the cause ranking |
| `latency_ms` | Time to analyze 1,000 points |

*Point-adjust* is the protocol used in most papers on SMD, but it inflates the numbers: a
single hit in a 50-point stretch takes F1 from about 0.04 to 1.0. That is why both versions
are reported, and promotion uses the unadjusted F1.

`best_f1` and `auc_pr` help separate two problems: a bad score (the model does not tell
anomalies apart) or a bad threshold (the model tells them apart, but the cut is in the wrong place).

## Calibrating the threshold on unseen data

The threshold is computed from the scores of normal data. If that data is the same used
in training, the errors are smaller than they would be on new data (the network already knows it), and the threshold
ends up too low. With `calibration_fraction: 0.2`, the last 20% of the training period is not
used to fit the network; it is used to compute the threshold and each variable's typical error.

Effect measured on the synthetic data:

| | Threshold | False alarms on new normal data | Test F1 | False alarms on the test set |
|---|---|---|---|---|
| Calibrated on the training data | 74.5 | 2.40% | 0.390 | 5.59% |
| Calibrated on unseen data | 116.3 | 0.00% | 0.455 | 3.04% |

## Results on SMD (`machine-1-1`)

| Version | F1 | PA-F1 | AUC-PR | Optimal F1 | False alarms | HitRate@100% | HitRate@150% |
|---|---|---|---|---|---|---|---|
| v1 (threshold calibrated on training) | 0.382 | 0.408 | 0.623 | 0.615 | 30.3% | 0.511 | 0.688 |
| v2 (threshold calibrated on unseen data) | 0.369 | 0.377 | 0.635 | 0.617 | 34.5% | 0.605 | 0.784 |

Interpretation:

- The score separates the anomalies well (AUC-PR of about 0.63; with the best possible threshold, F1
  would reach 0.62), and the cause analysis gets 60% to 78% of the responsible variables right.
- The fixed threshold produces many false alarms: the second half of SMD (used as the test set) behaves
  differently from the first (used in training). A fixed threshold does not follow that
  change. The original SPOT algorithm has a mode that adjusts the threshold continuously, but it requires
  keeping state between requests, and the API was designed to be stateless (see Phase 4).
- v2 improved the cause analysis, but had a lower F1 and was **rejected** by the promotion rule
  (Phase 3).

## Fix to the synthetic generator

The parameters that define the simulated system (means and scales) were drawn after the noise;
asking for a longer series produced a different system. They are now drawn first, and a longer series
extends the shorter one with the same initial values.

## Step-by-step reproduction

1. Train on synthetic data, register and promote:
   ```bash
   make train
   ```
   Expected result at the end: `version=1` and `promoted=True reason=no champion yet`.
2. Open the MLflow UI:
   ```bash
   make mlflow        # http://localhost:5050
   ```
   In the `mscvae-synthetic` experiment, open the run and check *Parameters*, *Metrics*
   (`train.loss` per epoch), *Artifacts* (`plots/test_scores.png`, `model_card.md`, `model/`) and the
   *Datasets* tab.
3. (Optional) Train on SMD (~30 MB download; a few minutes):
   ```bash
   make train-smd
   ```
4. (Optional) Train with a sample CSV in the your-own-data format:
   ```bash
   make sample-csv                                    # creates datasets/exemplo.csv
   make train CONFIG=configs/train_csv_example.yaml   # model "mscvae-exemplo"
   ```
5. Run this phase's tests:
   ```bash
   uv run pytest tests/unit/test_evaluate.py tests/unit/test_smd.py tests/unit/test_csv_source.py tests/integration -v
   ```
