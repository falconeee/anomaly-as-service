# Phase 1: Production-ready model

The original MSCVAE code was written for research: it trained and evaluated, but did not save the
model, did not validate the input columns and had a loop that might never end. This phase
makes the model safe to save, load from another process and use from an API.

## Main files

| File | Role |
|---|---|
| `src/anomaly_service/model/mscvae.py` | The model (with this phase's fixes) |
| `src/anomaly_service/model/persistence.py` | Saving and loading the model (`save_model`, `load_model`) |
| `src/anomaly_service/data/synthetic.py` | Synthetic data generator with known anomalies |
| `tests/unit/test_mscvae.py`, `test_persistence.py` | Model tests |

## Saving and loading the model safely

The model is saved as a **folder** with three files:

| File | Content |
|---|---|
| `weights.pt` | Neural network weights (numbers only, always saved for CPU) |
| `metadata.json` | Configuration, variable names, means and standard deviations used for normalization, threshold, library versions |
| `baselines.npz` | Typical error of each variable, used to explain the anomalies |

```python
model.save("artifacts/my_model")             # saves
model = MSCVAE.load("artifacts/my_model")    # loads (on CPU by default)
```

### Why not use `pickle`

The `pickle` format, Python's standard for saving objects, stores instructions that are **executed**
when the file is opened. A tampered file can run any code. Since the model is produced
by one process (training) and opened by another (the API), the project avoids that risk:

- the weights are read with `torch.load(..., weights_only=True)`, which refuses anything other than
  numbers;
- the rest is JSON and NPZ (read with `allow_pickle=False`).

The `format_version` field in `metadata.json` allows the format to evolve: a folder in an
unknown format is refused with a clear message.

## Validating the input columns

The model stores the names of the variables used in training (`model.feature_names`). On every
prediction:

- the columns are **reordered by name** into the training order;
- extra columns are ignored;
- if a variable is missing, the `FeatureSchemaError` error reports which ones are missing (`missing`) and which
  are extra (`extra`).

This check fixed a silent problem. pandas aligns operations by column **name**,
but the neural network reads the values by **position**. With the columns in another order, one
variable was read as if it were another, with no error at all:

```python
media = pd.Series([...], index=["b", "a"])   # training order: b, a
df = pd.DataFrame(..., columns=["a", "b"])   # data arrives in another order
(df - media).values                          # pandas reorders to (a, b); the network expects (b, a)
```

In an API that receives JSON, where key order is not guaranteed, this would happen
often.

## Reproducibility

- `set_deterministic` fixes the random seeds of Python, NumPy and PyTorch. Training (`fit`)
  does this again when it starts, so two trainings with the same configuration and the same data produce the
  same model.
- On CPU, the scores of two identical trainings match (the test uses a `1e-6` tolerance).
- A model trained on the Mac's GPU (MPS) and loaded on CPU produces scores with a relative difference
  of about `3e-7`, a normal rounding effect between devices.

## Computing the threshold without an infinite loop

The threshold uses the SPOT algorithm (Extreme Value Theory):

1. it separates the highest normal scores (above the 98th percentile), called the "tail";
2. it fits a statistical distribution (generalized Pareto) to that tail;
3. it picks the value that normal data would only exceed with probability `q = 0.0001`.

When there is no tail (all scores equal or too little data), the fit fails. The original code
kept retrying forever. Now there are at most 500 attempts; after that, the threshold
becomes the empirical `1 − q` percentile of the scores, and a warning is logged.

## Score and explanation in a single pass

`model.score_windows(df)` runs the network once and returns, for each window:

| Field | Meaning |
|---|---|
| `phi` | Anomaly score (comparable with `model.threshold`) |
| `s_diag` | Deviation of each variable taken **in isolation** |
| `s_off` | Deviation in each variable's **relationships** with the others |
| `s_val` | Deviation in the reconstruction of each variable's **raw value** |
| `recon_z` | Value the model considers normal (normalized) |
| `row_index` | Input row at which each window ends |

The deviations are measured relative to **the variable's own** typical error (median and MAD computed
on normal data) and are zero when the error is within normal. The API (Phase 4) uses these
channels to say whether the cause is one of **magnitude** (`s_diag`, `s_val`) or **correlation**
(`s_off`).

## Synthetic data for tests

The generator creates correlated variables (combinations of sine waves with noise) and injects
anomalies at known positions, which makes it possible to check whether the model detects and explains
them correctly:

| Type | What happens | How it shows up in the model |
|---|---|---|
| `spike` | The variable jumps a few standard deviations for a few points | Error in the raw value |
| `drift` | The variable moves away gradually | Error in the raw value, growing slowly |
| `correlation_break` | The variable starts moving opposite to the others, within the normal range | Error in the relationships; invisible to per-variable alerts |

## How the ML code is tested

There is no fixed "right value" to compare the score against. The tests check properties:

| Property | Example test |
|---|---|
| Reproducibility | Two trainings with the same seed give the same score |
| Invariance | Changing the column order or adding columns does not change the result |
| Round trip | Saving and loading keeps scores, threshold and explanations |
| Edge cases | Short series, missing values, constant scores |
| Behavior | An injected spike is detected and attributed to the right variable |

Training is the slow part, so `tests/conftest.py` trains a small model **only once** and
shares it across the tests.

## Fixes to the original code

- `np.round(X, decimals=5)` did not store its result (it had no effect); the rounding is now
  applied.
- The loss shown during training was divided by the number of samples, even though it was already a per-batch
  mean, which displayed it about 128 times smaller; it is now the per-batch mean.
- `predict` hid input errors and returned an empty result; the error is now reported.
- Without timestamps, `predict` returned the window number; it now returns the input row.
- A call to `gc.collect()` on every cause analysis was removed (it cost time with no benefit).
- The file goes through the same checks (ruff and mypy) as the rest of the project.

About the storage format: MLflow's default PyTorch model logging also uses
`pickle` and would not keep the rest of the state (normalization, threshold, typical errors); the ONNX
format would only cover the neural network. That is why a custom folder format was created.

## Step-by-step reproduction

1. Run the model tests:
   ```bash
   uv run pytest tests/unit/test_mscvae.py tests/unit/test_persistence.py -v
   ```
   Expected result: every test passing.
2. Train, save and load a small model:
   ```bash
   uv run python -c "
   from anomaly_service.data.synthetic import make_dataset
   from anomaly_service.model import MSCVAE
   train, test = make_dataset()
   m = MSCVAE(window_sizes=(5, 10, 20), device='cpu'); m.fit(train, epochs=5)
   loaded = MSCVAE.load(m.save('artifacts/demo'))
   print(loaded.feature_names, loaded.threshold)
   "
   ```
   Expected result: the list `['f00', ..., 'f07']` and a numeric threshold value.
3. Inspect the saved model:
   ```bash
   cat artifacts/demo/metadata.json
   ```
