import numpy as np
import pandas as pd
import pytest

from anomaly_service.model import MSCVAE, FeatureSchemaError
from anomaly_service.model.mscvae import AttributeMatrixGenerator
from conftest import FIT_KWARGS, MODEL_KWARGS

# --- Reproducibility --------------------------------------------------------------------


def test_fit_is_deterministic(fitted_model, train_df, test_series):
    other = MSCVAE(**MODEL_KWARGS)
    other.fit(train_df, **FIT_KWARGS)

    np.testing.assert_allclose(
        other.predict(test_series.data)["phi"], fitted_model.predict(test_series.data)["phi"], rtol=1e-6
    )
    assert other.threshold == pytest.approx(fitted_model.threshold, rel=1e-6)


def test_fit_reports_every_epoch_to_callback(train_df):
    calls = []
    model = MSCVAE(**MODEL_KWARGS)
    model.fit(
        train_df.iloc[:200],
        epochs=2,
        batch_size=64,
        verbose=False,
        callback=lambda e, m: calls.append((e, m)),
    )

    assert [epoch for epoch, _ in calls] == [1, 2]
    assert set(calls[0][1]) == {"loss", "beta", "epoch_time_s"}
    assert all(np.isfinite(m["loss"]) for _, m in calls)


# --- Feature schema ---------------------------------------------------------------------


def test_generator_aligns_columns_by_name_not_position(train_df):
    # Train on an UNSORTED column order: pandas would otherwise re-sort `df - mean` and
    # silently feed each variable into another variable's matrix row.
    reversed_cols = list(reversed(train_df.columns))
    generator = AttributeMatrixGenerator(window_sizes=(5, 10), step=1)
    generator.fit_scaler(train_df[reversed_cols])

    mats_trained_order, vals_trained_order = generator.generate(train_df[reversed_cols])
    mats_other_order, vals_other_order = generator.generate(train_df)

    assert torch_equal(mats_trained_order, mats_other_order)
    assert torch_equal(vals_trained_order, vals_other_order)


def test_column_order_does_not_change_scores(fitted_model, test_series):
    df = test_series.data
    shuffled = df[list(np.random.default_rng(0).permutation(df.columns))]

    np.testing.assert_array_equal(fitted_model.predict(shuffled)["phi"], fitted_model.predict(df)["phi"])


def test_missing_feature_raises_schema_error(fitted_model, test_series):
    df = test_series.data.drop(columns=["f03"]).assign(unknown=1.0)

    with pytest.raises(FeatureSchemaError) as exc:
        fitted_model.predict(df)

    assert exc.value.missing == ["f03"]
    assert exc.value.extra == ["unknown"]


def test_extra_columns_are_ignored(fitted_model, test_series):
    df = test_series.data
    with_extra = df.assign(unknown=np.arange(len(df)))

    np.testing.assert_array_equal(fitted_model.predict(with_extra)["phi"], fitted_model.predict(df)["phi"])


# --- Edge cases -------------------------------------------------------------------------


def test_series_shorter_than_largest_window_is_scored(fitted_model, test_series):
    short = test_series.data.iloc[:8]  # largest window is 20

    phi = fitted_model.predict(short)["phi"]

    assert len(phi) == 8
    assert np.all(np.isfinite(phi))


def test_missing_values_are_tolerated(fitted_model, test_series):
    df = test_series.data.copy()
    df.iloc[100:110, 2] = np.nan

    assert np.all(np.isfinite(fitted_model.predict(df)["phi"]))


def test_stride_maps_windows_to_input_rows(train_df):
    generator = AttributeMatrixGenerator(window_sizes=(5,), step=3)
    generator.fit_scaler(train_df)

    matrices, _ = generator.generate(train_df.iloc[:10])

    assert matrices.shape[0] == len(range(0, 10, 3))  # windows end at rows 0, 3, 6, 9


def test_pot_eval_terminates_on_constant_scores(fitted_model):
    # A constant series has no tail for SPOT to fit; this used to loop forever.
    assert fitted_model._pot_eval(np.ones(200)) == pytest.approx(1.0)


def test_pot_eval_threshold_sits_in_the_tail():
    scores = np.random.default_rng(0).exponential(size=5000)

    threshold = MSCVAE(device="cpu")._pot_eval(scores)

    assert np.quantile(scores, 0.98) < threshold < 20


# --- Scoring and root-cause analysis ----------------------------------------------------


def test_score_windows_matches_predict(fitted_model, test_series):
    scores = fitted_model.score_windows(test_series.data)

    np.testing.assert_allclose(scores.phi, fitted_model.predict(test_series.data)["phi"], rtol=1e-5)
    np.testing.assert_array_equal(scores.row_index, np.arange(len(test_series.data)))
    n_features = len(fitted_model.feature_names)
    for channel in (scores.s_diag, scores.s_off, scores.s_val, scores.recon_z):
        assert channel.shape == (len(test_series.data), n_features)
    assert (scores.s_diag >= 0).all() and (scores.s_off >= 0).all() and (scores.s_val >= 0).all()


def test_training_data_rarely_crosses_threshold(fitted_model, train_df):
    phi = fitted_model.predict(train_df)["phi"]

    assert (phi > fitted_model.threshold).mean() < 0.01


def test_detects_spike_and_blames_the_spiked_feature(fitted_model, test_series):
    spike = next(a for a in test_series.anomalies if a.kind == "spike")
    scores = fitted_model.score_windows(test_series.data)
    in_spike = (scores.row_index >= spike.start) & (scores.row_index < spike.end)

    assert scores.phi[in_spike].max() > fitted_model.threshold

    contrib = fitted_model.rca_window_scores(scores)[in_spike].sum(axis=0)
    assert scores.feature_names[int(np.argmax(contrib))] == spike.features[0]


def test_contribution_works_without_variable_metadata(fitted_model, test_series):
    contributions, reconstruction = fitted_model.contribution(test_series.data, top_k=2)

    assert len(contributions["VARIAVEL"]) == 2
    assert set(contributions["DESC"].values()) == {"NoDesc"}
    assert isinstance(reconstruction, pd.DataFrame)
    assert list(reconstruction.columns) == list(contributions["VARIAVEL"].values())


def torch_equal(a, b) -> bool:
    return a.shape == b.shape and bool((a == b).all())
