import json

import numpy as np
import pytest
import torch

from anomaly_service.model import MSCVAE, load_model, read_metadata
from anomaly_service.model.persistence import BASELINES_FILE, METADATA_FILE, WEIGHTS_FILE


@pytest.fixture
def bundle(fitted_model, tmp_path):
    return fitted_model.save(tmp_path / "bundle")


def test_round_trip_preserves_scores_threshold_and_rca(fitted_model, bundle, test_series):
    loaded = MSCVAE.load(bundle)

    assert loaded.feature_names == fitted_model.feature_names
    assert loaded.threshold == fitted_model.threshold
    np.testing.assert_allclose(
        loaded.predict(test_series.data)["phi"], fitted_model.predict(test_series.data)["phi"], rtol=1e-6
    )

    original, restored = (m.score_windows(test_series.data) for m in (fitted_model, loaded))
    for channel in ("s_diag", "s_off", "s_val", "recon_z"):
        np.testing.assert_allclose(getattr(restored, channel), getattr(original, channel), rtol=1e-6)


def test_bundle_is_free_of_pickled_objects(bundle):
    assert sorted(p.name for p in bundle.iterdir()) == sorted([WEIGHTS_FILE, METADATA_FILE, BASELINES_FILE])

    # Both loaders refuse anything that is not plain tensors / arrays.
    torch.load(bundle / WEIGHTS_FILE, weights_only=True)
    with np.load(bundle / BASELINES_FILE, allow_pickle=False) as baselines:
        assert len(baselines.files) == 6


def test_metadata_describes_the_model(fitted_model, bundle):
    metadata = read_metadata(bundle)

    assert metadata["features"] == fitted_model.feature_names
    assert metadata["config"]["window_sizes"] == [5, 10, 20]
    assert metadata["threshold"] == fitted_model.threshold
    assert metadata["versions"]["torch"] == torch.__version__


def test_unknown_format_version_is_rejected(bundle):
    metadata_path = bundle / METADATA_FILE
    metadata = json.loads(metadata_path.read_text())
    metadata["format_version"] = 999
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="format_version"):
        load_model(bundle)


def test_unfitted_model_cannot_be_saved(tmp_path):
    with pytest.raises(ValueError, match="fitted"):
        MSCVAE(device="cpu").save(tmp_path / "bundle")
