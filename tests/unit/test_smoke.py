import anomaly_service


def test_package_exposes_version():
    assert anomaly_service.__version__
