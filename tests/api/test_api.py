"""End-to-end API tests: FastAPI TestClient + fakeredis + Celery in eager mode.

Every endpoint runs for real (sync detection, the job queue, idempotency, webhooks, health,
metrics); only the infrastructure is replaced by in-memory fakes.
"""

import fakeredis
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from anomaly_service.api.main import create_app
from anomaly_service.config import Settings
from anomaly_service.inference.service import ModelManager
from anomaly_service.worker import tasks
from conftest import to_payload

API_KEY = "test-key"


def make_settings(**overrides) -> Settings:
    values = {
        "api_key": API_KEY,
        "model_refresh_interval_s": 0,
        "sync_max_points": 2000,
        "celery_task_always_eager": True,
        **overrides,
    }
    return Settings(_env_file=None, **values)


@pytest.fixture
def settings(bundle) -> Settings:
    return make_settings(model_path=str(bundle))


@pytest.fixture
def client(settings, monkeypatch):
    # The (eager) worker uses its own module-level manager: point it at the same bundle.
    monkeypatch.setattr(tasks, "manager", ModelManager(settings))
    app = create_app(settings, redis_client=fakeredis.FakeRedis(decode_responses=True))
    with TestClient(app, headers={"X-API-Key": API_KEY}) as test_client:
        yield test_client


# --- Synchronous detection --------------------------------------------------------------


def test_detect_finds_and_explains_the_injected_spike(client, test_series):
    response = client.post("/v1/detect", json=to_payload(test_series.data))

    assert response.status_code == 200
    body = response.json()
    assert body["model"]["name"] == "bundle"
    assert body["n_points"] == len(test_series.data)
    assert body["warmup_points"] == 19  # largest window (20) - 1

    spike = next(a for a in test_series.anomalies if a.kind == "spike")
    hits = [a for a in body["anomalies"] if a["start_index"] <= spike.end and a["end_index"] >= spike.start]
    assert hits, "the spike must be detected"
    anomaly = hits[0]
    assert anomaly["root_causes"][0]["variable"] == spike.features[0]
    assert anomaly["severity"] in {"low", "medium", "high"}
    assert anomaly["start"].startswith(str(test_series.data.index[anomaly["start_index"]].date()))
    assert anomaly["explanation"].startswith("Anomalia de severidade")


def test_detect_without_timestamps_reports_indices_in_english(client, test_series):
    payload = to_payload(test_series.data, language="en", include_scores=True)
    del payload["timestamps"]

    body = client.post("/v1/detect", json=payload).json()

    assert body["anomalies"][0]["explanation"].endswith(".")
    assert "between points" in body["anomalies"][0]["explanation"]
    assert "start" not in body["anomalies"][0]  # None fields are omitted
    assert len(body["scores"]) == len(test_series.data)


def test_column_order_and_extra_columns_do_not_matter(client, test_series):
    df = test_series.data
    shuffled = df[list(reversed(df.columns))].assign(extra=1.0)

    a = client.post("/v1/detect", json=to_payload(df)).json()
    b = client.post("/v1/detect", json=to_payload(shuffled)).json()

    assert b["ignored_features"] == ["extra"]
    assert [x["max_score"] for x in a["anomalies"]] == [x["max_score"] for x in b["anomalies"]]


def test_missing_values_are_accepted(client, test_series):
    payload = to_payload(test_series.data)
    payload["series"]["f01"][100:110] = [None] * 10

    assert client.post("/v1/detect", json=payload).status_code == 200


def test_missing_feature_returns_422_with_details(client, test_series):
    payload = to_payload(test_series.data.drop(columns=["f03"]))

    response = client.post("/v1/detect", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]["missing"] == ["f03"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["series"]["f00"].pop(),  # series of different lengths
        lambda p: p["timestamps"].pop(),  # timestamps of a different length
        lambda p: p["options"].update(top_k=0),  # invalid option
        lambda p: p.update(series={}),  # no series
    ],
)
def test_invalid_payloads_return_422(client, test_series, mutate):
    payload = to_payload(test_series.data.iloc[:50])
    mutate(payload)
    assert client.post("/v1/detect", json=payload).status_code == 422


def test_too_many_points_for_sync_returns_413(client, test_series):
    big = pd.concat([test_series.data] * 4)  # 2400 > sync_max_points (2000)

    response = client.post("/v1/detect", json=to_payload(big))

    assert response.status_code == 413
    assert "/v1/jobs" in response.json()["detail"]


def test_api_key_is_required(client, test_series):
    response = client.post("/v1/detect", json=to_payload(test_series.data), headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


# --- Asynchronous jobs ------------------------------------------------------------------


def test_job_lifecycle(client, test_series):
    submitted = client.post("/v1/jobs", json=to_payload(test_series.data))

    assert submitted.status_code == 202
    job = submitted.json()
    assert submitted.headers["Location"] == job["status_url"] == f"/v1/jobs/{job['job_id']}"

    state = client.get(job["status_url"]).json()
    assert state["status"] == "succeeded"
    assert state["result"]["n_points"] == len(test_series.data)
    assert state["result"]["anomalies"]


def test_idempotency_key_returns_the_same_job(client, test_series):
    headers = {"Idempotency-Key": "order-42"}
    first = client.post("/v1/jobs", json=to_payload(test_series.data), headers=headers)
    second = client.post("/v1/jobs", json=to_payload(test_series.data), headers=headers)

    assert first.status_code == 202 and second.status_code == 200
    assert first.json()["job_id"] == second.json()["job_id"]


def test_unknown_job_returns_404(client):
    assert client.get("/v1/jobs/does-not-exist").status_code == 404


def test_job_with_wrong_schema_is_rejected_at_submission(client, test_series):
    response = client.post("/v1/jobs", json=to_payload(test_series.data.drop(columns=["f00"])))
    assert response.status_code == 422


def test_job_result_is_posted_to_the_callback(client, test_series, monkeypatch):
    calls = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

    def fake_post(url, json, timeout):
        calls.append((url, json))
        return FakeResponse()

    monkeypatch.setattr(tasks.httpx, "post", fake_post)
    payload = to_payload(test_series.data) | {"callback_url": "https://example.com/hook"}

    job_id = client.post("/v1/jobs", json=payload).json()["job_id"]

    ((url, body),) = calls
    assert url == "https://example.com/hook"
    assert body["job_id"] == job_id and body["status"] == "succeeded"


# --- Model info, health and metrics -----------------------------------------------------


def test_model_info(client, fitted_model):
    info = client.get("/v1/models/current").json()

    assert info["source"] == "local"
    assert info["features"] == fitted_model.feature_names
    assert info["window_sizes"] == [5, 10, 20]
    assert np.isclose(info["threshold"], fitted_model.threshold)


def test_health_endpoints(client):
    assert client.get("/health/live").json() == {"status": "ok"}
    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["model"] is True and ready.json()["queue"] is True


def test_service_starts_without_a_model_and_reports_unavailable(tmp_path, test_series):
    settings = make_settings(model_path=str(tmp_path / "missing"))
    app = create_app(settings, redis_client=fakeredis.FakeRedis(decode_responses=True))

    with TestClient(app, headers={"X-API-Key": API_KEY}) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        response = client.post("/v1/detect", json=to_payload(test_series.data.iloc[:50]))
        assert response.status_code == 503
        assert "No model loaded" in response.json()["detail"]


def test_metrics_expose_model_and_detection_counters(client, test_series):
    client.post("/v1/detect", json=to_payload(test_series.data))

    metrics = client.get("/metrics").text

    assert 'anomaly_model_info{alias="-",name="bundle",version="local"} 1.0' in metrics
    assert 'anomaly_detections_total{mode="sync",outcome="ok"}' in metrics
    assert "anomaly_feature_drift_z" in metrics
    assert "anomaly_queue_depth" in metrics
    assert "http_request_duration_seconds" in metrics


def test_request_id_is_propagated(client):
    response = client.get("/health/live", headers={"X-Request-ID": "abc-123"})
    assert response.headers["X-Request-ID"] == "abc-123"
