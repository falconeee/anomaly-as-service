from anomaly_service.training.config import PromotionPolicy
from anomaly_service.training.promote import decide

POLICY = PromotionPolicy(metric="f1", tolerance=0.01, max_latency_ms=1000)


def metrics(f1: float, latency_ms: float = 100.0) -> dict[str, float]:
    return {"f1": f1, "latency_ms": latency_ms}


def test_first_model_is_promoted():
    promoted, reason = decide(metrics(0.3), None, POLICY)
    assert promoted and reason == "no champion yet"


def test_better_or_slightly_worse_challenger_wins():
    assert decide(metrics(0.60), metrics(0.50), POLICY)[0]
    assert decide(metrics(0.495), metrics(0.50), POLICY)[0]  # within tolerance


def test_clearly_worse_challenger_is_rejected():
    promoted, reason = decide(metrics(0.40), metrics(0.50), POLICY)
    assert not promoted and "< champion" in reason


def test_slow_challenger_is_rejected_even_if_better():
    promoted, reason = decide(metrics(0.9, latency_ms=5000), metrics(0.5), POLICY)
    assert not promoted and "latency" in reason
