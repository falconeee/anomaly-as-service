"""Model Registry lifecycle: challenger -> champion promotion, rollback and status.

Every new registered version gets the alias ``@challenger``. Promotion re-evaluates the
challenger AND the current ``@champion`` on the same labeled test set, loading both from the
registry exactly as the service would, and moves ``@champion`` only if the challenger passes
the policy (quality gate). The service always serves ``models:/<name>@champion``, so
promoting or rolling back is just moving an alias.

CLI::

    python -m anomaly_service.training.promote challenger --config configs/train_synthetic.yaml
    python -m anomaly_service.training.promote rollback --model mscvae-synthetic --version 3
    python -m anomaly_service.training.promote status --model mscvae-synthetic
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime

import mlflow.pyfunc
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from anomaly_service.logger import logger
from anomaly_service.model import MSCVAE
from anomaly_service.training.config import PromotionPolicy, TrainConfig, TrainData, load_data
from anomaly_service.training.evaluate import evaluate_model

CHAMPION = "champion"
CHALLENGER = "challenger"


@dataclass
class PromotionDecision:
    promoted: bool
    reason: str
    challenger_version: str
    champion_version: str | None
    challenger_metrics: dict[str, float]
    champion_metrics: dict[str, float] | None


def decide(
    challenger: dict[str, float], champion: dict[str, float] | None, policy: PromotionPolicy
) -> tuple[bool, str]:
    """Pure promotion rule (no MLflow), so it can be unit-tested in isolation."""
    if challenger["latency_ms"] > policy.max_latency_ms:
        return False, f"latency {challenger['latency_ms']:.0f}ms exceeds limit {policy.max_latency_ms:.0f}ms"
    if champion is None:
        return True, "no champion yet"
    metric = policy.metric
    new, old = challenger[metric], champion[metric]
    if new >= old - policy.tolerance:
        return True, f"{metric} {new:.4f} >= champion {old:.4f} - tolerance {policy.tolerance}"
    return False, f"{metric} {new:.4f} < champion {old:.4f} - tolerance {policy.tolerance}"


def get_alias(client: MlflowClient, name: str, alias: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def load_registered(name: str, version: str | int) -> MSCVAE:
    """Loads a registered version through pyfunc, the same path the service uses."""
    pyfunc_model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    model: MSCVAE = pyfunc_model.unwrap_python_model().model
    return model


def _evaluate_version(name: str, version: str, data: TrainData, policy: PromotionPolicy) -> dict[str, float]:
    model = load_registered(name, version)
    return evaluate_model(model, data.test, data.labels, data.events, latency_rows=policy.latency_rows)


def promote_challenger(
    name: str, data: TrainData, policy: PromotionPolicy, client: MlflowClient | None = None
) -> PromotionDecision:
    client = client or MlflowClient()
    challenger = get_alias(client, name, CHALLENGER)
    if challenger is None:
        raise ValueError(f"Registered model {name!r} has no @{CHALLENGER} version")
    champion = get_alias(client, name, CHAMPION)
    # Stores disagree on the type of `version` (int in SQLite, str over REST): normalize.
    new_version = str(challenger.version)
    old_version = str(champion.version) if champion is not None else None

    if old_version == new_version:
        logger.info("Version %s is already @%s", new_version, CHAMPION)
        return PromotionDecision(False, "already champion", new_version, old_version, {}, None)

    challenger_metrics = _evaluate_version(name, new_version, data, policy)
    champion_metrics = _evaluate_version(name, old_version, data, policy) if old_version else None
    promoted, reason = decide(challenger_metrics, champion_metrics, policy)

    # The decision and the numbers behind it are recorded on the version itself (audit trail).
    tags = {
        "promotion.decision": "promoted" if promoted else "rejected",
        "promotion.reason": reason,
        "promotion.compared_with": old_version or "none",
        "promotion.dataset_digest": data.digest,
        "promotion.evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        f"promotion.{policy.metric}": f"{challenger_metrics[policy.metric]:.4f}",
    }
    for key, value in tags.items():
        client.set_model_version_tag(name, new_version, key, value)

    if promoted:
        client.set_registered_model_alias(name, CHAMPION, new_version)
        if old_version is not None:
            client.set_model_version_tag(name, old_version, "replaced_by", new_version)
    logger.info("Challenger v%s %s: %s", new_version, "PROMOTED" if promoted else "rejected", reason)
    return PromotionDecision(promoted, reason, new_version, old_version, challenger_metrics, champion_metrics)


def rollback(name: str, version: str | int, client: MlflowClient | None = None) -> None:
    """Points @champion at an older version. The service picks it up on its next refresh."""
    client = client or MlflowClient()
    client.get_model_version(name, str(version))  # fails fast if the version does not exist
    client.set_registered_model_alias(name, CHAMPION, str(version))
    client.set_model_version_tag(name, str(version), "promotion.decision", "rollback")
    logger.info("@%s of %s now points to version %s", CHAMPION, name, version)


def status(name: str, client: MlflowClient | None = None) -> list[dict[str, str]]:
    client = client or MlflowClient()
    # search_model_versions does not fill `aliases`; the registered model has the alias map.
    aliases: dict[str, list[str]] = {}
    for alias, version in client.get_registered_model(name).aliases.items():
        aliases.setdefault(str(version), []).append(alias)
    rows = []
    for mv in sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version)):
        metrics = client.get_run(mv.run_id).data.metrics if mv.run_id else {}
        rows.append(
            {
                "version": str(mv.version),
                "aliases": ",".join(sorted(aliases.get(str(mv.version), []))) or "-",
                "f1": f"{metrics.get('f1', float('nan')):.4f}",
                "pa_f1": f"{metrics.get('pa_f1', float('nan')):.4f}",
                "auc_pr": f"{metrics.get('auc_pr', float('nan')):.4f}",
                "decision": mv.tags.get("promotion.decision", "-"),
                "git_sha": mv.tags.get("git_sha", "-"),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="promote", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_challenger = sub.add_parser("challenger", help="evaluate @challenger vs @champion and promote")
    p_challenger.add_argument("--config", required=True, help="training YAML (model name + test data)")

    p_rollback = sub.add_parser("rollback", help="point @champion at a given version")
    p_rollback.add_argument("--model", required=True)
    p_rollback.add_argument("--version", required=True)

    p_status = sub.add_parser("status", help="list versions, aliases and metrics")
    p_status.add_argument("--model", required=True)

    args = parser.parse_args(argv)
    if args.command == "challenger":
        cfg = TrainConfig.from_yaml(args.config)
        decision = promote_challenger(cfg.registered_model, load_data(cfg.data), cfg.promotion)
        print(f"promoted={decision.promoted} reason={decision.reason}")
    elif args.command == "rollback":
        rollback(args.model, args.version)
    else:
        rows = status(args.model)
        if not rows:
            print(f"No versions registered for {args.model!r}")
            return
        widths = {h: max(len(h), *(len(r[h]) for r in rows)) for h in rows[0]}
        print("  ".join(f"{h:<{w}}" for h, w in widths.items()))
        for row in rows:
            print("  ".join(f"{row[h]:<{w}}" for h, w in widths.items()))


if __name__ == "__main__":
    main()
