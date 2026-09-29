"""Lab 4 - drift detection and the re-execution decision engine.

The service is green and the model is answering. This asks a different question:
are the answers still right? Three things are measured, and the order matters.

  1. INPUT DRIFT   - has the feature distribution moved? PSI for magnitude, a KS
                     test for corroboration. PSI decides; KS never decides alone.
  2. PREDICTION DRIFT - has the score distribution moved? This is available
                     immediately, needs no labels, and is often the first signal.
  3. PERFORMANCE   - is the model actually worse? This needs labels, and for a
                     seven-day label they arrive seven days late. Here the batch
                     is synthetic so labels exist; in production this check runs a
                     week behind the other two, which is precisely why the first
                     two exist.

The decision follows from which of the three moved:
  input drift          -> re-run the DATA pipeline (the world changed; features
                          and the training set must be rebuilt before anything
                          is concluded about the model)
  performance drop but
  no input drift       -> re-run the ML pipeline (the relationship changed, not
                          the inputs: concept drift)
  neither              -> no action, and say so loudly enough that people trust
                          the monitor when it does fire
"""
import argparse, json
from pathlib import Path

import numpy as np, pandas as pd, mlflow
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

SERVED = ["units_7d", "units_28d", "trend_ratio", "days_since_last_sale",
          "quiet_days_28d", "avg_unit_price_7d", "lead_time_days", "promo_active"]
NAME, ALIAS = "freshmart_reorder_model", "champion"
PSI_THRESHOLD, AUC_DROP, LABEL_HORIZON = 0.20, 0.05, 7


def psi(ref, cur, bins=10):
    """Population Stability Index: sum (c-r) * ln(c/r) over reference quantile bins."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                      # a near-constant feature has no bins
        return 0.0
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def uncensored(df):
    """Drop rows whose 7-day label window is incomplete - same rule as Lab 2."""
    cutoff = pd.Timestamp(df.event_timestamp.max()) - pd.Timedelta(LABEL_HORIZON, unit="D")
    return df[df.event_timestamp <= cutoff]


def decide(ref_path, cur_path, tracking="sqlite:///artifacts/mlflow.db"):
    ref, cur = pd.read_parquet(ref_path), uncensored(pd.read_parquet(cur_path))

    # 1. input drift
    psi_scores = {c: round(psi(ref[c].values, cur[c].values), 4) for c in SERVED}
    ks_p = {c: float(ks_2samp(ref[c], cur[c]).pvalue) for c in SERVED}
    drifted = {k: v for k, v in psi_scores.items() if v > PSI_THRESHOLD}

    # 2. prediction drift + 3. performance, both from the live champion
    mlflow.set_tracking_uri(tracking)
    mv = mlflow.MlflowClient().get_model_version_by_alias(NAME, ALIAS)
    model = (mlflow.xgboost.load_model if mv.tags.get("model_type") == "xgboost"
             else mlflow.sklearn.load_model)(f"models:/{NAME}@{ALIAS}")
    p_ref = model.predict_proba(ref[SERVED])[:, 1]
    p_cur = model.predict_proba(cur[SERVED])[:, 1]
    pred_psi = round(psi(p_ref, p_cur), 4)

    auc_train = float(mv.tags.get("auc", "nan"))
    auc_now = float(roc_auc_score(cur.reorder_7d, p_cur))
    perf_drop = (auc_train - auc_now) > AUC_DROP

    if drifted:
        action = "rerun_data_pipeline"
        why = f"input drift on {sorted(drifted)} (PSI > {PSI_THRESHOLD})"
    elif perf_drop:
        action = "rerun_ml_pipeline"
        why = f"AUC {auc_train:.3f} -> {auc_now:.3f} with no input drift (concept drift)"
    else:
        action = "noop"
        why = "all signals within thresholds"

    return {
        "model_version": mv.version, "model_type": mv.tags.get("model_type"),
        "reference_rows": len(ref), "current_rows": len(cur),
        "psi": psi_scores, "ks_pvalues": {k: round(v, 6) for k, v in ks_p.items()},
        "drifted": drifted, "prediction_psi": pred_psi,
        "auc_at_training": auc_train, "auc_now": round(auc_now, 4),
        "label_rate_reference": round(float(ref.reorder_7d.mean()), 4),
        "label_rate_current": round(float(cur.reorder_7d.mean()), 4),
        "action": action, "reason": why,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="data/train.parquet")
    ap.add_argument("--current", default="data/current_batch.parquet")
    ap.add_argument("--out", default="artifacts/trigger_result.json")
    a = ap.parse_args()

    Path("artifacts").mkdir(exist_ok=True)
    rep = decide(a.reference, a.current)
    Path(a.out).write_text(json.dumps(rep, indent=2))

    top = sorted(rep["psi"].items(), key=lambda kv: -kv[1])[:3]
    print(f"[monitor] reference {rep['reference_rows']:,} rows vs current "
          f"{rep['current_rows']:,} rows, champion v{rep['model_version']}")
    print(f"[monitor] input drift  top PSI: " +
          ", ".join(f"{k}={v}" for k, v in top))
    print(f"[monitor] prediction drift PSI={rep['prediction_psi']}")
    print(f"[monitor] performance  AUC {rep['auc_at_training']:.3f} -> {rep['auc_now']:.3f}"
          f"   label rate {rep['label_rate_reference']} -> {rep['label_rate_current']}")
    print(f"[monitor] decision: {rep['action'].upper()}  ({rep['reason']})")
    print(f"[monitor] wrote {a.out}")
