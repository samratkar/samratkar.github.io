"""Lab 2 / Missions 3-7 - baseline -> XGBoost -> DL challenger -> gate -> register.

Three decisions in this file are worth more than the modelling:

  1. The split is TEMPORAL, not random. Rows are a store x product x day panel
     with rolling features, so a random split puts 5 August in train and 4 August
     in test for the same line. The model is then scored on a world it has
     already seen. Train on the past, test on the future - always, for panels.

  2. The features are exactly the eight the online store serves. Training on a
     ninth would guarantee training-serving skew the day Lab 3 goes live.

  3. Nothing is registered without lineage. The data_hash from Lab 1's generate
     stage is attached to the run and to the model version, so any prediction can
     be traced back to the extract that produced it.
"""
import time, json
from pathlib import Path
import numpy as np, pandas as pd
import mlflow, mlflow.sklearn, mlflow.xgboost
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

NAME      = "freshmart_reorder_model"
AUC_GATE  = 0.65
SPLIT_DAY = pd.Timestamp("2026-08-01")   # train on July, test on August
SERVED    = ["units_7d", "units_28d", "trend_ratio", "days_since_last_sale",
             "quiet_days_28d", "avg_unit_price_7d", "lead_time_days", "promo_active"]

Path("artifacts").mkdir(exist_ok=True)
mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")
mlflow.set_experiment("freshmart_reorder")

df = pd.read_parquet("data/train.parquet")
tr = df[df.event_timestamp <  SPLIT_DAY]
te = df[df.event_timestamp >= SPLIT_DAY]
Xtr, ytr = tr[SERVED], tr.reorder_7d
Xte, yte = te[SERVED], te.reorder_7d
data_hash = (Path("data/pos_raw.hash").read_text().strip()
             if Path("data/pos_raw.hash").exists() else "nohash")

print(f"train {len(tr):,} rows (to {SPLIT_DAY.date()}) | test {len(te):,} rows (after) | "
      f"base rate train={ytr.mean():.3f} test={yte.mean():.3f}")


ONE_ROW = Xte.head(1)          # what serving actually asks for: a single line


def evaluate(model, threshold=0.5):
    p = model.predict_proba(Xte)[:, 1]
    pred = (p >= threshold).astype(int)
    # Latency must be measured the way it will be incurred. Amortising a 6,561-row
    # batch gives 0.000 ms/row and would let any model through the gate; Lab 3
    # serves one store-product line per request, so time that instead.
    model.predict_proba(ONE_ROW)                                   # warm up
    t0 = time.perf_counter()
    for _ in range(200):
        model.predict_proba(ONE_ROW)
    latency_ms = (time.perf_counter() - t0) / 200 * 1000
    return {"auc": roc_auc_score(yte, p), "f1": f1_score(yte, pred),
            "precision": precision_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred),
            "latency_ms_single": latency_ms}


results = {}
with mlflow.start_run(run_name="build") as parent:
    mlflow.set_tag("data_hash", data_hash)
    mlflow.set_tag("split_day", str(SPLIT_DAY.date()))
    mlflow.log_params({"n_features": len(SERVED), "split": "temporal",
                       "train_rows": len(tr), "test_rows": len(te)})

    # -- Mission 3: the bar every real model must clear ------------------------
    base = DummyClassifier(strategy="stratified", random_state=0).fit(Xtr, ytr)
    results["baseline"] = evaluate(base)

    # the *honest* baseline, and a full candidate: cheap, interpretable, and
    # much harder to beat than a dummy. It is on the ballot in Mission 6.
    with mlflow.start_run(run_name="logreg", nested=True) as r_lr:
        t0 = time.perf_counter()
        lr = make_pipeline(StandardScaler(),
                           LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
        m = evaluate(lr); m["train_seconds"] = time.perf_counter() - t0
        mlflow.log_metrics(m); mlflow.set_tag("data_hash", data_hash)
        mlflow.sklearn.log_model(lr, name="model", serialization_format="pickle")
        results["logreg"] = {**m, "run_id": r_lr.info.run_id}

    # -- Mission 4: Candidate A - XGBoost --------------------------------------
    with mlflow.start_run(run_name="xgboost", nested=True) as r_xgb:
        t0 = time.perf_counter()
        # regularised deliberately: an unconstrained XGBoost on 11k rows and 8
        # smooth features memorises July and scores 0.63 on August
        xgb = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.03,
                            min_child_weight=20, reg_lambda=5,
                            subsample=0.8, colsample_bytree=0.8,
                            eval_metric="logloss", n_jobs=2,
                            random_state=0).fit(Xtr, ytr)
        train_s = time.perf_counter() - t0
        m = evaluate(xgb); m["train_seconds"] = train_s
        mlflow.log_metrics(m); mlflow.set_tag("data_hash", data_hash)
        mlflow.xgboost.log_model(xgb, name="model")
        results["xgboost"] = {**m, "run_id": r_xgb.info.run_id}

    # -- Mission 5: Candidate B - the deep-learning challenger -----------------
    with mlflow.start_run(run_name="mlp", nested=True) as r_mlp:
        t0 = time.perf_counter()
        mlp = make_pipeline(StandardScaler(),
                            MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400,
                                          random_state=0)).fit(Xtr, ytr)
        train_s = time.perf_counter() - t0
        m = evaluate(mlp); m["train_seconds"] = train_s
        mlflow.log_metrics(m); mlflow.set_tag("data_hash", data_hash)
        # pickle, not the skops default: skops refuses to round-trip the MLP's
        # AdamOptimizer, and a model you cannot reload is not a registered model
        mlflow.sklearn.log_model(mlp, name="model", serialization_format="pickle")
        results["mlp"] = {**m, "run_id": r_mlp.info.run_id}

    for k in ("baseline", "logreg", "xgboost", "mlp"):
        r = results[k]
        print(f"{k:<9} AUC={r['auc']:.3f}  F1@0.5={r['f1']:.3f}  "
              f"precision={r['precision']:.3f}  recall={r['recall']:.3f}")

    # -- Mission 6: the release gate ------------------------------------------
    candidates = {k: results[k] for k in ("logreg", "xgboost", "mlp")}
    best_name = max(candidates, key=lambda k: candidates[k]["auc"])
    best = candidates[best_name]
    checks = {
        "quality  (AUC >= %.2f)" % AUC_GATE: best["auc"] >= AUC_GATE,
        "beats the baseline":                best["auc"] > results["baseline"]["auc"] + 0.05,
        "lineage  (data_hash present)":      data_hash != "nohash",
        "latency  (< 5 ms/request)":         best["latency_ms_single"] < 5.0,
    }
    for label, ok in checks.items():
        print(f"  gate: {label:<28} {'PASS' if ok else 'FAIL'}")
    gate_pass = all(checks.values())
    print(f"release gate: {'PASS' if gate_pass else 'FAIL'} "
          f"(best={best_name}, AUC={best['auc']:.3f})")
    mlflow.log_metrics({"best_auc": best["auc"], "gate_pass": int(gate_pass)})

    # -- Mission 7: register, with the lineage tag ----------------------------
    if gate_pass:
        c = mlflow.MlflowClient()
        mv = mlflow.register_model(f"runs:/{best['run_id']}/model", NAME)
        for k, v in {"data_hash": data_hash, "model_type": best_name,
                     "auc": f"{best['auc']:.4f}", "split_day": str(SPLIT_DAY.date())}.items():
            c.set_model_version_tag(NAME, mv.version, k, v)
        c.set_registered_model_alias(NAME, "champion", mv.version)
        print(f"registered {NAME} v{mv.version} as @champion (data_hash={data_hash})")
    else:
        print("not registered - the gate is the point, not the paperwork")

Path("artifacts/results.json").write_text(json.dumps(
    {k: {kk: vv for kk, vv in v.items() if kk != "run_id"} for k, v in results.items()},
    indent=2, default=float))
