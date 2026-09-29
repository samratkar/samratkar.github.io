"""Lab 3 - the FreshMart reorder serving API.

The contract with the client is deliberately narrow: send an entity key, get a
probability. The client never sends features. It could not compute them correctly
anyway - units_7d is a trailing window over the POS log - and if it tried, its
arithmetic would drift from the training pipeline's within a sprint. That drift
is training-serving skew, and this design makes it structurally impossible: the
features come from the same Feast definition Lab 2 trained on.

Three details worth reading closely:

  * the model is loaded by ALIAS, not by path. models:/...@champion is whatever
    Lab 2's gate last promoted; Lab 5 will move that alias and this service will
    pick up the new model on restart with no code change.

  * liveness and readiness are separate endpoints and answer to different
    actors. If the registry or the online store is unreachable at startup, the
    process is still alive - it must not be restart-looped - but it is not ready
    and must not receive traffic.

  * an unknown entity key returns 404, not 500 and not a guess. Lab 1's Mission 4
    showed Feast returning None for a key it never materialised; this is where
    that null becomes an honest answer to the caller.
"""
import os
from contextlib import asynccontextmanager

import pandas as pd, mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from feast import FeatureStore

TRACKING = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///artifacts/mlflow.db")
NAME     = "freshmart_reorder_model"
ALIAS    = os.getenv("MODEL_ALIAS", "champion")
REPO     = os.getenv("FEAST_REPO", "feature_repo")

SERVED = ["units_7d", "units_28d", "trend_ratio", "days_since_last_sale",
          "quiet_days_28d", "avg_unit_price_7d", "lead_time_days", "promo_active"]
FEATURES = [f"reorder_features:{c}" for c in SERVED]

# The tier boundaries are NOT round numbers pulled from the air. 0.10 is the
# cost-optimal threshold Lab 2's Mission 3 found for a 5:1 stockout-to-overstock
# cost ratio; 0.25 is where precision is still above 0.4. Change the cost ratio
# and these change with it.
ACT_NOW, WATCH = 0.10, 0.25

S = {"ready": False, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup work belongs here, and it is allowed to fail without killing the
    # process - that is the whole point of having a readiness probe.
    try:
        mlflow.set_tracking_uri(TRACKING)
        mv = mlflow.MlflowClient().get_model_version_by_alias(NAME, ALIAS)
        load = (mlflow.xgboost.load_model if mv.tags.get("model_type") == "xgboost"
                else mlflow.sklearn.load_model)
        S["model"] = load(f"models:/{NAME}@{ALIAS}")
        S["ver"], S["tags"] = mv.version, dict(mv.tags)
        S["fs"] = FeatureStore(repo_path=REPO)
        S["ready"] = True
    except Exception as exc:                       # noqa: BLE001 - report, do not crash
        S["error"] = f"{type(exc).__name__}: {exc}"
    yield
    S.clear()


app = FastAPI(title="FreshMart Reorder API", lifespan=lifespan)


class Req(BaseModel):
    store_id: str
    product_id: str


@app.get("/health")            # liveness: is the process up?
def health():
    return {"status": "alive"}


@app.get("/ready")             # readiness: can it actually serve?
def ready():
    if not S.get("ready"):
        raise HTTPException(503, f"not ready: {S.get('error') or 'still loading'}")
    return {"status": "ready", "model_version": S["ver"], "alias": ALIAS,
            "model_type": S["tags"].get("model_type"),
            "data_hash": S["tags"].get("data_hash")}


@app.post("/predict")
def predict(r: Req):
    if not S.get("ready"):
        raise HTTPException(503, "not ready")
    feats = S["fs"].get_online_features(
        features=FEATURES,
        entity_rows=[{"store_id": r.store_id, "product_id": r.product_id}]).to_dict()
    row = {c: feats[c][0] for c in SERVED}
    if any(v is None for v in row.values()):
        raise HTTPException(404, f"no features for {r.store_id}/{r.product_id}")

    prob = float(S["model"].predict_proba(pd.DataFrame([row])[SERVED])[0][1])
    return {"store_id": r.store_id, "product_id": r.product_id,
            "reorder_probability": round(prob, 4),
            "action": ("order now" if prob >= WATCH else
                       "add to watch list" if prob >= ACT_NOW else "no action"),
            "model_version": S["ver"]}
