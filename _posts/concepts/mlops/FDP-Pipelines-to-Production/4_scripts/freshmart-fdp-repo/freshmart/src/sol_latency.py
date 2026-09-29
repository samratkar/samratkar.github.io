# CHALLENGE solution: measure the serving path and find the bottleneck, instead of
# guessing that "the model is slow".
#
# The serving path has two legs: fetch the features from the online store, then
# score them. Lab 2 already measured the second leg at about 0.6 ms. Measure the
# first, and the answer changes what you would optimise.
#
# A third number is measured here that most latency scripts omit: how OLD the
# served features are. The online store holds the last observed row per line, and
# a line that has not sold for three weeks is scored on three-week-old features.
# That is not a latency problem, but it is discovered by the same script - and it
# is the freshness pillar Lab 4 monitors.
import statistics as st
import time
import pandas as pd, mlflow
from feast import FeatureStore

NAME, ALIAS = "freshmart_reorder_model", "champion"
SERVED = ["units_7d", "units_28d", "trend_ratio", "days_since_last_sale",
          "quiet_days_28d", "avg_unit_price_7d", "lead_time_days", "promo_active"]

mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")
mv = mlflow.MlflowClient().get_model_version_by_alias(NAME, ALIAS)
model = (mlflow.xgboost.load_model if mv.tags.get("model_type") == "xgboost"
         else mlflow.sklearn.load_model)(f"models:/{NAME}@{ALIAS}")
fs = FeatureStore(repo_path="feature_repo")

off = pd.read_parquet("feature_repo/data/reorder_features.parquet")
latest = off.sort_values("event_timestamp").groupby(["store_id", "product_id"]).tail(1)
sample = latest.sample(50, random_state=0)
as_of = pd.Timestamp(off.event_timestamp.max())

lookup, predict, age = [], [], []
for r in sample.itertuples():
    t0 = time.perf_counter()
    f = fs.get_online_features(features=[f"reorder_features:{c}" for c in SERVED],
                               entity_rows=[{"store_id": r.store_id,
                                             "product_id": r.product_id}]).to_dict()
    t1 = time.perf_counter()
    X = pd.DataFrame([{c: f[c][0] for c in SERVED}])
    if X.isna().any().any():
        continue
    model.predict_proba(X)
    t2 = time.perf_counter()
    lookup.append((t1 - t0) * 1000)
    predict.append((t2 - t1) * 1000)
    age.append((as_of - r.event_timestamp).days)

total = [l + p for l, p in zip(lookup, predict)]
print(f"n={len(total)} single-entity requests, model v{mv.version} ({mv.tags.get('model_type')})")
print(f"  feature lookup  p50={st.median(lookup):>6.2f} ms   p95={sorted(lookup)[int(.95*len(lookup))]:>6.2f} ms")
print(f"  model predict   p50={st.median(predict):>6.2f} ms   p95={sorted(predict)[int(.95*len(predict))]:>6.2f} ms")
print(f"  end to end      p50={st.median(total):>6.2f} ms   p95={sorted(total)[int(.95*len(total))]:>6.2f} ms")
slower, faster = (("lookup", "model") if st.median(lookup) > st.median(predict)
                  else ("model", "lookup"))
ratio = max(st.median(lookup), st.median(predict)) / min(st.median(lookup), st.median(predict))
print(f"  the {slower} costs {ratio:.1f}x the {faster} on this machine")
print(f"\nfeature age at {as_of.date()}: p50={st.median(age):.0f} days, max={max(age)} days")
print("A quiet line is scored on the last day it sold - that is a data problem,")
print("not a latency one, and it is Lab 4's. On latency: whichever leg dominates,")
print("measure before optimising. Here the whole request is ~2 ms against a budget")
print("that is usually 100 ms, so the correct action is to change nothing.")
