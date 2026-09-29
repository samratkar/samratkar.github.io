# CHALLENGE solution: batch scoring - ONE Feast lookup and ONE vectorised predict
# for N store-product lines, instead of N round trips through /predict.
#
# The interface should match the load. A planner scoring one line wants the API.
# The morning replenishment run scores every line in a store and wants this: the
# online store is asked once, the model is called once, and the ranking comes back
# in a single frame.
import pandas as pd, mlflow
from feast import FeatureStore

NAME, ALIAS = "freshmart_reorder_model", "champion"
SERVED = ["units_7d", "units_28d", "trend_ratio", "days_since_last_sale",
          "quiet_days_28d", "avg_unit_price_7d", "lead_time_days", "promo_active"]
ACT_NOW, WATCH = 0.10, 0.25

mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")
mv = mlflow.MlflowClient().get_model_version_by_alias(NAME, ALIAS)
model = (mlflow.xgboost.load_model if mv.tags.get("model_type") == "xgboost"
         else mlflow.sklearn.load_model)(f"models:/{NAME}@{ALIAS}")

# every line carried by one store - the shape of a real morning run
pairs = (pd.read_parquet("feature_repo/data/reorder_features.parquet")
           .query("store_id == 'S001'")[["store_id", "product_id"]]
           .drop_duplicates().head(200))
rows = pairs.to_dict("records")

fs = FeatureStore(repo_path="feature_repo")
feats = fs.get_online_features(features=[f"reorder_features:{c}" for c in SERVED],
                               entity_rows=rows).to_df()
scored = feats.dropna(subset=SERVED).copy()
scored["reorder_probability"] = model.predict_proba(scored[SERVED])[:, 1]
scored["action"] = pd.cut(scored.reorder_probability, [-1, ACT_NOW, WATCH, 2],
                          labels=["no action", "add to watch list", "order now"])

top = scored.sort_values("reorder_probability", ascending=False).head(5)
print(f"scored {len(scored)} lines for store S001 in ONE lookup + ONE predict "
      f"(model v{mv.version})")
print(f"mean probability {scored.reorder_probability.mean():.3f} | "
      f"order now: {(scored.action == 'order now').sum()} | "
      f"watch: {(scored.action == 'add to watch list').sum()}")
print("\ntop 5 lines for the morning replenishment run")
print(top[["store_id", "product_id", "units_7d", "days_since_last_sale",
           "reorder_probability", "action"]].to_string(index=False))
