# CHALLENGE solution: a shadow-mode comparison harness.
#
# Shadow is the safest first exposure of a new model: it scores real inputs in
# parallel with the champion and its answers are logged and thrown away. No user
# sees them, so the blast radius is zero - and unlike an offline test set, the
# inputs are exactly what production sends, including the odd ones.
#
# The candidate here is real: Lab 2's XGBoost, registered as @candidate. It passed
# every gate except "best AUC", which is precisely the model a team would want to
# shadow before switching.
#
# What matters to a planner is not the probability but the ACTION. Two models can
# differ by 0.03 on average and still disagree about what to do on 8% of lines,
# and that disagreement rate is the number to take to the promotion meeting.
import numpy as np, pandas as pd, mlflow

NAME = "freshmart_reorder_model"
SERVED = ["units_7d", "units_28d", "trend_ratio", "days_since_last_sale",
          "quiet_days_28d", "avg_unit_price_7d", "lead_time_days", "promo_active"]
ACT_NOW, WATCH = 0.10, 0.25
DELTA_TOLERANCE, DISAGREE_TOLERANCE = 0.05, 0.10

mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")
c = mlflow.MlflowClient()


def load(alias):
    mv = c.get_model_version_by_alias(NAME, alias)
    loader = (mlflow.xgboost.load_model if mv.tags.get("model_type") == "xgboost"
              else mlflow.sklearn.load_model)
    return loader(f"models:/{NAME}@{alias}"), mv


def action(p):
    return np.where(p >= WATCH, "order now",
                    np.where(p >= ACT_NOW, "watch", "no action"))


champ, cmv = load("champion")
cand,  dmv = load("candidate")

# real serving inputs: the latest row per line, which is what the online store holds
off = pd.read_parquet("feature_repo/data/reorder_features.parquet")
X = (off.sort_values("event_timestamp").groupby(["store_id", "product_id"]).tail(1)
        .sample(500, random_state=0))

pc = champ.predict_proba(X[SERVED])[:, 1]
pd_ = cand.predict_proba(X[SERVED])[:, 1]
delta = np.abs(pc - pd_)
ac, ad = action(pc), action(pd_)
disagree = float((ac != ad).mean())

print(f"champion  v{cmv.version} ({cmv.tags.get('model_type')}, AUC {cmv.tags.get('auc')})")
print(f"candidate v{dmv.version} ({dmv.tags.get('model_type')}, AUC {dmv.tags.get('auc')})")
print(f"shadowed {len(X)} real serving inputs - candidate answers logged, never returned\n")
print(f"  mean |delta| = {delta.mean():.4f}   p95 |delta| = {np.quantile(delta, .95):.4f}"
      f"   max = {delta.max():.4f}")
print(f"  action disagreement = {disagree:.1%} of lines")
print(pd.crosstab(pd.Series(ac, name="champion"), pd.Series(ad, name="candidate")).to_string())

ok_delta = delta.mean() <= DELTA_TOLERANCE
ok_disagree = disagree <= DISAGREE_TOLERANCE
ok_auc = float(dmv.tags.get("auc", 0)) >= float(cmv.tags.get("auc", 1))
print(f"\npromotion rule")
print(f"  mean |delta| <= {DELTA_TOLERANCE}          {'PASS' if ok_delta else 'FAIL'}")
print(f"  action disagreement <= {DISAGREE_TOLERANCE:.0%}   {'PASS' if ok_disagree else 'FAIL'}")
print(f"  candidate AUC >= champion AUC   {'PASS' if ok_auc else 'FAIL'}")
print(f"  -> {'PROMOTE' if (ok_delta and ok_disagree and ok_auc) else 'DO NOT PROMOTE'}")
print("\nShadow served nothing to anyone. It only logged the comparison.")
