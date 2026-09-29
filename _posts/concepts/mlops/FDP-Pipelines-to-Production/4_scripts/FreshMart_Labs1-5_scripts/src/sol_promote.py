# CHALLENGE solution: a promotion gate that compares the candidate against what is
# actually in production - on accuracy AND on behaviour.
#
# Two models with the same AUC are not interchangeable. They can rank equally well
# and still disagree about which lines to act on, and the planner experiences the
# disagreement, not the AUC. So the gate checks three things:
#
#   1. accuracy on a HELD-OUT recent batch, not on the numbers each model recorded
#      at its own training time - those were measured on different data;
#   2. action agreement, because a large behavioural change needs a human decision
#      even when the metrics improve;
#   3. lineage, because a model without a data_hash cannot be reproduced.
import numpy as np, pandas as pd, mlflow
from sklearn.metrics import roc_auc_score

NAME = "freshmart_reorder_model"
SERVED = ["units_7d", "units_28d", "trend_ratio", "days_since_last_sale",
          "quiet_days_28d", "avg_unit_price_7d", "lead_time_days", "promo_active"]
ACT_NOW, AUC_TOLERANCE, MIN_AGREEMENT = 0.10, 0.005, 0.90

mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")
c = mlflow.MlflowClient()


def load(a):
    mv = c.get_model_version_by_alias(NAME, a)
    loader = (mlflow.xgboost.load_model if mv.tags.get("model_type") == "xgboost"
              else mlflow.sklearn.load_model)
    return loader(f"models:/{NAME}@{a}"), mv


# evaluate BOTH on the same recent labelled batch - the drifted one Lab 4 flagged
df = pd.read_parquet("data/current_batch.parquet")
df = df[df.event_timestamp <= pd.Timestamp(df.event_timestamp.max()) - pd.Timedelta(7, unit="D")]
X, y = df[SERVED], df.reorder_7d

prod, prod_mv = load("production")
cand, cand_mv = load("candidate")
p_prod = prod.predict_proba(X)[:, 1]
p_cand = cand.predict_proba(X)[:, 1]

auc_prod, auc_cand = roc_auc_score(y, p_prod), roc_auc_score(y, p_cand)
agreement = float(np.mean((p_prod >= ACT_NOW) == (p_cand >= ACT_NOW)))

print(f"evaluation batch: {len(df):,} labelled rows from the drifted period\n")
print(f"{'':<14}{'version':>9}{'type':>10}{'AUC at train':>14}{'AUC on batch':>14}")
print(f"{'production':<14}v{prod_mv.version:<8}{prod_mv.tags.get('model_type',''):>10}"
      f"{prod_mv.tags.get('auc','-'):>14}{auc_prod:>14.4f}")
print(f"{'candidate':<14}v{cand_mv.version:<8}{cand_mv.tags.get('model_type',''):>10}"
      f"{cand_mv.tags.get('auc','-'):>14}{auc_cand:>14.4f}")
print(f"\naction agreement at threshold {ACT_NOW}: {agreement:.1%}")

checks = {
    f"accuracy   candidate AUC >= production - {AUC_TOLERANCE}": auc_cand >= auc_prod - AUC_TOLERANCE,
    f"behaviour  action agreement >= {MIN_AGREEMENT:.0%}":       agreement >= MIN_AGREEMENT,
    "lineage    candidate has a data_hash":                      bool(cand_mv.tags.get("data_hash")),
}
for label, ok in checks.items():
    print(f"  {label:<48} {'PASS' if ok else 'FAIL'}")
print(f"\ndecision: {'PROMOTE candidate to @production' if all(checks.values()) else 'HOLD - do not promote'}")
