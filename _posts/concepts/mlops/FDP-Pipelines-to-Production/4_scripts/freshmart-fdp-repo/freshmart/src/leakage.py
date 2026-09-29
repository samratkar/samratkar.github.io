"""Lab 2 / Missions 1-2 - define the target, then expose two kinds of leakage.

Leakage is any feature carrying information not available at prediction time. It
produces spectacular offline numbers and a model that collapses in production,
which is why it gets a whole mission.

Two demonstrations, because they fail differently and are caught differently:

  (1) THE LOUD ONE - a feature computed from the label. AUC goes to 1.000, which
      is the tell. Nobody plans this; it arrives when a well-meaning colleague
      joins in a column that already encodes the outcome.

  (2) THE QUIET ONE - a feature whose window looks forward. units_7d_centred is a
      seven-day window centred on the day, so it includes three days that have
      not happened yet. Nothing is computed from the label, the AUC lift looks
      like a good feature rather than a bug, and - the part worth stopping on -
      its correlation with the target is 0.39, far below any screening threshold
      you would set. This one is caught by asking WHEN each value is knowable,
      never by a correlation report.

Lab 1's engineer stage builds every window with .rolling(n) and no centring for
exactly this reason.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SERVED = ["units_7d", "units_28d", "trend_ratio", "days_since_last_sale",
          "quiet_days_28d", "avg_unit_price_7d", "lead_time_days", "promo_active"]
SPLIT_DAY = pd.Timestamp("2026-08-01")     # train on July, test on August


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


train = pd.read_parquet("data/train.parquet")
y = train["reorder_7d"]
print(f"rows={len(train):,}  base rate={y.mean():.3f}  "
      f"one row = one store x product x day")

# ---- (1) the loud one: a feature derived from the label ---------------------
rng = np.random.default_rng(0)
train["planner_flagged"] = y * 0.9 + rng.normal(0, 0.05, len(y))

auc_clean  = cross_val_score(lr(), train[SERVED], y, cv=3, scoring="roc_auc").mean()
auc_leaked = cross_val_score(lr(), train[SERVED + ["planner_flagged"]], y, cv=3,
                             scoring="roc_auc").mean()
print(f"\nAUC without leak         : {auc_clean:.3f}")
print(f"AUC with planner_flagged : {auc_leaked:.3f}   <- too good to be true = red flag")
print(f"  corr(planner_flagged, y) = {train.planner_flagged.corr(y):.3f}  "
      f"-> any correlation screen catches this")

# ---- (2) the quiet one: a window that looks forward -------------------------
# rebuild it from the feature table, as a careless colleague would
feats = (pd.read_parquet("feature_repo/data/reorder_features.parquet")
           .sort_values(["store_id", "product_id", "event_timestamp"]))
# same right-censoring rule the training set uses
feats = feats[feats.event_timestamp <= pd.Timestamp(feats.event_timestamp.max()) - pd.Timedelta(7, unit="D")]
feats["units_7d_centred"] = (feats.groupby(["store_id", "product_id"], group_keys=False)["units"]
                                  .transform(lambda s: s.rolling(7, center=True, min_periods=1).sum()))
past   = feats[feats.event_timestamp <  SPLIT_DAY]
future = feats[feats.event_timestamp >= SPLIT_DAY]

for cols, label in [(SERVED, "backward windows only "),
                    (SERVED + ["units_7d_centred"], "+ centred 7-day window")]:
    m = lr().fit(past[cols], past.reorder_7d)
    auc = roc_auc_score(future.reorder_7d, m.predict_proba(future[cols])[:, 1])
    print(f"{label}: AUC={auc:.3f}")
print(f"  corr(units_7d_centred, y) = {feats.units_7d_centred.corr(feats.reorder_7d):.3f}  "
      f"-> a 0.5 screen does NOT catch this")
print("\nLesson: the loud leak is caught by a number. The quiet one is caught only by")
print("        asking, feature by feature, what was knowable at prediction time.")
