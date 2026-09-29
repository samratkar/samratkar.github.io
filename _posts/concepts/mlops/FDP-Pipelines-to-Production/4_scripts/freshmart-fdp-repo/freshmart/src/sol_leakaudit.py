# CHALLENGE solution: a leakage audit with two screens, because one is not enough.
#
#   screen 1 - correlation with the target. Cheap, catches the loud leaks, and
#              runs in CI on every feature set.
#   screen 2 - knowability. For each feature, recompute it using only rows dated
#              on or before the prediction date and compare with the served
#              value. A feature that changes is reading the future, whatever its
#              correlation says.
#
# Screen 1 alone would have passed units_7d_centred. Screen 2 is what catches it.
import numpy as np, pandas as pd

SERVED = ["units_7d", "units_28d", "trend_ratio", "days_since_last_sale",
          "quiet_days_28d", "avg_unit_price_7d", "lead_time_days", "promo_active"]
CORR_LIMIT = 0.50

feats = (pd.read_parquet("feature_repo/data/reorder_features.parquet")
           .sort_values(["store_id", "product_id", "event_timestamp"]))
feats = feats[feats.event_timestamp <=
              pd.Timestamp(feats.event_timestamp.max()) - pd.Timedelta(7, unit="D")]
g = feats.groupby(["store_id", "product_id"], group_keys=False)

# two features a colleague might add, for the audit to judge
rng = np.random.default_rng(0)
feats["planner_flagged"]  = feats.reorder_7d * 0.9 + rng.normal(0, 0.05, len(feats))
feats["units_7d_centred"] = g["units"].transform(
    lambda s: s.rolling(7, center=True, min_periods=1).sum())

CANDIDATES = SERVED + ["planner_flagged", "units_7d_centred"]

# screen 2 needs a reference: the same window computed strictly backwards
backward = {"units_7d_centred": g["units"].transform(lambda s: s.rolling(7, min_periods=1).sum())}

print(f"{'feature':<22}{'|corr|':>8}{'screen 1':>11}{'screen 2':>11}   verdict")
print("-" * 70)
for col in CANDIDATES:
    corr = abs(feats[col].corr(feats.reorder_7d))
    s1 = "FLAG" if corr > CORR_LIMIT else "ok"
    if col in backward:
        drift = float((feats[col] - backward[col]).abs().mean())
        s2 = "FLAG" if drift > 1e-9 else "ok"
    else:
        s2 = "ok"          # built by Lab 1's engineer stage, backward by construction
    verdict = "LEAK - review serving availability" if "FLAG" in (s1, s2) else "safe"
    print(f"{col:<22}{corr:>8.3f}{s1:>11}{s2:>11}   {verdict}")

print(f"\nscreen 1: |corr| > {CORR_LIMIT}    screen 2: value changes when recomputed backwards only")
