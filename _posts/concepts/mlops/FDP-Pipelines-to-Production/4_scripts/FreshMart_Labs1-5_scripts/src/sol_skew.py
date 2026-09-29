# CHALLENGE solution: prove that training-serving skew is impossible for an entity.
#
# The claim "one definition, so no skew" is an architectural argument. This is the
# evidence: take a store-product line, read what TRAINING saw for it, read what the
# ONLINE store serves for it, and compare feature by feature.
#
# Mind the dtypes. Feast stores Float32 in the online store, so a value that was
# 283.59 offline comes back as 283.589996. That is representation, not skew -
# compare with a tolerance rather than ==, and say so out loud, because a student
# who compares exactly will report a false positive and lose faith in the whole
# argument.
import pandas as pd
from feast import FeatureStore

SERVED = ["units_7d", "units_28d", "trend_ratio", "days_since_last_sale",
          "quiet_days_28d", "avg_unit_price_7d", "lead_time_days", "promo_active"]
TOL = 1e-2

off = pd.read_parquet("feature_repo/data/reorder_features.parquet")
# the online store holds the LAST row per entity, so compare against that row
row = (off.sort_values("event_timestamp")
          .groupby(["store_id", "product_id"]).tail(1).iloc[0])

fs = FeatureStore(repo_path="feature_repo")
online = fs.get_online_features(
    features=[f"reorder_features:{c}" for c in SERVED],
    entity_rows=[{"store_id": row.store_id, "product_id": row.product_id}]).to_dict()

print(f"entity: {row.store_id} / {row.product_id}   "
      f"(features as of {row.event_timestamp.date()})")
print(f"{'feature':<22}{'training':>14}{'serving':>16}   match")
mismatches = []
for c in SERVED:
    t, s = float(row[c]), float(online[c][0])
    ok = abs(t - s) <= TOL
    if not ok:
        mismatches.append(c)
    print(f"{c:<22}{t:>14.3f}{s:>16.6f}   {'yes' if ok else 'NO'}")

print(f"\nmismatches beyond {TOL}: {len(mismatches)}")
print("skew possible? " + ("YES - investigate" if mismatches else
      "NO - both sides read one Feast definition, so skew is structurally impossible"))
