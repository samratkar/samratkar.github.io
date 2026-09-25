# CHALLENGE solution: prove that a naive key merge leaks the future, and that
# Feast's as-of join does not. Same store, same product, same label date, two
# join strategies.
import pandas as pd
from feast import FeatureStore

feats = pd.read_parquet("feature_repo/data/reorder_features.parquet")

# pick the busiest store-product line, and label an EARLY day on it
pair = (feats.groupby(["store_id", "product_id"]).size()
             .sort_values(ascending=False).index[0])
line = feats[(feats.store_id == pair[0]) & (feats.product_id == pair[1])] \
          .sort_values("event_timestamp")
label_row = line.iloc[1]
entity_df = pd.DataFrame([{"store_id": pair[0], "product_id": pair[1],
                           "event_timestamp": label_row.event_timestamp}])

# (1) naive merge on the entity key alone: pandas keeps every match, and a
#     careless .iloc[-1] (or a groupby().last()) takes the most recent row -
#     which is in the future relative to the label. This is leakage.
naive = entity_df.merge(
    line[["store_id", "product_id", "units_7d", "event_timestamp"]]
        .rename(columns={"event_timestamp": "feature_timestamp"}),
    on=["store_id", "product_id"], how="left").iloc[-1]

# (2) Feast: as-of join, feature timestamp <= label timestamp
fs = FeatureStore(repo_path="feature_repo")
pit = fs.get_historical_features(entity_df=entity_df,
                                 features=["reorder_features:units_7d"]).to_df()

leaked = naive.feature_timestamp > label_row.event_timestamp
print(f"line            : {pair[0]} / {pair[1]}  ({len(line)} days with sales)")
print(f"label date      : {label_row.event_timestamp.date()}")
print(f"naive merge  -> units_7d={naive.units_7d} from {naive.feature_timestamp.date()}"
      f"{'   *** FUTURE - LEAKED ***' if leaked else ''}")
print(f"point-in-time-> units_7d={pit['units_7d'].iloc[0]}  (as known on the label date)")
