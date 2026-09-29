"""Lab 2 / step 0 - build the training set THROUGH Feast, not around it.

Lab 1 ended with one FeatureView served two ways. The temptation now is to read
feature_repo/data/reorder_features.parquet directly - it is right there, and it
is faster. Do that and you have quietly created two definitions of a feature: the
one training reads, and the one serving reads. They agree today and drift apart
the first time someone edits the parquet or the view.

So training pulls from get_historical_features, exactly as Lab 3's serving will
pull from get_online_features. The entity dataframe carries the label and its
timestamp; Feast performs the as-of join. This is the point-in-time correctness
Lab 1's Mission 4 proved, now used for the purpose it exists for.
"""
import argparse
from pathlib import Path
import pandas as pd
from feast import FeatureStore

LABEL_HORIZON = 7        # reorder_7d looks seven days ahead

SERVED = ["units_7d", "units_28d", "trend_ratio", "days_since_last_sale",
          "quiet_days_28d", "avg_unit_price_7d", "lead_time_days", "promo_active"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="feature_repo")
    ap.add_argument("--out", default="data/train.parquet")
    a = ap.parse_args()

    # the entity dataframe: one row per labelled event, with the time the label
    # refers to. Everything else comes from the store.
    labels = pd.read_parquet(f"{a.repo}/data/reorder_features.parquet")[
        ["store_id", "product_id", "event_timestamp", "reorder_7d"]]

    fs = FeatureStore(repo_path=a.repo)
    train = fs.get_historical_features(
        entity_df=labels,
        features=[f"reorder_features:{f}" for f in SERVED]).to_df()

    # Feast hands back UTC-aware timestamps; the rest of the lab compares them
    # against plain dates, so normalise once here rather than in five places.
    train["event_timestamp"] = train["event_timestamp"].dt.tz_convert(None)
    train = train.sort_values("event_timestamp").reset_index(drop=True)

    # Right-censoring: the last seven days of the panel cannot have a complete
    # seven-day future window, so their reorder_7d is a truncated 0 rather than a
    # real negative. Lab 1 keeps those rows because SERVING needs the most recent
    # day; TRAINING must drop them, or the model learns that mid-August is quiet.
    cutoff = pd.Timestamp(train.event_timestamp.max()) - pd.Timedelta(LABEL_HORIZON, unit="D")
    censored = int((train.event_timestamp > cutoff).sum())
    train = train[train.event_timestamp <= cutoff]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(a.out, index=False)
    print(f"[training-set] {len(train):,} rows x {len(SERVED)} served features "
          f"-> {a.out}")
    print(f"[training-set] dropped {censored:,} right-censored rows after "
          f"{cutoff.date()} (incomplete {LABEL_HORIZON}-day label window)")
    print(f"[training-set] label reorder_7d base rate {train.reorder_7d.mean():.3f} | "
          f"{train.event_timestamp.min().date()} to {train.event_timestamp.max().date()}")
