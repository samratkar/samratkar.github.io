"""Lab 1 / stage: engineer - clean POS -> the ML-ready feature table.

M2's handoff named the grain (Store x Product x Day) and the candidate signals:
recent sales, trend, inventory pressure, lead time, promotion and category
context, and history of going quiet. This stage builds exactly those, and the
label the Build beat will train on: reorder_7d - did this store move this product
again within the next seven days?

Everything here is computed ONCE and served two ways (offline for training,
online for serving) from a single Feast definition. Anything a modeller
recomputes in a notebook instead is training-serving skew waiting to happen.

Note what is deliberately absent: no aggregate built from reorder_7d, and every
rolling window looks backwards only. Target encoding in a feature pipeline is the
leakage trap Lab 2 opens with, and a feature stage is exactly where it enters.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

SPIKE = 1.5      # an off-cycle order is needed when next-week demand exceeds
                 # 1.5x the trailing week - a business parameter, set with Supply
                 # Chain, not a fact discovered in the data

FEATURE_COLS = ["units_7d", "units_28d", "trend_ratio", "days_since_last_sale",
                "avg_unit_price_7d", "lead_time_days", "promo_active", "quiet_days_28d"]


def _forward_7d_units(daily: pd.DataFrame) -> np.ndarray:
    """Units sold in the seven days AFTER each row, per store-product line."""
    out = []
    for _, g in daily.groupby(["store_id", "product_id"], sort=False):
        days = g.day.values.astype("datetime64[D]").astype(int)
        cum  = np.concatenate([[0], np.cumsum(g.units.values)])
        end  = np.searchsorted(days, days + 7, side="right")     # inclusive of day+7
        out.append(cum[end] - cum[np.arange(len(days)) + 1])      # exclude today
    return np.concatenate(out)


def engineer(pos: pd.DataFrame, stores: pd.DataFrame,
             products: pd.DataFrame, ops: pd.DataFrame) -> pd.DataFrame:
    # 1. collapse transactions to the grain M2 specified
    daily = (pos.assign(day=pos.transaction_date.dt.normalize())
                .groupby(["store_id", "product_id", "day"], as_index=False)
                .agg(units=("quantity", "sum"), revenue=("revenue", "sum"),
                     txns=("transaction_id", "count"),
                     avg_unit_price=("unit_price", "mean"))
                .sort_values(["store_id", "product_id", "day"]))
    daily["avg_unit_price"] = daily.avg_unit_price.round(2)

    # 2. backward-looking windows, per store-product line
    g = daily.groupby(["store_id", "product_id"], group_keys=False)
    daily["units_7d"]  = g["units"].transform(lambda s: s.rolling(7,  min_periods=1).sum()).astype("int64")
    daily["units_28d"] = g["units"].transform(lambda s: s.rolling(28, min_periods=1).sum()).astype("int64")
    daily["trend_ratio"] = (daily.units_7d / (daily.units_28d / 4).clip(lower=0.1)).round(3)
    daily["avg_unit_price_7d"] = g["avg_unit_price"].transform(
        lambda s: s.rolling(7, min_periods=1).mean()).round(2)
    daily["days_since_last_sale"] = (g["day"].transform(lambda s: s.diff().dt.days)
                                     .fillna(0).astype("int64"))
    # how often this line went quiet recently - M2's "historical stockout" proxy
    daily["quiet_days_28d"] = g["days_since_last_sale"].transform(
        lambda s: s.rolling(28, min_periods=1).sum()).astype("int64")

    # 3. the label: demand over the NEXT seven days, compared with the trailing
    #    week. FreshMart replenishes on a weekly cycle, so the question worth
    #    predicting is which lines will spike hard enough to need an off-cycle
    #    order. This is the only forward-looking quantity in the file, and it is
    #    the label - never a feature.
    daily["future_units_7d"] = _forward_7d_units(daily)
    daily["reorder_7d"] = (daily.future_units_7d >
                           SPIKE * daily.units_7d.clip(lower=1)).astype("int64")
    daily = daily.drop(columns="future_units_7d")

    # 4. context that lives in reference and operational feeds, not in the POS log
    out = (daily.merge(stores[["store_id", "store_type", "city"]], on="store_id", how="left")
                .merge(products[["product_id", "category"]], on="product_id", how="left")
                .merge(ops[["product_id", "lead_time_days", "promo_active"]],
                       on="product_id", how="left"))

    out = out.rename(columns={"day": "event_timestamp"})
    out["created_timestamp"] = pd.Timestamp.now("UTC").tz_convert(None)
    return out.reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/pos_clean.parquet")
    ap.add_argument("--stores", default="data/store_master.parquet")
    ap.add_argument("--products", default="data/product_master.parquet")
    ap.add_argument("--ops", default="data/product_ops.parquet")
    ap.add_argument("--out", default="feature_repo/data/reorder_features.parquet")
    a = ap.parse_args()

    feats = engineer(pd.read_parquet(a.inp), pd.read_parquet(a.stores),
                     pd.read_parquet(a.products), pd.read_parquet(a.ops))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f"[features] {len(feats):,} rows, {feats.shape[1]} cols "
          f"at Store x Product x Day -> {out}")
    print(f"[features] reorder_7d base rate {feats.reorder_7d.mean():.3f} | "
          f"engineered: {', '.join(FEATURE_COLS)}")
