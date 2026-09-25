"""Lab 1 / stage: ops - the replenishment feed that changes daily.

Store and product masters are reference data: they change when a store opens or a
product is listed. Lead times and promotions are operational: a supplier slips a
day, a category goes on promotion this week. M2 wrote all of it in one gen.py
because it only needed a snapshot. A pipeline separates them, because they change
at different rates - and a stage that re-runs when the wrong thing changes is how
a 90-second pipeline becomes a 40-minute one.

This is the mutable input in Mission 3's challenge: change it, and features must
be recomputed without re-ingesting or re-cleaning the immutable POS log.
"""
import argparse
from pathlib import Path
import hashlib
import numpy as np, pandas as pd

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default="2026-08-15", help="the day this feed was pulled")
    ap.add_argument("--masters", default="data/product_master.parquet")
    ap.add_argument("--out", default="data/product_ops.parquet")
    a = ap.parse_args()

    products = pd.read_parquet(a.masters)
    # seed from the as-of date, hashed stably: Python's hash() is salted per
    # process, which would make this stage irreproducible and defeat DVC
    seed = int(hashlib.sha256(a.asof.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    ops = products[["product_id"]].copy()
    ops["lead_time_days"] = rng.integers(1, 8, len(ops)).astype("int64")
    ops["promo_active"]   = (rng.random(len(ops)) < 0.15).astype("int64")
    ops["asof"] = a.asof

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    ops.to_parquet(a.out, index=False)
    print(f"[ops] {len(ops)} products  as of {a.asof}  "
          f"| mean lead time {ops.lead_time_days.mean():.1f}d "
          f"| {int(ops.promo_active.sum())} on promotion -> {a.out}")
