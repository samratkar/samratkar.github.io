"""Lab 1 / stage: generate - the overnight POS extract.

This is the same FreshMart feed you profiled in M2. The generator is M2's gen.py,
unchanged in its draw order, so seed 42 reproduces the exact extract M2 reported:
31,598 rows, 2026-07-03 to 2026-08-15, 250 injected missing quantities and 150
duplicated transactions. In M2 you ran it once by hand to have something to look
at. Here it stands in for the 06:10 batch, and the imperfections are no longer a
teaching device - they are the thing the contract has to stop.

Two properties earn their keep downstream: a fixed seed makes the extract
byte-identical run to run, which is what makes DVC's content hashing (and so
selective re-execution) demonstrable on a laptop; and the run writes
data/pos_raw.hash, the anchor for the lineage pillar.

--drift produces the shifted batch Lab 4 monitors: demand pulled toward Metro
stores and larger baskets. Same schema, different distribution - exactly the
failure a schema check cannot catch.
"""
import argparse, hashlib
from pathlib import Path
import numpy as np, pandas as pd

STORES   = [f"S{i:03d}" for i in range(1, 21)]
PRODUCTS = [f"P{i:04d}" for i in range(1, 101)]
CALENDAR = pd.date_range("2026-07-01", "2026-08-15", freq="D")


def build_pos(seed: int, scale: float, drift: bool):
    """M2's generator, with the same draw order so the fixture is reproducible."""
    rng = np.random.default_rng(seed)
    open_offset = rng.integers(0, 30, len(STORES))     # days after 1 Jul each store opens
    traffic     = rng.uniform(0.4, 1.6, len(STORES))   # relative volume multiplier
    store_open  = dict(zip(STORES, open_offset))
    store_traf  = dict(zip(STORES, traffic))

    rows, tid = [], 0
    for s in STORES:
        active = CALENDAR[store_open[s]:]                       # this store's active window
        n_s = int(1500 * store_traf[s] * scale)                 # its transaction count
        for _ in range(n_s):
            d  = rng.choice(active)
            q  = rng.poisson(4 if drift else 3) + 1             # drift: bigger baskets
            up = round(float(rng.uniform(20, 500)), 2)
            rows.append((f"T{tid:07d}", d, s, rng.choice(PRODUCTS), q, up, round(q * up, 2)))
            tid += 1

    df = pd.DataFrame(rows, columns=["transaction_id", "transaction_date", "store_id",
                                     "product_id", "quantity", "unit_price", "revenue"])
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # the imperfections M2 catalogued, reproduced at batch scale
    n_missing = int(250 * scale)
    df.loc[rng.choice(df.index, n_missing, replace=False), "quantity"] = np.nan
    df = pd.concat([df, df.sample(int(150 * scale), random_state=7)], ignore_index=True)
    return df, store_open


def build_masters(seed: int):
    """Reference data: stable properties of stores and products."""
    rng = np.random.default_rng(seed + 1)
    stores = pd.DataFrame({
        "store_id": STORES,
        "city": rng.choice(["Pune", "Mumbai", "Bengaluru", "Hyderabad"], len(STORES)),
        "store_type": rng.choice(["Metro", "Express", "Neighbourhood"], len(STORES))})
    products = pd.DataFrame({
        "product_id": PRODUCTS,
        "category": rng.choice(["Dairy", "Beverages", "Snacks", "Personal Care", "Staples"],
                               len(PRODUCTS))})
    return stores, products


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiply batch volume")
    ap.add_argument("--rows", type=int, default=0, help="target row count (overrides --scale)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--out", default="data/pos_raw.parquet")
    a = ap.parse_args()

    scale = a.rows / 31_598 if a.rows else a.scale
    pos, _ = build_pos(a.seed, scale, a.drift)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    pos.to_parquet(out, index=False)

    stores, products = build_masters(a.seed)
    stores.to_parquet(out.parent / "store_master.parquet", index=False)
    products.to_parquet(out.parent / "product_master.parquet", index=False)

    h = hashlib.sha256(out.read_bytes()).hexdigest()[:12]
    out.with_suffix(".hash").write_text(h)          # data/pos_raw.parquet -> data/pos_raw.hash
    tag = " [DRIFTED]" if a.drift else ""
    print(f"[generate] {len(pos):,} rows -> {out}  "
          f"{pos.transaction_date.min().date()} to {pos.transaction_date.max().date()}  "
          f"data_hash={h}{tag}")
    print(f"[generate] masters: {len(stores)} stores, {len(products)} products  "
          f"| injected: {int(pos.quantity.isna().sum())} missing quantity, "
          f"{int(pos.transaction_id.duplicated().sum())} duplicate ids")
