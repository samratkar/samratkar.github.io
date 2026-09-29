# Improved FreshMart generator: stores now differ in maturity and volume,
# so ranking-by-different-metrics is a real, demonstrable phenomenon.
from pathlib import Path
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
Path("freshmart_m2/data").mkdir(parents=True, exist_ok=True)

stores   = [f"S{i:03d}" for i in range(1, 21)]
products = [f"P{i:04d}" for i in range(1, 101)]
full = pd.date_range("2026-07-01", "2026-08-15", freq="D")

# store heterogeneity: some opened late (fewer active days), some low-traffic
open_offset = rng.integers(0, 30, len(stores))          # days after 1 Jul each store "opens"
traffic     = rng.uniform(0.4, 1.6, len(stores))        # relative volume multiplier
store_open  = dict(zip(stores, open_offset))
store_traf  = dict(zip(stores, traffic))

rows = []
tid = 0
for s in stores:
    active = full[store_open[s]:]                         # this store's active window
    n_s = int(1500 * store_traf[s])                       # its transaction count
    for _ in range(n_s):
        d = rng.choice(active)
        q = rng.poisson(3) + 1
        up = round(float(rng.uniform(20, 500)), 2)
        rows.append((f"T{tid:07d}", d, s, rng.choice(products), q, up, round(q*up,2)))
        tid += 1
df = pd.DataFrame(rows, columns=["transaction_id","transaction_date","store_id",
                                 "product_id","quantity","unit_price","revenue"])
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
# imperfections
df.loc[rng.choice(df.index, 250, replace=False), "quantity"] = np.nan
df = pd.concat([df, df.sample(150, random_state=7)], ignore_index=True)
df.to_csv("freshmart_m2/data/pos_transactions.csv", index=False)

pd.DataFrame({"store_id":stores,
    "city":rng.choice(["Pune","Mumbai","Bengaluru","Hyderabad"],len(stores)),
    "store_type":rng.choice(["Metro","Express","Neighbourhood"],len(stores))
}).to_csv("freshmart_m2/data/store_master.csv", index=False)
pd.DataFrame({"product_id":products,
    "category":rng.choice(["Dairy","Beverages","Snacks","Personal Care","Staples"],len(products)),
    "lead_time_days":rng.integers(1,8,len(products))
}).to_csv("freshmart_m2/data/product_master.csv", index=False)
print(df.shape, df.transaction_date.min().date(), df.transaction_date.max().date())
