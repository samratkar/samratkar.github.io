# CHALLENGE 2 solution: turn the profile into a triage table (valid / suspicious / invalid).
import pandas as pd
df = pd.read_csv("freshmart_m2/data/pos_transactions.csv", parse_dates=["transaction_date"])
checks = []
checks.append(("transaction_id near-unique", df.transaction_id.nunique()/len(df),
    ">0.99 expected", "VALID - IDs should be unique; the small gap is the injected dups"))
checks.append(("quantity missing rate", df.quantity.isna().mean(),
    "should be ~0", "SUSPICIOUS - investigate the feed, do not silently impute"))
checks.append(("unit_price range", (df.unit_price.min(), df.unit_price.max()),
    "20-500 by design", "VALID - within the business range"))
for name,val,expect,verdict in checks:
    print(f"{name:26} value={val}  expect={expect}\n   -> {verdict}")
