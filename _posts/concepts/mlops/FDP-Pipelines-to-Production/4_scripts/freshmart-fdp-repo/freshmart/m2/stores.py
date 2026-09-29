# Mission 3 - which stores drive the growth? Compare on volume AND behaviour.
import pandas as pd
df = pd.read_csv("freshmart_m2/data/pos_transactions.csv", parse_dates=["transaction_date"])

store_day = (df.dropna(subset=["quantity"])
    .groupby(["store_id", "transaction_date"], as_index=False)
    .agg(units=("quantity", "sum"), revenue=("revenue", "sum"),
         transactions=("transaction_id", "nunique")))

store_summary = (store_day.groupby("store_id", as_index=False)
    .agg(units=("units", "sum"), revenue=("revenue", "sum"),
         active_days=("transaction_date", "nunique")))
store_summary["revenue_per_day"] = store_summary["revenue"] / store_summary["active_days"]

print(store_summary.sort_values("revenue", ascending=False).head(10).to_string(index=False))
