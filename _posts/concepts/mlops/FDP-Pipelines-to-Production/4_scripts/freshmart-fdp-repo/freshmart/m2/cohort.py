# Mission 5 - the store-cohort question. Group stores by their first active month.
import pandas as pd
df = pd.read_csv("freshmart_m2/data/pos_transactions.csv", parse_dates=["transaction_date"])
stores_df = pd.read_csv("freshmart_m2/data/store_master.csv")

x = df.dropna(subset=["quantity"]).merge(stores_df, on="store_id").copy()
x["month"] = x["transaction_date"].dt.to_period("M").astype(str)
first_month = x.groupby("store_id")["transaction_date"].min().dt.to_period("M").astype(str)
x["cohort"] = x["store_id"].map(first_month)

cohort = (x.groupby(["cohort", "month"], as_index=False)
            .agg(units=("quantity", "sum"), revenue=("revenue", "sum")))
pivot = cohort.pivot(index="cohort", columns="month", values="revenue")
print(pivot.round(0).to_string())
