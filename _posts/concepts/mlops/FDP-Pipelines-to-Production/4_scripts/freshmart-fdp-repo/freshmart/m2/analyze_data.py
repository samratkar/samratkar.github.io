# Mission 1 - what actually arrived? Establish shape, grain, coverage, quality.
import pandas as pd

df = pd.read_csv("freshmart_m2/data/pos_transactions.csv", parse_dates=["transaction_date"])
print("shape:", df.shape)
print("date :", df["transaction_date"].min().date(), "to", df["transaction_date"].max().date())
print("stores:", df["store_id"].nunique(), "| products:", df["product_id"].nunique())
print("\nmissing values per column:")
print(df.isna().sum())
print("\nduplicate transaction_ids:", df["transaction_id"].duplicated().sum())
print("\nnumeric summary:")
print(df[["quantity", "unit_price", "revenue"]].describe().round(2))
