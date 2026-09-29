# CHALLENGE 5 solution: a second cohort dimension (store_type) that aids replenishment.
import pandas as pd
df = pd.read_csv("freshmart_m2/data/pos_transactions.csv", parse_dates=["transaction_date"])
sm = pd.read_csv("freshmart_m2/data/store_master.csv")
d = df.dropna(subset=["quantity"]).merge(sm, on="store_id")
d["month"] = d.transaction_date.dt.to_period("M").astype(str)

# cohort by store_type (a replenishment-relevant segmentation, not just first-month)
pivot = d.pivot_table(index="store_type", columns="month", values="revenue",
                      aggfunc="sum").round(0)
print("revenue by store_type x month:")
print(pivot.to_string())
# why it helps replenishment: different formats have different velocity/lead-time needs
share = (d.groupby("store_type").revenue.sum()/d.revenue.sum()*100).round(1)
print("\nrevenue share by store_type (%):"); print(share.to_string())
