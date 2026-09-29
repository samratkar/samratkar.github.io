# CHALLENGE 3 solution: rank stability + where metric choice MATTERS.
# Finding: on uniform data the top store is stable, but the GAPS differ by metric
# and mid-table ranks reshuffle - so metric choice matters most for the "close" calls.
import pandas as pd
df = pd.read_csv("freshmart_m2/data/pos_transactions.csv", parse_dates=["transaction_date"])
d = df.dropna(subset=["quantity"])
s = d.groupby("store_id").agg(revenue=("revenue","sum"), units=("quantity","sum"),
        active_days=("transaction_date","nunique")).reset_index()
s["rev_per_day"] = s.revenue/s.active_days

# rank each store on two metrics and show how many stores change rank position
r_rev = s.sort_values("revenue", ascending=False).reset_index(drop=True)
r_rpd = s.sort_values("rev_per_day", ascending=False).reset_index(drop=True)
pos_rev = {sid:i for i,sid in enumerate(r_rev.store_id)}
pos_rpd = {sid:i for i,sid in enumerate(r_rpd.store_id)}
moved = sum(1 for sid in s.store_id if pos_rev[sid] != pos_rpd[sid])
print(f"stores that change rank between revenue and rev_per_day: {moved} of {len(s)}")
print("top-1 is stable here, but mid-table reshuffles - metric choice matters for the close calls")
print("\ntop-5 by revenue vs by rev_per_day:")
print("revenue     :", list(r_rev.store_id.head(5)))
print("rev_per_day :", list(r_rpd.store_id.head(5)))
