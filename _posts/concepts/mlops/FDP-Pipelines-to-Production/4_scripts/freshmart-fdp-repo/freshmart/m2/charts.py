# Mission 4 - the visualisation decision. Choose the chart, then the library.
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import plotly.express as px
df = pd.read_csv("freshmart_m2/data/pos_transactions.csv", parse_dates=["transaction_date"])

# 1) a TIME question -> line
daily = (df.dropna(subset=["quantity"])
    .groupby("transaction_date", as_index=False)
    .agg(revenue=("revenue", "sum")))
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(daily["transaction_date"], daily["revenue"])
ax.set_title("FreshMart Revenue Trend"); ax.set_xlabel("Date"); ax.set_ylabel("Revenue")
plt.tight_layout(); plt.savefig("freshmart_m2/trend.png", dpi=110)

# 2) a RANKING question -> bar (reuse store_summary from Mission 3)
store_summary = (df.dropna(subset=["quantity"])
    .groupby("store_id", as_index=False).agg(revenue=("revenue", "sum")))
top = store_summary.sort_values("revenue", ascending=False).head(10)
plt.figure(figsize=(7, 4)); sns.barplot(data=top, y="store_id", x="revenue")
plt.title("Top Stores by Revenue"); plt.tight_layout(); plt.savefig("freshmart_m2/top_stores.png", dpi=110)

# 3) a RELATIONSHIP question -> interactive scatter (Plotly)
fig = px.scatter(df.sample(min(4000, len(df)), random_state=42),
                 x="unit_price", y="quantity", color="store_id", title="Price vs Quantity")
fig.write_html("freshmart_m2/price_qty.html")
print("charts written: trend.png, top_stores.png, price_qty.html")
