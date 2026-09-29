# CHALLENGE 4 solution: the store-ranking bar chart in Altair vs seaborn.
# Altair states the ENCODING declaratively (x/y/sort) - closer to "grammar of graphics".
import pandas as pd, altair as alt
df = pd.read_csv("freshmart_m2/data/pos_transactions.csv")
top = (df.dropna(subset=["quantity"]).groupby("store_id", as_index=False)
         .agg(revenue=("revenue","sum")).sort_values("revenue", ascending=False).head(10))

chart = (alt.Chart(top).mark_bar().encode(
            x=alt.X("revenue:Q", title="Revenue"),
            y=alt.Y("store_id:N", sort="-x", title="Store"))
         .properties(title="Top Stores by Revenue (Altair)"))
chart.save("freshmart_m2/top_stores_altair.html")
print("Altair chart -> freshmart_m2/top_stores_altair.html")
print("Note: encoding is declared (x=revenue, y=store sorted by -x); the library figures out the rest.")
