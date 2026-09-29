# Mission 2 - the automated second opinion. A broad profile as an accelerator.
import pandas as pd
from ydata_profiling import ProfileReport      # see the deprecation note below

df = pd.read_csv("freshmart_m2/data/pos_transactions.csv", parse_dates=["transaction_date"])
profile = ProfileReport(df, title="FreshMart POS Data Profile", minimal=True)
profile.to_file("freshmart_m2/profile.html")
print("profile written to freshmart_m2/profile.html - open it in a browser")
