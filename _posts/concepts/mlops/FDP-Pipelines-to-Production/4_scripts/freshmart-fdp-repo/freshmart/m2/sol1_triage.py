# CHALLENGE 1 solution: classify each quality finding as ENGINEERING (contract) or LOCAL (analysis fix).
import pandas as pd
df = pd.read_csv("freshmart_m2/data/pos_transactions.csv", parse_dates=["transaction_date"])

findings = []
dup = df["transaction_id"].duplicated().sum()
findings.append(("duplicate transaction_ids", dup,
    "ENGINEERING", "uniqueness is a contract guarantee the pipeline must enforce upstream"))
missing_q = df["quantity"].isna().sum()
findings.append(("missing quantity", missing_q,
    "ENGINEERING", "completeness of a core measure belongs in the data contract, not each notebook"))
neg_rev = (df["revenue"] < 0).sum()
findings.append(("negative revenue", neg_rev,
    "LOCAL/none", "none found here; would be a contract range-check if present"))

print(f"{'finding':28} {'count':>7}  {'route':12} why")
for name,cnt,route,why in findings:
    print(f"{name:28} {cnt:>7}  {route:12} {why}")
