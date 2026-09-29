# CHALLENGE solution: AUC 0.70 with F1 0.05 is not a broken model, it is a badly
# chosen operating point. AUC scores the RANKING; F1 scores one THRESHOLD on it.
# At a 20% base rate almost nothing clears 0.5, so almost nothing is flagged.
#
# Two ways to choose better:
#   (a) maximise F1 - the statistician's answer, no business input required;
#   (b) minimise expected cost - the planner's answer. A missed reorder is a
#       stockout (lost margin, unhappy customer); a false alarm is one unnecessary
#       case pack sitting in the back room. Those are not equal, so 0.5 is not the
#       answer, and neither is 0.5 after balancing the classes.
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

SERVED = ["units_7d", "units_28d", "trend_ratio", "days_since_last_sale",
          "quiet_days_28d", "avg_unit_price_7d", "lead_time_days", "promo_active"]
SPLIT_DAY = pd.Timestamp("2026-08-01")
COST_MISS, COST_FALSE_ALARM = 40.0, 8.0        # rupees per line, agreed with Supply Chain

df = pd.read_parquet("data/train.parquet")
tr, te = df[df.event_timestamp < SPLIT_DAY], df[df.event_timestamp >= SPLIT_DAY]
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[SERVED], tr.reorder_7d)
p, y = m.predict_proba(te[SERVED])[:, 1], te.reorder_7d.to_numpy()

print(f"AUC = {roc_auc_score(y, p):.3f}  (the ranking never changes below)")
print(f"base rate = {y.mean():.3f}\n")
print(f"{'threshold':>10}{'flagged':>9}{'precision':>11}{'recall':>9}{'F1':>7}{'cost/1k lines':>15}")
rows = []
for t in [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]:
    pred = (p >= t).astype(int)
    misses = int(((pred == 0) & (y == 1)).sum())
    alarms = int(((pred == 1) & (y == 0)).sum())
    cost = (misses * COST_MISS + alarms * COST_FALSE_ALARM) / len(y) * 1000
    rows.append((t, f1_score(y, pred), cost))
    print(f"{t:>10.2f}{pred.sum():>9}{precision_score(y, pred, zero_division=0):>11.3f}"
          f"{recall_score(y, pred):>9.3f}{f1_score(y, pred):>7.3f}{cost:>15,.0f}")

best_f1   = max(rows, key=lambda r: r[1])
best_cost = min(rows, key=lambda r: r[2])
print(f"\nbest F1   at threshold {best_f1[0]:.2f}  (F1={best_f1[1]:.3f}, "
      f"{best_f1[1]/rows[0][1]:.0f}x the F1 at 0.5)")
print(f"best cost at threshold {best_cost[0]:.2f}  (Rs {best_cost[2]:,.0f} per 1,000 lines, "
      f"vs Rs {rows[0][2]:,.0f} at 0.5 - a {1-best_cost[2]/rows[0][2]:.0%} saving)")

# the cost-optimal threshold is a function of the cost ratio, not of the model
print("\ncost-optimal threshold as the miss:false-alarm ratio changes")
grid = [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
for miss in (15.0, 25.0, 40.0, 80.0):
    costs = []
    for t in grid:
        pred = (p >= t).astype(int)
        costs.append((t, (((pred == 0) & (y == 1)).sum() * miss +
                          ((pred == 1) & (y == 0)).sum() * COST_FALSE_ALARM) / len(y) * 1000))
    t_best, c_best = min(costs, key=lambda r: r[1])
    print(f"  miss=Rs{miss:>5.0f}  false alarm=Rs{COST_FALSE_ALARM:.0f}  "
          f"({miss/COST_FALSE_ALARM:>4.1f}:1)  ->  threshold {t_best:.2f}  "
          f"(Rs {c_best:,.0f}/1k lines)")
print("\nThe model never changed; only the price of being wrong did. The threshold")
print("is a business decision informed by the model, not a property of the model.")
