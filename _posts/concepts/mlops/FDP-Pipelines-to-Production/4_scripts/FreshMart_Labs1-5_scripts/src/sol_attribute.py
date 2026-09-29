# CHALLENGE solution: attribute the decision to features, and expose where KS
# misleads.
#
# A monitor that says "drift detected" sends an engineer looking through eight
# features by hand. A monitor that says "units_7d, PSI 0.81" sends them to the
# right one. Rank every feature by PSI, print its KS p-value beside it, and flag
# the rows where the two disagree - because those rows are where a team that
# trusts KS p-values alone will retrain for no reason.
import numpy as np, pandas as pd
from scipy.stats import ks_2samp

SERVED = ["units_7d", "units_28d", "trend_ratio", "days_since_last_sale",
          "quiet_days_28d", "avg_unit_price_7d", "lead_time_days", "promo_active"]
PSI_THRESHOLD, KS_ALPHA = 0.20, 0.05


def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    r = np.clip(np.histogram(ref, edges)[0] / len(ref), 1e-4, None)
    c = np.clip(np.histogram(cur, edges)[0] / len(cur), 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


ref = pd.read_parquet("data/train.parquet")
cur = pd.read_parquet("data/current_batch.parquet")
cur = cur[cur.event_timestamp <= pd.Timestamp(cur.event_timestamp.max()) - pd.Timedelta(7, unit="D")]

rows = [(f, psi(ref[f].values, cur[f].values), float(ks_2samp(ref[f], cur[f]).pvalue))
        for f in SERVED]
rows.sort(key=lambda r: -r[1])

print(f"n_ref={len(ref):,}  n_cur={len(cur):,}")
print(f"{'feature':<22}{'PSI':>9}{'KS p':>12}   verdict")
print("-" * 72)
for f, ps, kp in rows:
    psi_flag, ks_flag = ps > PSI_THRESHOLD, kp < KS_ALPHA
    if psi_flag and ks_flag:
        verdict = "DRIFTED - both agree"
    elif not psi_flag and not ks_flag:
        verdict = "stable - both agree"
    elif ks_flag:
        verdict = "KS significant, PSI negligible - sample-size artefact"
    else:
        verdict = "PSI large, KS not significant - check the tails"
    print(f"{f:<22}{ps:>9.4f}{kp:>12.2e}   {verdict}")

drivers = [f for f, ps, _ in rows if ps > PSI_THRESHOLD]
print(f"\nattribution: {drivers} drive the decision.")
print(f"{sum(1 for _, ps, kp in rows if kp < KS_ALPHA and ps <= PSI_THRESHOLD)} feature(s) are "
      f"'statistically significant' with a PSI under {PSI_THRESHOLD} - on {len(cur):,} rows KS")
print("detects shifts far too small to act on. Decide on PSI magnitude; read KS as corroboration.")
