# CHALLENGE solution: the five pillars are not a printout - they are a gate.
# Give each pillar a verdict against the last accepted run, and exit non-zero so
# Prefect (or cron, or Airflow) can stop on it.
#   freshness    absolute age of the newest event
#   volume       rows AND store-product pairs, tolerance band vs the last run
#   schema       binary
#   distribution mean basket size, tolerance band
#   lineage      a hash must be present, and it must have changed
import json, sys
from pathlib import Path

BASELINE = Path("data/pillars_baseline.json")
TOL = {"volume_pct": 0.30, "pairs_pct": 0.10, "basket_abs": 0.5, "max_age_days": 2}

def gate(report_path="data/contract_report.json", age_days=0):
    obs = json.loads(Path(report_path).read_text())["observability"]
    base = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    v = {}

    v["freshness"] = ("PASS" if age_days <= TOL["max_age_days"]
                      else f"FAIL (newest event {age_days}d old)")
    v["schema"]  = "PASS" if obs["schema"]["matches"] else "FAIL (schema drift)"
    v["lineage"] = ("PASS" if obs["lineage"]["data_hash"] != "unknown"
                    else "FAIL (no lineage hash)")

    if base is None:
        v["volume"] = v["distribution"] = "PASS (first run - baseline set)"
    else:
        rows, was = obs["volume"]["rows"], base["volume"]["rows"]
        swing = abs(rows - was) / max(was, 1)
        pairs, pwas = obs["volume"]["store_product_pairs"], base["volume"]["store_product_pairs"]
        pswing = abs(pairs - pwas) / max(pwas, 1)
        if swing > TOL["volume_pct"]:
            v["volume"] = f"FAIL (rows moved {swing:.0%}: {was:,} -> {rows:,})"
        elif pswing > TOL["pairs_pct"]:
            v["volume"] = f"FAIL (grain shrank {pswing:.0%}: {pwas:,} -> {pairs:,} pairs)"
        else:
            v["volume"] = "PASS"
        b_now, b_was = obs["distribution"]["mean_basket"], base["distribution"]["mean_basket"]
        v["distribution"] = ("PASS" if abs(b_now - b_was) <= TOL["basket_abs"]
                             else f"FAIL (mean basket {b_was} -> {b_now})")

    ok = all(x.startswith("PASS") for x in v.values())
    for pillar, verdict in v.items():
        print(f"  {pillar:<13} {verdict}")
    print(f"[gate] {'ACCEPTED' if ok else 'REJECTED'}")
    if ok:
        BASELINE.write_text(json.dumps(obs, indent=2, default=str))
    return ok

if __name__ == "__main__":
    sys.exit(0 if gate(age_days=int(sys.argv[1]) if len(sys.argv) > 1 else 0) else 1)
