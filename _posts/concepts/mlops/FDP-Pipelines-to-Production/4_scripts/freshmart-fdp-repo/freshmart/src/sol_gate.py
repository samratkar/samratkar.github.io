# CHALLENGE solution: a release gate is a conjunction, not a leaderboard.
# Quality AND reproducibility AND latency AND a beaten baseline. A candidate that
# wins on AUC and fails on lineage is not a better model - it is an unauditable
# one, and it does not ship.
import json
from pathlib import Path

AUC_FLOOR, LATENCY_CEILING_MS, BASELINE_AUC = 0.65, 5.0, 0.501

def gate(c):
    checks = {
        "quality      AUC >= %.2f" % AUC_FLOOR:            c["auc"] >= AUC_FLOOR,
        "improvement  > baseline + 0.05":                  c["auc"] > BASELINE_AUC + 0.05,
        "latency      < %.1f ms/request" % LATENCY_CEILING_MS: c["latency_ms_single"] < LATENCY_CEILING_MS,
        "lineage      data_hash present":                  bool(c.get("data_hash")),
    }
    return checks, all(checks.values())

# three candidates: the one we trained, plus two hypotheticals worth arguing about
results = json.loads(Path("artifacts/results.json").read_text())
candidates = {
    "logreg (this run)":   {**results["logreg"],  "data_hash": "e0ce921bf845"},
    "xgboost (this run)":  {**results["xgboost"], "data_hash": "e0ce921bf845"},
    "notebook_model_v3":   {"auc": 0.82, "latency_ms_single": 0.9, "data_hash": None},
}

for name, c in candidates.items():
    checks, ok = gate(c)
    print(f"{name}   AUC={c['auc']:.3f}")
    for label, passed in checks.items():
        print(f"    {label:<32} {'PASS' if passed else 'FAIL'}")
    print(f"    -> {'REGISTER' if ok else 'BLOCKED'}\n")

print("notebook_model_v3 has the best AUC in the room and is blocked. Nobody can")
print("say which extract it was trained on, so nobody can reproduce it, defend it")
print("to an auditor, or tell Lab 4 what to compare against when it drifts. That")
print("is not a technicality - it is the difference between a model and a file.")
