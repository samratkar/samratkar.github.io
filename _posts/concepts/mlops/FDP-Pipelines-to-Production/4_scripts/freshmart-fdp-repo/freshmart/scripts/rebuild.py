"""Rebuild the generated state for whichever checkpoint is checked out.

Code lives in Git; data, the feature store and the model registry do not.
After `git switch -c <branch> <tag>`, run this once from the repository root:

    python scripts/rebuild.py          # asks before deleting generated files
    python scripts/rebuild.py --yes    # no prompt

It deletes ONLY generated files (data/, artifacts/, mlruns/, the Feast
databases, m2/freshmart_m2/, dvc.lock) and re-runs, in lab order, every step whose code
exists at this checkpoint. Your source files are never touched.
"""
import argparse, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ["data", "artifacts", "mlruns", "feature_repo/data",
             "feature_repo/registry.db", "feature_repo/online_store.db",
             "m2/freshmart_m2", "dvc.lock"]
PY = sys.executable

def have(p):
    return (ROOT / p).exists()

def run(label, cmd, cwd=ROOT):
    print(f"\n==> {label}\n    $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"\n[rebuild] FAILED at: {label}. Fix the error above, then run "
                 f"this script again.")

def steps():
    s = []
    if have("m2/gen.py"):
        s.append(("M2   generate the analysis extract", [PY, "gen.py"], ROOT / "m2"))
    if have("dvc.yaml"):
        s.append(("Lab 1 data pipeline + feature store (dvc repro)", ["dvc", "repro"], ROOT))
    if have("src/make_broken_batches.py"):
        s.append(("Lab 1 broken fixtures for the challenges", [PY, "src/make_broken_batches.py"], ROOT))
    if have("src/train.py"):
        s.append(("Lab 2 training set through Feast", [PY, "src/build_training_set.py"], ROOT))
        s.append(("Lab 2 train, gate, register @champion", [PY, "src/train.py"], ROOT))
    if have("src/register_candidate.py"):
        s.append(("Lab 3 register the XGBoost run as @candidate", [PY, "src/register_candidate.py"], ROOT))
    if have("src/monitor.py"):
        s.append(("Lab 4 festival batch: generate", [PY, "src/generate_data.py", "--drift", "--seed", "7",
                  "--out", "data/current_raw.parquet"], ROOT))
        s.append(("Lab 4 festival batch: clean", [PY, "src/clean_pos.py", "--in", "data/current_raw.parquet",
                  "--out", "data/current_clean.parquet", "--report", "data/current_clean_report.json"], ROOT))
        s.append(("Lab 4 festival batch: features", [PY, "src/engineer_features.py", "--in",
                  "data/current_clean.parquet", "--out", "data/current_batch.parquet"], ROOT))
        s.append(("Lab 4 monitor and trigger decision", [PY, "src/monitor.py"], ROOT))
    if have("pipelines/ci_cd_local.py"):
        s.append(("Lab 5 local CI/CD pipeline (@production)", [PY, "pipelines/ci_cd_local.py"], ROOT))
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="do not ask before deleting generated files")
    a = ap.parse_args()
    if not (ROOT / ".git").exists():
        sys.exit("[rebuild] run this inside the cloned repository")
    plan = steps()
    print("[rebuild] generated files to delete:", ", ".join(p for p in GENERATED if have(p)) or "none")
    print("[rebuild] steps to run:")
    for label, *_ in plan:
        print("   -", label)
    if not a.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        sys.exit("[rebuild] cancelled - nothing changed")
    for p in GENERATED:
        t = ROOT / p
        if t.is_dir():
            shutil.rmtree(t)
        elif t.exists():
            t.unlink()
    (ROOT / "data").mkdir(exist_ok=True)
    for label, cmd, cwd in plan:
        run(label, cmd, cwd)
    print("\n[rebuild] done - this checkpoint's state is rebuilt. Carry on with the next lab.")

if __name__ == "__main__":
    main()
