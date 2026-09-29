"""Lab 5 - the local CI/CD pipeline, run as jobs with gates between them.

This reproduces on a laptop what a cloud pipeline does with runners and approval
steps: test, build, gate, promote, deploy, smoke. Each job can fail the run, and
the gates are the reason the pipeline exists - a pipeline without them is just a
slower way to copy files.

The promotion step is the part worth reading twice. It records @previous BEFORE
moving @production, so rollback is already possible the instant the new version
goes live. Nobody has to remember to prepare for it, which is the only way it
gets done at 02:00.

Note what is NOT here: no model file is copied, no image is rebuilt to promote.
Promotion is an alias move on the registry, and the deploy step reads that alias.
"""
import os, signal, subprocess, sys, time
from pathlib import Path

import mlflow

TRACKING  = "sqlite:///artifacts/mlflow.db"
NAME      = "freshmart_reorder_model"
AUC_FLOOR = 0.65
PORT      = 8066


def job(title):
    print(f"\n\u2500\u2500 JOB: {title} " + "\u2500" * max(2, 46 - len(title)))


def run(cmd, **kw):
    r = subprocess.run(cmd, **kw)
    return r.returncode


def main():
    Path("artifacts").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING)
    c = mlflow.MlflowClient()

    job("test (data contract)")
    if run([sys.executable, "src/validate_data.py"],
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        print("[ci] data contract FAILED - build blocked, nothing else runs")
        sys.exit(1)
    print("[ci] data contract: PASS")

    job("build (re-execute pipeline, retrain, register)")
    run(["dvc", "repro", "-q"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run([sys.executable, "src/build_training_set.py"], stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    if run([sys.executable, "src/train.py"], stdout=subprocess.DEVNULL,
           stderr=subprocess.DEVNULL) != 0:
        print("[ci] training FAILED - blocked")
        sys.exit(1)
    champ = c.get_model_version_by_alias(NAME, "champion")
    auc = float(champ.tags.get("auc", 0))
    print(f"[ci] @champion = v{champ.version} ({champ.tags.get('model_type')}) AUC={auc:.4f} "
          f"data_hash={champ.tags.get('data_hash')}")

    job("gate (quality + lineage)")
    if auc < AUC_FLOOR:
        print(f"[ci] quality gate FAILED (AUC {auc:.4f} < {AUC_FLOOR}) - not promoted")
        sys.exit(1)
    if not champ.tags.get("data_hash"):
        print("[ci] lineage gate FAILED (no data_hash) - not promoted")
        sys.exit(1)
    print(f"[ci] quality gate PASS (AUC {auc:.4f} >= {AUC_FLOOR})")
    print("[ci] lineage gate PASS (data_hash present)")

    job("promote (@production <- @champion)")
    try:
        prev = c.get_model_version_by_alias(NAME, "production")
        c.set_registered_model_alias(NAME, "previous", prev.version)
        print(f"[ci] rollback pointer @previous -> v{prev.version}")
    except Exception:
        print("[ci] no existing @production - this is the first promotion")
    c.set_registered_model_alias(NAME, "production", champ.version)
    print(f"[ci] @production -> v{champ.version}")

    job("deploy + smoke test (@production)")
    env = dict(os.environ, MODEL_ALIAS="production", MLFLOW_TRACKING_URI=TRACKING)
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "serving.api:app",
                             "--port", str(PORT)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import requests, pandas as pd
        ready = False
        for _ in range(60):
            try:
                if requests.get(f"http://localhost:{PORT}/ready", timeout=2).status_code == 200:
                    ready = True
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        if not ready:
            print("[ci] deploy FAILED - service never became ready")
            sys.exit(1)
        info = requests.get(f"http://localhost:{PORT}/ready", timeout=5).json()
        print(f"[ci] service ready: serving v{info['model_version']} via @{info['alias']}")

        line = pd.read_parquet("feature_repo/data/reorder_features.parquet").iloc[0]
        r = requests.post(f"http://localhost:{PORT}/predict",
                          json={"store_id": line.store_id, "product_id": line.product_id},
                          timeout=10)
        body = r.json()
        assert r.status_code == 200 and 0 <= body["reorder_probability"] <= 1
        print(f"[ci] smoke test HTTP {r.status_code}: {body['store_id']}/{body['product_id']} "
              f"-> {body['reorder_probability']} ({body['action']})")
        print("\n[ci] PIPELINE GREEN - production is live and serving.")
    finally:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=15)
        print("[ci] service torn down")


if __name__ == "__main__":
    main()
