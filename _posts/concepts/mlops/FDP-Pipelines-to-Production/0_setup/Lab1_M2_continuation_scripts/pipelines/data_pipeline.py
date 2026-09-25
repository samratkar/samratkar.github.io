"""Lab 1 - the data pipeline as a Prefect flow.

Prefect turns each stage into a task with logging, retries and a visible run in
the Prefect UI. The flow: generate -> clean -> validate (contract gate) -> ops ->
engineer -> feast apply + materialize. A failed contract raises, and Prefect
stops the flow before a single feature is built on data nobody trusts.
"""
import subprocess, sys, json
from pathlib import Path
from prefect import flow, task, get_run_logger

def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout: print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip()); raise RuntimeError(f"stage failed: {' '.join(cmd)}")

@task(retries=1)
def generate(drift):
    cmd = [sys.executable, "src/generate_data.py"]
    if drift: cmd.append("--drift")
    _run(cmd)

@task
def clean():
    _run([sys.executable, "src/clean_pos.py"])

@task
def validate():
    _run([sys.executable, "src/validate_data.py"])          # contract gate
    rep = json.loads(Path("data/contract_report.json").read_text())
    if not rep["success"]:
        raise ValueError(f"data contract failed: {rep['failed_expectations']}")
    return rep["observability"]

@task(retries=2)
def ops(asof):
    _run([sys.executable, "src/fetch_ops.py", "--asof", asof])

@task
def engineer():
    _run([sys.executable, "src/engineer_features.py"])

@task
def feast_apply_materialize():
    _run(["feast", "-c", "feature_repo", "apply"])
    _run(["feast", "-c", "feature_repo", "materialize-incremental",
          "2026-12-31T00:00:00"])

@flow(name="reorder-data-pipeline")
def data_pipeline(asof: str = "2026-08-15", drift: bool = False):
    log = get_run_logger()
    generate(drift)
    clean()
    obs = validate()
    log.info(f"observability: {obs['volume']['rows']} rows, "
             f"{obs['volume']['store_product_pairs']} store-product pairs, "
             f"latest event {obs['freshness']['latest_event']}")
    ops(asof)
    engineer()
    feast_apply_materialize()
    log.info("data pipeline complete - offline features written, online store materialised")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default="2026-08-15")
    ap.add_argument("--drift", action="store_true")
    a = ap.parse_args()
    data_pipeline(asof=a.asof, drift=a.drift)
