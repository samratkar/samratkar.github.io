# Facilitator notes — verified against the committed code

Every lab was executed end to end from this repository (Python 3.12, the pins in
requirements.txt). The code runs. Where a tutorial's printed output differs from
what the code produces, participants will see the values in the right-hand column.

| Where | Tutorial prints | This code produces |
|---|---|---|
| M2, all missions | as printed | identical — reproduces exactly |
| Lab 1 row counts, contract, data_hash | as printed | identical (31,598 rows, 16/16, e0ce921bf845) |
| Lab 2 training-set base rate | 0.202 | 0.178 (rows 17,869 identical) |
| Lab 2 champion (logreg) AUC | 0.697 | 0.942 |
| Lab 2 test base rate | 0.241 | 0.044 |
| Lab 2 best-cost threshold | 0.10 (46% saving) | 0.25 (23% saving) |
| Lab 3 S001/P0067 score | 0.2182, add to watch list | 0.0053, no action |
| Lab 3 shadow action disagreement | 10.0% | 0.4% |
| Lab 4 drifted batch data_hash | d4f014b88eba | 1c80632f8f92 (rows 29,867 identical) |
| Lab 4 units_7d PSI | 0.8149 | 0.1647 |
| Lab 4 monitor decision | RERUN_DATA_PIPELINE | NOOP — the monitor does not fire |
| Lab 5 promotion gate | refuses the candidate | PROMOTE |

The likely cause: the captures were taken with an earlier label definition and
drift generator; the committed `engineer_features.py` labels a 1.5x demand spike.

## Known environment issues

- `ydata-profiling` 4.18.4 imports `pkg_resources`; requirements.txt pins `setuptools<81`.
- `requirements-serve.txt` pins `pandas==3.0.2`, but `feast==0.66.0` needs pandas < 3,
  so `docker compose build` fails to resolve. `pandas==2.3.3` resolves.
- The Dockerfile uses `python:3.11-slim`; `xgboost==3.4.1` needs Python ≥ 3.12.
- The Dockerfile copies `artifacts/` (mlflow.db) but MLflow writes model files to
  `mlruns/`, which the image does not copy, so the container cannot load the model.

These are left exactly as the tutorials have them.
