# FreshMart — From Pipelines to Production

Reference code for the BITS Pilani WILP · CSIS Faculty Development Programme
*From Pipelines to Production*. Every file here is taken verbatim from the six
FreshMart tutorials (M2 v8, Labs 1–5 v9), committed in lab order.

## Quick start

```bash
git clone <repository-or-bundle> freshmart && cd freshmart
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python check_setup.py
```

## Catch-up checkpoints

Each lab's finished state is a Git tag. Fell behind? Save your work, then:

```bash
git switch -c catchup-lab2 lab2-done   # code as the Lab 2 tutorial ends
python scripts/rebuild.py              # regenerate data, feature store, registry
```

| Tag | Code as it stands when… |
|---|---|
| `start` | the environment is ready, before M2 |
| `m2-done` | M2 (analysis anchor) is finished |
| `lab1-done` | Lab 1 (DataOps pipeline + Feast) is finished |
| `lab2-done` | Lab 2 (ML pipeline + MLflow registry) is finished |
| `lab3-done` | Lab 3 (serving API + planner app) is finished |
| `lab4-done` | Lab 4 (monitor + container) is finished |
| `lab5-done` | Lab 5 (CI/CD, canary, rollback) is finished |

Data is never committed: every dataset is regenerated from a fixed seed, so
`rebuild.py` gives everyone identical numbers. See the participant guide
*FreshMart Catch-up Checkpoints — Git Guide* for the full walkthrough.
