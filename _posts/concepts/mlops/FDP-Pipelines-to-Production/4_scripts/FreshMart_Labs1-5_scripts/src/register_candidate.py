# Lab 3 / Mission 6 setup: promote Lab 2's XGBoost run to @candidate.
#
# Shadow mode is meaningless if the candidate IS the champion - the deltas are all
# zero and the harness proves nothing. Lab 2 already trained a second model that
# passed every gate except "best AUC"; that is exactly what a real candidate looks
# like. Register it under the same name with the candidate alias.
import mlflow

NAME = "freshmart_reorder_model"
mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")
c = mlflow.MlflowClient()

exp = c.get_experiment_by_name("freshmart_reorder")
runs = c.search_runs([exp.experiment_id],
                     filter_string="attributes.run_name = 'xgboost'",
                     order_by=["attributes.start_time DESC"], max_results=1)
if not runs:
    raise SystemExit("no xgboost run found - run src/train.py first")
run = runs[0]

mv = mlflow.register_model(f"runs:/{run.info.run_id}/model", NAME)
for k, v in {"model_type": "xgboost",
             "data_hash": run.data.tags.get("data_hash", "unknown"),
             "auc": f"{run.data.metrics['auc']:.4f}"}.items():
    c.set_model_version_tag(NAME, mv.version, k, v)
c.set_registered_model_alias(NAME, "candidate", mv.version)
print(f"registered v{mv.version} as @candidate "
      f"(xgboost, AUC={run.data.metrics['auc']:.4f}, "
      f"data_hash={run.data.tags.get('data_hash')})")
