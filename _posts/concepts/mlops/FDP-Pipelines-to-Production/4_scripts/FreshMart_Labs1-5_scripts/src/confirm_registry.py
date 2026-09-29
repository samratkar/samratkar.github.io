import mlflow
mlflow.set_tracking_uri("sqlite:///artifacts/mlflow.db")
c = mlflow.MlflowClient()
mv = c.get_model_version_by_alias("freshmart_reorder_model", "champion")
print(f"champion = v{mv.version}  run_id={mv.run_id[:12]}")
print(f"tags     = {mv.tags}")
m = c.get_run(mv.run_id).data.metrics
print(f"metrics  = auc={m['auc']:.3f} f1={m['f1']:.3f} latency={m['latency_ms_single']:.2f} ms/request")
