import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(experiment_ids=["0"], order_by=["start_time DESC"])
print(f"Total runs: {len(runs)}")
for r in runs:
    p = r.data.params
    m = r.data.metrics
    print(f"Run {r.info.run_id[:8]} | n_estimators={p.get('n_estimators')} max_depth={p.get('max_depth')} min_samples_split={p.get('min_samples_split')} | accuracy={m.get('accuracy'):.4f} f1={m.get('f1_score'):.4f}")
