"""MLflow tracking wrapper with safe offline fallback."""

from typing import Dict, Any, Optional
import os


class MlflowTracker:
    """Safe MLflow tracking interface with automatic fallback if server is inactive."""

    def __init__(self, experiment_name: str = "BillGuard_Anomaly_Detection"):
        self.experiment_name = experiment_name
        self.is_available = False
        try:
            import mlflow
            self.mlflow = mlflow
            self.mlflow.set_experiment(experiment_name)
            self.is_available = True
        except Exception as e:
            self.mlflow = None
            print(f"[MLflow Tracker] MLflow disabled or offline: {e}")

    def log_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Log an experiment run."""
        if not self.is_available or self.mlflow is None:
            return None

        try:
            with self.mlflow.start_run(run_name=run_name) as run:
                self.mlflow.log_params(params)
                self.mlflow.log_metrics(metrics)
                if tags:
                    self.mlflow.set_tags(tags)
                return run.info.run_id
        except Exception as e:
            print(f"[MLflow] Run logging failed: {e}")
            return None
