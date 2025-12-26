import os
from typing import Dict, Optional, Any
import mlflow
from mlflow.tracking import MlflowClient
from rich.console import Console
from rich.text import Text
import questionary

console = Console()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")


class MLFlowManager:
    """
    A manager for working with MLflow, providing structured interaction with
    the server and experiment management.
    """

    def __init__(self, experiment_name: str, track_experiment: bool = True):
        """
        Args:
            experiment_name: The name of the experiment.
            track_experiment: A flag indicating whether the experiment should be tracked.
        """
        self.uri = MLFLOW_TRACKING_URI
        self.experiment_name = experiment_name
        self.track_experiment = track_experiment
        self.client = MlflowClient()

    def connect(self) -> None:
        """Connect to the MLflow server and set up the experiment."""
        if not self.track_experiment:
            return

        with console.status("[bold cyan]Connecting to MLflow...[/bold cyan]"):
            mlflow.set_tracking_uri(self.uri)
            mlflow.set_experiment(self.experiment_name)
        console.print("[green]✓[/green] Connected to MLflow server")

    def start_run(self, run_name: Optional[str] = None) -> Optional[mlflow.ActiveRun]:
        """
        Start a new experiment run.

        Args:
            run_name: The run name. If None, it will be requested from the user.

        Returns:
            An active run object or None if tracking is disabled.
        """
        if not self.track_experiment:
            return None

        if run_name is None:
            run_name = questionary.text(
                "Enter run name",
                default="Test"
            ).ask()

        console.print(f'→ [MLFlow] Run name set to: [blue]{run_name}[/blue]') 
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log experiment parameters."""
        if not self.track_experiment:
            return

        with console.status("[bold cyan]Logging parameters...[/bold cyan]"):
            mlflow.log_params(params)
        console.print('→ [MLFlow] Parameters logged') 

    def log_artifact(
        self,
        artifact_path: str,
        run_id: Optional[str] = None
    ) -> None:
        """
        Log an artifact to the experiment.

        Args:
            artifact_path: Path to the artifact.
            run_id: Run ID (if None, the active run is used).
        """
        if not self.track_experiment:
            return

        with console.status("[bold cyan]Logging artifact...[/bold cyan]"):
            if run_id is None:
                mlflow.log_artifact(artifact_path)
            else:
                self.client.log_artifact(run_id, artifact_path)
        console.print('→ [MLFlow] The artifact is registered') 

    def log_model(
        self,
        model: Any,
        name: str,
        signature: Any,
        run_id: Optional[str] = None
    ) -> None:
        """
        Register a model in the experiment.

        Args:
            model: The model to register.
            name: Model name.
            signature: Model signature.
            run_id: Run ID (if None, the active run is used).
        """
        if not self.track_experiment:
            return

        with console.status("[bold cyan]Logging model...[/bold cyan]"):
            if run_id is None:
                mlflow.pytorch.log_model(model, name, signature=signature)
            else:
                with mlflow.start_run(run_id=run_id):
                    mlflow.pytorch.log_model(model, name, signature=signature)
        console.print('→ [MLFlow] The model is registered') 

    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
        run_id: Optional[str] = None
    ) -> None:
        """
        Log a numeric metric to the experiment.

        Args:
            key: Metric name (e.g., "loss", "accuracy").
            value: Metric value (a number).
            step: Step/iteration (optional). If specified, the metric will be associated with this step.
            run_id: Run ID (if None, the active run is used).
        """
        if not self.track_experiment:
            return

        if run_id is None:
            mlflow.log_metric(key, value, step=step)
        else:
            self.client.log_metric(run_id, key, value, step=step)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        run_id: Optional[str] = None
    ) -> None:
        """
        Log a set of metrics to the experiment.

        Args:
            metrics: A dictionary of metrics in the form {name: value}.
            step: Step/iteration (optional). If specified, all metrics will be associated with this step.
            run_id: Run ID (if None, the active run is used).
        """
        if not self.track_experiment:
            return

        # with console.status("[bold cyan]Logging metrics...[/bold cyan]"):
        if run_id is None:
            mlflow.log_metrics(metrics, step=step)
        else:
            for key, value in metrics.items():
                self.client.log_metric(run_id, key, value, step=step)

    def end_run(self) -> None:
        """End the active run."""
        if not self.track_experiment:
            return

        mlflow.end_run()
        console.print('→ [MLFlow] Run ended') 
