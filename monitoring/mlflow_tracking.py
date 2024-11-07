import mlflow
import logging

class MLflowTracking:
    def __init__(self, experiment_name="Default Experiment"):
        """
        Initializes MLflow tracking for a given experiment.

        Parameters:
        - experiment_name (str): The name of the MLflow experiment to log under.
        """
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_experiment()

    def setup_experiment(self):
        """
        Sets up the MLflow experiment based on the provided experiment name.
        """
        mlflow.set_experiment(self.experiment_name)
        self.logger.info(f"MLflow experiment '{self.experiment_name}' set up successfully.")

    def start_run(self, run_name=None):
        """
        Starts an MLflow run with an optional name.

        Parameters:
        - run_name (str): Optional name for the run.

        Returns:
        - mlflow.ActiveRun: The active MLflow run object.
        """
        run = mlflow.start_run(run_name=run_name)
        self.logger.info(f"Started MLflow run: '{run_name}'")
        return run

    def end_run(self):
        """
        Ends the current MLflow run.
        """
        mlflow.end_run()
        self.logger.info("MLflow run ended.")

    def log_params(self, params):
        """
        Logs parameters for the current MLflow run.

        Parameters:
        - params (dict): Dictionary of parameters to log.
        """
        mlflow.log_params(params)
        self.logger.info(f"Logged parameters: {params}")

    def log_metrics(self, metrics):
        """
        Logs metrics for the current MLflow run.

        Parameters:
        - metrics (dict): Dictionary of metrics to log.
        """
        mlflow.log_metrics(metrics)
        self.logger.info(f"Logged metrics: {metrics}")

    def log_param(self, key, value):
        """
        Logs a single parameter for the current MLflow run.

        Parameters:
        - key (str): Parameter name.
        - value (Any): Parameter value.
        """
        mlflow.log_param(key, value)
        self.logger.info(f"Logged parameter: {key} = {value}")

    def log_metric(self, key, value):
        """
        Logs a single metric for the current MLflow run.

        Parameters:
        - key (str): Metric name.
        - value (float): Metric value.
        """
        mlflow.log_metric(key, value)
        self.logger.info(f"Logged metric: {key} = {value}")

    def log_artifact(self, file_path, artifact_path=None):
        """
        Logs a file or directory as an artifact for the current MLflow run.

        Parameters:
        - file_path (str): Path to the file or directory to log.
        - artifact_path (str, optional): Optional artifact path within the run.
        """
        mlflow.log_artifact(file_path, artifact_path=artifact_path)
        self.logger.info(f"Logged artifact: {file_path}")

    def log_artifacts(self, dir_path, artifact_path=None):
        """
        Logs a directory of artifacts for the current MLflow run.

        Parameters:
        - dir_path (str): Path to the directory to log.
        - artifact_path (str, optional): Optional artifact path within the run.
        """
        mlflow.log_artifacts(dir_path, artifact_path=artifact_path)
        self.logger.info(f"Logged artifacts from directory: {dir_path}")