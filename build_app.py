# The build_app.py file will serve as a utility to encapsulate the entire app-building process, setting up configuration, initializing classes, and loading necessary assets. This setup will make everything accessible for main.py without exposing internal details directly.

# The build_app.py will:
# DB setup etc..
# Read Configuration: Load configurations from a configuration file.
# Set up Dependencies: Initialize components like logging, MLflow tracking, model, pipeline, etc.
# Return the Assembled Components: Return initialized instances and configurations for use in main.py.
import pickle
import yaml
from logging_config import LoggerConfig
from mlflow_tracking import MLflowTracking
from feature_engineering.engineer_features import FeatureEngineer
from post_processing import PostProcessor
from fastapi import FastAPI
from api_server import app as fastapi_app
import logging

class AppBuilder:
    def __init__(self, config_path="config.yaml"):
        """
        Initializes AppBuilder with configurations from the given YAML config file.

        Parameters:
        - config_path (str): Path to the configuration YAML file.
        """
        self.config = self._load_config(config_path)
        self.logger = None
        self.model = None
        self.feature_engineer = None
        self.post_processor = None
        self.mlflow_tracking = None

    def _load_config(self, config_path):
        """
        Reads the configuration from a YAML file.

        Parameters:
        - config_path (str): Path to the YAML configuration file.

        Returns:
        - dict: Parsed configuration dictionary.
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        return config

    def setup_logging(self):
        """
        Sets up logging based on configuration settings.
        """
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        log_to_file = log_config.get("to_file", True)
        log_filename = log_config.get("filename", "app.log")

        logger_config = LoggerConfig(log_level=log_level, log_to_file=log_to_file, log_filename=log_filename)
        self.logger = logger_config.get_logger("AppBuilder")
        self.logger.info("Logging setup complete.")

    def load_model_and_pipeline(self):
        """
        Loads the trained model and feature engineering pipeline from specified paths.
        """
        try:
            model_path = self.config["model"]["path"]
            pipeline_path = self.config["pipeline"]["path"]

            with open(model_path, "rb") as model_file:
                self.model = pickle.load(model_file)
                self.logger.info("Model loaded successfully.")

            with open(pipeline_path, "rb") as pipeline_file:
                self.feature_engineer = pickle.load(pipeline_file)
                self.logger.info("Feature engineering pipeline loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load model or pipeline: {e}")
            raise

    def setup_post_processor(self):
        """
        Initializes the PostProcessor with max_value from configuration for inverse transformations.
        """
        max_value = self.config["post_processing"].get("max_value")
        self.post_processor = PostProcessor(max_value=max_value)
        self.logger.info("PostProcessor initialized.")

    def setup_mlflow_tracking(self):
        """
        Initializes MLflow tracking with the experiment name from configuration.
        """
        experiment_name = self.config["mlflow"]["experiment_name"]
        self.mlflow_tracking = MLflowTracking(experiment_name=experiment_name)
        self.logger.info("MLflow tracking initialized.")

    def build(self):
        """
        Builds and returns all initialized components as a dictionary.

        Returns:
        - dict: A dictionary containing initialized components.
        """
        self.setup_logging()
        self.load_model_and_pipeline()
        self.setup_post_processor()
        self.setup_mlflow_tracking()

        # Return the fully assembled app components
        return {
            "logger": self.logger,
            "model": self.model,
            "feature_engineer": self.feature_engineer,
            "post_processor": self.post_processor,
            "mlflow_tracking": self.mlflow_tracking,
            "app": fastapi_app,
        }
