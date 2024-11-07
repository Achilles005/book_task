import pickle
import numpy as np
import pandas as pd
from feature_engineer.engineer_features import FeatureEngineer
from pyspark.sql import SparkSession
from pyspark.ml.functions import vector_to_array

class ModelPredictor:
    def __init__(self, model_path="best_model.pkl"):
        """
        Initializes the ModelPredictor with a trained model and recreates the feature engineering pipeline.

        Parameters:
        - model_path (str): Path to the saved model file.
        """
        self.spark = SparkSession.builder.appName("ModelPrediction").getOrCreate()
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load the trained model from a specified path.

        Parameters:
        - model_path (str): Path to the model file.

        Returns:
        - model: Loaded model object.
        """
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return model

    def preprocess_data(self, data):
        """
        Preprocesses the input data using the FeatureEngineer class.

        Parameters:
        - data (pd.DataFrame or dict): Raw data to preprocess.

        Returns:
        - Spark DataFrame: Processed data ready for prediction.
        """
        # If the input is a dictionary (single data point), convert to Pandas DataFrame
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        # Convert Pandas DataFrame to Spark DataFrame
        spark_data = self.spark.createDataFrame(data)

        # Apply the feature engineering steps
        feature_engineer = FeatureEngineer(spark_data)
        processed_data = (
            feature_engineer.create_date_features()
            .encode_era_feature()
            .add_text_features()
            .add_sentiment_analysis()
            .add_author_publisher_frequency()
            .add_author_publisher_embeddings()
            .add_category_feature_hashing()
            .final_processed_data()
        )

        # Convert assembled features to an array for model compatibility
        processed_data = processed_data.withColumn("assembled_features_array", vector_to_array("assembled_features"))
        return processed_data

    def predict(self, data):
        """
        Make predictions on the input data.

        Parameters:
        - data (pd.DataFrame or dict): Input data for prediction.

        Returns:
        - np.array: Predictions for the input data.
        """
        # Preprocess data using the feature engineering pipeline
        processed_data = self.preprocess_data(data)
        processed_data_pd = processed_data.select("assembled_features_array").toPandas()
        X = np.array(processed_data_pd["assembled_features_array"].tolist())

        # Generate predictions
        predictions = self.model.predict(X)
        return predictions

    def predict_from_csv(self, input_data_path):
        """
        Run batch predictions from a CSV file.

        Parameters:
        - input_data_path (str): Path to the CSV file containing the data.

        Returns:
        - np.array: Predictions for the batch data.
        """
        # Load CSV file into a Pandas DataFrame
        data = pd.read_csv(input_data_path)
        return self.predict(data)

    def predict_single(self, single_data_point):
        """
        Run prediction for a single data point.

        Parameters:
        - single_data_point (dict): Dictionary representing a single data instance.

        Returns:
        - np.array: Prediction for the single data point.
        """
        prediction = self.predict(single_data_point)
        print("Single Prediction:", prediction)
        return prediction

# Example usage of ModelPredictor
if __name__ == "__main__":

    # Initialize ModelPredictor
    predictor = ModelPredictor(model_path="best_model.pkl")
    predictions = predictor.predict_from_csv("batch_data.csv")
    print("Batch Predictions:", predictions)