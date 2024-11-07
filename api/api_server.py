from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
from feature_engineering.engineer_features import FeatureEngineer  # Import FeatureEngineer for data processing
from monitoring.mlflow_tracking import MLflowTracking  # Import MLflowTracking for tracking requests if needed
import logging

# Initialize the FastAPI app
app = FastAPI()

# Load model and feature engineering pipeline at startup
with open("best_model.pkl", "rb") as model_file, open("preprocessing_pipeline.pkl", "rb") as pipeline_file:
    model = pickle.load(model_file)
    feature_engineer = pickle.load(pipeline_file)
    logger = logging.getLogger("api_server")
    logger.info("Model and preprocessing pipeline loaded successfully.")

# Input data model
class PredictionRequest(BaseModel):
    publishedYear: int
    book_age: int
    published_decade: int
    published_century: int
    author_frequency: int
    publisher_frequency: int
    published_era_encoded: int
    Category_Index: int
    title_sentiment_encoded: int
    description_sentiment_encoded: int
    title_tfidf_dense: float
    desc_tfidf_dense: float
    author_publisher_combined_embedding: float

# Prediction endpoint
@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Convert input data to DataFrame for processing
        input_data = pd.DataFrame([request.dict()])

        # Preprocess input data using FeatureEngineer pipeline
        processed_data = feature_engineer.assemble_features(input_data)

        # Convert processed data to a format suitable for model prediction
        X = np.array(processed_data.select("assembled_features").collect()).reshape(1, -1)

        # Make prediction
        prediction = model.predict(X)
        return {"prediction": prediction[0]}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs for details.")

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "API is running"}

# Start the server with: uvicorn api_server:app --reload
