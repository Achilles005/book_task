# book_task
This project is a production-ready data science project built with FastAPI, PySpark, and MLflow. The repo provides endpoints for model predictions and is designed for easy deployment and scaling. With feature engineering, post-processing, and MLflow tracking integrated, this server can handle large datasets and supports a CI/CD pipeline for continuous testing, building, and deployment.

# Table of Contents
1. Overview
2. Features
3. Directory Structure
4. Setup and Installation
5. Usage
6. API Endpoints
7. Configuration
8. CI/CD Pipeline
9. Contributing

# Overview
This API server is built for scalable machine learning inference in a production environment. With a modular design, it separates feature engineering, post-processing, and prediction tasks, making it easy to maintain and extend. The server is containerized with Docker, and a GitHub Actions CI/CD pipeline ensures quality and reliability by automating tests, builds, and deployments.

# Features
Model Serving with FastAPI: Serve predictions with a robust FastAPI server.
Feature Engineering: Data preprocessing and transformation using PySpark.
Post-Processing: Apply custom transformations to model outputs before returning them to clients.
MLflow Integration: Track model performance and log metrics.
CI/CD Pipeline: Automate linting, testing, and deployment with GitHub Actions.
Dockerized Deployment: Ensure a consistent environment using Docker.


# Setup and Installation
1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```
2. Create a Virtual Environment and Install Dependencies
It is recommended to set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```
3. Configuration
Edit config.yaml to set paths and parameters based on your environment and preferences.

yaml
```bash
logging:
  level: INFO
  to_file: true
  filename: "app.log"

model:
  path: "best_model.pkl"

pipeline:
  path: "preprocessing_pipeline.pkl"

post_processing:
  max_value: 100  # Used for inverse log transformation

mlflow:
  experiment_name: "Production Predictions"
4. Run the Application
Start the application with:
```
```bash
python main.py
5. Run with Docker
To build and run the server in a Docker container:
```
```bash
docker build -t api_server .
docker run -p 8000:8000 api_server
Usage
Endpoints
POST /predict: Accepts JSON data and returns a prediction.
Request Body:
json
{
  "publishedYear": 2005,
  "book_age": 16,
  "published_decade": 2000,
  "published_century": 21,
  "author_frequency": 10,
  "publisher_frequency": 5,
  "published_era_encoded": 1,
  "Category_Index": 2,
  "title_sentiment_encoded": 1,
  "description_sentiment_encoded": 0,
  "title_tfidf_dense": 0.23,
  "desc_tfidf_dense": 0.17,
  "author_publisher_combined_embedding": 0.3
}
Response:
json
{
  "prediction": 25.7
}
GET /health: Health check endpoint to verify if the server is running.

Testing
To run unit tests, execute:
```
```bash
pytest
Configuration
Configuration is managed through config.yaml and includes settings for:

Logging: Control logging levels and output file.
Model and Pipeline Paths: Paths to saved model and preprocessing pipeline.
Post-Processing: Specify parameters for output transformations.
MLflow: Set experiment name for tracking.
CI/CD Pipeline
This project uses a GitHub Actions workflow (.github/workflows/ci-cd.yml) to automate the CI/CD process.

Workflow Steps
Linting: Uses flake8 to check code quality.
Testing: Runs unit tests using pytest.
Build and Push Docker Image:
Builds the Docker image and pushes it to Docker Hub.
Requires DOCKER_USERNAME and DOCKER_PASSWORD secrets in GitHub.
Deploy: Placeholder step for deployment. Customize this step based on your cloud environment.
Example GitHub Actions Workflow
yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install flake8
      - name: Lint with flake8
        run: flake8 --max-line-length=88 .

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      - name: Run tests
        run: pytest

  build_and_push:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v2
      - name: Log in to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      - name: Build and push Docker image
        run: |
          docker build -t your_dockerhub_username/your_image_name:latest .
          docker push your_dockerhub_username/your_image_name:latest

  deploy:
    runs-on: ubuntu-latest
    needs: build_and_push
    steps:
      - name: Deploy to server
        run: |
          # Add deployment script here (e.g., SSH into server, pull Docker image, restart container)
```