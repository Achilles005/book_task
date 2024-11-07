import mlflow
import numpy as np
import pandas as pd
import time
import pickle
import lightgbm as lgb
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import KFold

class ModelTrainer:
    def __init__(self, X_train, y_train,experiment_name="Model Training Experiment"):
        """
        Initializes the ModelTrainer with training data and an MLflow experiment.

        Parameters:
        - X_train: Training features as a numpy array.
        - y_train: Training target values as a numpy array.
        - experiment_name: Name of the MLflow experiment.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.models = {}
        mlflow.set_experiment(experiment_name)

    def _calculate_metrics(self, model, X, y, metric_storage):
        """
        Calculates and stores RMSE and MAPE for a given model, dataset (X, y), and storage.

        Parameters:
        - model: Trained model for predictions.
        - X: Features as a numpy array.
        - y: Target values as a numpy array.
        - metric_storage: Dictionary with keys 'rmse' and 'mape' to store metrics.
        """
        y_pred = model.predict(X)
        metric_storage['rmse'].append(mean_squared_error(y, y_pred, squared=False))
        metric_storage['mape'].append(mean_absolute_percentage_error(y, y_pred))

    def _cross_validate(self, model, model_name, params=None):
        """
        Perform cross-validation, log results in MLflow, and return metrics.

        Parameters:
        - model: The model instance.
        - model_name: Name of the model.
        - params: Model parameters for logging in MLflow.
        """
        with mlflow.start_run(run_name=f"{model_name}_CV"):
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            train_metrics = {'rmse': [], 'mape': []}
            val_metrics = {'rmse': [], 'mape': []}
            feature_importance_agg = np.zeros(self.X_train.shape[1]) if hasattr(model, "feature_importances_") else None
            start_time = time.time()

            for train_index, val_index in kf.split(self.X_train):
                X_train_fold, X_val_fold = self.X_train[train_index], self.X_train[val_index]
                y_train_fold, y_val_fold = self.y_train[train_index], self.y_train[val_index]
                model.fit(X_train_fold, y_train_fold)

                # Calculate metrics for training and validation sets
                self._calculate_metrics(model, X_train_fold, y_train_fold, train_metrics)
                self._calculate_metrics(model, X_val_fold, y_val_fold, val_metrics)

                # Aggregate feature importance if applicable
                if feature_importance_agg is not None:
                    feature_importance_agg += model.feature_importances_

            # Log parameters, metrics, and feature importance to MLflow
            if params:
                for param, value in params.items():
                    mlflow.log_param(param, value)
            mlflow.log_metric("rmse_cv", np.mean(val_metrics['rmse']))
            mlflow.log_metric("mape_cv", np.mean(val_metrics['mape']))
            mlflow.log_metric("rmse_train_cv", np.mean(train_metrics['rmse']))
            mlflow.log_metric("mape_train_cv", np.mean(train_metrics['mape']))
            mlflow.log_metric("training_time", time.time() - start_time)

            # Log average feature importance if available
            if feature_importance_agg is not None:
                feature_importance = feature_importance_agg / kf.get_n_splits()
                mlflow.log_param("feature_importance", dict(enumerate(feature_importance)))

            print(f"{model_name} - RMSE CV: {np.mean(val_metrics['rmse'])}, MAPE CV: {np.mean(val_metrics['mape'])}")


    def train_decision_tree(self):
        """
        Trains a Decision Tree Regressor with cross-validation and logs results.
        """
        dt_params = {
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42
        }
        model = DecisionTreeRegressor(**dt_params)
        self._cross_validate(model, "DecisionTree", dt_params)

    def train_linear_regression(self):
        """
        Trains a Linear Regression model with cross-validation and logs results.
        """
        model = LinearRegression()
        self._cross_validate(model, "LinearRegression")

    def train_ridge(self):
        """
        Trains a Ridge Regression model with cross-validation and logs results.
        """
        model = Ridge(alpha=1.0)
        self._cross_validate(model, "RidgeRegression", {"alpha": 1.0})

    def train_lasso(self):
        """
        Trains a Lasso Regression model with cross-validation and logs results.
        """
        model = Lasso(alpha=0.1)
        self._cross_validate(model, "LassoRegression", {"alpha": 0.1})
    
    
    def _log_and_plot_feature_importance(self, model, model_name, feature_names):
        """
        Logs and plots feature importance for a model in MLflow and generates a bar chart.

        Parameters:
        - model: The trained model with feature importance (e.g., LightGBM, XGBoost, RandomForest).
        - model_name: Name of the model for logging and display.
        - feature_names: List of feature names corresponding to model input.
        """
        if hasattr(model, "feature_importances_"):
            # Retrieve and normalize feature importances
            feature_importance = model.feature_importances_
            feature_importance = feature_importance / feature_importance.sum()  # Normalize

            # Log feature importances in MLflow
            mlflow.log_param(f"{model_name}_feature_importance", dict(zip(feature_names, feature_importance.tolist())))

            # Plot feature importance
            feature_importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": feature_importance
            }).sort_values(by="importance", ascending=False)
        else:
            print(f"{model_name} does not support feature importance extraction.")

    def train_lightgbm(self, feature_names):
        """
        Trains a LightGBM model with cross-validation, logs results, and plots feature importance.
        """
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.1,
            'max_depth': 5,
            'n_estimators': 10,
            'random_state': 42
        }
        model = lgb.LGBMRegressor(**lgb_params)
        self._cross_validate(model, "LightGBM", lgb_params)

        # Log feature importance
        self._log_and_plot_feature_importance(model, "LightGBM", feature_names)

    def train_xgboost(self, feature_names):
        """
        Trains an XGBoost model with cross-validation, logs results, and plots feature importance.
        """
        xgb_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 5,
            'n_estimators': 10,
            'eval_metric': 'rmse',
            'random_state': 42
        }
        model = xgb.XGBRegressor(**xgb_params)
        self._cross_validate(model, "XGBoost", xgb_params)

        # Log feature importance
        self._log_and_plot_feature_importance(model, "XGBoost", feature_names)

    def train_random_forest(self, feature_names):
        """
        Trains a Random Forest Regressor with cross-validation, logs results, and plots feature importance.
        """
        rf_params = {
            'n_estimators': 10,
            'max_depth': 5,
            'min_samples_split': 5,
            'random_state': 42
        }
        model = RandomForestRegressor(**rf_params)
        self._cross_validate(model, "RandomForest", rf_params)

        # Log feature importance
        self._log_and_plot_feature_importance(model, "RandomForest", feature_names)

    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluates a trained model on the test set and logs the results.

        Parameters:
        - model: The trained model to evaluate.
        - X_test: Test features as a numpy array.
        - y_test: Test target values as a numpy array.
        - model_name: Name of the model for logging and display.
        """
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print(f"{model_name} Test Results - RMSE: {rmse}, MAPE: {mape}")

        # Log metrics in MLflow
        with mlflow.start_run(run_name=f"{model_name}_Test_Evaluation"):
            mlflow.log_metric("test_rmse", rmse)
            mlflow.log_metric("test_mape", mape)
            mlflow.log_metric("test_r2", r2)
        
        return {"model_name": model_name, "rmse": rmse, "mape": mape}

    def select_and_save_best_model(self, X_test, y_test):
        """
        Evaluates all models, selects the best model based on RMSE, and saves it.

        Parameters:
        - X_test: Test features as a numpy array.
        - y_test: Test target values as a numpy array.
        """
        # Store each model's test metrics
        evaluation_results = []

        # Evaluate each model (assuming each model has been trained and stored in self.models)
        for model_name, model in self.models.items():
            results = self.evaluate_model(model, X_test, y_test, model_name)
            evaluation_results.append(results)

        # Select the model with the lowest RMSE
        best_model_info = min(evaluation_results, key=lambda x: x["rmse"])
        best_model = self.models[best_model_info["model_name"]]

        print(f"Selected Best Model: {best_model_info['model_name']} with RMSE: {best_model_info['rmse']}")

        # Save the best model
        with open(f"{best_model_info['model_name']}_best_model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        mlflow.log_artifact(f"{best_model_info['model_name']}_best_model.pkl")

        return best_model_info
    
    def train_all_models_and_select_best(self, feature_names, X_test, y_test):
        """
        Trains all specified models, logs results, and selects the best model based on RMSE.

        Parameters:
        - feature_names: List of feature names corresponding to model input (for feature importance logging).
        - X_test: Test features as a numpy array.
        - y_test: Test target values as a numpy array.

        Returns:
        - dict: Information about the best model including its name, metrics, and path to the saved model.
        """
        # Train each model and store it in self.models
        self.models["DecisionTree"] = DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=42)
        self.models["LinearRegression"] = LinearRegression()
        self.models["RidgeRegression"] = Ridge(alpha=1.0)
        self.models["LassoRegression"] = Lasso(alpha=0.1)
        self.models["LightGBM"] = lgb.LGBMRegressor(objective="regression", metric="rmse", learning_rate=0.1, max_depth=5, n_estimators=10, random_state=42)
        self.models["XGBoost"] = xgb.XGBRegressor(objective="reg:squarederror", learning_rate=0.1, max_depth=5, n_estimators=10, eval_metric="rmse", random_state=42)
        self.models["RandomForest"] = RandomForestRegressor(n_estimators=10, max_depth=5, min_samples_split=5, random_state=42)

        # Train and log each model
        for model_name, model in self.models.items():
            self._cross_validate(model, model_name, params=model.get_params() if hasattr(model, 'get_params') else {})
            if model_name in ["LightGBM", "XGBoost", "RandomForest"]:
                self._log_and_plot_feature_importance(model, model_name, feature_names)

        # Select and save the best model
        best_model_info = self.select_and_save_best_model(X_test, y_test)
        return best_model_info