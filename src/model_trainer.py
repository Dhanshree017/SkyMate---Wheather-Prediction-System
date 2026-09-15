import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_pipeline import WeatherDataPipeline


class WeatherModelTrainer:
    def __init__(self, artifact_dir: str = "saved_models"):
        self.artifact_dir = artifact_dir
        self.models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "RidgeRegression": Ridge(alpha=1.0)
        }
        self.best_model = None
        self.best_model_name = ""
        self.best_score = -float("inf")

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Trains multiple regression models and prints performance benchmarks along with accuracies."""
        results = {}

        print("\n" + "=" * 50)
        print("STARTING MODEL TRAINING & EVALUATION")
        print("=" * 50)

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)

            # Accuracy (R² score) on training data
            train_acc = model.score(X_train, y_train) * 100

            # Predict on test set
            predictions = model.predict(X_test)

            # Accuracy (R² score) on testing data
            test_acc = model.score(X_test, y_test) * 100

            # Compute evaluation metrics
            r2 = r2_score(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(y_test, predictions)

            results[name] = {
                "Train_Accuracy": train_acc,
                "Test_Accuracy": test_acc,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae
            }

            print(f"  • Training Accuracy : {train_acc:.2f}%")
            print(f"  • Testing Accuracy  : {test_acc:.2f}%")
            print(f"  • R² Score          : {r2:.4f}")
            print(f"  • RMSE              : {rmse:.4f}")
            print(f"  • MAE               : {mae:.4f}")

            # Track top performer based on R2 Score / Test Accuracy
            if r2 > self.best_score:
                self.best_score = r2
                self.best_model = model
                self.best_model_name = name

        print("\n" + "=" * 50)
        print(f"🏆 BEST MODEL SELECTED: {self.best_model_name} (Test Accuracy = {self.best_score * 100:.2f}%)")
        print("=" * 50)

        return results

    def save_best_model(self):
        """Exports the winning model artifact to disk."""
        if self.best_model is None:
            raise ValueError("No model trained yet! Run train_and_evaluate first.")

        os.makedirs(self.artifact_dir, exist_ok=True)
        model_path = os.path.join(self.artifact_dir, "model.pkl")
        joblib.dump(self.best_model, model_path)
        print(f"\nSaved best model ({self.best_model_name}) to '{model_path}'")


if __name__ == "__main__":
    # Resolve project root dataset path
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")

    # 1. Run Data Pipeline to get engineered train/test splits
    pipeline = WeatherDataPipeline(dataset_dir=dataset_path)
    df_raw = pipeline.load_and_merge_data()
    df_engineered = pipeline.engineer_features(df_raw)
    X_train, X_test, y_train, y_test = pipeline.encode_and_prepare(df_engineered)

    # Save preprocessing metadata
    pipeline.save_artifacts()

    # 2. Train Models and Save the Best Artifact
    trainer = WeatherModelTrainer()
    trainer.train_and_evaluate(X_train, X_test, y_train, y_test)
    trainer.save_best_model()