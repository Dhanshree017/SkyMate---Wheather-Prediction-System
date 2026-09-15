import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


class WeatherDataPipeline:
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self.files = {
            "air_quality": "Air quality information.xlsx",
            "astronomical": "Astronomical.xlsx",
            "location": "Location information.xlsx",
            "weather": "Weather data.xlsx"
        }
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def load_and_merge_data(self) -> pd.DataFrame:
        """Loads all four Excel files and merges them on 'last_updated_epoch'."""
        dfs = []
        for key, filename in self.files.items():
            file_path = os.path.join(self.dataset_dir, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Missing expected dataset file: {file_path}")
            
            df = pd.read_excel(file_path)
            dfs.append(df)
            print(f"Loaded {filename} successfully with shape: {df.shape}")

        # Merge iteratively on primary key 'last_updated_epoch'
        merged_df = dfs[0]
        for next_df in dfs[1:]:
            # Drop duplicate columns except key if any exist across files
            duplicate_cols = [c for c in next_df.columns if c in merged_df.columns and c != 'last_updated_epoch']
            next_df_cleaned = next_df.drop(columns=duplicate_cols)
            merged_df = pd.merge(merged_df, next_df_cleaned, on='last_updated_epoch', how='inner')

        print(f"\nFinal Merged Master Dataframe Shape: {merged_df.shape}")
        return merged_df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs cleaning, datetime transformations, and feature engineering."""
        df = df.copy()

        # Handle Datetime Features
        if 'last_updated' in df.columns:
            df['datetime'] = pd.to_datetime(df['last_updated'])
        else:
            df['datetime'] = pd.to_datetime(df['last_updated_epoch'], unit='s')

        # Temporal/Cyclical Feature Extraction
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month

        # Sine/Cosine transformations for cyclical time attributes
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

        # Drop non-predictive metadata & duplicate target variations
        cols_to_drop = [
            'last_updated_epoch', 'last_updated', 'datetime', 
            'timezone', 'temperature_fahrenheit'
        ]
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        # Fill missing values using median for numeric and mode for categorical
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

        return df

    def encode_and_prepare(self, df: pd.DataFrame, target_col: str = 'temperature_celsius'):
        """Encodes categorical variables, splits into features/target, and scales numerical values."""
        df_processed = df.copy()

        # Categorical Encoding
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le

        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]

        # Save Feature Column Names for Inference Pipeline
        self.feature_names = list(X.columns)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Feature Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def save_artifacts(self, artifact_dir: str = "saved_models"):
        """Saves scaler, encoders, and feature names for prediction time."""
        os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(artifact_dir, "scaler.pkl"))
        joblib.dump(self.label_encoders, os.path.join(artifact_dir, "label_encoders.pkl"))
        joblib.dump(self.feature_names, os.path.join(artifact_dir, "feature_names.pkl"))
        print(f"Data Pipeline Artifacts successfully saved to '{artifact_dir}/'")


if __name__ == "__main__":
    # Test Data Pipeline execution locally
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
    pipeline = WeatherDataPipeline(dataset_dir=dataset_path)
    
    df_raw = pipeline.load_and_merge_data()
    df_engineered = pipeline.engineer_features(df_raw)
    X_train, X_test, y_train, y_test = pipeline.encode_and_prepare(df_engineered)
    pipeline.save_artifacts()
    
    print("\nData Pipeline Execution Complete!")
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")