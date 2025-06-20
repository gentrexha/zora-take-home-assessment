from pathlib import Path
from typing import Any, Tuple

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import typer

from ztha.config import CONFIG

MODELS_DIR = Path(CONFIG.artifacts_path)
PROCESSED_DATA_DIR = Path(CONFIG.data.raw_data_path).parent / "processed"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

app = typer.Typer()


class ModelTrainer:
    """Handles model training for churn prediction."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None

    def prepare_data(
        self, features: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """Prepare features and target for model training."""
        logger.info("Preparing training data...")

        # Separate features and target
        X = features.drop(["is_churned"], axis=1)
        y = features["is_churned"]

        # Store feature names for interpretation
        self.feature_names = X.columns.tolist()
        logger.debug(f"Feature names: {self.feature_names[:5]}... (showing first 5)")

        # Train/test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=CONFIG.model.test_size,
            random_state=CONFIG.model.random_state,
            stratify=y,
        )
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)

        # Scale features
        logger.debug("Scaling features with StandardScaler...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Log data preparation summary
        data_summary = {
            "training_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "total_features": X_train.shape[1],
            "training_churn_rate": f"{y_train.mean():.2%}",
            "test_churn_rate": f"{y_test.mean():.2%}",
            "test_size_ratio": CONFIG.model.test_size,
        }
        logger.info(f"Data Preparation Summary: {data_summary}")
        logger.success("Data preparation completed successfully")

        return X_train_scaled, X_test_scaled, y_train, y_test  # type: ignore

    def train_model(self, X_train: np.ndarray, y_train: pd.Series | np.ndarray) -> Any:
        """Train the specified model type."""
        logger.info(f"Training {CONFIG.model.model_type} model...")

        if CONFIG.model.model_type == "gradient_boosting":
            model_params = {
                "n_estimators": CONFIG.model.gb_n_estimators,
                "learning_rate": CONFIG.model.gb_learning_rate,
                "max_depth": CONFIG.model.gb_max_depth,
                "random_state": CONFIG.model.random_state,
                "validation_fraction": CONFIG.model.gb_validation_fraction,
                "n_iter_no_change": CONFIG.model.gb_n_iter_no_change,
            }

            self.model = GradientBoostingClassifier(**model_params)

        elif CONFIG.model.model_type == "logistic_regression":
            model_params = {
                "max_iter": CONFIG.model.lr_max_iter,
                "C": CONFIG.model.lr_C,
                "random_state": CONFIG.model.random_state,
            }

            self.model = LogisticRegression(**model_params)
        else:
            error_msg = f"Unsupported model type: {CONFIG.model.model_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.debug(f"Starting model training with {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)

        logger.success("Model training completed successfully!")

        return self.model

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the trained model."""
        if self.model is None:
            raise ValueError("Model must be trained first")

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_  # type: ignore
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_[0])  # type: ignore
        else:
            raise ValueError("Model does not support feature importance")

        feature_importance = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        return feature_importance


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = MODELS_DIR / "churn_model.pkl",
):
    """
    Main function to train the churn prediction model.

    Args:
        features_path (Path): Path to the features CSV file.
        model_path (Path): Path to save the trained model.
    """
    logger.info("Starting model training process...")

    # Load data
    logger.info(f"Loading features from {features_path}")
    features = pd.read_csv(features_path, index_col="wallet_address")

    # Train model
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.prepare_data(features)
    model = trainer.train_model(X_train, y_train)

    # Save artifacts
    logger.info(f"Saving model to {model_path}")
    joblib.dump(model, model_path)

    scaler_path = model_path.parent / "feature_scaler.pkl"
    logger.info(f"Saving scaler to {scaler_path}")
    joblib.dump(trainer.scaler, scaler_path)

    feature_names_path = model_path.parent / "feature_names.json"
    logger.info(f"Saving feature names to {feature_names_path}")
    pd.Series(trainer.feature_names).to_json(feature_names_path, indent=2)

    # Save feature importance
    feature_importance = trainer.get_feature_importance()
    feature_importance_path = model_path.parent / "feature_importance.csv"
    logger.info(f"Saving feature importance to {feature_importance_path}")
    feature_importance.to_csv(feature_importance_path, index=False)

    # Note: Evaluation is not part of this script.
    # It is handled by the evaluator component.
    logger.success("Model training complete.")


if __name__ == "__main__":
    app()
