"""
Model training utilities for the churn prediction pipeline.
"""

from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ztha.config import CONFIG, log


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
        log.info("Preparing training data...")
        log.log_config(CONFIG.model, "Model Training")

        # Separate features and target
        X = features.drop(["is_churned"], axis=1)
        y = features["is_churned"]

        # Store feature names for interpretation
        self.feature_names = X.columns.tolist()
        log.debug(f"Feature names: {self.feature_names[:5]}... (showing first 5)")

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
        log.debug("Scaling features with StandardScaler...")
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
        log.log_metrics(data_summary, "Data Preparation")
        log.success("Data preparation completed successfully")

        return X_train_scaled, X_test_scaled, y_train, y_test  # type: ignore

    def train_model(self, X_train: np.ndarray, y_train: pd.Series | np.ndarray) -> Any:
        """Train the specified model type."""
        log.info(f"Training {CONFIG.model.model_type} model...")

        if CONFIG.model.model_type == "gradient_boosting":
            model_params = {
                "n_estimators": CONFIG.model.gb_n_estimators,
                "learning_rate": CONFIG.model.gb_learning_rate,
                "max_depth": CONFIG.model.gb_max_depth,
                "random_state": CONFIG.model.random_state,
                "validation_fraction": CONFIG.model.gb_validation_fraction,
                "n_iter_no_change": CONFIG.model.gb_n_iter_no_change,
            }
            log.log_metrics(model_params, "Gradient Boosting Parameters")

            self.model = GradientBoostingClassifier(**model_params)

        elif CONFIG.model.model_type == "logistic_regression":
            model_params = {
                "max_iter": CONFIG.model.lr_max_iter,
                "C": CONFIG.model.lr_C,
                "random_state": CONFIG.model.random_state,
            }
            log.log_metrics(model_params, "Logistic Regression Parameters")

            self.model = LogisticRegression(**model_params)
        else:
            error_msg = f"Unsupported model type: {CONFIG.model.model_type}"
            log.error(error_msg)
            raise ValueError(error_msg)

        log.debug(f"Starting model training with {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)

        log.success("Model training completed successfully!")

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
