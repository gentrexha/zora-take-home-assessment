"""
Tests for the model training module.
"""

import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

from ztha.modeling.train import ModelTrainer


@pytest.mark.parametrize("model_type", ["gradient_boosting", "logistic_regression"])
def test_train_and_evaluate_with_cv(
    engineered_features: pd.DataFrame, model_type: str, monkeypatch
):
    """
    Test the full cross-validation training and evaluation process.
    This test is parameterized to run for each supported model type.
    """
    # Use monkeypatch to set the model type for this test run
    monkeypatch.setattr("ztha.config.CONFIG.model.model_type", model_type)

    # Ensure there's enough data for 5-fold split
    assert engineered_features["is_churned"].value_counts().min() >= 5, (
        "Not enough samples in the minority class for 5-fold CV"
    )

    trainer = ModelTrainer()
    model, scaler, cv_results = trainer.train_and_evaluate_with_cv(engineered_features)

    # 1. Check return types
    assert isinstance(model, BaseEstimator)
    assert isinstance(scaler, StandardScaler)
    assert isinstance(cv_results, dict)

    # 2. Check content of the results dictionary
    expected_keys = [
        "evaluation_type",
        "model_type",
        "recall_at_top_k_percent",
        "auroc",
        "classification_report",
        "feature_importance",
        "auroc_std",
    ]
    for key in expected_keys:
        assert key in cv_results, f"Key '{key}' missing from cv_results"

    # 3. Check specific result values
    assert cv_results["evaluation_type"] == "cross_validation"
    assert isinstance(cv_results["auroc"], float)
    assert 0.0 <= cv_results["auroc"] <= 1.0
    assert isinstance(cv_results["recall_at_top_k_percent"], float)
    assert 0.0 <= cv_results["recall_at_top_k_percent"] <= 1.0
    # Model type is stored as the class name, not the config string
    expected_class_name = (
        "GradientBoostingClassifier"
        if model_type == "gradient_boosting"
        else "LogisticRegression"
    )
    assert cv_results["model_type"] == expected_class_name
    assert isinstance(cv_results["feature_importance"], pd.DataFrame)
    assert not cv_results["feature_importance"].empty
