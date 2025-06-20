"""
Tests for the feature engineering module.
"""

from typing import Tuple

import numpy as np
import pandas as pd


def test_feature_engineering_output_shape_and_target(
    engineered_features: pd.DataFrame, sample_data: Tuple[pd.DataFrame, pd.DataFrame]
):
    """Test the output shape, index, and presence of the target variable."""
    collectors, _ = sample_data
    assert isinstance(engineered_features, pd.DataFrame)
    assert not engineered_features.empty
    # The output should have the same number of rows as the collectors input
    assert engineered_features.shape[0] == collectors.shape[0]
    # The target variable must be the last column
    assert engineered_features.columns[-1] == "is_churned"
    # Index should be wallet_address from original collectors df
    assert engineered_features.index.name == "wallet_address"
    assert engineered_features.index.equals(
        collectors.set_index("wallet_address").index
    )


def test_no_missing_or_infinite_values(engineered_features: pd.DataFrame):
    """Test that the final feature set contains no NaN or infinite values."""
    assert not engineered_features.isnull().sum().sum() > 0, (
        "Found NaN values in features"
    )
    assert (
        not np.isinf(engineered_features.select_dtypes(include=np.number)).any().any()
    ), "Found infinite values in features"


def test_profile_features(engineered_features: pd.DataFrame):
    """Test a few key profile features."""
    assert "account_age_days" in engineered_features.columns
    assert "total_social_links" in engineered_features.columns
    assert engineered_features["account_age_days"].min() >= 0
    assert engineered_features.loc["0x1", "total_social_links"] == 2
    assert engineered_features.loc["0x2", "total_social_links"] == 0


def test_activity_summary_features(engineered_features: pd.DataFrame):
    """Test a few key activity summary features."""
    assert "total_collections" in engineered_features.columns
    assert "unique_collections" in engineered_features.columns
    assert "primary_chain_ETH" in engineered_features.columns
    assert engineered_features.loc["0x1", "total_collections"] == 3
    assert engineered_features.loc["0x2", "unique_collections"] == 1
    assert engineered_features.loc["0x1", "primary_chain_ETH"] == 1
    assert engineered_features.loc["0x2", "primary_chain_OPTIMISM"] == 1


def test_temporal_features(engineered_features: pd.DataFrame):
    """Test a few key temporal features."""
    assert "days_since_last_collection" in engineered_features.columns
    assert "collections_last_30_days" in engineered_features.columns
    assert engineered_features["days_since_last_collection"].min() >= 0
