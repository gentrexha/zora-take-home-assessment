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
    """Test a few key profile features that should survive feature selection."""
    # These basic features should generally survive feature selection
    profile_features = ["has_username", "has_bio", "total_social_links"]
    found_features = [f for f in profile_features if f in engineered_features.columns]

    # At least some profile features should be present
    assert len(found_features) > 0, (
        f"Expected some profile features, found: {list(engineered_features.columns)}"
    )

    # Check that the features have reasonable values
    for feature in found_features:
        assert engineered_features[feature].notna().all(), (
            f"Feature {feature} should not have NaN values"
        )


def test_activity_summary_features(engineered_features: pd.DataFrame):
    """Test a few key activity summary features that should survive feature selection."""
    # These features are often important and should survive selection
    activity_features = [
        "total_number_collected",
        "collection_size_variance",
        "unique_chains",
    ]
    found_features = [f for f in activity_features if f in engineered_features.columns]

    # At least some activity features should be present
    assert len(found_features) > 0, (
        f"Expected some activity features, found: {list(engineered_features.columns)}"
    )

    # Check that numeric features are non-negative
    for feature in found_features:
        if "variance" not in feature:  # variance can be 0
            assert (engineered_features[feature] >= 0).all(), (
                f"Feature {feature} should be non-negative"
            )


def test_temporal_features(engineered_features: pd.DataFrame):
    """Test that some temporal-related features are present."""
    # Look for any temporal-related features that might survive selection
    temporal_keywords = ["days", "age", "last", "recent", "gap"]
    temporal_features = [
        col
        for col in engineered_features.columns
        if any(keyword in col.lower() for keyword in temporal_keywords)
    ]

    # At least some temporal features should be present
    assert len(temporal_features) > 0, (
        f"Expected some temporal features, found temporal-related: {temporal_features}"
    )

    # Check that date-based features have reasonable ranges
    for feature in temporal_features:
        if "days" in feature.lower():
            # Days should be non-negative and reasonable (not more than 10 years)
            assert (engineered_features[feature] >= 0).all(), (
                f"Feature {feature} should be non-negative"
            )
            assert (engineered_features[feature] <= 3650).all(), (
                f"Feature {feature} should be reasonable (< 10 years)"
            )
