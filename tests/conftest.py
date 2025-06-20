"""
Configuration for pytest, including shared fixtures.
"""

from pathlib import Path

import pandas as pd
import pytest

from ztha.dataset import DataLoader
from ztha.features import FeatureEngineer


@pytest.fixture
def data_loader(monkeypatch) -> DataLoader:
    """
    Fixture to provide a DataLoader pointing to test fixture data.
    """
    fixture_path = Path(__file__).parent / "fixtures"
    monkeypatch.setattr("ztha.config.CONFIG.data.raw_data_path", str(fixture_path))
    monkeypatch.setattr(
        "ztha.config.CONFIG.data.collectors_file", "sample_collectors.csv"
    )
    monkeypatch.setattr("ztha.config.CONFIG.data.activity_file", "sample_activity.csv")

    # Patch pd.to_datetime to handle empty values and timezone issues properly in tests
    original_to_datetime = pd.to_datetime

    def patched_to_datetime(arg, *args, **kwargs):
        # For our test data, always use errors='coerce' to handle NaN values
        if "errors" not in kwargs:
            kwargs["errors"] = "coerce"
        result = original_to_datetime(arg, *args, **kwargs)
        # Convert timezone-aware dates to timezone-naive for test consistency
        if hasattr(result, "dt") and result.dt.tz is not None:
            result = result.dt.tz_convert(None)
        elif hasattr(result, "tz") and result.tz is not None:
            result = result.tz_convert(None)
        return result

    monkeypatch.setattr("pandas.to_datetime", patched_to_datetime)

    # This ensures the DataLoader uses the patched config
    return DataLoader()


@pytest.fixture
def sample_data(data_loader: DataLoader) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load sample collectors and activity data."""
    return data_loader.load_data()


@pytest.fixture
def engineered_features(sample_data: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    """Run feature engineering on sample data."""
    collectors, activity = sample_data
    feature_engineer = FeatureEngineer()
    return feature_engineer.engineer_features(collectors, activity)
