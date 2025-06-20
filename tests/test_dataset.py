"""
Tests for the dataset loading module.
"""

import pandas as pd

from ztha.dataset import DataLoader


def test_load_collectors(data_loader: DataLoader):
    """Test loading and preprocessing of collectors data."""
    collectors_df = data_loader.load_collectors()
    assert isinstance(collectors_df, pd.DataFrame)
    assert not collectors_df.empty
    assert "is_churned" in collectors_df.columns
    assert pd.api.types.is_datetime64_any_dtype(collectors_df["account_created_at"])
    assert collectors_df["is_churned"].isin([0, 1]).all()
    assert collectors_df.shape[0] == 10


def test_load_activity(data_loader: DataLoader):
    """Test loading of collection activity data."""
    activity_df = data_loader.load_activity()
    assert isinstance(activity_df, pd.DataFrame)
    assert not activity_df.empty
    assert pd.api.types.is_datetime64_any_dtype(activity_df["date"])
    assert activity_df.shape[0] == 11


def test_load_data(data_loader: DataLoader):
    """Test loading both datasets."""
    collectors_df, activity_df = data_loader.load_data()
    assert isinstance(collectors_df, pd.DataFrame)
    assert isinstance(activity_df, pd.DataFrame)
    assert not collectors_df.empty
    assert not activity_df.empty
