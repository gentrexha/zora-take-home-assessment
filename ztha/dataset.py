"""
Data loading utilities for the churn prediction pipeline.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from ztha.config import CONFIG, log


class DataLoader:
    """Handles loading and basic preprocessing of collectors and activity data."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path or CONFIG.data.raw_data_path)

    def load_collectors(self) -> pd.DataFrame:
        """Load and preprocess collectors data."""
        log.info("Loading collectors data...")

        collectors = pd.read_csv(self.data_path / CONFIG.data.collectors_file)

        # Parse dates
        collectors["account_created_at"] = pd.to_datetime(
            collectors["account_created_at"]
        )
        collectors["churned_at_date"] = pd.to_datetime(collectors["churned_at_date"])

        # Create binary churn target
        collectors["is_churned"] = collectors["churned_at_date"].notna().astype(int)

        # Log data summary
        data_summary = {
            "total_collectors": len(collectors),
            "churn_rate": f"{collectors['is_churned'].mean():.2%}",
            "churned_count": collectors["is_churned"].sum(),
            "active_count": (collectors["is_churned"] == 0).sum(),
        }
        log.log_metrics(data_summary, "Collectors Data")

        return collectors

    def load_activity(self) -> pd.DataFrame:
        """Load and preprocess collection activity data with chunked processing."""
        log.info("Loading collection activity data...")
        log.debug(f"Using chunk size: {CONFIG.data.chunk_size}")

        activity_chunks = []
        chunk_count = 0

        for chunk in pd.read_csv(
            self.data_path / CONFIG.data.activity_file, chunksize=CONFIG.data.chunk_size
        ):
            # Basic cleaning
            chunk = chunk.dropna(subset=["wallet_address"])
            chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
            activity_chunks.append(chunk)
            chunk_count += 1
            log.debug(f"Processed chunk {chunk_count} with {len(chunk)} records")

        activity = pd.concat(activity_chunks, ignore_index=True)

        # Remove rows with missing dates
        initial_count = len(activity)
        activity = activity.dropna(subset=["date"])

        # Log activity data summary
        activity_summary = {
            "total_activities": len(activity),
            "chunks_processed": chunk_count,
            "date_range_start": str(activity["date"].min()),
            "date_range_end": str(activity["date"].max()),
            "unique_wallets": activity["wallet_address"].nunique(),
            "missing_dates_removed": initial_count - len(activity),
        }
        log.log_metrics(activity_summary, "Activity Data")

        return activity

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both collectors and activity data."""
        collectors = self.load_collectors()
        activity = self.load_activity()
        return collectors, activity
