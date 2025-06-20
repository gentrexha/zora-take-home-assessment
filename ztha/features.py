"""
Feature engineering for the churn prediction pipeline.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.feature_selection import SelectKBest, f_classif

from ztha.config import CONFIG, log


class FeatureEngineer:
    """Handles feature engineering for churn prediction."""

    def __init__(self):
        self.reference_date = None
        self.collectors_df = None
        self.activity_df = None
        self.selected_features = None

    def engineer_features(
        self, collectors: pd.DataFrame, activity: pd.DataFrame
    ) -> pd.DataFrame:
        """Create comprehensive feature set for churn prediction."""
        log.info("Engineering features...")
        log.log_config(CONFIG.features, "Feature Engineering")

        # Set reference date and dataframes
        self.reference_date = activity["date"].max()
        log.info(f"Reference date: {self.reference_date}")
        self.collectors_df = collectors.set_index("wallet_address")
        self.activity_df = activity

        # Start with the base collectors
        features = pd.DataFrame(index=self.collectors_df.index)

        # Build feature set step-by-step
        log.debug("Adding profile features...")
        features = self._add_profile_features(features)

        log.debug("Adding activity summary features...")
        features = self._add_activity_summary_features(features)

        log.debug("Adding temporal pattern features...")
        features = self._add_temporal_pattern_features(features)

        log.debug("Adding value proxy features...")
        features = self._add_value_proxy_features(features)

        log.debug("Adding decline and recency features...")
        features = self._add_decline_and_recency_features(features)

        log.debug("Adding risk indicator features...")
        features = self._add_risk_indicator_features(features)

        log.debug("Adding data quality features...")
        features = self._add_data_quality_features(features)

        log.debug("Adding derived and interaction features...")
        features = self._add_derived_features(features)

        log.debug("Encoding categorical features...")
        features = self._encode_categorical_features(features)

        log.debug("Cleaning final features...")
        features = self._clean_features(features)

        # Add target variable
        features["is_churned"] = self.collectors_df["is_churned"]

        # Apply feature selection if requested
        if CONFIG.features.apply_feature_selection:
            log.info(
                f"Applying feature selection to reduce from {len(features.columns) - 1} to {CONFIG.features.max_features} features..."
            )
            features = self._apply_feature_selection(
                features, CONFIG.features.max_features
            )

        log.success("Feature engineering completed successfully")
        return features

    def _apply_feature_selection(
        self, features: pd.DataFrame, n_features: int = 25
    ) -> pd.DataFrame:
        """Apply multiple feature selection techniques to reduce dimensionality."""
        X = features.drop(["is_churned"], axis=1)
        y = features["is_churned"]

        # Handle any remaining NaN values
        X = X.fillna(0)

        # Method 1: Remove low-variance features (constant or nearly constant)
        variances = X.var()
        high_variance_mask = variances > CONFIG.features.variance_threshold
        high_variance_features = X.columns[high_variance_mask].tolist()
        log.info(
            f"Removed {len(X.columns) - len(high_variance_features)} low-variance features"
        )

        X_filtered = X[high_variance_features]

        # Method 2: Correlation-based filtering (remove highly correlated features)
        corr_matrix = X_filtered.corr().abs()  # type: ignore
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr_features = [
            column
            for column in upper_triangle.columns
            if (upper_triangle[column] > CONFIG.features.correlation_threshold).any()
        ]

        X_filtered = X_filtered.drop(columns=high_corr_features)
        log.info(f"Removed {len(high_corr_features)} highly correlated features")

        # Method 3: If still too many features, apply univariate selection
        if len(X_filtered.columns) > n_features:
            # Use f_classif for univariate feature selection
            selector = SelectKBest(
                score_func=f_classif, k=min(n_features, len(X_filtered.columns))
            )
            selector.fit(X_filtered, y)
            selected_feature_names = X_filtered.columns[selector.get_support()].tolist()

            log.info(
                f"Univariate selection kept {len(selected_feature_names)} features"
            )
        else:
            selected_feature_names = X_filtered.columns.tolist()

        # Store selected features for future reference
        self.selected_features = selected_feature_names

        # Create final feature set
        final_features = features[selected_feature_names + ["is_churned"]].copy()

        log.info(
            f"Feature selection complete: {len(final_features.columns) - 1} features selected"
        )
        log.info(f"Top selected features: {selected_feature_names[:10]}")

        return pd.DataFrame(final_features)

    def _add_profile_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Adds basic profile features from the collectors table."""
        assert self.collectors_df is not None
        assert self.reference_date is not None
        # Account Characteristics
        features["account_age_days"] = (
            self.reference_date - self.collectors_df["account_created_at"]
        ).dt.days
        features["has_username"] = self.collectors_df["username"].notna().astype(int)
        features["has_display_name"] = (
            self.collectors_df["display_name"].notna().astype(int)
        )
        features["has_bio"] = self.collectors_df["bio"].notna().astype(int)
        features["bio_length"] = self.collectors_df["bio"].str.len().fillna(0)

        # Social Media Integration
        social_cols = ["linked_farcaster", "linked_instagram", "linked_twitter"]
        for col in social_cols:
            features[col] = self.collectors_df[col].astype(int)
        features["total_social_links"] = self.collectors_df[social_cols].sum(axis=1)

        # Geographic & Acquisition (to be one-hot encoded later)
        features["country"] = self.collectors_df["country"]
        features["referral_source"] = self.collectors_df["referral_source"]

        return features

    def _add_activity_summary_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Adds features summarizing collection activity."""
        assert self.activity_df is not None
        # Aggregate activity by wallet
        agg_cols = {
            "date": ["count", "min", "max"],
            "number_collected": ["sum", "mean", "std", "max"],
            "collection_address": "nunique",
            "chain_name": "nunique",
            "file_type": "nunique",
            "creator": "nunique",
            "commented": "sum",
        }
        wallet_activity = self.activity_df.groupby("wallet_address").agg(agg_cols)
        wallet_activity.columns = ["_".join(col) for col in wallet_activity.columns]
        wallet_activity = wallet_activity.rename(  # type: ignore
            columns={
                "date_count": "total_collections",
                "date_min": "first_collection_date",
                "date_max": "last_collection_date",
                "number_collected_sum": "total_number_collected",
                "number_collected_mean": "avg_collection_size",
                "number_collected_std": "collection_size_variance",
                "number_collected_max": "max_single_collection",
                "collection_address_nunique": "unique_collections",
                "chain_name_nunique": "unique_chains",
                "file_type_nunique": "unique_file_types",
                "creator_nunique": "unique_creators",
                "commented_sum": "total_comments",
            }
        )

        # Collection Diversity
        wallet_activity["collection_diversity_ratio"] = (
            wallet_activity["unique_collections"] / wallet_activity["total_collections"]
        )
        wallet_activity["creator_loyalty_ratio"] = self.activity_df.groupby(
            "wallet_address"
        )["creator"].apply(
            lambda x: x.value_counts().max() / len(x) if len(x) > 0 else 0
        )
        wallet_activity["chain_concentration_ratio"] = self.activity_df.groupby(
            "wallet_address"
        )["chain_name"].apply(
            lambda x: x.value_counts().max() / len(x) if len(x) > 0 else 0
        )

        # Chain & Media Type Counts (One-hot encoded style)
        chain_counts = pd.crosstab(
            self.activity_df["wallet_address"], self.activity_df["chain_name"]
        )
        media_counts = pd.crosstab(
            self.activity_df["wallet_address"], self.activity_df["file_type"]
        )
        wallet_activity = wallet_activity.join(chain_counts, how="left").join(
            media_counts, how="left"
        )

        # Primary chain and media type
        if not chain_counts.empty:
            wallet_activity["primary_chain"] = chain_counts.idxmax(axis=1)
        if not media_counts.empty:
            wallet_activity["primary_media_type"] = media_counts.idxmax(axis=1)

        # Media type diversity
        wallet_activity["media_type_diversity_score"] = self.activity_df.groupby(
            "wallet_address"
        )["file_type"].apply(
            lambda x: entropy(x.value_counts(normalize=True)) if len(x) > 0 else 0
        )

        return features.join(wallet_activity, how="left")  # type: ignore

    def _add_temporal_pattern_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Adds features related to the timing and patterns of activity."""
        assert self.reference_date is not None
        assert self.activity_df is not None
        assert self.collectors_df is not None
        # Recency and Tenure
        features["days_since_first_collection"] = (
            self.reference_date - features["first_collection_date"]
        ).dt.days
        features["days_since_last_collection"] = (
            self.reference_date - features["last_collection_date"]
        ).dt.days

        # Activity Window
        features["days_active"] = (
            features["last_collection_date"] - features["first_collection_date"]
        ).dt.days + 1

        # Activity in last N days
        for days in [30, 60, 90]:
            recent_date = self.reference_date - timedelta(days=days)
            recent_activity = (
                self.activity_df[self.activity_df["date"] >= recent_date]
                .groupby("wallet_address")["date"]
                .count()
            )
            features[f"collections_last_{days}_days"] = recent_activity

        # Activity Regularity
        sorted_activity = self.activity_df.sort_values(["wallet_address", "date"])
        time_diffs = sorted_activity.groupby("wallet_address")["date"].diff().dt.days
        features["time_between_collections_avg"] = time_diffs.groupby(
            self.activity_df["wallet_address"]
        ).mean()
        features["time_between_collections_std"] = time_diffs.groupby(
            self.activity_df["wallet_address"]
        ).std()
        features["collection_gaps_count_30d"] = (
            time_diffs[time_diffs > 30]
            .groupby(self.activity_df["wallet_address"])
            .count()
        )
        features["dormancy_periods_60d"] = (
            time_diffs[time_diffs > 60]
            .groupby(self.activity_df["wallet_address"])
            .count()
        )

        # Collection Burst Frequency
        daily_counts = (
            self.activity_df.groupby(["wallet_address", "date"]).size().reset_index()
        )
        daily_counts.columns = ["wallet_address", "date", "counts"]
        burst_days = (
            daily_counts[daily_counts["counts"] > 1]
            .groupby("wallet_address")["date"]
            .count()
        )
        features["collection_burst_frequency"] = burst_days

        # Weekend vs Weekday
        features["weekend_activity_ratio"] = self.activity_df.groupby("wallet_address")[
            "date"
        ].apply(lambda x: (pd.to_datetime(x).dt.dayofweek >= 5).mean())

        # Early Adopter Score
        thirty_days_after_creation = pd.to_datetime(
            self.collectors_df["account_created_at"]
        ) + timedelta(days=30)
        early_activity = self.activity_df.join(
            thirty_days_after_creation.rename("deadline"), on="wallet_address"
        )
        early_collections = (
            early_activity[early_activity["date"] <= early_activity["deadline"]]
            .groupby("wallet_address")["date"]
            .count()
        )
        features["early_adopter_score"] = early_collections

        return features

    def _add_value_proxy_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Adds features related to the economic value of collections."""
        assert self.activity_df is not None
        assert self.collectors_df is not None
        # High Volume Collections
        features["high_volume_collections"] = (
            self.activity_df[self.activity_df["number_collected"] > 100]
            .groupby("wallet_address")["date"]
            .count()
        )

        # Whale Behavior
        features["is_whale"] = (
            features["total_number_collected"]
            >= features["total_number_collected"].quantile(
                CONFIG.features.whale_percentile
            )
        ).astype(int)

        # Top 10% Collection Size
        features["top_10_pct_collection_size"] = self.activity_df.groupby(
            "wallet_address"
        )["number_collected"].apply(lambda x: x.quantile(0.9) if not x.empty else 0)

        # Collection Size Percentiles
        for p in [0.25, 0.50, 0.75, 0.90]:
            features[f"collection_size_p{int(p * 100)}"] = self.activity_df.groupby(
                "wallet_address"
            )["number_collected"].quantile(p)

        return features

    def _add_decline_and_recency_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Adds features that capture declining activity and recency-based risk."""
        assert self.activity_df is not None
        assert self.reference_date is not None

        # Dormancy flags based on days_since_last_collection
        for days in [14, 30, 60]:
            features[f"is_dormant_{days}d"] = (
                features["days_since_last_collection"] > days
            ).astype(int)

        # Activity decline (30d vs 31-60d)
        collections_30d = features["collections_last_30_days"]
        collections_60d = features["collections_last_60_days"]
        collections_31_to_60d = collections_60d - collections_30d
        with np.errstate(divide="ignore", invalid="ignore"):
            features["activity_decline_30_60_days"] = (
                (collections_30d - collections_31_to_60d) / collections_31_to_60d
            ).replace([np.inf, -np.inf], 0)

        # Longest recent gap in last 90 days
        recent_activity_90d = self.activity_df[
            self.activity_df["date"] >= self.reference_date - timedelta(days=90)
        ]
        sorted_recent_activity = recent_activity_90d.sort_values(  # type: ignore
            ["wallet_address", "date"]
        )
        recent_gaps = (
            sorted_recent_activity.groupby("wallet_address")["date"].diff().dt.days
        )
        features["longest_recent_gap_90d"] = recent_gaps.groupby(
            self.activity_df["wallet_address"]
        ).max()

        # Recent vs historical activity ratio
        historical_daily_avg = (
            features["total_collections"] / features["days_active"]
        ).replace([np.inf, -np.inf], 0)
        recent_daily_avg = (features["collections_last_30_days"] / 30).replace(
            [np.inf, -np.inf], 0
        )
        features["recent_vs_historical_ratio"] = (
            recent_daily_avg / historical_daily_avg
        ).replace([np.inf, -np.inf], 0)

        return features

    def _add_risk_indicator_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Adds features that are direct signals of churn risk."""
        assert self.activity_df is not None
        # Declining Activity Trend (simple version)
        activity_30d = features["collections_last_30_days"]
        activity_30_to_60d = features["collections_last_60_days"] - activity_30d
        features["declining_activity_trend"] = (
            activity_30d < activity_30_to_60d
        ).astype(int)

        # Low Engagement Flag
        comment_rate = features["total_comments"] / features["total_collections"]
        features["low_engagement_flag"] = (
            comment_rate < comment_rate.quantile(0.25)
        ).astype(int)

        # Other Risk Flags
        features["single_chain_risk"] = (features["unique_chains"] == 1).astype(int)
        features["no_social_links_risk"] = (features["total_social_links"] == 0).astype(
            int
        )
        features["long_gap_since_last_collection"] = (
            features["days_since_last_collection"] > 60
        ).astype(int)
        features["meaningful_collection_ratio"] = (
            self.activity_df[self.activity_df["number_collected"] > 1]
            .groupby("wallet_address")
            .size()
            / features["total_collections"]
        )

        return features

    def _add_data_quality_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Adds features about the completeness of the data for each user."""
        assert self.activity_df is not None
        missing_counts = (
            self.activity_df.isna().groupby(self.activity_df["wallet_address"]).sum()
        )
        features["missing_dates_count"] = missing_counts["date"]
        features["missing_token_ids_count"] = missing_counts["token_id"]
        features["data_completeness_score"] = 1 - (
            missing_counts.sum(axis=1)
            / (len(self.activity_df.columns) * features["total_collections"])
        )
        return features

    def _add_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add derived and interaction features."""
        assert self.collectors_df is not None
        # Ratios (handle division by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            features["collection_frequency_per_month"] = features[
                "total_collections"
            ] / (features["account_age_days"] / 30)
            features["comment_rate"] = (
                features["total_comments"] / features["total_collections"]
            )
            features["avg_collected_per_day_active"] = (
                features["total_number_collected"] / features["days_active"]
            )
            features["recent_activity_ratio"] = (
                features["collections_last_30_days"] / features["total_collections"]
            )
            features["multi_chain_user"] = (features["unique_chains"] > 1).astype(int)

        return features

    def _encode_categorical_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Handles one-hot encoding for categorical features."""
        assert self.collectors_df is not None
        # For high cardinality, group rare categories into 'Other'
        for col in ["country", "referral_source"]:
            if col in features.columns:
                top_cats = features[col].value_counts().nlargest(5).index.tolist()
                features[col] = features[col].where(
                    features[col].isin(top_cats), "Other"
                )

        # Apply one-hot encoding
        categorical_cols = [
            "country",
            "referral_source",
            "primary_chain",
            "primary_media_type",
        ]
        # Filter out columns that might not exist if activity is empty
        cols_to_encode = [col for col in categorical_cols if col in features.columns]
        features = pd.get_dummies(
            features, columns=cols_to_encode, prefix=cols_to_encode, dummy_na=True
        )

        # Re-align index with the original collectors DataFrame to ensure consistency
        # This adds wallets that might not have had any activity and removes wallets
        # that might have been in activity logs but not in the collectors table.
        assert self.collectors_df is not None
        features = features.reindex(self.collectors_df.index.tolist())

        return features

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare final feature set."""
        # Drop temporary columns
        features = features.drop(
            columns=["first_collection_date", "last_collection_date"], errors="ignore"
        )

        # Replace infinite values with NaN
        features = features.replace([np.inf, -np.inf], np.nan)

        # Fill NaNs. This can be improved with more sophisticated imputation.
        for col in features.columns:
            # Convert boolean columns created by get_dummies to int
            if features[col].dtype == "bool":
                features[col] = features[col].astype(int)

            if features[col].dtype in [np.number, "float64", "int64"]:
                features[col] = features[col].fillna(0)  # Simple fill with 0 for now

        return features
