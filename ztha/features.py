"""
Feature engineering for the churn prediction pipeline.
"""

from datetime import timedelta

import numpy as np
import pandas as pd

from ztha.config import CONFIG, log


class FeatureEngineer:
    """Handles feature engineering for churn prediction."""

    def __init__(self):
        self.reference_date = None

    def engineer_features(
        self, collectors: pd.DataFrame, activity: pd.DataFrame
    ) -> pd.DataFrame:
        """Create comprehensive feature set for churn prediction."""
        log.info("Engineering features...")
        log.log_config(CONFIG.features, "Feature Engineering")

        # Set reference date for temporal features
        self.reference_date = activity["date"].max()
        log.info(f"Reference date: {self.reference_date}")

        # Build feature set step by step
        log.debug("Building RFM features...")
        features = self._build_rfm_features(activity)

        log.debug("Adding behavioral features...")
        features = self._add_behavioral_features(features, activity)

        log.debug("Adding collector profile features...")
        features = self._add_collector_profile_features(features, collectors)

        log.debug("Adding derived features...")
        features = self._add_derived_features(features)

        log.debug("Cleaning features...")
        features = self._clean_features(features)

        # Log feature engineering summary
        feature_summary = {
            "total_features": features.shape[1] - 1,  # Exclude target variable
            "total_collectors": features.shape[0],
            "churn_rate": f"{features['is_churned'].mean():.2%}",
            "feature_categories": {
                "rfm_enabled": CONFIG.features.include_rfm_features,
                "behavioral_enabled": CONFIG.features.include_behavioral_features,
                "social_enabled": CONFIG.features.include_social_features,
                "demographic_enabled": CONFIG.features.include_demographic_features,
            },
        }
        log.log_metrics(feature_summary, "Feature Engineering")
        log.success("Feature engineering completed successfully")

        return features

    def _build_rfm_features(self, activity: pd.DataFrame) -> pd.DataFrame:
        """Build RFM (Recency, Frequency, Monetary) features."""
        if not CONFIG.features.include_rfm_features:
            return pd.DataFrame()

        # Aggregate activity by wallet
        wallet_activity = (
            activity.groupby("wallet_address")
            .agg(
                {
                    "date": ["min", "max", "count"],
                    "number_collected": ["sum", "mean", "std"],
                    "collection_address": "nunique",
                    "chain_name": "nunique",
                    "file_type": "nunique",
                    "creator": "nunique",
                    "commented": lambda x: x.sum(),
                }
            )
            .round(2)
        )

        # Flatten column names
        wallet_activity.columns = [
            f"{col[0]}_{col[1]}" for col in wallet_activity.columns
        ]
        wallet_activity = wallet_activity.rename(  # type: ignore
            columns={
                "date_min": "first_activity_date",
                "date_max": "last_activity_date",
                "date_count": "total_activities",
                "number_collected_sum": "total_collected",
                "number_collected_mean": "avg_collected_per_activity",
                "number_collected_std": "collected_volatility",
                "collection_address_nunique": "unique_collections",
                "chain_name_nunique": "unique_chains",
                "file_type_nunique": "unique_file_types",
                "creator_nunique": "unique_creators",
                "commented_<lambda>": "total_comments",
            }
        )

        # Calculate temporal RFM metrics (convert to numeric)
        wallet_activity["days_since_last_activity"] = (
            self.reference_date - wallet_activity["last_activity_date"]
        ).dt.days

        wallet_activity["days_active"] = (
            wallet_activity["last_activity_date"]
            - wallet_activity["first_activity_date"]
        ).dt.days + 1

        wallet_activity["activity_frequency"] = (
            wallet_activity["total_activities"] / wallet_activity["days_active"]
        )

        # Drop datetime columns to avoid scaling issues
        wallet_activity = wallet_activity.drop(
            ["first_activity_date", "last_activity_date"], axis=1
        )

        # Handle infinite/null values
        wallet_activity["activity_frequency"] = wallet_activity[
            "activity_frequency"
        ].replace([np.inf, -np.inf], np.nan)
        wallet_activity["collected_volatility"] = wallet_activity[
            "collected_volatility"
        ].fillna(0)

        return wallet_activity

    def _add_behavioral_features(
        self, features: pd.DataFrame, activity: pd.DataFrame
    ) -> pd.DataFrame:
        """Add behavioral pattern features."""
        if not CONFIG.features.include_behavioral_features:
            return features

        # Recent activity (configurable time window)
        assert self.reference_date is not None
        recent_date = self.reference_date - timedelta(
            days=CONFIG.features.recent_activity_days
        )
        recent_activity = (
            activity[activity["date"] >= recent_date]
            .groupby("wallet_address")
            .agg(
                {
                    "date": "count",
                    "number_collected": "sum",
                    "commented": lambda x: x.sum(),
                }
            )
            .rename(  # type: ignore
                columns={
                    "date": "recent_activities",
                    "number_collected": "recent_collected",
                    "commented": "recent_comments",
                }
            )
        )

        # Chain activity patterns
        chain_activity = (
            activity.groupby("wallet_address")["chain_name"]
            .apply(lambda x: x.value_counts().max() / len(x) if len(x) > 0 else 0)
            .rename("chain_concentration")  # type: ignore
        )

        # File type preferences
        file_type_activity = (
            activity.groupby("wallet_address")["file_type"]
            .apply(lambda x: x.value_counts().max() / len(x) if len(x) > 0 else 0)
            .rename("file_type_concentration")  # type: ignore
        )

        # Combine behavioral features
        behavioral_features = pd.concat(
            [recent_activity, chain_activity, file_type_activity], axis=1
        )
        behavioral_features = behavioral_features.fillna(0)

        return features.join(behavioral_features, how="left").fillna(0)

    def _add_collector_profile_features(
        self, features: pd.DataFrame, collectors: pd.DataFrame
    ) -> pd.DataFrame:
        """Add collector profile and demographic features."""
        collectors_clean = collectors.set_index("wallet_address")

        profile_features = pd.DataFrame(index=collectors_clean.index)

        if CONFIG.features.include_social_features:
            # Social engagement score
            profile_features["social_score"] = (
                collectors_clean["linked_farcaster"].astype(int)
                + collectors_clean["linked_instagram"].astype(int)
                + collectors_clean["linked_twitter"].astype(int)
            )

            # Individual social platform indicators
            profile_features["has_farcaster"] = collectors_clean[
                "linked_farcaster"
            ].astype(int)
            profile_features["has_instagram"] = collectors_clean[
                "linked_instagram"
            ].astype(int)
            profile_features["has_twitter"] = collectors_clean["linked_twitter"].astype(
                int
            )

        # Account characteristics
        profile_features["account_age_days"] = (
            self.reference_date - collectors_clean["account_created_at"]
        ).dt.days

        if CONFIG.features.include_demographic_features:
            # Referral source patterns
            profile_features["referral_organic"] = (
                collectors_clean["referral_source"]
                .isin(["twitter", "farcaster"])
                .astype(int)
            )
            profile_features["referral_discord"] = (
                collectors_clean["referral_source"] == "discord"
            ).astype(int)

            # Geographic indicators
            profile_features["is_us"] = (collectors_clean["country"] == "USA").astype(
                int
            )
            profile_features["is_europe"] = (
                collectors_clean["country"]
                .isin(["UK", "Germany", "France"])
                .astype(int)
            )

        # Add target variable
        profile_features["is_churned"] = collectors_clean["is_churned"]

        return features.join(profile_features, how="inner")

    def _add_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add derived and interaction features."""
        # Engagement ratios
        features["comment_rate"] = (
            features["total_comments"] / features["total_activities"]
        )
        features["collection_diversity"] = (
            features["unique_collections"] / features["total_activities"]
        )
        features["creator_diversity"] = (
            features["unique_creators"] / features["total_activities"]
        )
        features["avg_collected_per_day"] = (
            features["total_collected"] / features["days_active"]
        )

        # Recent vs historical activity ratios
        features["recent_activity_ratio"] = (
            features["recent_activities"] / features["total_activities"]
        )
        features["recent_collection_ratio"] = (
            features["recent_collected"] / features["total_collected"]
        )

        # Value segmentation features
        features["is_whale"] = (
            features["total_collected"]
            >= features["total_collected"].quantile(CONFIG.features.whale_percentile)
        ).astype(int)

        # Early adopter identification
        features["is_early_adopter"] = (
            (
                features["creator_diversity"]
                >= features["creator_diversity"].quantile(
                    CONFIG.features.early_adopter_creator_percentile
                )
            )
            & (
                features["account_age_days"]
                >= features["account_age_days"].quantile(
                    CONFIG.features.early_adopter_account_age_percentile
                )
            )
        ).astype(int)

        # Activity intensity features
        features["activity_intensity"] = (
            features["total_activities"] / features["account_age_days"]
        )
        features["collection_intensity"] = (
            features["total_collected"] / features["account_age_days"]
        )

        return features

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare final feature set."""
        # Replace infinite values with NaN, then fill with appropriate values
        features = features.replace([np.inf, -np.inf], np.nan)

        # Fill NaN values with median for numeric columns (except target)
        numeric_columns = features.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != "is_churned"]

        for col in numeric_columns:
            features[col] = features[col].fillna(features[col].median())

        return features
