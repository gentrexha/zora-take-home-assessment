"""
Configuration settings and logging setup for the churn prediction pipeline.
"""

from pathlib import Path
import random
import sys
from typing import Literal

from loguru import logger
import numpy as np
from pydantic import BaseModel, Field, field_validator


class LoggingConfig(BaseModel):
    """Logging configuration with Loguru."""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        description="Log message format",
    )
    colorize: bool = Field(default=True, description="Enable colored output")

    @field_validator("level")
    def validate_level(cls, v):
        valid_levels = [
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()


class DataConfig(BaseModel):
    """Data-related configuration."""

    raw_data_path: str = Field(default="data/raw", description="Path to raw data files")
    processed_path: str = Field(
        default="data/processed", description="Path to processed data files"
    )
    collectors_file: str = Field(
        default="collectors.csv", description="Collectors data filename"
    )
    activity_file: str = Field(
        default="collection_activity.csv", description="Activity data filename"
    )
    chunk_size: int = Field(
        default=50000, ge=1000, description="Chunk size for processing large files"
    )


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""

    recent_activity_days: int = Field(
        default=30, ge=1, le=365, description="Days to consider for recent activity"
    )
    whale_percentile: float = Field(
        default=0.8,
        ge=0.5,
        le=0.99,
        description="Percentile threshold for whale identification",
    )
    early_adopter_creator_percentile: float = Field(
        default=0.7,
        ge=0.5,
        le=0.99,
        description="Creator diversity threshold for early adopters",
    )
    early_adopter_account_age_percentile: float = Field(
        default=0.7,
        ge=0.5,
        le=0.99,
        description="Account age threshold for early adopters",
    )

    # Feature categories to include
    include_rfm_features: bool = Field(
        default=True, description="Include RFM (Recency, Frequency, Monetary) features"
    )
    include_behavioral_features: bool = Field(
        default=True, description="Include behavioral pattern features"
    )
    include_social_features: bool = Field(
        default=True, description="Include social media connection features"
    )
    include_demographic_features: bool = Field(
        default=True, description="Include demographic and geographic features"
    )

    # Feature selection configuration
    apply_feature_selection: bool = Field(
        default=True, description="Apply feature selection to reduce dimensionality"
    )
    max_features: int = Field(
        default=25, ge=5, le=100, description="Maximum number of features to select"
    )
    variance_threshold: float = Field(
        default=0.01,
        ge=0.0,
        le=0.1,
        description="Minimum variance threshold for feature selection",
    )
    correlation_threshold: float = Field(
        default=0.95,
        ge=0.8,
        le=0.99,
        description="Maximum correlation threshold for feature selection",
    )


class ModelConfig(BaseModel):
    """Model training configuration."""

    model_type: Literal["gradient_boosting", "logistic_regression"] = Field(
        default="logistic_regression", description="Type of model to train"
    )
    test_size: float = Field(
        default=0.2, ge=0.1, le=0.5, description="Proportion of data for testing"
    )
    random_state: int = Field(default=42, description="Random seed for reproducibility")

    # SMOTE configuration
    apply_smote: bool = Field(
        default=True, description="Apply SMOTE for training data balancing"
    )
    smote_sampling_strategy: str = Field(
        default="auto",
        description="SMOTE sampling strategy (auto, minority, majority, not minority, not majority, all)",
    )
    smote_k_neighbors: int = Field(
        default=5, ge=1, le=20, description="Number of neighbors for SMOTE"
    )

    # Gradient Boosting parameters
    gb_n_estimators: int = Field(
        default=100, ge=10, le=1000, description="Number of boosting stages"
    )
    gb_learning_rate: float = Field(
        default=0.1, ge=0.01, le=1.0, description="Learning rate for gradient boosting"
    )
    gb_max_depth: int = Field(
        default=6, ge=1, le=20, description="Maximum depth of trees"
    )
    gb_validation_fraction: float = Field(
        default=0.2,
        ge=0.1,
        le=0.5,
        description="Fraction of training data for validation",
    )
    gb_n_iter_no_change: int = Field(
        default=10, ge=5, le=50, description="Early stopping patience"
    )

    # Logistic Regression parameters
    lr_max_iter: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum iterations for logistic regression",
    )
    lr_C: float = Field(
        default=1.0,
        ge=0.001,
        le=1000.0,
        description="Regularization strength (inverse)",
    )


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    # Name of the business metric to be displayed in the report
    business_metric: str = Field(
        "Recall@Top10%", description="Primary metric for business reports"
    )
    # The percentile of collectors to target for recall calculation
    recall_at_k_percent: float = Field(
        0.1,
        description="Percentage for Recall@TopK metric (e.g., 0.1 for top 10%)",
    )
    save_predictions: bool = Field(
        default=True, description="Save individual predictions to CSV"
    )
    save_probabilities: bool = Field(
        default=True, description="Save prediction probabilities"
    )


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""

    artifacts_path: str = Field(
        default="artifacts", description="Directory to save model artifacts"
    )
    random_seed: int = Field(default=42, description="Global random seed")

    # Sub-configurations
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)  # type: ignore

    class Config:
        validate_assignment = True

    def model_post_init(self, __context):
        """Ensure artifacts directory exists and setup logging."""
        Path(self.artifacts_path).mkdir(exist_ok=True)
        self.setup_logging()

    def setup_logging(self):
        """Configure Loguru logging based on configuration."""
        # Remove default logger
        logger.remove()

        # Add configured logger
        logger.add(
            sys.stderr,
            level=self.logging.level,
            format=self.logging.format,
            colorize=self.logging.colorize,
        )

        # Add file logging for production
        log_file = Path(self.artifacts_path) / "pipeline.log"
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days",
            compression="zip",
        )


class ChurnLogger:
    """Centralized logging class for the churn prediction pipeline."""

    def __init__(self, name: str = "ChurnPipeline"):
        self.logger = logger.bind(name=name)

    def info(self, message: str, **kwargs):
        """Log info level message."""
        self.logger.info(message, **kwargs)

    def success(self, message: str, **kwargs):
        """Log success level message."""
        self.logger.success(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error level message."""
        self.logger.error(message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self.logger.debug(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical level message."""
        self.logger.critical(message, **kwargs)

    def log_metrics(self, metrics: dict, prefix: str = ""):
        """Log a dictionary of metrics in a structured way."""
        prefix_str = f"{prefix} " if prefix else ""
        for key, value in metrics.items():
            if isinstance(value, float):
                self.info(f"{prefix_str}{key}: {value:.4f}")
            else:
                self.info(f"{prefix_str}{key}: {value}")

    def log_config(self, config_section: BaseModel, section_name: str):
        """Log configuration section in a structured way."""
        self.info(f"=== {section_name.upper()} CONFIGURATION ===")
        for field_name, field_value in config_section.model_dump().items():
            self.info(f"{field_name}: {field_value}")


# Global config instance
CONFIG = PipelineConfig()

# Global logger instance
log = ChurnLogger()


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # Potentially add seeds for other libraries like torch, tensorflow if used.
