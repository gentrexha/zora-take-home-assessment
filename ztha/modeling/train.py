import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight
import typer

from ztha.config import CONFIG, log
from ztha.modeling.predict import ModelEvaluator


def random_oversample(X, y, random_state=42):
    """Random oversampling using sklearn.utils.resample to balance classes."""
    # Combine X and y for resampling
    df = pd.concat([X, y], axis=1)

    # Get class counts
    class_counts = y.value_counts()
    majority_count = class_counts.max()

    # Oversample minority classes
    balanced_dfs = []
    for class_label in class_counts.index:
        class_df = df[df[y.name] == class_label]

        if len(class_df) < majority_count:
            # Oversample this class
            oversampled = resample(
                class_df,
                replace=True,
                n_samples=majority_count,
                random_state=random_state,
            )
            balanced_dfs.append(oversampled)
        else:
            balanced_dfs.append(class_df)

    # Combine all classes
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    # Split back into X and y
    X_balanced = balanced_df.drop(columns=[y.name])
    y_balanced = balanced_df[y.name]

    return X_balanced, y_balanced


MODELS_DIR = Path(CONFIG.artifacts_path)
PROCESSED_DATA_DIR = Path(CONFIG.data.raw_data_path).parent / "processed"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

app = typer.Typer()


class ModelTrainer:
    """Handles model training and evaluation using cross-validation."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.evaluator = ModelEvaluator()

    def _get_model(self) -> Any:
        """Instantiate model based on config."""
        if CONFIG.model.model_type == "gradient_boosting":
            model_params = {
                "n_estimators": CONFIG.model.gb_n_estimators,
                "learning_rate": CONFIG.model.gb_learning_rate,
                "max_depth": CONFIG.model.gb_max_depth,
                "random_state": CONFIG.model.random_state,
                "validation_fraction": CONFIG.model.gb_validation_fraction,
                "n_iter_no_change": CONFIG.model.gb_n_iter_no_change,
            }
            return GradientBoostingClassifier(**model_params)
        elif CONFIG.model.model_type == "logistic_regression":
            model_params = {
                "max_iter": CONFIG.model.lr_max_iter,
                "C": CONFIG.model.lr_C,
                "random_state": CONFIG.model.random_state,
                "class_weight": "balanced",
            }
            return LogisticRegression(**model_params)
        else:
            error_msg = f"Unsupported model type: {CONFIG.model.model_type}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def train_and_evaluate_with_cv(
        self, features: pd.DataFrame
    ) -> Tuple[Any, StandardScaler, Dict[str, Any]]:
        """
        Train and evaluate the model using stratified 5-fold cross-validation.

        A final model is trained on the entire dataset after evaluation.
        """
        log.info("Starting 5-fold stratified cross-validation...")

        X = features.drop(["is_churned"], axis=1)
        y = features["is_churned"]
        self.feature_names = X.columns.tolist()

        # Data validation checks
        self._validate_data(X, pd.Series(y) if not isinstance(y, pd.Series) else y)

        skf = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=CONFIG.model.random_state
        )
        all_eval_results = []

        for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
            log.info(f"--- FOLD {fold}/5 ---")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Apply SMOTE to training data if configured
            if CONFIG.model.apply_smote:
                log.debug("Applying random oversampling to balance training data...")
                X_train_balanced, y_train_balanced = random_oversample(
                    X_train, y_train, random_state=CONFIG.model.random_state
                )

                # Log balancing results
                original_counts = y_train.value_counts().sort_index()
                balanced_counts = (
                    pd.Series(y_train_balanced).value_counts().sort_index()
                )
                log.info(
                    f"Random oversampling applied - Original: {original_counts.tolist()}, Balanced: {balanced_counts.tolist()}"
                )

                X_train_for_scaling = X_train_balanced
                y_train_for_model = y_train_balanced
            else:
                X_train_for_scaling = X_train
                y_train_for_model = y_train

            # Scale features for the current fold
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_for_scaling)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = self._get_model()
            if isinstance(model, GradientBoostingClassifier):
                if CONFIG.model.apply_smote:
                    # No need for sample weights when data is balanced by SMOTE
                    model.fit(X_train_scaled, y_train_for_model)
                else:
                    # Ensure y is a pandas Series for compute_sample_weight
                    y_for_weights = (
                        pd.Series(y_train_for_model)
                        if not isinstance(y_train_for_model, pd.Series)
                        else y_train_for_model
                    )
                    sample_weights = compute_sample_weight(
                        class_weight="balanced", y=y_for_weights
                    )
                    model.fit(
                        X_train_scaled, y_train_for_model, sample_weight=sample_weights
                    )
            else:
                if CONFIG.model.apply_smote:
                    # No need for class_weight when data is balanced by SMOTE
                    model_params = {
                        "max_iter": CONFIG.model.lr_max_iter,
                        "C": CONFIG.model.lr_C,
                        "random_state": CONFIG.model.random_state,
                    }
                    model = LogisticRegression(**model_params)
                model.fit(X_train_scaled, y_train_for_model)

            # Evaluate model on unbalanced test data
            eval_results = self.evaluator.evaluate_model(
                model, scaler, self.feature_names, np.asarray(X_test_scaled), y_test
            )
            all_eval_results.append(eval_results)
            log.info(f"Fold {fold} AUROC: {eval_results['auroc']:.4f}")

        log.info("--- Cross-Validation Summary ---")
        # Aggregate results
        mean_auroc = np.mean([res["auroc"] for res in all_eval_results])
        std_auroc = np.std([res["auroc"] for res in all_eval_results])
        mean_recall_at_k = np.mean(
            [res["recall_at_top_k_percent"] for res in all_eval_results]
        )
        std_recall_at_k = np.std(
            [res["recall_at_top_k_percent"] for res in all_eval_results]
        )

        log.log_metrics(  # type: ignore
            {
                "mean_auroc": mean_auroc,
                "std_auroc": std_auroc,
                "mean_recall_at_k": mean_recall_at_k,
                "std_recall_at_k": std_recall_at_k,
            },
            "CV Results",
        )

        # Create a final aggregated report
        final_report = self._create_aggregated_report(all_eval_results)

        # Train final model on all data
        log.info("Training final model on the entire dataset...")

        # Apply SMOTE to full dataset for final model if configured
        if CONFIG.model.apply_smote:
            log.info("Applying random oversampling to full dataset for final model...")
            X_balanced, y_balanced = random_oversample(
                X, y, random_state=CONFIG.model.random_state
            )

            self.scaler.fit(X_balanced)
            X_scaled = self.scaler.transform(X_balanced)
            final_model = self._get_model()

            if isinstance(final_model, GradientBoostingClassifier):
                final_model.fit(X_scaled, y_balanced)
            else:
                # Adjust logistic regression for balanced data
                model_params = {
                    "max_iter": CONFIG.model.lr_max_iter,
                    "C": CONFIG.model.lr_C,
                    "random_state": CONFIG.model.random_state,
                }
                final_model = LogisticRegression(**model_params)
                final_model.fit(X_scaled, y_balanced)
        else:
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            final_model = self._get_model()
            if isinstance(final_model, GradientBoostingClassifier):
                sample_weights = compute_sample_weight(class_weight="balanced", y=y)
                final_model.fit(X_scaled, y, sample_weight=sample_weights)
            else:
                final_model.fit(X_scaled, y)

        log.success("Final model training complete.")

        # Add feature importance from the final model to the report
        final_report["feature_importance"] = self.evaluator._get_feature_importance(
            final_model, self.feature_names
        )
        final_report["feature_names"] = self.feature_names

        return final_model, self.scaler, final_report

    def _validate_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate data quality and log warnings for potential issues."""
        log.info("=== DATA VALIDATION ===")

        # Check for missing values
        missing_counts = X.isnull().sum()
        if missing_counts.any():
            missing_features_count = (missing_counts > 0).sum()
            log.warning(f"Found missing values in {missing_features_count} features")

        # Check for infinite values
        inf_counts = pd.Series(
            [np.isinf(X[col]).sum() for col in X.columns], index=X.columns
        )
        if inf_counts.any():
            inf_features_count = (inf_counts > 0).sum()  # type: ignore
            log.warning(f"Found infinite values in {inf_features_count} features")

        # Check class balance
        class_balance = y.value_counts(normalize=True)
        minority_class_ratio = class_balance.min()
        log.info(f"Class balance - Minority class: {minority_class_ratio:.2%}")
        if minority_class_ratio < 0.05:
            log.warning("Severe class imbalance detected (minority class < 5%)")

        # Check feature/sample ratio
        n_features = len(X.columns)
        n_samples = len(X)
        ratio = n_samples / n_features
        log.info(f"Sample/feature ratio: {ratio:.1f} (samples per feature)")
        if ratio < 10:
            log.warning(f"Low sample/feature ratio ({ratio:.1f}). Risk of overfitting!")
        elif ratio < 5:
            log.error(
                f"Very low sample/feature ratio ({ratio:.1f}). High overfitting risk!"
            )

        # Check feature variance
        low_variance_mask = X.var() < 0.01
        low_variance_features = low_variance_mask.sum()  # type: ignore
        if low_variance_features > 0:
            log.info(f"Found {low_variance_features} low-variance features")

        log.info("=== DATA VALIDATION COMPLETE ===")

    def _create_aggregated_report(
        self, all_eval_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create an aggregated report from all fold evaluation results."""
        # Average key metrics
        final_report = {
            "auroc": np.mean([r["auroc"] for r in all_eval_results]),
            "auroc_std": np.std([r["auroc"] for r in all_eval_results]),
            "recall_at_top_k_percent": np.mean(
                [r["recall_at_top_k_percent"] for r in all_eval_results]
            ),
            "recall_at_top_k_percent_std": np.std(
                [r["recall_at_top_k_percent"] for r in all_eval_results]
            ),
            "optimal_threshold": np.mean(
                [r["optimal_threshold"] for r in all_eval_results]
            ),
            "model_type": all_eval_results[0]["model_type"],
            "evaluation_type": "cross_validation",
            "precision_recall_curve": all_eval_results[0]["precision_recall_curve"],
        }

        # Sum confusion matrices
        total_cm = np.sum(
            [np.array(res["confusion_matrix"]) for res in all_eval_results], axis=0
        )
        final_report["confusion_matrix"] = total_cm.tolist()

        # Average classification reports
        avg_class_report = all_eval_results[0]["classification_report"].copy()
        for class_label, metrics in avg_class_report.items():
            if isinstance(metrics, dict):
                for metric_name in metrics:
                    values = [
                        res["classification_report"][class_label][metric_name]
                        for res in all_eval_results
                    ]
                    avg_class_report[class_label][metric_name] = np.mean(values)
            else:
                # Handle top-level metrics like 'accuracy'
                values = [
                    res["classification_report"][class_label]
                    for res in all_eval_results
                ]
                avg_class_report[class_label] = np.mean(values)
        final_report["classification_report"] = avg_class_report

        return final_report


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = MODELS_DIR / "churn_model.pkl",
):
    """
    Main function to train the churn prediction model.

    Args:
        features_path (Path): Path to the features CSV file.
        model_path (Path): Path to save the trained model.
    """
    log.info("Starting model training process...")

    # Load data
    log.info(f"Loading features from {features_path}")
    features = pd.read_csv(features_path, index_col="wallet_address")

    # Train model
    trainer = ModelTrainer()
    final_model, scaler, final_report = trainer.train_and_evaluate_with_cv(features)

    # Save artifacts
    log.info(f"Saving model to {model_path}")
    joblib.dump(final_model, model_path)

    scaler_path = model_path.parent / "feature_scaler.pkl"
    log.info(f"Saving scaler to {scaler_path}")
    joblib.dump(scaler, scaler_path)

    feature_names_path = model_path.parent / "feature_names.json"
    log.info(f"Saving feature names to {feature_names_path}")
    pd.Series(trainer.feature_names).to_json(feature_names_path, indent=2)

    # Save feature importance
    feature_importance = trainer.evaluator._get_feature_importance(
        final_model, trainer.feature_names
    )
    feature_importance_path = model_path.parent / "feature_importance.csv"
    log.info(f"Saving feature importance to {feature_importance_path}")
    feature_importance.to_csv(feature_importance_path, index=False)

    # Save final report
    final_report_path = model_path.parent / "final_report.json"
    log.info(f"Saving final report to {final_report_path}")
    with open(final_report_path, "w") as f:
        json.dump(final_report, f)

    # Note: Evaluation is not part of this script.
    # It is handled by the evaluator component.
    log.success("Model training complete.")


if __name__ == "__main__":
    app()
