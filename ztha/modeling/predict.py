import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import typer

from ztha.config import CONFIG, log

app = typer.Typer()


class PredictionConfig(BaseModel):
    features_path: Path
    model_path: Path
    scaler_path: Path
    predictions_path: Path


class PredictionResult(BaseModel):
    """Data model for a single prediction result."""

    wallet_address: str
    churn_probability: float
    predicted_label: int


class ModelEvaluator:
    """Handles model evaluation and artifact saving."""

    def __init__(self, artifacts_path: Path = Path(CONFIG.artifacts_path)):
        self.artifacts_dir = artifacts_path
        self.artifacts_dir.mkdir(exist_ok=True, parents=True)

    def evaluate_model(
        self,
        model: Any,
        scaler: StandardScaler,
        feature_names: List[str],
        X: np.ndarray,
        y: pd.Series,
    ) -> Dict[str, Any]:
        """
        Evaluate the model's performance on a given dataset.
        Includes threshold tuning to optimize for F1-score of the positive class.
        """
        log.debug(f"Evaluating model on {X.shape[0]} samples.")
        y_pred_proba = model.predict_proba(X)[:, 1]

        # --- Threshold Tuning ---
        precisions, recalls, thresholds = precision_recall_curve(y, y_pred_proba)
        # Calculate F1 score for each threshold, avoiding division by zero
        f1_scores = [
            (2 * p * r) / (p + r) if (p + r) > 0 else 0
            for p, r in zip(precisions, recalls)
        ]
        # Find the optimal threshold that maximizes F1 score for the positive class
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        log.info(f"Optimal threshold found: {optimal_threshold:.4f}")

        # Use the optimal threshold for final predictions
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        auroc = roc_auc_score(y, y_pred_proba)
        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        precision_at_k = self._precision_at_top_k(y, y_pred_proba)

        evaluation_results = {
            "auroc": auroc,
            "precision_at_top_k_percent": precision_at_k,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "optimal_threshold": optimal_threshold,
            "model_type": type(model).__name__,
            "precision_recall_curve": {
                "precision": precisions.tolist(),
                "recall": recalls.tolist(),
                "thresholds": thresholds.tolist(),  # type: ignore
            },
        }
        return evaluation_results

    def save_artifacts(
        self,
        model: Any,
        scaler: StandardScaler,
        evaluation_results: Dict[str, Any],
    ):
        """
        Save all model artifacts using the aggregated results from cross-validation.
        """
        log.info("Saving artifacts from aggregated CV results...")

        # Save the final model and the scaler fit on all data
        joblib.dump(model, self.artifacts_dir / "churn_model.pkl")
        joblib.dump(scaler, self.artifacts_dir / "feature_scaler.pkl")

        # Save feature names from the evaluation report
        feature_names = evaluation_results["feature_names"]
        with open(self.artifacts_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f, indent=2)

        # Save feature importance from the evaluation report
        # It might be a DataFrame or a list of dicts, handle both
        feature_importance_data = evaluation_results["feature_importance"]
        if isinstance(feature_importance_data, pd.DataFrame):
            feature_importance_df = feature_importance_data
        else:
            feature_importance_df = pd.DataFrame(feature_importance_data)

        feature_importance_df.to_csv(
            self.artifacts_dir / "feature_importance.csv", index=False
        )

        # Structure and save the final evaluation summary
        report_data = {
            "auroc": evaluation_results["auroc"],
            "auroc_std": evaluation_results.get("auroc_std"),
            "precision_at_top_k_percent": evaluation_results[
                "precision_at_top_k_percent"
            ],
            "precision_at_top_k_percent_std": evaluation_results.get(
                "precision_at_top_k_percent_std"
            ),
            "optimal_threshold": evaluation_results["optimal_threshold"],
            "model_type": evaluation_results["model_type"],
            "evaluation_type": evaluation_results.get("evaluation_type"),
            "classification_report": evaluation_results["classification_report"],
            "confusion_matrix": evaluation_results["confusion_matrix"],
            "feature_importance": feature_importance_df.to_dict(orient="records"),
        }
        evaluation_path = self.artifacts_dir / "evaluation_results.json"
        log.info(f"Saving comprehensive evaluation results to {evaluation_path}")
        with open(evaluation_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        log.success(f"Artifacts saved to {self.artifacts_dir}")

    def _precision_at_top_k(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> float:
        """Calculate Precision@Top K% based on the value in the project config."""
        k_fraction = CONFIG.evaluation.precision_at_k_percent
        if not (0 < k_fraction <= 1):
            raise ValueError("precision_at_k_percent must be between 0 and 1")

        top_k = int(len(y_true) * k_fraction)
        if top_k == 0:
            log.warning(
                f"Top k% ({k_fraction * 100:.1f}%) resulted in 0 users. "
                "Returning precision of 0. Consider a larger dataset or k%."
            )
            return 0.0

        top_indices = np.argsort(y_pred_proba)[-top_k:]
        return float(y_true.iloc[top_indices].mean())

    def _get_feature_importance(
        self, model: Any, feature_names: List[str]
    ) -> pd.DataFrame:
        """Extract feature importance from the model."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            # For models like Logistic Regression
            importances = model.coef_[0]

        feature_importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )
        feature_importance_df["importance_abs"] = feature_importance_df[
            "importance"
        ].abs()
        return feature_importance_df.sort_values(
            "importance_abs", ascending=False
        ).drop(columns=["importance_abs"])

    def generate_business_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a business-friendly summary report."""
        auroc = evaluation_results["auroc"]
        precision_at_k = evaluation_results["precision_at_top_k_percent"]
        # Ensure 'feature_importance' is a DataFrame
        feature_importance_df = pd.DataFrame(evaluation_results["feature_importance"])
        top_features = feature_importance_df.head(5)

        top_features_str = "\n".join(
            [
                f"  {i}. {row['feature']} ({row['importance']:.3f})"
                for i, (_, row) in enumerate(top_features.iterrows(), 1)
            ]
        )

        report = f"""
# Churn Prediction Business Report

## 1. Executive Summary
- **Model AUROC**: {auroc:.3f}
- **Precision at Top {CONFIG.evaluation.precision_at_k_percent * 100:.0f}%**: {precision_at_k:.3f}
- **Optimal Threshold (Avg. over CV)**: {evaluation_results.get("optimal_threshold", "N/A"):.3f}
- **Key Insight**: The model shows a clear ability to distinguish between churning and non-churning users. By targeting the top {CONFIG.evaluation.precision_at_k_percent * 100:.0f}% of users most likely to churn, we can focus retention efforts effectively.

## 2. Key Churn Drivers
The top 5 features driving churn predictions are:
{top_features_str}

## 3. Actionable Insights
- **High-Risk Segment**: The model has identified a high-risk segment of users. Focusing retention campaigns on this group is recommended.
- **Intervention Strategy**: The feature importances suggest that declining activity and reduced engagement are key churn indicators. Proactive engagement campaigns for users showing these signs could be effective.
"""
        # Save report
        with open(self.artifacts_dir / "business_report.txt", "w") as f:
            f.write(report)

        return report


@app.command()
def predict(
    model_path: Path = typer.Option(
        Path(CONFIG.artifacts_path) / "churn_model.pkl",
        "--model-path",
        "-m",
        help="Path to the trained model file.",
    ),
    features_path: Path = typer.Option(
        Path(CONFIG.data.processed_path) / "features.csv",
        "--features-path",
        "-f",
        help="Path to the features CSV file.",
    ),
    predictions_path: Path = typer.Option(
        Path(CONFIG.artifacts_path) / "predictions.csv",
        "--predictions-path",
        "-p",
        help="Path to save the predictions.",
    ),
):
    """Load a trained model and generate predictions on new data."""
    log.info("Starting prediction process...")

    artifacts_path = model_path.parent
    scaler_path = artifacts_path / "feature_scaler.pkl"
    feature_names_path = artifacts_path / "feature_names.json"

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(feature_names_path) as f:
            feature_names = json.load(f)

        features_df = pd.read_csv(features_path, index_col="wallet_address")
        X_test = scaler.transform(features_df[feature_names])

        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Here you could use a fixed optimal threshold saved from evaluation
        # For now, using 0.5 as a default for prediction script
        y_pred = (y_pred_proba >= 0.5).astype(int)

        predictions_df = pd.DataFrame(
            {
                "wallet_address": features_df.index,
                "churn_probability": y_pred_proba,
                "predicted_label": y_pred,
            }
        )

        predictions_df.to_csv(predictions_path, index=False)
        log.success(f"Predictions saved successfully to {predictions_path}")

    except FileNotFoundError as e:
        log.error(f"Error: {e}. Please check file paths.")
        raise typer.Exit(code=1)
    except Exception as e:
        log.error(f"An unexpected error occurred: {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
