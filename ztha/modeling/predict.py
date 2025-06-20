from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import typer

from ztha.config import CONFIG, log

app = typer.Typer()


class PredictionConfig(BaseModel):
    features_path: Path
    model_path: Path
    scaler_path: Path
    predictions_path: Path


class ModelEvaluator:
    """Handles model evaluation and artifact saving."""

    def __init__(self, artifacts_path: Optional[str] = None):
        self.artifacts_path = Path(artifacts_path or CONFIG.artifacts_path)
        self.artifacts_path.mkdir(exist_ok=True)

    def evaluate_model(
        self,
        model,
        scaler,
        feature_names: list,
        X_test: np.ndarray,
        y_test: pd.Series | np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate model performance with comprehensive metrics."""
        log.info("Evaluating model performance...")
        log.log_config(CONFIG.evaluation, "Evaluation")

        # Get predictions and probabilities
        log.debug("Generating predictions and probabilities...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Required metrics
        auroc = roc_auc_score(y_test, y_pred_proba)

        # Business metric: Precision@Top K%
        top_k_percent = int(CONFIG.evaluation.precision_at_k_percent * len(y_test))
        top_indices = np.argsort(y_pred_proba)[-top_k_percent:]
        if isinstance(y_test, pd.Series):
            precision_at_k = y_test.iloc[top_indices].mean()
        else:
            precision_at_k = y_test[top_indices].mean()

        # Additional classification metrics
        log.debug("Computing classification metrics...")
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Feature importance
        log.debug("Extracting feature importance...")
        feature_importance = self._get_feature_importance(model, feature_names)

        # Compile results
        results = {
            "auroc": float(auroc),
            "precision_at_top_k_percent": float(precision_at_k),
            "precision_at_k_description": f"Precision@Top{CONFIG.evaluation.precision_at_k_percent * 100:.0f}%",
            "classification_report": classification_rep,
            "confusion_matrix": conf_matrix.tolist(),
            "feature_importance": feature_importance,
            "model_type": type(model).__name__,
            "evaluation_timestamp": datetime.now().isoformat(),
            "test_set_size": len(y_test),
            "k_percent_value": CONFIG.evaluation.precision_at_k_percent,
            "top_k_sample_size": top_k_percent,
        }

        # Add predictions if requested
        if CONFIG.evaluation.save_predictions:
            log.debug("Including predictions in results...")
            results["predictions"] = {
                "y_test": y_test.tolist(),
                "y_pred": y_pred.tolist(),
            }

        if CONFIG.evaluation.save_probabilities:
            log.debug("Including probabilities in results...")
            results["predictions"] = results.get("predictions", {})
            results["predictions"]["y_pred_proba"] = y_pred_proba.tolist()

        # Log key metrics
        eval_metrics = {
            "auroc": auroc,
            f"precision_at_top_{CONFIG.evaluation.precision_at_k_percent * 100:.0f}_percent": precision_at_k,
            "overall_precision": classification_rep["1"]["precision"],  # type: ignore
            "overall_recall": classification_rep["1"]["recall"],  # type: ignore
            "overall_f1_score": classification_rep["1"]["f1-score"],  # type: ignore
            "test_samples": len(y_test),
            "top_k_sample_size": top_k_percent,
        }
        log.log_metrics(eval_metrics, "Model Evaluation")

        # Log top features
        top_features = feature_importance.head(5)
        log.info("Top 5 Important Features:")
        for _, row in top_features.iterrows():
            log.info(f"  {row['feature']}: {row['importance']:.4f}")

        log.success("Model evaluation completed successfully")

        return results

    def _get_feature_importance(self, model, feature_names: list) -> pd.DataFrame:
        """Extract feature importance from the model."""
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
        else:
            # Return empty DataFrame if no importance available
            return pd.DataFrame(columns=["feature", "importance"])  # type: ignore

        feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        return feature_importance

    def save_artifacts(self, model, scaler, evaluation_results: Dict[str, Any]) -> None:
        """Save all model artifacts and evaluation results."""
        log.info("Saving artifacts...")

        artifacts_saved = []

        try:
            # Save model and scaler
            log.debug("Saving trained model...")
            joblib.dump(model, self.artifacts_path / "churn_model.pkl")
            artifacts_saved.append("churn_model.pkl")

            log.debug("Saving feature scaler...")
            joblib.dump(scaler, self.artifacts_path / "feature_scaler.pkl")
            artifacts_saved.append("feature_scaler.pkl")

            # Save feature names
            log.debug("Saving feature names...")
            with open(self.artifacts_path / "feature_names.json", "w") as f:
                json.dump(
                    evaluation_results["feature_importance"]["feature"].tolist(),
                    f,
                    indent=2,
                )
            artifacts_saved.append("feature_names.json")

            # Save feature importance as CSV
            log.debug("Saving feature importance...")
            evaluation_results["feature_importance"].to_csv(
                self.artifacts_path / "feature_importance.csv", index=False
            )
            artifacts_saved.append("feature_importance.csv")

            # Save detailed evaluation results
            log.debug("Saving evaluation results...")
            eval_results_for_json = evaluation_results.copy()
            # Convert DataFrame to dict for JSON serialization
            eval_results_for_json["feature_importance"] = evaluation_results[
                "feature_importance"
            ].to_dict("records")

            with open(self.artifacts_path / "evaluation_results.json", "w") as f:
                json.dump(eval_results_for_json, f, indent=2, default=str)
            artifacts_saved.append("evaluation_results.json")

            # Save predictions and probabilities as CSV for easy analysis
            if "predictions" in evaluation_results:
                log.debug("Saving predictions...")
                predictions_df = pd.DataFrame(evaluation_results["predictions"])
                predictions_df.to_csv(
                    self.artifacts_path / "predictions.csv", index=False
                )
                artifacts_saved.append("predictions.csv")

            # Save model summary with key business metrics
            log.debug("Saving model summary...")
            is_cv = (
                evaluation_results.get("evaluation_type") == "5-Fold Cross-Validation"
            )
            summary = {
                "model_type": evaluation_results["model_type"],
                "evaluation_strategy": "5-Fold CV" if is_cv else "Train-Test Split",
                "mean_auroc": evaluation_results["auroc"],
                "auroc_std": evaluation_results.get("auroc_std"),
                "mean_precision_at_top_k_percent": evaluation_results[
                    "precision_at_top_k_percent"
                ],
                "precision_at_top_k_percent_std": evaluation_results.get(
                    "precision_at_top_k_percent_std"
                ),
                "precision_at_k_description": evaluation_results[
                    "precision_at_k_description"
                ],
                "total_features": len(evaluation_results["feature_importance"]),
                "dataset_size": evaluation_results["test_set_size"],
                "training_timestamp": evaluation_results["evaluation_timestamp"],
                "top_5_features": evaluation_results["feature_importance"]
                .head(5)
                .to_dict("records"),
                "config_summary": {
                    "model_type": CONFIG.model.model_type,
                    "test_size": "N/A (CV)" if is_cv else CONFIG.model.test_size,
                    "precision_at_k_percent": CONFIG.evaluation.precision_at_k_percent,
                    "whale_percentile": CONFIG.features.whale_percentile,
                    "recent_activity_days": CONFIG.features.recent_activity_days,
                },
            }

            with open(self.artifacts_path / "model_summary.json", "w") as f:
                json.dump(summary, f, indent=2, default=str)
            artifacts_saved.append("model_summary.json")

            # Create a business-friendly metrics file
            log.debug("Saving business metrics...")
            auroc_std_str = (
                f" ± {evaluation_results['auroc_std']:.3f}"
                if "auroc_std" in evaluation_results
                else ""
            )
            prec_std_str = (
                f" ± {evaluation_results['precision_at_top_k_percent_std']:.3f}"
                if "precision_at_top_k_percent_std" in evaluation_results
                else ""
            )
            business_metrics = {
                "key_performance_indicators": {
                    "evaluation_strategy": "5-Fold Cross-Validation"
                    if is_cv
                    else "Single Train-Test Split",
                    "model_discriminative_power_auroc": f"{evaluation_results['auroc']:.3f}{auroc_std_str}",
                    "precision_for_top_risk_users": f"{evaluation_results['precision_at_top_k_percent']:.3f}{prec_std_str}",
                    "description": f"Of users flagged as top {CONFIG.evaluation.precision_at_k_percent * 100:.0f}% churn risk, an average of {evaluation_results['precision_at_top_k_percent']:.1%} actually churned.",
                },
                "actionable_insights": {
                    "total_high_risk_users_identified": evaluation_results[
                        "top_k_sample_size"
                    ],
                    "expected_churners_in_high_risk_group": int(
                        evaluation_results["precision_at_top_k_percent"]
                        * evaluation_results["top_k_sample_size"]
                    ),
                    "top_churn_drivers": evaluation_results["feature_importance"]
                    .head(3)["feature"]
                    .tolist(),
                },
                "model_deployment_readiness": {
                    "auroc_threshold_met": evaluation_results["auroc"] >= 0.7,
                    "precision_threshold_met": evaluation_results[
                        "precision_at_top_k_percent"
                    ]
                    >= 0.5,
                    "ready_for_production": (
                        evaluation_results["auroc"] >= 0.7
                        and evaluation_results["precision_at_top_k_percent"] >= 0.5
                    ),
                },
            }

            with open(self.artifacts_path / "business_metrics.json", "w") as f:
                json.dump(business_metrics, f, indent=2, default=str)
            artifacts_saved.append("business_metrics.json")

            # Log successful artifact saving
            log.log_metrics(
                {
                    "artifacts_directory": str(self.artifacts_path),
                    "total_files_saved": len(artifacts_saved),
                },
                "Artifacts",
            )

            log.info("Saved artifacts:")
            for artifact in artifacts_saved:
                log.info(f"  ✓ {artifact}")

            log.success("All artifacts saved successfully!")

        except Exception as e:
            log.error(f"Error saving artifacts: {str(e)}")
            raise

    def generate_business_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a business-friendly summary report."""
        auroc = evaluation_results["auroc"]
        precision_at_k = evaluation_results["precision_at_top_k_percent"]
        top_features = evaluation_results["feature_importance"].head(5)
        is_cv = evaluation_results.get("evaluation_type") == "5-Fold Cross-Validation"

        auroc_std_str = (
            f" (± {evaluation_results['auroc_std']:.3f})"
            if "auroc_std" in evaluation_results
            else ""
        )
        prec_std_str = (
            f" (± {evaluation_results['precision_at_top_k_percent_std']:.2%})"
            if "precision_at_top_k_percent_std" in evaluation_results
            else ""
        )

        report = f"""
COLLECTOR CHURN PREDICTION - MODEL PERFORMANCE SUMMARY
=====================================================

METHODOLOGY
• Evaluation Strategy: {"5-Fold Stratified Cross-Validation" if is_cv else "Single Train-Test Split"}
• This approach provides a more robust estimate of model performance.

📊 KEY BUSINESS METRICS
• Mean Model Accuracy (AUROC): {auroc:.3f}{auroc_std_str} {"✅ Strong" if auroc >= 0.8 else "⚠️ Moderate" if auroc >= 0.7 else "❌ Weak"}
• Mean Precision@Top{CONFIG.evaluation.precision_at_k_percent * 100:.0f}%: {precision_at_k:.1%}{prec_std_str}
• Interpretation: Of users flagged as highest churn risk, we can expect {precision_at_k:.1%} to actually churn on average.

🎯 ACTIONABILITY
• Target {evaluation_results["top_k_sample_size"]} users for retention campaigns (based on a single data split)
• Expected to capture ~{int(precision_at_k * evaluation_results["top_k_sample_size"])} churners
• Focus interventions on top risk factors below

🔍 TOP CHURN DRIVERS (from final model trained on all data)
"""
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            report += f"  {i}. {row['feature']}: {row['importance']:.3f}\n"

        report += f"""
📈 BUSINESS IMPACT POTENTIAL
• Current model can identify {precision_at_k:.1%} of high-risk churners with good reliability.
• Recommended for {"immediate deployment" if auroc >= 0.8 and precision_at_k >= 0.6 else "pilot testing" if auroc >= 0.7 else "further development"}
• Focus retention efforts on top {CONFIG.evaluation.precision_at_k_percent * 100:.0f}% risk users for maximum ROI

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        # Save report
        with open(self.artifacts_path / "business_report.txt", "w") as f:
            f.write(report)

        return report


@app.command()
def main(
    features_path: Path = typer.Option(
        "artifacts/test_features.csv", help="Path to test features data."
    ),
    model_path: Path = typer.Option(
        "artifacts/churn_model.pkl", help="Path to the trained model."
    ),
    scaler_path: Path = typer.Option(
        "artifacts/feature_scaler.pkl", help="Path to the feature scaler."
    ),
    predictions_path: Path = typer.Option(
        "artifacts/predictions.csv", help="Path to save predictions."
    ),
):
    """
    Load a trained model and generate predictions on new data.
    """
    log.info("Starting prediction process...")

    prediction_config = PredictionConfig(
        features_path=features_path,
        model_path=model_path,
        scaler_path=scaler_path,
        predictions_path=predictions_path,
    )
    log.log_config(
        prediction_config,
        "Prediction",
    )

    try:
        # Load model, scaler, and features
        log.debug(f"Loading model from {model_path}...")
        model = joblib.load(model_path)

        log.debug(f"Loading scaler from {scaler_path}...")
        scaler = joblib.load(scaler_path)

        log.debug(f"Loading features from {features_path}...")
        features_df = pd.read_csv(features_path)
        X_test = scaler.transform(features_df)

        # Generate predictions
        log.info("Generating predictions...")
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        # Create predictions DataFrame
        predictions_df = features_df.copy()
        predictions_df["churn_probability"] = y_pred_proba
        predictions_df["churn_prediction"] = y_pred

        # Save predictions
        log.info(f"Saving predictions to {predictions_path}...")
        predictions_path.parent.mkdir(exist_ok=True, parents=True)
        predictions_df.to_csv(predictions_path, index=False)

        log.success(f"Predictions saved successfully to {predictions_path}")
        log.info(
            f"Top 5 users with highest churn probability:\n{predictions_df.sort_values('churn_probability', ascending=False).head(5)}"
        )

    except FileNotFoundError as e:
        log.error(f"Error: {e}. Please check file paths.")
        raise typer.Exit(code=1)
    except Exception as e:
        log.error(f"An unexpected error occurred: {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
