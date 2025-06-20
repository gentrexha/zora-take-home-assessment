"""
Collector Churn Prediction Pipeline

This module orchestrates the complete machine learning pipeline to predict
collector churn, with a focus on interpretability and business actionability.
"""

from ztha.config import CONFIG, log, set_seeds
from ztha.dataset import DataLoader
from ztha.features import FeatureEngineer
from ztha.modeling.train import ModelTrainer


class ChurnPredictor:
    """Main orchestrator for the collector churn prediction pipeline."""

    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self) -> None:
        """Execute the complete churn prediction pipeline."""
        log.info("=== COLLECTOR CHURN PREDICTION PIPELINE ===")

        pipeline_info = {
            "model_type": CONFIG.model.model_type,
            "evaluation_strategy": "5-Fold Stratified Cross-Validation",
            "business_metric": f"Recall@Top{CONFIG.evaluation.precision_at_k_percent * 100:.0f}%",
            "artifacts_path": CONFIG.artifacts_path,
        }
        log.log_metrics(pipeline_info, "Pipeline Configuration")

        # Step 1: Load data
        log.info("Step 1: Loading data...")
        collectors, activity = self.data_loader.load_data()

        # Step 2: Engineer features
        log.info("Step 2: Engineering features...")
        features = self.feature_engineer.engineer_features(collectors, activity)

        # Step 3: Train model using cross-validation and get final artifacts
        log.info("Step 3: Training and evaluating model with cross-validation...")
        final_model, scaler, cv_results = self.model_trainer.train_and_evaluate_with_cv(
            features
        )

        # Step 4: Save artifacts
        log.info("Step 4: Saving artifacts from final model...")
        self.model_trainer.evaluator.save_artifacts(final_model, scaler, cv_results)

        # Step 5: Generate business report
        log.info("Step 5: Generating business report from CV results...")
        self.model_trainer.evaluator.generate_business_report(cv_results)

        # Final summary
        final_summary = {
            "mean_auroc": cv_results["auroc"],
            "auroc_std": cv_results.get("auroc_std", "N/A"),
            f"mean_recall_at_top_{CONFIG.evaluation.precision_at_k_percent * 100:.0f}_percent": cv_results[
                "precision_at_top_k_percent"
            ],
            "artifacts_saved_to": CONFIG.artifacts_path,
        }
        log.log_metrics(final_summary, "Final CV Results")

        log.success("=== PIPELINE COMPLETED ===")
        log.info("Key output files:")
        log.info("  • predictions.csv - Individual user predictions and probabilities")
        log.info("  • feature_importance.csv - Model interpretability insights")


def main():
    """Main entry point for the pipeline."""
    set_seeds(CONFIG.random_seed)
    log.info(f"Random seeds set to {CONFIG.random_seed} for reproducibility.")
    predictor = ChurnPredictor()
    predictor.run_pipeline()


if __name__ == "__main__":
    main()
