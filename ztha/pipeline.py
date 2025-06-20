"""
Collector Churn Prediction Pipeline

This module orchestrates the complete machine learning pipeline to predict
collector churn, with a focus on interpretability and business actionability.
"""

from ztha.config import CONFIG, log
from ztha.dataset import DataLoader
from ztha.evaluator import ModelEvaluator
from ztha.features import FeatureEngineer
from ztha.model_trainer import ModelTrainer


class ChurnPredictor:
    """Main orchestrator for the collector churn prediction pipeline."""

    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()

    def run_pipeline(self) -> None:
        """Execute the complete churn prediction pipeline."""
        log.info("=== COLLECTOR CHURN PREDICTION PIPELINE ===")

        pipeline_info = {
            "model_type": CONFIG.model.model_type,
            "business_metric": f"Precision@Top{CONFIG.evaluation.precision_at_k_percent * 100:.0f}%",
            "artifacts_path": CONFIG.artifacts_path,
        }
        log.log_metrics(pipeline_info, "Pipeline Configuration")

        # Step 1: Load data
        log.info("Step 1: Loading data...")
        collectors, activity = self.data_loader.load_data()

        # Step 2: Engineer features
        log.info("Step 2: Engineering features...")
        features = self.feature_engineer.engineer_features(collectors, activity)

        # Step 3: Prepare training data
        log.info("Step 3: Preparing training data...")
        X_train, X_test, y_train, y_test = self.model_trainer.prepare_data(features)

        # Step 4: Train model
        log.info("Step 4: Training model...")
        model = self.model_trainer.train_model(X_train, y_train)

        # Step 5: Evaluate model
        log.info("Step 5: Evaluating model...")
        assert self.model_trainer.feature_names is not None
        evaluation_results = self.evaluator.evaluate_model(
            model,
            self.model_trainer.scaler,
            self.model_trainer.feature_names,
            X_test,
            y_test,
        )

        # Step 6: Save artifacts
        log.info("Step 6: Saving artifacts...")
        self.evaluator.save_artifacts(
            model, self.model_trainer.scaler, evaluation_results
        )

        # Step 7: Generate business report
        log.info("Step 7: Generating business report...")
        self.evaluator.generate_business_report(evaluation_results)

        # Final summary
        final_results = {
            "auroc": evaluation_results["auroc"],
            f"precision_at_top_{CONFIG.evaluation.precision_at_k_percent * 100:.0f}_percent": evaluation_results[
                "precision_at_top_k_percent"
            ],
            "artifacts_saved_to": CONFIG.artifacts_path,
        }
        log.log_metrics(final_results, "Final Results")

        log.success("=== PIPELINE COMPLETED ===")
        log.info("Key output files:")
        log.info("  • predictions.csv - Individual user predictions and probabilities")
        log.info("  • business_metrics.json - Business-focused performance metrics")
        log.info("  • feature_importance.csv - Model interpretability insights")
        log.info("  • model_summary.json - Technical model performance summary")


def main():
    """Main entry point for the pipeline."""
    predictor = ChurnPredictor()
    predictor.run_pipeline()


if __name__ == "__main__":
    main()
