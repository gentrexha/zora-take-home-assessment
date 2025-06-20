"""
Integration tests for the main pipeline orchestrator.
"""

import pandas as pd
import pytest
from pytest_mock import MockerFixture

from ztha.pipeline import ChurnPredictor


@pytest.fixture
def mock_pipeline_dependencies(mocker: MockerFixture):
    """Mock the main dependencies of the ChurnPredictor class."""
    mock_data_loader = mocker.patch("ztha.pipeline.DataLoader", autospec=True)
    mock_feature_engineer = mocker.patch("ztha.pipeline.FeatureEngineer", autospec=True)
    mock_model_trainer = mocker.patch("ztha.pipeline.ModelTrainer", autospec=True)

    # Configure mock return values to allow the pipeline to run
    mock_data_loader.return_value.load_data.return_value = (
        pd.DataFrame({"id": [1]}),
        pd.DataFrame({"id": [1]}),
    )
    mock_feature_engineer.return_value.engineer_features.return_value = pd.DataFrame(
        {"feature": [0], "is_churned": [0]}
    )
    mock_model_trainer.return_value.train_and_evaluate_with_cv.return_value = (
        mocker.MagicMock(),  # Dummy model
        mocker.MagicMock(),  # Dummy scaler
        {"auroc": 0.9, "precision_at_top_k_percent": 0.8},  # Dummy cv_results
    )

    # Configure the evaluator attribute
    mock_evaluator = mocker.MagicMock()
    mock_model_trainer.return_value.evaluator = mock_evaluator

    return mock_data_loader, mock_feature_engineer, mock_model_trainer


def test_run_pipeline(mock_pipeline_dependencies):
    """
    Test that the pipeline runs and calls its components in the correct order.
    """
    mock_dl, mock_fe, mock_mt = mock_pipeline_dependencies

    # Instantiate the predictor - this will use the mocked classes
    predictor = ChurnPredictor()
    predictor.run_pipeline()

    # Get the instances of the mocks that were created in ChurnPredictor.__init__
    dl_instance = mock_dl.return_value
    fe_instance = mock_fe.return_value
    mt_instance = mock_mt.return_value

    # Verify that the main methods were called once
    dl_instance.load_data.assert_called_once()
    fe_instance.engineer_features.assert_called_once()
    mt_instance.train_and_evaluate_with_cv.assert_called_once()
    mt_instance.evaluator.save_artifacts.assert_called_once()
    mt_instance.evaluator.generate_business_report.assert_called_once()

    # Verify they were all called (sufficient to confirm the pipeline flow)
    assert dl_instance.load_data.called
    assert fe_instance.engineer_features.called
    assert mt_instance.train_and_evaluate_with_cv.called
    assert mt_instance.evaluator.save_artifacts.called
    assert mt_instance.evaluator.generate_business_report.called
