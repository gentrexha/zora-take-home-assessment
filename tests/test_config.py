"""
Tests for the configuration module.
"""

from pydantic import BaseModel
import pytest

from ztha.config import CONFIG, ChurnLogger, PipelineConfig, log, set_seeds


def test_config_object():
    """Test that the main CONFIG object is loaded correctly."""
    assert isinstance(CONFIG, PipelineConfig)
    assert isinstance(CONFIG.data, BaseModel)
    assert isinstance(CONFIG.model, BaseModel)
    assert isinstance(CONFIG.features, BaseModel)
    assert isinstance(CONFIG.evaluation, BaseModel)
    assert CONFIG.artifacts_path == "artifacts"


def test_logger_object():
    """Test that the logger object is an instance of ChurnLogger."""
    assert isinstance(log, ChurnLogger)


def test_set_seeds():
    """Test that the set_seeds function runs without error."""
    try:
        set_seeds(123)
        assert True
    except Exception as e:
        pytest.fail(f"set_seeds function raised an exception: {e}")
