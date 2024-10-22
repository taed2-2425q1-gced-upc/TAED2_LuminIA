import os
import pytest
import yaml
import json
from unittest.mock import patch, mock_open
from pathlib import Path
from src.models.evaluate import load_params, create_data_dict, save_yaml_file, evaluate_model  
from src.config.config import (
    PARAMS_PATH,
    YAML_FILE,
    METRICS_DIR,
    WEIGHTS_PATH,
)

@pytest.fixture
def mock_yaml_file(tmp_path):
    """Create a temporary YAML file for testing."""
    params = {
        "prepare": {
            "names": {0: "stop_sign", 1: "traffic_light", 2: "pedestrian", 3: "bicycle"},  # Adjusted to 4 classes
            "nc": 4,  # Updated number of classes
            "random_state": 42
        }
    }
    yaml_file = tmp_path / "params.yaml"
    with open(yaml_file, "w") as f:
        yaml.dump(params, f)
    return yaml_file

@pytest.fixture
def mock_metrics_dir(tmp_path):
    """Create a temporary directory for metrics."""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return metrics_dir

def test_load_params(mock_yaml_file, monkeypatch):
    """Test loading parameters from a YAML file."""
    # Replace the YAML file path with the path of the mock
    monkeypatch.setattr("src.config.config.PARAMS_PATH", str(mock_yaml_file))
    
    params = load_params()
    
    assert params is not None, "Parameters did not load correctly"
    assert params["nc"] == 4, "The number of classes is not as expected"  # Changed to 4
    assert "names" in params, "Missing names in parameters"

def test_create_data_dict():
    """Test creating a data dictionary for YOLO."""
    prepare_params = {
        "names": {0: "stop_sign", 1: "traffic_light", 2: "pedestrian", 3: "bicycle"},  # Adjusted to 4 classes
        "nc": 4  # Updated number of classes
    }
    data_dict = create_data_dict(prepare_params)

    assert data_dict["nc"] == 4, "The number of classes in the dictionary is incorrect"  # Changed to 4
    assert data_dict["names"] == {0: "stop_sign", 1: "traffic_light", 2: "pedestrian", 3: "bicycle"}, "Class names do not match"
    assert 'train' in data_dict, "The dictionary does not contain the 'train' key"
    assert 'val' in data_dict, "The dictionary does not contain the 'val' key"

def test_save_yaml_file(tmp_path):
    """Test for saving a YAML file."""
    data = {
        'train': 'train_images_path',
        'val': 'test_images_path',
        'nc': 4,
        'names': {0: 'class0', 1: 'class1', 2: 'class2', 3: 'class3'}
    }

    save_yaml_file(data)  # Call the function you are testing

    # Verify that the file was created
    assert YAML_FILE.exists(), "The YAML file was not created."

    # Verify that the content of the file is as expected
    with open(YAML_FILE, 'r') as file:
        content = yaml.safe_load(file)
        assert content == data, "The content of the YAML file is not as expected."

@patch('src.models.evaluate.YOLO')
def test_evaluate_model(mock_yolo):
    """Test to evaluate the model and verify the returned metrics."""
    
    class MockBox:
        def __init__(self):
            self.map50 = 0.5
            self.map = 0.4
            self.mp = 0.6
            self.mr = 0.7

    class MockValResult:
        def __init__(self):
            self.box = MockBox()

    mock_instance = mock_yolo.return_value
    mock_instance.val.return_value = MockValResult()  
    
    metrics = evaluate_model()
    
    assert metrics["mAP50B"] == 0.5, "The mAP at 50 is not as expected"
    assert metrics["mAP50-95B"] == 0.4, "The mAP between 50 and 95 is not as expected"
    assert metrics["precisionB"] == 0.6, "The precision is not as expected"
    assert metrics["recallB"] == 0.7, "The recall is not as expected"
