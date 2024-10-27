"""Test suite for functions in the src.models.train module."""

import os
import json
from pathlib import Path
from unittest.mock import patch
import pytest
import yaml
import pandas as pd
from src.models.train import (
    update_datasets_dir,
    read_params,
    create_data_dict,
    save_yaml,
    train_model,
    log_emissions_to_mlflow,
    save_model_to_directory,
    print_exp_folders,
)
from src.config.config import (
    TRAIN_IMAGES_PATH,
    TEST_IMAGES_PATH,
    METRICS_DIR,
)


@pytest.fixture
def mock_yaml_file(tmp_path):
    """Create a temporary YAML file for testing."""
    params = {
        "prepare": {
            "names": {0: "stop_sign", 1: "traffic_light", 2: "pedestrian", 3: "bicycle"},
            "nc": 4,
            "random_state": 42
        },
        "train": {
            "epochs": 5
        }
    }
    yaml_file = tmp_path / "params.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(params, f)
    return yaml_file


@pytest.fixture
def mock_json_file(tmp_path):
    """Create a temporary JSON file for testing."""
    json_data = {
        "datasets_dir": "/old/path"
    }
    json_file = tmp_path / "settings.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f)
    return json_file


def test_update_datasets_dir(mock_json_file):
    """Test to update the datasets directory in a JSON file."""
    new_dir = "/new/path"
    update_datasets_dir(str(mock_json_file), new_dir)

    with open(mock_json_file, 'r', encoding="utf-8") as file:
        data = json.load(file)

    assert data['datasets_dir'] == new_dir, "The datasets directory was not updated correctly."


def test_read_params(mock_yaml_file):
    """Test to read parameters from a YAML file."""
    prepare_params, train_params = read_params(str(mock_yaml_file))

    assert prepare_params is not None, "Prepare parameters were not loaded correctly."
    assert train_params is not None, "Train parameters were not loaded correctly."
    assert prepare_params["nc"] == 4, "The number of classes is not as expected."
    assert len(prepare_params["names"]) == 4, "Expected 4 class names."


def test_create_data_dict():
    """Test to create a data dictionary for training."""
    prepare_params = {
        "names": {0: "stop_sign", 1: "traffic_light", 2: "pedestrian", 3: "bicycle"},
        "nc": 4
    }
    data_dict = create_data_dict(prepare_params)

    assert data_dict["nc"] == 4, "The number of classes in the dictionary is incorrect."
    assert data_dict["train"] == str(TRAIN_IMAGES_PATH), "The training path is not as expected."
    assert data_dict["val"] == str(TEST_IMAGES_PATH), "The validation path is not as expected."


def test_save_yaml(tmp_path):
    """Test to save a dictionary as a YAML file."""
    data = {
        'train': 'train_images_path',
        'val': 'test_images_path',
        'nc': 4,
        'names': {0: 'class0', 1: 'class1', 2: 'class2', 3: 'class3'}
    }
    yaml_file = tmp_path / "output.yaml"

    save_yaml(data, yaml_file)

    # Verify that the file was created
    assert yaml_file.exists(), "The YAML file was not created."

    # Verify that the file content is as expected
    with open(yaml_file, 'r', encoding="utf-8") as file:
        content = yaml.safe_load(file)
        assert content == data, "The content of the YAML file is not as expected."


@patch('src.models.train.YOLO')
@patch('src.models.train.EmissionsTracker')
def test_train_model(mock_emissions_tracker, mock_yolo, mock_yaml_file):
    """Test to train the YOLO model."""
    prepare_params, train_params = read_params(str(mock_yaml_file))

    mock_instance = mock_yolo.return_value
    mock_instance.train.return_value = None

    # Pass the mock_yaml_file path
    emissions_output_folder = train_model(str(mock_yaml_file), train_params)

    assert mock_instance.train.called, "The train() method of the model was not called."
    assert emissions_output_folder == METRICS_DIR, "The emissions folder does not match."


@patch('src.models.train.pd.read_csv')
@patch('src.models.train.mlflow')
def test_log_emissions_to_mlflow(mock_mlflow, mock_read_csv):
    """Test to log CO2 emissions to MLflow."""
    mock_read_csv.return_value = pd.DataFrame({
        "CO2": [1.0, 2.0, 3.0],
        "other_metric": [0.1, 0.2, 0.3],
    })

    emissions_output_folder = Path("some_folder")  # Simulating the path of the emissions folder
    log_emissions_to_mlflow(emissions_output_folder)

    assert mock_mlflow.log_params.called, "Log parameters in MLflow were not called."
    assert mock_mlflow.log_metrics.called, "Log metrics in MLflow were not called."


@patch('shutil.copy')
def test_save_model_to_directory(mock_copy):
    """Test to save the trained model to the models directory."""
    # Simulate the existence of the experiments folder
    train_runs_path = Path("some/fake/path/runs/detect")
    os.makedirs(train_runs_path, exist_ok=True)
    os.makedirs(train_runs_path / "train_1", exist_ok=True)
    (train_runs_path / "train_1/weights").mkdir(parents=True, exist_ok=True)
    (train_runs_path / "train_1/weights/best.pt").touch()

    # Patch the glob function to simulate that there are experiment folders
    with patch('pathlib.Path.glob', return_value=[train_runs_path / "train_1"]):
        save_model_to_directory()

    assert mock_copy.called, "The copy() function was not called to save the model."


@patch('pathlib.Path.glob')
def test_print_exp_folders(mock_glob, tmp_path):
    """Test to print the experiment folders."""

    # Simulate the existence of experiment directories
    experiment_folder_1 = tmp_path / "runs/detect/train_1"
    experiment_folder_2 = tmp_path / "runs/detect/train_2"
    experiment_folder_1.mkdir(parents=True, exist_ok=True)
    experiment_folder_2.mkdir(parents=True, exist_ok=True)

    # Make glob return these simulated folders
    mock_glob.return_value = [experiment_folder_1, experiment_folder_2]

    with patch('builtins.print') as mock_print:
        print_exp_folders()

        assert mock_print.called, "The print() function was not called."
        mock_print.assert_any_call("Experiment folders:")
        mock_print.assert_any_call(experiment_folder_1)
        mock_print.assert_any_call(experiment_folder_2)


if __name__ == "__main__":
    pytest.main()
