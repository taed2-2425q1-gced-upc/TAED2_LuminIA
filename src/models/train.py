"""Train a YOLO model for traffic signs detection."""

import random
import json
import shutil
import numpy as np
import mlflow
import pandas as pd
import yaml
from codecarbon import EmissionsTracker
from ultralytics import YOLO

from src.config.config import (PROJ_ROOT, METRICS_DIR, MODELS_DIR,
                                TRAIN_IMAGES_PATH, TEST_IMAGES_PATH,
                                SETTINGS_PATH, PARAMS_PATH)

def update_datasets_dir(file_path, new_datasets_dir):
    """Update the datasets directory in the JSON configuration."""
    with open(file_path, 'r', encoding='utf8') as file:
        data = json.load(file)
    data['datasets_dir'] = new_datasets_dir
    with open(file_path, 'w', encoding='utf8') as file:
        json.dump(data, file, indent=4)
    print("The JSON file has been updated.")

def read_params(params_path):
    """Read parameters from a YAML file."""
    with open(params_path, "r", encoding="utf8") as params_file:
        try:
            params = yaml.safe_load(params_file)
            return params["prepare"], params["train"]
        except yaml.YAMLError as exc:
            print(exc)
            return None, None

def create_data_dict(prepare_params):
    """Create the data dictionary for training."""
    class_names = list(prepare_params["names"].values())
    nc = prepare_params["nc"]

    return {
        'train': str(TRAIN_IMAGES_PATH),
        'val': str(TEST_IMAGES_PATH),
        'nc': nc,
        'names': dict(enumerate(class_names))
    }

def save_yaml(data, yaml_file_path):
    """Save the data dictionary as a YAML file."""
    with open(yaml_file_path, 'w', encoding='utf8') as file:
        yaml.dump(data, file, default_flow_style=False)

# ============== #
# MODEL TRAINING #
# ============== #

def train_model(yaml_file_path, train_params):
    """Train the YOLO model."""
    # Set the random state for reproducibility
    prepare_params, _ = read_params(PARAMS_PATH)
    np.random.seed(prepare_params["random_state"])
    random.seed(prepare_params["random_state"])

    ts_model = YOLO("yolov8m.yaml")
    emissions_output_folder = METRICS_DIR

    with EmissionsTracker(
        project_name="traffic-signs-detection",
        measure_power_secs=1,
        tracking_mode="process",
        output_dir=emissions_output_folder,
        output_file="emissions.csv",
        on_csv_write="append",
        default_cpu_power=45,
    ):
        ts_model.train(data=yaml_file_path,  # Pass the YAML file path instead of the dict
                       epochs=train_params["epochs"])

    return emissions_output_folder


def log_emissions_to_mlflow(emissions_output_folder):
    """Log CO2 emissions to MLflow."""
    emissions = pd.read_csv(emissions_output_folder / "emissions.csv")
    emissions_metrics = emissions.iloc[-1, 4:13].to_dict()
    emissions_params = emissions.iloc[-1, 13:].to_dict()

    mlflow.log_params(emissions_params)
    mlflow.log_metrics(emissions_metrics)

def save_model_to_directory():
    """Save the trained model to the models directory."""
    train_runs_path = PROJ_ROOT / "runs" / "detect"
    if not train_runs_path.exists():
        print(f"Error: The directory {train_runs_path} does not exist.")
        return
    exp_folders = sorted(train_runs_path.glob("train*"), key=lambda p: p.stat().st_mtime)
    if not exp_folders:
        print("No experiment folders found in the directory.")
        return
    latest_exp_folder = exp_folders[-1]
    default_model_path = latest_exp_folder / "weights/best.pt"
    if not default_model_path.exists():
        print(f"Error: The model {default_model_path} does not exist.")
        return
    model_save_path = MODELS_DIR / "ts_model.pt"
    shutil.copy(default_model_path, model_save_path)
    print(f"Model from {default_model_path} copied to {model_save_path}")

def print_exp_folders():
    """Print the list of experiment folders."""
    train_runs_path = PROJ_ROOT / "runs" / "detect"
    exp_folders = sorted(train_runs_path.glob("train*"), key=lambda p: p.stat().st_mtime)

    if exp_folders:
        print("Experiment folders:")
        for folder in exp_folders:
            print(folder)
    else:
        print("No experiment folders found matching 'train*'.")

    latest_exp_folder = exp_folders[-1]
    print("Last folder:")
    print(latest_exp_folder)

# Update datasets directory in the JSON file
update_datasets_dir(SETTINGS_PATH, str(PROJ_ROOT))

mlflow.set_experiment("traffic-signs")
mlflow.sklearn.autolog(log_model_signatures=False, log_datasets=False)

with mlflow.start_run():
    # Read parameters from the config
    prepare_params_outer, train_params_outer = read_params(PARAMS_PATH)

    # Create data dictionary
    data_dict_outer = create_data_dict(prepare_params_outer)

    # Save the data dictionary as a YAML file
    YAML_FILE_PATH = 'dataset.yaml'
    save_yaml(data_dict_outer, YAML_FILE_PATH)

    # Train the model and track emissions
    emissions_output_folder_outer = train_model(YAML_FILE_PATH, train_params_outer)

    # Log emissions to MLflow
    log_emissions_to_mlflow(emissions_output_folder_outer)

    # Save the model to the models directory
    save_model_to_directory()
