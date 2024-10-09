"""
This module trains a YOLO model for traffic sign detection.

It performs the following tasks:
1. Sets up the MLflow experiment.
2. Reads configuration parameters from a YAML file.
3. Prepares the dataset paths and class names.
4. Tracks CO2 emissions during model training.
5. Trains the model and logs metrics to MLflow.
6. Saves the trained model to a specified path.
"""

from pathlib import Path

import random

import numpy as np
import pandas as pd
import yaml
from codecarbon import EmissionsTracker
from ultralytics import YOLO

import mlflow
from src.config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, DATA_DIR

mlflow.set_experiment("traffic-signs")
mlflow.sklearn.autolog(log_model_signatures=False, log_datasets=False)

TRAIN_IMAGES_PATH = PROCESSED_DATA_DIR / "train_images.txt"
TEST_IMAGES_PATH = PROCESSED_DATA_DIR / "test_images.txt"

with mlflow.start_run():
    # Path of the parameters file
    params_path = Path("params.yaml")
    # Read parameters
    with open(params_path, "r", encoding="utf8") as params_file:
        try:
            params = yaml.safe_load(params_file)
            prepare_params = params["prepare"]
            train_params = params["train"]
        except yaml.YAMLError as exc:
            print(exc)

    # Read class names and number of classes from prepare parameters
    class_names = list(prepare_params["names"].values())
    nc = prepare_params["nc"]

    # Create the data dictionary using data lists and data preparation parameters
    data = {
        'train': (
            "/mnt/c/Users/evaji/OneDrive/Documents/UNI/7/TAED2/LuminIA/TAED2_LuminIA/"
            "data/processed/train_images.txt"
        ),
        # prepare_params["train"],
        'val': (
            "/mnt/c/Users/evaji/OneDrive/Documents/UNI/7/TAED2/LuminIA/TAED2_LuminIA/"
            "data/processed/test_images.txt"
        ),        # 'train': str(TRAIN_IMAGES_PATH),
        # 'val': str(TEST_IMAGES_PATH),
        'nc': nc,
        'names': dict(enumerate(class_names))
    }

    # Write the dictionary to a YAML file if needed (for the YOLO model)
    YAML_FILE = 'dataset.yaml'
    with open(YAML_FILE, 'w', encoding='utf-8') as file:
        yaml.dump(data, file, default_flow_style=False)

    # ============== #
    # MODEL TRAINING #
    # ============== #

    # For the sake of reproducibility, set the random_state for both Numpy and built-in libraries
    np.random.seed(prepare_params["random_state"])
    random.seed(prepare_params["random_state"])

    # Model specification
    ts_model = YOLO("yolov8m.yaml")

    # Track the CO2 emissions of training the model
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
        # Then fit the model to the training data
        ts_model.train(data=YAML_FILE,
                       epochs=train_params["epochs"],
                       # batch_size=train_params["batch_size"],
                       # learning_rate=train_params["learning_rate"]
                       )

    # Log the CO2 emissions to MLflow
    emissions = pd.read_csv(emissions_output_folder / "emissions.csv")
    emissions_metrics = emissions.iloc[-1, 4:13].to_dict()
    emissions_params = emissions.iloc[-1, 13:].to_dict()

    mlflow.log_params(emissions_params)
    mlflow.log_metrics(emissions_metrics)

    # Save the model as a pickle file
    # ts_model.save(str(MODELS_DIR / "ts_model.pt"))
    ts_model.save(
        "/mnt/c/Users/evaji/OneDrive/Documents/UNI/7/TAED2/LuminIA/TAED2_LuminIA/models/"
        "ts_model.pt"
    )
