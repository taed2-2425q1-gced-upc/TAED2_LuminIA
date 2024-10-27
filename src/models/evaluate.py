"""
Model evaluation script for YOLO object detection.
"""

from pathlib import Path
import random
import json
import yaml
import numpy as np
import mlflow
from ultralytics import YOLO

from src.config.config import (
    METRICS_DIR,
    TRAIN_IMAGES_PATH,
    TEST_IMAGES_PATH,
    PARAMS_PATH,
    YAML_FILE,
    WEIGHTS_PATH,
)


def load_params():
    """Load parameters from a YAML file."""
    with open(PARAMS_PATH, "r", encoding="utf-8") as params_file:
        try:
            params = yaml.safe_load(params_file)
            return params["prepare"]  # Extract prepare parameters
        except yaml.YAMLError as exc:
            print(exc)
            return None


def create_data_dict(prepare_params):
    """Create a data dictionary for YOLO from prepare parameters."""
    class_names = list(prepare_params["names"].values())
    nc = prepare_params["nc"]

    data = {
        'train': str(TRAIN_IMAGES_PATH),
        'val': str(TEST_IMAGES_PATH),
        'nc': nc,
        'names': dict(enumerate(class_names))  # Simplified with dict() directly
    }

    return data


def save_yaml_file(data):
    """Save the data dictionary to a YAML file."""
    with open(YAML_FILE, 'w', encoding='utf-8') as file:  # Added encoding
        yaml.dump(data, file, default_flow_style=False)


# ================ #
# MODEL EVALUATION #
# ================ #


def evaluate_model(yaml_file_path, prepare_params):
    """Evaluate the YOLO model using the provided YAML file."""
    # Load the model
    model = YOLO(WEIGHTS_PATH)

    # Validate using the YAML file
    val_results = model.val(data=yaml_file_path)

    # Check for predictions for all classes
    class_names = prepare_params["names"]

    # Gather metrics
    metrics_dict = {
        "mAP50B": val_results.results_dict['metrics/mAP50(B)'],
        "mAP50-95B": val_results.results_dict['metrics/mAP50-95(B)'],
        "precisionB": val_results.results_dict['metrics/precision(B)'],
        "recallB": val_results.results_dict['metrics/recall(B)'],
    }

    class_map = {}
    for i, class_name in enumerate(class_names.values()):
        # Use index to fetch corresponding map values
        if i < len(val_results.box.maps):
            class_map[class_name] = {
                "mAP50-95": val_results.box.maps[i],  # Directly assign from maps
            }
        else:
            class_map[class_name] = {"mAP50-95": None}

    return metrics_dict, class_map


# Load parameters
global_prepare_params = load_params()

# Create the data dictionary
global_data = create_data_dict(global_prepare_params)

# Save the dictionary to a YAML file
save_yaml_file(global_data)

# Create metrics folder if it doesn't exist
Path(METRICS_DIR).mkdir(exist_ok=True)

# Set MLflow experiment
mlflow.set_experiment("traffic-signs")

with mlflow.start_run():
    # Set random seeds for reproducibility
    np.random.seed(global_prepare_params["random_state"])
    random.seed(global_prepare_params["random_state"])

    # Evaluate the model
    global_metrics_dict, global_class_map = evaluate_model(YAML_FILE, global_prepare_params)

    # Log overall metrics to MLflow
    mlflow.log_metrics(global_metrics_dict)

    # Log class-wise mAP to MLflow, skipping classes with no predictions
    for name, mAPs in global_class_map.items():
        if mAPs["mAP50-95"] is not None:  # Check if mAP50-95 is not None
            mlflow.log_metric(f"class_{name}_mAP50-95", mAPs["mAP50-95"])
        else:
            print(f"No predictions for class '{name}', skipping logging.")

    # Merge overall metrics and class-wise mAPs into a single dictionary
    merged_metrics = {
        "overall_metrics": global_metrics_dict,
        "class_wise_mAPs": global_class_map
    }

    # Save evaluation metrics to a JSON file
    with open(METRICS_DIR / "scores.json", "w", encoding='utf-8') as scores_file:  # Added encoding
        json.dump(merged_metrics, scores_file, indent=4)

    print("Evaluation completed.")
