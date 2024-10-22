from pathlib import Path
import mlflow
import numpy as np
import random
import yaml
import json
from ultralytics import YOLO

from src.config.config import (
    METRICS_DIR, 
    PROCESSED_DATA_DIR, 
    MODELS_DIR, 
    TRAIN_IMAGES_PATH, 
    TEST_IMAGES_PATH, 
    PARAMS_PATH, 
    YAML_FILE,
    WEIGHTS_PATH,
)

def load_params():
    """Load parameters from a YAML file."""
    with open(PARAMS_PATH, "r", encoding="utf8") as params_file:
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
        'names': {i: name for i, name in enumerate(class_names)}
    }

    return data

def save_yaml_file(data):
    """Save the data dictionary to a YAML file."""
    with open(YAML_FILE, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

# ================ #
# MODEL EVALUATION #
# ================ #

def evaluate_model():
    """Evaluate the model and return the metrics."""
    ts_model = YOLO(WEIGHTS_PATH)
    metrics = ts_model.val(data=YAML_FILE)

    # Extraer las métricas usando los atributos correctos (sin paréntesis)
    metrics_dict = {
        "mAP50B": metrics.box.map50,       # mAP at IoU 50 for bounding boxes
        "mAP50-95B": metrics.box.map,      # mAP at IoU 50-95 for bounding boxes
        "precisionB": metrics.box.mp,      # Mean precision for bounding boxes (sin paréntesis)
        "recallB": metrics.box.mr          # Mean recall for bounding boxes (sin paréntesis)
    }

    return metrics_dict


def main():
    """Main evaluation function."""
    # Load parameters
    prepare_params = load_params()
    if prepare_params is None:
        print("Failed to load parameters.")
        return

    # Create the data dictionary
    data = create_data_dict(prepare_params)

    # Save the dictionary to a YAML file
    save_yaml_file(data)

    # Create metrics folder if it doesn't exist
    Path(METRICS_DIR).mkdir(exist_ok=True)

    # Set MLflow experiment
    mlflow.set_experiment("traffic-signs")

    with mlflow.start_run():
        # Set random seeds for reproducibility
        np.random.seed(prepare_params["random_state"])
        random.seed(prepare_params["random_state"])

        # Evaluate the model
        metrics_dict = evaluate_model()

        # Log metrics to MLflow
        mlflow.log_metrics(metrics_dict)

        # Save evaluation metrics to a JSON file
        with open(METRICS_DIR / "scores.json", "w") as scores_file:
            json.dump(metrics_dict, scores_file, indent=4)

        print("Evaluation completed.")


if __name__ == "__main__":
    main()