from pathlib import Path
import mlflow
import numpy as np
import random
import yaml
import json
from ultralytics import YOLO

from src.config import METRICS_DIR, PROCESSED_DATA_DIR, MODELS_DIR

# DATA LOADING
# Path to the models folder
MODELS_FOLDER_PATH = Path("models")
TRAIN_IMAGES_PATH = PROCESSED_DATA_DIR / "train_images.txt"
TEST_IMAGES_PATH = PROCESSED_DATA_DIR / "test_images.txt"

params_path = Path("params.yaml")

# Read data preparation parameters
with open(params_path, "r", encoding="utf8") as params_file:
    try:
        params = yaml.safe_load(params_file)
        prepare_params = params["prepare"]  # Extract prepare parameters
    except yaml.YAMLError as exc:
        print(exc)

# Read class names and number of classes from prepare parameters
print("prepare_params['names']:", prepare_params["names"])
print("Type of prepare_params['names']:", type(prepare_params["names"]))

class_names = list(prepare_params["names"].values())
nc = prepare_params["nc"]

# Create the data dictionary using data lists and data preparation parameters
data = {
    'train': str(TRAIN_IMAGES_PATH), 
    'val': str(TEST_IMAGES_PATH), 
    'nc': nc,
    'names': {i: name for i, name in enumerate(class_names)} 
}

# Write the dictionary to a YAML file if needed (for the YOLO model)
yaml_file = 'dataset.yaml'
with open(yaml_file, 'w') as file:
    yaml.dump(data, file, default_flow_style=False)


# ================ #
# MODEL EVALUATION #
# ================ #

# Path to the metrics folder
Path("metrics").mkdir(exist_ok=True)
metrics_folder_path = METRICS_DIR

mlflow.set_experiment("traffic-signs")

with mlflow.start_run():
    np.random.seed(prepare_params["random_state"])
    random.seed(prepare_params["random_state"])  

    # Load the model
    ts_model = YOLO(MODELS_DIR / "ts_model.pt")

    # Perform validation and save the evaluation metrics in runs directory
    metrics = ts_model.val(data=yaml_file)

    # Create a dictionary to hold the metrics to log
    metrics_dict = {
        "mAP50B": metrics.results_dict['metrics/mAP50(B)'],   
        "mAP50-95B": metrics.results_dict['metrics/mAP50-95(B)'],    
        "precisionB": metrics.results_dict['metrics/precision(B)'],   
        "recallB": metrics.results_dict['metrics/recall(B)']       
    }

    # Log metrics to MLflow
    mlflow.log_metrics(metrics_dict)

    # Save the evaluation metrics to a JSON file
    with open(metrics_folder_path / "scores.json", "w") as scores_file:
        json.dump(metrics_dict, scores_file, indent=4)

    print("Evaluation completed.")