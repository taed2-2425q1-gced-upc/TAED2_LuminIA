import json
import pickle
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from taed2_luminia.config import METRICS_DIR, PROCESSED_DATA_DIR

# Path to the models folder
MODELS_FOLDER_PATH = Path("models")

def load_validation_data(input_folder_path: Path):

    X_valid = pd.read_csv(input_folder_path / "X_valid.csv")

    return X_valid

def evaluate_model(model_file_name, x):
    with open(MODELS_FOLDER_PATH / model_file_name, "rb") as pickled_model:
        traffic_model = pickle.load(pickled_model)

        # Compute predictions using the model
    metrics = traffic_model.val(data=x)
    return metrics



if __name__ == "__main__":
    # Path to the metrics folder
    Path("metrics").mkdir(exist_ok=True)
    metrics_folder_path = METRICS_DIR

    X_valid = load_validation_data(PROCESSED_DATA_DIR)

    mlflow.set_experiment("traffic-signs-detection")

    with mlflow.start_run():
        # Load the model
        metrics = evaluate_model(
            "traffic-model.pkl", X_valid
        )

        metrics_dict = {
            "mAP50B": metrics['mAP50B'],  # Example
            "mAP50-95B": metrics['mAP50-95B'],  # Example
            "precisionB": metrics['precisionB'],  # Example
            "recallB": metrics['recallB']  # Example
        }

        # Log the evaluation metrics to MLflow
        mlflow.log_metrics(metrics_dict)

        # Save the evaluation metrics to a JSON file
        with open(metrics_folder_path / "scores.json", "w") as scores_file:
            json.dump(
                metrics_dict,
                scores_file,
                indent=4,
            )

        print("Evaluation completed.")