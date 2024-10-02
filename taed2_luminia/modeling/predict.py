from pathlib import Path
import mlflow
import json
import pickle
import yaml

from taed2_luminia.config import METRICS_DIR, PROCESSED_DATA_DIR

# Path to the models folder
MODELS_FOLDER_PATH = Path("models")

def load_validation(input_folder_path: Path):

    with open(input_folder_path / "test_images.txt", "r") as file:
        X_valid_list = file.readlines()
    X_valid_list = [x.strip() for x in X_valid_list]  # To remove newline characters.

    class_names = ['prohibitory',
    'danger',
    'mandatory',
    'other']

    # Create a dictionary for the YAML structure
    data = {
        'validation': X_valid_list,
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }

    # Write the dictionary to a YAML file
    yaml_file = 'dataset.yaml'
    with open(yaml_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    return yaml_file

def evaluate_model(model_file_name, x):
    with open(MODELS_FOLDER_PATH / model_file_name, "rb") as pickled_model:
        traffic_model = pickle.load(pickled_model)

    # Compute metrics using the model
    valid_metrics = traffic_model.val(data=x)

    return valid_metrics

if __name__ == '__main__':
    # Path to the metrics folder
    Path("metrics").mkdir(exist_ok=True)
    metrics_folder_path = METRICS_DIR

    X_valid = load_validation(PROCESSED_DATA_DIR)

    mlflow.set_experiment("traffic-sign-detection")

    with mlflow.start_run():
        metrics = evaluate_model(
            "traffic-sign-model.pkl", X_valid)
        
        metrics_dict = {
            "mAP50B": metrics['mAP50B'],  
            "mAP50-95B": metrics['mAP50-95B'],
            "precisionB": metrics['precisionB'], 
            "recallB": metrics['recallB']  
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
