from pathlib import Path
import pickle

from ultralytics import YOLO
import pandas as pd
import json
import mlflow
import pickle
from pathlib import Path
import yaml
from codecarbon import EmissionsTracker

import typer
from loguru import logger
from tqdm import tqdm

from taed2_luminia.config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR

mlflow.set_experiment("traffic-signs-detection")
mlflow.sklearn.autolog(log_model_signatures=False, log_datasets=False)

with mlflow.start_run():
    
    # Path of the parameters file
    params_path = Path("params.yaml")

    # Path of the prepared data folder
    input_folder_path = PROCESSED_DATA_DIR

    # Read training images list
    with open(input_folder_path / "train_images.txt", "r") as file:
        X_train_list = file.readlines()
    X_train_list = [x.strip() for x in X_train_list]  # To remove newline characters


    # This will currently give us the random state
    with open(params_path, "r", encoding="utf8") as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["train"]
        except yaml.YAMLError as exc:
            print(exc)
    
    class_names = ['prohibitory',
    'danger',
    'mandatory',
    'other']

    # Create a dictionary for the YAML structure
    data = {
        'train': X_train_list,
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }

    # Write the dictionary to a YAML file
    yaml_file = 'dataset.yaml'
    with open(yaml_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    
    # ------------- TRAINING -------------

    model = YOLO("yolov8m.yaml", random_state=params["random_state"])
    
    emissions_output_folder = METRICS_DIR
    with EmissionsTracker(
        project_name="iowa-house-prices",
        measure_power_secs=1,
        tracking_mode="process",
        output_dir=emissions_output_folder,
        output_file="emissions.csv",
        on_csv_write="append",
        default_cpu_power=45,
    ):
        model.train(data=yaml_file, epochs=100)
    
    # Log the CO2 emissions to MLflow
    emissions = pd.read_csv(emissions_output_folder / "emissions.csv")
    emissions_metrics = emissions.iloc[-1, 4:13].to_dict()
    emissions_params = emissions.iloc[-1, 13:].to_dict()
    mlflow.log_params(emissions_params)
    mlflow.log_metrics(emissions_metrics)  

    # Save the model as a pickle file
    Path("models").mkdir(exist_ok=True)

    with open(MODELS_DIR / "traffic-sign-model.pkl", "wb") as pickle_file:
        pickle.dump(model, pickle_file)
