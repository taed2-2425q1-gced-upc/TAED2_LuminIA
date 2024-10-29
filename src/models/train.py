from pathlib import Path
import numpy as np
import random
import mlflow
import pandas as pd
import yaml
from codecarbon import EmissionsTracker
from ultralytics import YOLO
import json
import shutil
import torch 

from src.config import PROJ_ROOT, METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, DATA_DIR

import json


file_path = 'settings.json'
# Variable con el nuevo valor para 'datasets_dir'
nuevo_datasets_dir = str(PROJ_ROOT)
# Leer el archivo JSON
with open(file_path, 'r') as file:
    data = json.load(file)
# Modificar el valor de 'datasets_dir'
data['datasets_dir'] = nuevo_datasets_dir
with open(file_path, 'w') as file:
    json.dump(data, file, indent=4)
print("El archivo JSON ha sido actualizado.")


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
        'train': str(TRAIN_IMAGES_PATH), 
        'val': str(TEST_IMAGES_PATH), 
        'nc': nc,
        'names': {i: name for i, name in enumerate(class_names)} 
    }

    # Write the dictionary to a YAML file if needed (for the YOLO model)
    yaml_file = 'dataset.yaml'
    with open(yaml_file, 'w') as file:
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
        ts_model.train(data=yaml_file, 
                       epochs=train_params["epochs"], 
                       #batch_size=train_params["batch_size"], 
                       #learning_rate=train_params["learning_rate"]
                       )

    # Log the CO2 emissions to MLflow
    emissions = pd.read_csv(emissions_output_folder / "emissions.csv")
    emissions_metrics = emissions.iloc[-1, 4:13].to_dict()
    emissions_params = emissions.iloc[-1, 13:].to_dict()

    mlflow.log_params(emissions_params)
    mlflow.log_metrics(emissions_metrics)

    # Save the model as a pickle file
    #ts_model.save(str(MODELS_DIR / "ts_model.pt"))
    #ts_model.save("ts_model.pt")

    #Save the model in the models directory    
    train_runs_path = Path("runs/detect")
    exp_folders = sorted(train_runs_path.glob("train*"), key=lambda p: p.stat().st_mtime)
    latest_exp_folder = exp_folders[-1]
    default_model_path = latest_exp_folder / "weights/best.pt"
    model_save_path = MODELS_DIR / "ts_model.pt"
    shutil.copy(default_model_path, model_save_path)
    print(f"Model from {default_model_path} copied to {model_save_path}")