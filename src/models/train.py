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

from src.config.config import PROJ_ROOT, METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR, DATA_DIR, TRAIN_IMAGES_PATH, TEST_IMAGES_PATH, SETTINGS_PATH, PARAMS_PATH

import json


def update_datasets_dir(file_path, new_datasets_dir):
    """Update the datasets directory in the JSON configuration."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    data['datasets_dir'] = new_datasets_dir
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print("El archivo JSON ha sido actualizado.")


def read_params(params_path):
    """Read parameters from a YAML file."""
    with open(params_path, "r", encoding="utf8") as params_file:
        try:
            params = yaml.safe_load(params_file)
            return params["prepare"], params["train"]
        except yaml.YAMLError as exc:
            print(exc)
            return None, None



with mlflow.start_run():
    # Path of the parameters file
    params_path = PARAMS_PATH
    # Read parameters
    with open(PARAMS_PATH, "r", encoding="utf8") as params_file:
        try:
            params = yaml.safe_load(params_file)
            prepare_params = params["prepare"]  
            train_params = params["train"]
        except yaml.YAMLError as exc:
            print(exc)
    
def create_data_dict(prepare_params):
    """Create the data dictionary for training."""
    class_names = list(prepare_params["names"].values())
    nc = prepare_params["nc"]
    
    return {
        'train': str(TRAIN_IMAGES_PATH), 
        'val': str(TEST_IMAGES_PATH), 
        'nc': nc,
        'names': {i: name for i, name in enumerate(class_names)} 
    }


def save_yaml(data, yaml_file):
    """Save the data dictionary as a YAML file."""
    with open(yaml_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

# ============== #
# MODEL TRAINING #
# ============== #

def train_model(data_dict, train_params):
    """Train the YOLO model."""
    # Set the random state for reproducibility
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
        ts_model.train(data=data_dict, 
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
    # Define la ruta de train_runs_path usando PROJ_ROOT
    train_runs_path = PROJ_ROOT / "runs" / "detect"
    # Verifica que el directorio existe
    if not train_runs_path.exists():
        print(f"Error: El directorio {train_runs_path} no existe.")
        return
    # Obtiene las carpetas de experimentos ordenadas por fecha de modificación
    exp_folders = sorted(train_runs_path.glob("train*"), key=lambda p: p.stat().st_mtime)
    # Verifica si hay carpetas de experimentos disponibles
    if not exp_folders:
        print("No se encontraron carpetas de experimentos en el directorio.")
        return
    # Toma la carpeta más reciente
    latest_exp_folder = exp_folders[-1]
    # Define la ruta del modelo a copiar
    default_model_path = latest_exp_folder / "weights/best.pt"
    # Verifica si el modelo por defecto existe
    if not default_model_path.exists():
        print(f"Error: El modelo {default_model_path} no existe.")
        return
    # Define la ruta donde se guardará el modelo
    model_save_path = MODELS_DIR / "ts_model.pt"
    # Copia el modelo al directorio de modelos
    shutil.copy(default_model_path, model_save_path)
    print(f"Model from {default_model_path} copied to {model_save_path}")




def print_exp_folders():
    """Print the list of experiment folders."""
    train_runs_path = PROJ_ROOT / "runs" / "detect"  # Cambia esta línea según tu estructura de directorios
    exp_folders = sorted(train_runs_path.glob("train*"), key=lambda p: p.stat().st_mtime)
    
    if exp_folders:
        print("Experiment folders:")
        for folder in exp_folders:
            print(folder)
    else:
        print("No experiment folders found matching 'train*'.")

    latest_exp_folder = exp_folders[-1]
    print("last folder") 
    print(latest_exp_folder)



def main():

    # Update datasets directory in the JSON file
    update_datasets_dir(SETTINGS_PATH, str(PROJ_ROOT))

    mlflow.set_experiment("traffic-signs")
    mlflow.sklearn.autolog(log_model_signatures=False, log_datasets=False)

    with mlflow.start_run():
        # Read parameters from the config
        prepare_params, train_params = read_params(PARAMS_PATH)
        if prepare_params is None or train_params is None:
            return  # Exit if params cannot be read

        # Create data dictionary
        data_dict = create_data_dict(prepare_params)

        # Save the data dictionary as a YAML file
        yaml_file = 'dataset.yaml'
        save_yaml(data_dict, yaml_file)

        # Train the model and track emissions
        emissions_output_folder = train_model(yaml_file, train_params)

        # Log emissions to MLflow
        log_emissions_to_mlflow(emissions_output_folder)

        # Save the model to the models directory
        save_model_to_directory()
    
if __name__ == "__main__":
    main()