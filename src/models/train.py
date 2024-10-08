import pickle
from pathlib import Path
import numpy as np
import random
import mlflow
import pandas as pd
import yaml
from codecarbon import EmissionsTracker
from ultralytics import YOLO

from src.config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR

mlflow.set_experiment("traffic-signs")
mlflow.sklearn.autolog(log_model_signatures=False, log_datasets=False)

#config_path = os.path.join(os.getcwd(), 'settings.json')
#os.environ['ULTRALYTICS_CONFIG'] = config_path
## Load your settings from JSON
#with open(config_path, 'r') as f:
#    config = json.load(f)

TRAIN_IMAGES_PATH = "data/processed/train_images.txt"
TEST_IMAGES_PATH = "data/processed/test_images.txt"

with mlflow.start_run():
    # Path of the parameters file
    params_path = Path("params.yaml")

    # Read data preparation parameters
    with open(params_path, "r", encoding="utf8") as params_file:
        try:
            params = yaml.safe_load(params_file)
            prepare_params = params["prepare"]  # Extract prepare parameters
            train_params = params["train"]  # Extract train parameters
        except yaml.YAMLError as exc:
            print(exc)

    # Read class names and number of classes from prepare parameters
    print("prepare_params['names']:", prepare_params["names"])
    print("Type of prepare_params['names']:", type(prepare_params["names"]))

    class_names = list(prepare_params["names"].values())
    nc = prepare_params["nc"]

    # Create the data dictionary using data lists and data preparation parameters
    data = {
        'train': "/home/ramon/Desktop/GCED/TAED2_LuminIA/data/processed/train_images.txt", #prepare_params["train"],
        'val': "/home/ramon/Desktop/GCED/TAED2_LuminIA/data/processed/test_images.txt", #prepare_params["test"],
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
    Path(MODELS_DIR).mkdir(exist_ok=True)

    with open(MODELS_DIR / "ts_model.pkl", "wb") as pickle_file:
        pickle.dump(ts_model, pickle_file)