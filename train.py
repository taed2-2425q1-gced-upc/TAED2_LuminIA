from ultralytics import YOLO
import pandas as pd
import json
import mlflow
import pickle
from pathlib import Path
import yaml
from codecarbon import EmissionsTracker
from taed2_luminia.config import METRICS_DIR, MODELS_DIR, PROCESSED_DATA_DIR

mlflow.set_experiment("traffic-signs-detection")
mlflow.sklearn.autolog(log_model_signatures=False, log_datasets=False)

with mlflow.start_run():
    params_path = Path("params.yaml")
    input_folder_path = PROCESSED_DATA_DIR  # Path of the prepared data folder
    
    # Read training dataset
    X_train = pd.read_csv(input_folder_path / "X_train.csv")
    
    # Read data preparation parameters
    with open(params_path, "r", encoding="utf8") as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["train"]
        except yaml.YAMLError as exc:
            print(exc)
    
    # -------------------- TRAINING --------------------

    model = YOLO("yolov8m.yaml")
    traffic_model = model(random_state=params["random_state"]) # A DEFINIR

    emissions_output_folder = METRICS_DIR
    #with EmissionsTracker(
     #   project_name="traffic-sign-detection",
      #  measure_power_secs=1,
       # tracking_mode="process",
        #output_dir=emissions_output_folder,
        #output_file="emissions.csv",
        #on_csv_write="append",
        #default_cpu_power=45,
    #):
    # Then fit the model to the training data
    traffic_model.train(data=X_train, epochs=params["epochs"])

    # Log the CO2 emissions to MLflow
    #emissions = pd.read_csv(emissions_output_folder / "emissions.csv")
    #emissions_metrics = emissions.iloc[-1, 4:13].to_dict()
    #emissions_params = emissions.iloc[-1, 13:].to_dict()
    #mlflow.log_params(emissions_params)
    #mlflow.log_metrics(emissions_metrics)

    # Save the model as a pickle file
    Path("models").mkdir(exist_ok=True)

    with open(MODELS_DIR / "traffic-model.pkl", "wb") as pickle_file:
        pickle.dump(traffic_model, pickle_file)
