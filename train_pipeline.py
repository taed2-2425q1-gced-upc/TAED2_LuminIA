from ultralytics import YOLO
import json

def train_eval(yaml_file_path, 
               model_save_path='/Users/adriancerezuelahernandez/Desktop/Q7/TAED2/project1/results/yolov8_model.pt', 
               metrics_save_path='/Users/adriancerezuelahernandez/Desktop/Q7/TAED2/project1/results/metrics.json'):

    model = YOLO("yolov8m.yaml")
    results = model.train(
    data=yaml_file_path, 
    epochs = 100
    )
    metrics = model.val(data=yaml_file_path)
    model.save(model_save_path)

    # Save the evaluation metrics to a JSON file
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f)

    # Return the trained model and metrics
    return model, metrics, model_save_path
