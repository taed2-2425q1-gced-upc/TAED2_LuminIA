from ultralytics import YOLO
import os
import json

from src.config.config import TRAIN_IMAGES_PATH,WEIGHTS_PATH, METRICS_DIR


#############################################################
# NUMBER OF PARAMS
#############################################################

model = YOLO(WEIGHTS_PATH)

total_params = sum(p.numel() for p in model.parameters())

merged_params = {
    "total_params": total_params
}

with open(METRICS_DIR / "total_params.json", "w", encoding='utf-8') as file:
    json.dump(merged_params, file, indent=4)

###############################################################
# DATASET SIZE
###############################################################

def compute_weightsum(txt_path):
    wsum = 0
    
    with open(txt_path, 'r') as file:
        for line in file:
            img_path = os.path.abspath(line.strip())

            if os.path.exists(img_path):
                img_size = os.path.getsize(img_path)
                wsum += img_size
            else:
                print(f"Image not found: {img_path}")

    return wsum

weight_total_sum = compute_weightsum(TRAIN_IMAGES_PATH)

merged_metrics = {
    "total_weight_sum": weight_total_sum
}


with open(METRICS_DIR / "dataset_size.json", "w", encoding='utf-8') as file:
    json.dump(merged_metrics, file, indent=4)