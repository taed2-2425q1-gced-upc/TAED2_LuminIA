!pip install ultralytics

!pip install -U ipywidgets

!pip install wandb

import os
import shutil
from sklearn.model_selection import train_test_split
import cv2
from ultralytics import YOLO
import glob
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import iplot
import yaml

output_directory = '/data/processed/'

# Iterate over all the files and folders in the directory
for filename in os.listdir(output_directory):
    file_path = os.path.join(output_directory, filename)
    
    # Check if it's a file or directory and remove it accordingly
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)  # Remove the file
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)  # Remove the directory
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')

print("All files and folders have been removed.")


# class_names = ['prohibitory',
# 'danger',
# 'mandatory',
# 'other']

from pathlib import Path
params_path = Path("params.yaml")

with open(params_path, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["prepare"]
    except yaml.YAMLError as exc:
        print(exc)

data_dir = '/data/raw/ts'

image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]

train_image_list=params["train"]
test_image_list=params["test"]

# Get list of all image files
image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]

# Split into training and validation sets
train_files, test_files = train_test_split(image_files, test_size=params["test_size"],
    random_state=params["random_state"])

# Write the split file lists to disk
with open(train_image_list, 'w') as f:
    for item in train_files:
        f.write("%s\n" % os.path.join(data_dir, item))

with open(test_image_list, 'w') as f:
    for item in test_files:
        f.write("%s\n" % os.path.join(data_dir, item))

# # Create a dictionary for the YAML structure
# data = {
#     'train': train_image_list,
#     'test': test_image_list,
#     'nc': len(class_names),
#     'names': {i: name for i, name in enumerate(class_names)}
# }

# # Write the dictionary to a YAML file
# yaml_file = 'dataset.yaml'
# with open(yaml_file, 'w') as file:
#     yaml.dump(data, file, default_flow_style=False)