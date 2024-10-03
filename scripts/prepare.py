import os
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path

#Path from the parameters file
params_path = Path("params.yaml")

#Read data preparation parameters
with open(params_path, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)
        params = params["prepare"]
    except yaml.YAMLError as exc:
        print(exc)

# Define paths
raw_data_dir = Path('data/raw/ts')
output_directory = Path('data/processed')

#Define output files
train_image_list = output_directory / params["train"]
test_image_list = output_directory / params["test"]

# Get list of all image files
image_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.jpg')]

# Split into training and validation sets
train_files, test_files = train_test_split(image_files, test_size=params["test_size"], random_state=params["random_state"])

# Write the split file lists to disk
with open(train_image_list, 'w') as f:
    for item in train_files:
        f.write("%s\n" % os.path.join(raw_data_dir, item))

with open(test_image_list, 'w') as f:
    for item in test_files:
        f.write("%s\n" % os.path.join(raw_data_dir, item))
