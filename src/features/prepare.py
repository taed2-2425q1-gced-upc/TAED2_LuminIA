import os
from sklearn.model_selection import train_test_split
import yaml
from src.config.config import (
    PARAMS_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR_TS,
)

def load_parameters():
    """Load parameters from the global PARAMS_PATH."""
    with open(PARAMS_PATH, "r") as params_file:  # Usa PARAMS_PATH directamente
        try:
            params = yaml.safe_load(params_file)
            return params.get("prepare", {})
        except yaml.YAMLError as exc:
            print(exc)
            return {}


def get_image_files():
    """Get a list of image files from the raw data directory."""
    return [f for f in os.listdir(RAW_DATA_DIR_TS) if f.endswith('.jpg')]


def split_data(image_files, params):
    """Split the dataset into training and testing sets."""
    train_files, test_files = train_test_split(image_files, test_size=params["test_size"], random_state=params["random_state"])
    return train_files, test_files

def write_file_list(file_list, output_path):
    """Write a list of file paths to a specified output file with a base directory prefix."""
    with open(output_path, 'w') as f:
        for item in file_list:
            # Concatenar la ruta base con el nombre del archivo
            full_path = f"{RAW_DATA_DIR_TS}/{item}"
            f.write("%s\n" % full_path)
            

def prepare_data():
    """Main function to prepare the data."""
    # Load parameters
    params = load_parameters()

    # Get list of all image files
    image_files = get_image_files()

    # Split into training and validation sets
    train_files, test_files = split_data(image_files, params)

    # Define output files
    train_image_list = PROCESSED_DATA_DIR / params["train"]
    test_image_list = PROCESSED_DATA_DIR / params["test"]

    # Write the split file lists to disk
    write_file_list(train_files, train_image_list)
    write_file_list(test_files, test_image_list)

if __name__ == "__main__":
    prepare_data()