import os
from sklearn.model_selection import train_test_split
import yaml

def data():
    # CREATING YAML FILE AND SPLITTING DATA INTO TRAIN AND VAL --------------------------------------------

    class_names = ['prohibitory', 'danger', 'mandatory', 'other']
    data_dir = '/Users/adriancerezuelahernandez/Desktop/Q7/TAED2/project1/ts'
    train_image_list = '/Users/adriancerezuelahernandez/Desktop/Q7/TAED2/project1/results/train_images.txt'
    val_image_list = '/Users/adriancerezuelahernandez/Desktop/Q7/TAED2/project1/results/val_images.txt'

    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    # Split into training and validation sets
    train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

    # Write the split file lists to disk
    with open(train_image_list, 'w') as f:
        for item in train_files:
            f.write("%s\n" % os.path.join(data_dir, item))

    with open(val_image_list, 'w') as f:
        for item in val_files:
            f.write("%s\n" % os.path.join(data_dir, item))

    # Create a dictionary for the YAML structure
    data = {
        'train': train_image_list,
        'val': val_image_list,
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }

    # Write the dictionary to a YAML file
    yaml_file = 'dataset.yaml'
    with open(yaml_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    return os.path.abspath(yaml_file)  # Return full path of the YAML file