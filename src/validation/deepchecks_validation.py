from pathlib import Path
import numpy as np
from PIL import Image
import torch
from deepchecks.vision import VisionData
from deepchecks.vision.suites import data_integrity, train_test_validation
from src.config.config import(
     REPORTS_DIR, PROJ_ROOT, TRAIN_IMAGES_PATH, TEST_IMAGES_PATH
)
import random

OUTPUT_FILE = REPORTS_DIR / "deepchecks_validation.html"

def load_image_paths(split_file):
    """
    Load image paths from the split file.
    """
    image_paths = []
    split_file_path = Path(split_file)
    
    if not split_file_path.exists():
        raise FileNotFoundError(f"Split file {split_file} does not exist.")
    
    with open(split_file_path, 'r') as f:
        for line in f:
            image_path = line.strip()
            image_paths.append(PROJ_ROOT / image_path)
    
    return image_paths

def load_yolo_annotations(image_paths):
    """
    Load the YOLO annotations for the given image paths and convert them to Deepchecks format.
    """
    annotations = []
    for image_path in image_paths:
        annotation_file = image_path.with_suffix('.txt')
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                boxes = []
                for line in f:
                    # Each line contains class_id, x_center, y_center, width, height
                    # Parse the line and convert YOLO format to Deepchecks format
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    img_width, img_height = Image.open(image_path).size
                    x_min = int((x_center - width / 2) * img_width)
                    y_min = int((y_center - height / 2) * img_height)
                    w = int(width * img_width)
                    h = int(height * img_height)
                    boxes.append([int(class_id), x_min, y_min, w, h])
                annotations.append(torch.tensor(boxes))
        else:
            annotations.append(torch.tensor([]))
    return annotations

def load_images(image_paths):
    """
    Load images and convert them to NumPy arrays.
    """
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            images.append(np.array(img))
    return images

def create_batches(image_paths, annotations, batch_size=32):
    """
    Create batches of data in the required format.
    """
    combined = list(zip(image_paths, annotations))
    random.shuffle(combined)
    image_paths[:], annotations[:] = zip(*combined)
    
    for i in range(0, len(image_paths), batch_size):
        batch_images = image_paths[i:i + batch_size]
        batch_annotations = annotations[i:i + batch_size]
        yield {
            'images': load_images(batch_images),
            'labels': batch_annotations
        }


# Load the YOLO annotations for both train and test datasets
train_annotations = load_yolo_annotations(TRAIN_IMAGES_PATH)
test_annotations = load_yolo_annotations(TEST_IMAGES_PATH)

# Debugging: Print the first few images and annotations passed to Deepchecks
print("First few images and annotations passed to Deepchecks:")
for i, (image, annotation) in enumerate(zip(TRAIN_IMAGES_PATH[:3], train_annotations[:3])):
    print(f"Image {i}: {image}")
    print(f"Annotation {i}: {annotation}")

# Create VisionData objects for train and test datasets with the correct task_type (Object Detection)
train_data = VisionData(batch_loader=create_batches(TRAIN_IMAGES_PATH, train_annotations), task_type='object_detection', reshuffle_data=False)
test_data = VisionData(batch_loader=create_batches(TEST_IMAGES_PATH, test_annotations), task_type='object_detection', reshuffle_data=False)

# Create the custom suite
custom_suite = data_integrity()
custom_suite.add(train_test_validation())

# Run the suite
result = custom_suite.run(train_data, test_data)

# Save the result as HTML
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

result.save_as_html(str(OUTPUT_FILE))