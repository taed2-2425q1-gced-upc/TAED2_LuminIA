"""Validate datasets for object detection tasks using Deepchecks."""

from pathlib import Path
import random  # Mover a la parte superior
import numpy as np
from PIL import Image
import torch
from deepchecks.vision import VisionData
from deepchecks.vision.suites import data_integrity, train_test_validation
from src.config.config import REPORTS_DIR, TRAIN_IMAGES_PATH, TEST_IMAGES_PATH

OUTPUT_FILE = REPORTS_DIR / "deepchecks_validation.html"

def load_image_paths(split_file):
    """
    Load image paths from the split file (which contains paths to images).
    """
    image_paths = []
    split_file_path = Path(split_file)

    if not split_file_path.exists():
        raise FileNotFoundError(f"Split file {split_file} does not exist.")

    with open(split_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            image_path = line.strip()
            # Convert to Path object and ensure it's absolute
            image_paths.append(Path(image_path).resolve())

    return image_paths

def load_yolo_annotations(image_paths):
    """
    Load the YOLO annotations for the given image paths and convert them to Deepchecks format.
    """
    annotations = []
    for image_path in image_paths:
        annotation_file = image_path.with_suffix('.txt')
        if annotation_file.exists():
            with open(annotation_file, 'r', encoding='utf-8') as f:
                boxes = []
                for line in f:
                    # Each line contains class_id, x_center, y_center, width, height
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

# Load the image paths from the text files containing paths to images
train_image_paths = load_image_paths(TRAIN_IMAGES_PATH)
test_image_paths = load_image_paths(TEST_IMAGES_PATH)

# Load the YOLO annotations for both train and test datasets
train_annotations = load_yolo_annotations(train_image_paths)
test_annotations = load_yolo_annotations(test_image_paths)

# Debugging: Print the lengths and first few items of the image paths and annotations
print("Train Image Paths:", len(train_image_paths), "First few:", train_image_paths[:3])
print("Train Annotations:", len(train_annotations), "First few:", train_annotations[:3])
print("Test Image Paths:", len(test_image_paths), "First few:", test_image_paths[:3])
print("Test Annotations:", len(test_annotations), "First few:", test_annotations[:3])

# Create VisionData objects for train and test datasets
train_data = VisionData(
    batch_loader=create_batches(train_image_paths, train_annotations),
    task_type='object_detection',
    reshuffle_data=False
)

test_data = VisionData(
    batch_loader=create_batches(test_image_paths, test_annotations),
    task_type='object_detection',
    reshuffle_data=False
)

# Create the custom suite
custom_suite = data_integrity()
custom_suite.add(train_test_validation())

# Run the suite
result = custom_suite.run(train_data, test_data)

# Save the result as HTML
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

result.save_as_html(str(OUTPUT_FILE))
