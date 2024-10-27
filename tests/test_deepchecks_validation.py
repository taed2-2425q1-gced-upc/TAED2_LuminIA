"""Test suite for the deepchecks_validation module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.validation.deepchecks_validation import (
    load_image_paths,
    load_yolo_annotations,
    load_images,
    create_batches,
)

@pytest.fixture(scope="module")
def setup_files():
    """Set up temporary image files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepare temporary image files for testing
        test_image_paths = [
            Path(temp_dir) / "image1.jpg",
            Path(temp_dir) / "image2.jpg",
            Path(temp_dir) / "image3.jpg",
        ]

        # Create a temporary file for the test image paths
        test_image_paths_file = Path(temp_dir) / "test_images.txt"

        # Create the file with test image paths
        with open(test_image_paths_file, 'w', encoding='utf-8') as f:
            for path in test_image_paths:
                f.write(str(path) + "\n")

        # Create empty test images
        for path in test_image_paths:
            img = Image.new('RGB', (100, 100))  # Create a black image of 100x100
            img.save(path)

        yield test_image_paths_file, test_image_paths  # Return the file and paths for use in tests

def test_load_image_paths(setup_files):
    """Test the load_image_paths function."""
    test_image_paths_file, image_paths = setup_files
    loaded_image_paths = load_image_paths(test_image_paths_file)
    assert len(loaded_image_paths) == 3  # Ensure it loads the 3 images
    for original_path, loaded_path in zip(image_paths, loaded_image_paths):
        assert loaded_path.name == Path(original_path).name

def test_load_yolo_annotations(setup_files):
    """Test the load_yolo_annotations function."""
    test_image_paths_file, image_paths = setup_files
    # Create test YOLO annotation files
    annotations_content = "0 0.5 0.5 1 1\n"  # A simple example of annotation
    for path in image_paths:  # Use paths generated in setup_files
        annotation_file = Path(path).with_suffix('.txt')
        with open(annotation_file, 'w', encoding='utf-8') as f:
            f.write(annotations_content)

    loaded_image_paths = [Path(p) for p in image_paths]
    annotations = load_yolo_annotations(loaded_image_paths)

    assert len(annotations) == len(image_paths)
    for annotation in annotations:
        assert isinstance(annotation, torch.Tensor)
        assert annotation.size(1) == 5  # Should have 5 columns (class_id, x_min, y_min, w, h)

def test_load_images(setup_files):
    """Test the load_images function."""
    test_image_paths_file, image_paths = setup_files
    loaded_image_paths = [Path(p) for p in image_paths]
    images = load_images(loaded_image_paths)

    assert len(images) == len(image_paths)
    for img in images:
        assert isinstance(img, np.ndarray)
        assert img.shape == (100, 100, 3)  # Should be an RGB image of 100x100

def test_create_batches(setup_files):
    """Test the create_batches function."""
    test_image_paths_file, image_paths = setup_files
    loaded_image_paths = [Path(p) for p in image_paths]
    dummy_annotations = [torch.tensor([[0, 0, 0, 100, 100]]) for _ in image_paths]

    batches = list(create_batches(loaded_image_paths, dummy_annotations, batch_size=2))

    assert len(batches) == 2  # Should have 2 batches for 3 images (2 + 1 in the last)
    assert len(batches[0]['images']) == 2  # The first batch should contain 2 images
    assert len(batches[1]['images']) == 1  # The second batch should contain 1 image

# Ensure all tests are executed
if __name__ == "__main__":
    pytest.main()
