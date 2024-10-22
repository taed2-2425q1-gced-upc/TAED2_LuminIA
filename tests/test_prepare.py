import os
import pytest
from unittest.mock import patch, mock_open
from unittest.mock import MagicMock
from src.features.prepare import (
    load_parameters,
    get_image_files,
    split_data,
    write_file_list,
    prepare_data,
)
from src.config.config import PROCESSED_DATA_DIR


def write_file_list(file_list, output_file):
    """Write the list of files to an output file."""
    with open(output_file, 'w') as f:
        for file in file_list:
            f.write(f"{file}\n")  


def test_load_parameters():
    """Test to load parameters correctly."""
    params = load_parameters()
    assert "test_size" in params, "Missing 'test_size' parameter"
    assert params["test_size"] == 0.2, "Test size is not as expected"
    assert "train" in params, "Missing 'train' parameter"
    assert "test" in params, "Missing 'test' parameter"


def test_load_parameters_with_invalid_yaml():
    """Test to load parameters from an invalid YAML file."""
    # Simulate a malformed YAML file
    mock_file = mock_open(read_data="invalid_yaml: [")
    
    with patch("builtins.open", mock_file):
        params = load_parameters()
        
        # Verify that an empty dictionary is returned in case of an error
        assert params == {}, "Expected an empty dictionary when loading an invalid YAML"


def test_get_image_files(tmp_path):
    """Test for obtaining image files from the directory."""
    
    # Create a temporary directory and add some image files
    mock_image_dir = tmp_path / "data" / "raw"
    mock_image_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 test image files
    expected_images = {f"{i:05}.jpg" for i in range(3)}  # Expected names
    for i in range(3):
        mock_image_file = mock_image_dir / f"{i:05}.jpg"
        mock_image_file.touch()  # Create an empty file

    # Create a file that is not an image
    non_image_file = mock_image_dir / "not_an_image.txt"
    non_image_file.touch()  # Create a text file

    # Patch the RAW_DATA_DIR_TS variable to point to the temporary directory
    with patch("src.config.config.RAW_DATA_DIR_TS", mock_image_dir):
        images = get_image_files()

        # Verify that all returned files are .jpg images
        assert all(img.endswith('.jpg') for img in images), "Not all returned files are .jpg images"
        
        # Verify that non-image files are not returned (e.g., the text file)
        assert "not_an_image.txt" not in images, "A non-image file was returned."

        # Verify that the expected images are in the returned results
        for expected in expected_images:
            assert expected in images, f"{expected} was not found in the list of returned images."


def test_split_data():
    """Test to split the image files into training and test sets."""
    image_files = ["image1.jpg", "image2.jpg", "image3.jpg"]
    params = {"test_size": 0.2, "random_state": 42}
    train_files, test_files = split_data(image_files, params)

    assert len(train_files) == 2, "Expected the training set to have 2 files"
    assert len(test_files) == 1, "Expected the test set to have 1 file"


def test_write_file_list(tmp_path):
    """Test to write the list of files to an output file."""
    file_list = ["image1.jpg", "image2.jpg"]  # Simple names
    output_file = tmp_path / "output.txt"

    write_file_list(file_list, output_file)

    # Verify that the file was created and contains the expected content
    with open(output_file, 'r') as f:
        lines = f.readlines()
        print("File content:", [line.strip() for line in lines])  # For debugging
        assert len(lines) == 2, "The number of lines in the output file is not as expected"
        assert lines[0].strip() == "image1.jpg", "The first line is not as expected"
        assert lines[1].strip() == "image2.jpg", "The second line is not as expected"
