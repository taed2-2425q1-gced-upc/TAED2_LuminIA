import os
import pytest
import torch
from ultralytics import YOLO
from src.utils import load_config, load_image
from PIL import Image
from src.config.config import (
    PROJ_ROOT,
    CONFIG_PATH,
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    METRICS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    PARAMS_PATH,
    WEIGHTS_PATH,
    RAW_DATA_DIR_TS,
)
from pathlib import Path



@pytest.fixture
def yolo_model():
    
    model = YOLO(WEIGHTS_PATH)
    return model

def test_model_loading(yolo_model):
    """Test to verify that the YOLO model loads correctly and applies weights."""
    assert yolo_model is not None, "Model did not load correctly"
    
    # Simulate inference with a dummy input tensor
    dummy_input = torch.randn(1, 3, 416, 416)  # Dummy image of size 416x416
    output = yolo_model(dummy_input)

    assert output is not None, "Model output is None"
    assert len(output) > 0, "Model output is empty"

@pytest.fixture
def yolo_validation_data():
    """Fixture to load a specific set of images for validation."""
    image_paths = [
        RAW_DATA_DIR_TS / "00426.jpg",
        RAW_DATA_DIR_TS / "00317.jpg",
        RAW_DATA_DIR_TS / "00624.jpg"
    ]
    
    images = [load_image(path) for path in image_paths]
    
    return images, image_paths


def test_load_config():
    """Test to ensure the YOLO configuration file is loaded correctly."""
    config = load_config(CONFIG_PATH)

    assert config is not None, "Configuration file did not load correctly"
    print("Loaded configuration sections:", config.sections())
    assert 'net' in config, "Configuration file must contain 'net' section"
    assert 'convolutional' in config, "Configuration file must contain at least one 'convolutional' section"
    
    assert 'batch' in config['net'], "The 'net' section must contain 'batch' parameter"
    assert 'subdivisions' in config['net'], "The 'net' section must contain 'subdivisions' parameter"



def test_load_image():
    """Test to verify that images load correctly."""
    valid_image_path = RAW_DATA_DIR_TS / "00426.jpg"  # Usar una de las imágenes válidas
    invalid_image_path = os.path.join('models', 'invalid_image.txt')

    try:
        img = load_image(valid_image_path)
        assert isinstance(img, Image.Image), "Loaded image is not of type PIL.Image.Image"
    except Exception as e:
        pytest.fail(f"Exception occurred while loading valid image: {e}")

    with pytest.raises(IOError):
        load_image(invalid_image_path)  # Attempt to load an invalid image


from pathlib import Path
import os

def compare_predictions_with_expected(yolo_model, image_path):
    """Compare model predictions with expected labels."""
    input_image = load_image(image_path)
    predictions = yolo_model.predict(input_image)

    formatted_predictions = []
    for pred in predictions:
        if hasattr(pred, 'boxes'):
            for box in pred.boxes:
                formatted_predictions.append({
                    'Class': box.cls.item(),
                    'Confidence': box.conf.item(),
                    'Coordinates': box.xyxy[0].tolist() if len(box.xyxy) > 0 else []
                })

    # Cambiar la extensión de .jpg a .txt usando with_suffix
    expected_output_path = Path(image_path).with_suffix(".txt")  # Cambia .jpg a .txt

    if expected_output_path.exists():
        with open(expected_output_path, 'r') as file:
            expected_output = file.read().strip()

            expected_values = expected_output.split()
            expected_class = float(expected_values[0])
            expected_coords = [float(val) for val in expected_values[2:]]

            for pred in formatted_predictions:
                assert 'Class' in pred, "Missing 'Class' key in prediction"
                assert 'Confidence' in pred, "Missing 'Confidence' key in prediction"
                assert 'Coordinates' in pred, "Missing 'Coordinates' key in prediction"

                coordinates = pred['Coordinates']
                assert len(coordinates) == 4, "Coordinates must contain 4 values"

    else:
        print(f"No ground truth label found for {image_path}.")

def test_yolo_output_structure(yolo_model, yolo_validation_data):
    """Test to verify the structure of YOLO predictions against expected labels."""
    for img_path in yolo_validation_data[1]:  # Usar los paths de la fixture
        compare_predictions_with_expected(yolo_model, img_path)

def test_yolo_class_prediction(yolo_model, yolo_validation_data):
    """Test to verify YOLO model predicts the correct classes for images with multiple objects."""
    for img_path in yolo_validation_data[1]:  # Usar los paths de la fixture
        input_image = load_image(img_path)
        predictions = yolo_model.predict(input_image)

        # Extraer todas las clases predichas de las cajas (bounding boxes)
        predicted_classes = []
        for pred in predictions:
            if hasattr(pred, 'boxes'):
                for box in pred.boxes:
                    predicted_classes.append(box.cls.item())

        # Cambiar la extensión de .jpg a .txt usando with_suffix
        expected_output_path = Path(img_path).with_suffix(".txt")  # Cambia .jpg a .txt

        if expected_output_path.exists():
            with open(expected_output_path, 'r') as file:
                expected_output = file.read().strip()

                # Obtener todas las clases esperadas del archivo
                expected_classes = [float(line.split()[0]) for line in expected_output.splitlines()]

                print(f"Image: {img_path}")
                print(f"Expected Classes: {expected_classes}")
                print(f"Predicted Classes: {predicted_classes}")

                # Verificar que todas las clases esperadas estén en las clases predichas
                for expected_class in expected_classes:
                    assert expected_class in predicted_classes, f"Expected class {expected_class} was not predicted."
