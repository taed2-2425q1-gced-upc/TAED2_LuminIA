import os
import pytest
import torch
from ultralytics import YOLO
from src.utils import load_config, load_image
from PIL import Image


# Model configuration path
CONFIG_PATH = 'models/yolov3_ts_train.cfg'
# Path to the model weights file
WEIGHTS_PATH = 'models/best.pt'


@pytest.fixture
def yolo_model():
    """Fixture to load the YOLO model."""
    WEIGHTS_PATH = 'C:/Users/laia2/Desktop/TAED2/TAED2_LuminIA/models/best.pt'  # Cambia a la ruta correcta
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
        "data/raw/ts/00426.jpg",
        "data/raw/ts/00317.jpg",
        "data/raw/ts/00624.jpg"
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
    valid_image_path = "data/raw/ts/00426.jpg"  # Usar una de las imágenes válidas
    invalid_image_path = os.path.join('models', 'invalid_image.txt')

    try:
        img = load_image(valid_image_path)
        assert isinstance(img, Image.Image), "Loaded image is not of type PIL.Image.Image"
    except Exception as e:
        pytest.fail(f"Exception occurred while loading valid image: {e}")

    with pytest.raises(IOError):
        load_image(invalid_image_path)  # Attempt to load an invalid image


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

    expected_output_path = image_path.replace(".jpg", ".txt")

    if os.path.exists(expected_output_path):
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

        # Leer las clases esperadas desde el archivo correspondiente
        expected_output_path = img_path.replace(".jpg", ".txt")
        if os.path.exists(expected_output_path):
            with open(expected_output_path, 'r') as file:
                expected_output = file.read().strip()

                # Obtener todas las clases esperadas del archivo (suponiendo que el archivo tiene una clase por línea)
                expected_classes = [float(line.split()[0]) for line in expected_output.splitlines()]

                print(f"Image: {img_path}")
                print(f"Expected Classes: {expected_classes}")
                print(f"Predicted Classes: {predicted_classes}")

                # Verificar que todas las clases esperadas estén en las clases predichas
                for expected_class in expected_classes:
                    assert expected_class in predicted_classes, f"Expected class {expected_class} was not predicted."
