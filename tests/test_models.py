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
    model = YOLO(WEIGHTS_PATH)
    return model


def test_load_config():
    """Test to ensure the YOLO configuration file is loaded correctly."""
    config = load_config(CONFIG_PATH)

    assert config is not None, "Configuration file did not load correctly"

    print("Loaded configuration sections:", config.sections())
    assert 'net' in config, "Configuration file must contain 'net' section"
    assert 'convolutional' in config, "Configuration file must contain at least one 'convolutional' section"
    
    assert 'batch' in config['net'], "The 'net' section must contain 'batch' parameter"
    assert 'subdivisions' in config['net'], "The 'net' section must contain 'subdivisions' parameter"


def test_model_loading(yolo_model):
    """Test to verify that the YOLO model loads correctly and applies weights."""
    assert yolo_model is not None, "Model did not load correctly"
    
    # Simulate inference with a dummy input tensor
    dummy_input = torch.randn(1, 3, 416, 416)  # Dummy image of size 416x416
    output = yolo_model(dummy_input)

    assert output is not None, "Model output is None"
    assert len(output) > 0, "Model output is empty"


def test_load_image():
    """Test to verify that images load correctly."""
    valid_image_path = os.path.join('models', 'valid_image.jpg')
    invalid_image_path = os.path.join('models', 'invalid_image.txt')

    try:
        img = load_image(valid_image_path)
        assert isinstance(img, Image.Image), "Loaded image is not of type PIL.Image.Image"
    except Exception as e:
        pytest.fail(f"Exception occurred while loading valid image: {e}")

    with pytest.raises(IOError):
        load_image(invalid_image_path)  # Attempt to load an invalid image


@pytest.fixture
def yolo_validation_data():
    """Fixture to load validation image data."""
    val_data_path = "data/processed/test_images.txt"
    with open(val_data_path, 'r') as file:
        image_paths = [line.strip() for line in file.readlines()]
    
    images = [load_image(path) for path in image_paths]
    return images


def test_yolo_with_validation_data(yolo_model, yolo_validation_data):
    """Test YOLO model inference with validation data."""
    assert yolo_validation_data is not None, "No validation images loaded"
    assert len(yolo_validation_data) > 0, "Validation image list is empty"

    for img in yolo_validation_data:
        output = yolo_model(img)  # Perform inference with the model


@pytest.fixture
def test_images():
    """Fixture to load image paths from a text file."""
    test_images_path = "data/processed/test_images.txt"
    with open(test_images_path, 'r') as file:
        image_paths = [line.strip() for line in file.readlines()]
    return image_paths


@pytest.fixture
def yolo_validation_data_reduced():
    """Fixture to load a specific reduced set of images for validation."""
    image_paths = [
        "data/raw/ts/00426.jpg",
        "data/raw/ts/00317.jpg",
        "data/raw/ts/00624.jpg"
    ]
    
    images = [load_image(path) for path in image_paths]
    return images


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


def test_yolo_output_structure_reduced(yolo_model, yolo_validation_data_reduced):
    """Test to verify the structure of YOLO predictions against expected labels."""
    for img_path in [
        "data/raw/ts/00426.jpg",
        "data/raw/ts/00317.jpg",
        "data/raw/ts/00624.jpg"
    ]:
        compare_predictions_with_expected(yolo_model, img_path)


def test_yolo_class_prediction(yolo_model, yolo_validation_data_reduced):
    """Test to verify YOLO model predicts the correct class for a reduced set of images."""
    for img_path in [
        "data/raw/ts/00426.jpg",
        "data/raw/ts/00317.jpg",
        "data/raw/ts/00624.jpg"
    ]:
        input_image = load_image(img_path)
        predictions = yolo_model.predict(input_image)

        predicted_classes = []
        for pred in predictions:
            if hasattr(pred, 'boxes'):
                for box in pred.boxes:
                    predicted_classes.append(box.cls.item())

        expected_output_path = img_path.replace(".jpg", ".txt")

        if os.path.exists(expected_output_path):
            with open(expected_output_path, 'r') as file:
                expected_output = file.read().strip()

                expected_values = expected_output.split()
                expected_class = float(expected_values[0])

                print(f"Image: {img_path}")
                print(f"Expected Class: {expected_class}")
                print(f"Predicted Classes: {predicted_classes}")

                assert expected_class in predicted_classes, f"Expected class {expected_class} was not predicted."


@pytest.fixture
def yolo_metrics_validation_data():
    """Fixture to load all images listed in data/processed/test_images.txt for validation."""
    test_image_file_path = "data/processed/train_images.txt"
    
    with open(test_image_file_path, 'r') as file:
        image_paths = [line.strip() for line in file.readlines() if line.strip()]

    images = [load_image(path) for path in image_paths if os.path.exists(path)]
    
    return images, image_paths
    
"""

def test_yolo_performance(yolo_model, yolo_metrics_validation_data):

    images, image_paths = yolo_metrics_validation_data  # Desempaquetar las imágenes y rutas      

    # Inicializar conteos totales
    total_true_positive = 0
    total_false_positive = 0
    total_false_negative = 0

    for img_path in image_paths:
        # Realizar la predicción con el modelo
        input_image = load_image(img_path)
        predictions = yolo_model.predict(input_image)

        # Formatear las predicciones
        formatted_predictions = []
        for pred in predictions:
            if hasattr(pred, 'boxes'):
                for box in pred.boxes:
                    formatted_predictions.append({
                        'Clase': box.cls.item(),  # Clase predicha
                        'Confianza': box.conf.item(),  # Confianza de la predicción
                        'Coordenadas': box.xyxy[0].tolist() if len(box.xyxy) > 0 else []  # Extraer solo los valores
                    })

        # Obtener la etiqueta real del archivo de texto
        expected_output_path = img_path.replace(".jpg", ".txt")

        if os.path.exists(expected_output_path):
            with open(expected_output_path, 'r') as file:
                expected_output = file.read().strip()
                expected_values = expected_output.split()
                expected_classes = [float(val) for val in expected_values]  # Clases reales

                # Inicializar conteos para esta imagen
                true_positive = 0
                false_positive = 0
                false_negative = 0

                # Contar las clases esperadas
                expected_count = {cls: expected_classes.count(cls) for cls in set(expected_classes)}

                # Contar las clases predichas
                predicted_count = {cls: 0 for cls in [pred['Clase'] for pred in formatted_predictions]}  # Extraer solo las clases

                # Contar TP, FP y FN para la imagen actual
                for cls in expected_count:
                    if cls in predicted_count:
                        # True positives son los que coinciden
                        tp = min(expected_count[cls], predicted_count[cls])
                        true_positive += tp
                        # Falsos negativos son los que no se predijeron
                        false_negative += expected_count[cls] - tp
                    else:
                        # No hay predicciones para esta clase, así que todos son falsos negativos
                        false_negative += expected_count[cls]

                # Falsos positivos son aquellos que se predijeron pero no estaban en las clases esperadas
                for cls in predicted_count:
                    if cls not in expected_count:
                        false_positive += predicted_count[cls]

                # Acumular los resultados
                total_true_positive += true_positive
                total_false_positive += false_positive
                total_false_negative += false_negative

    # Calcular precisión, recall y F1 Score
    precision = total_true_positive / (total_true_positive + total_false_positive) if (total_true_positive + total_false_positive) > 0 else 0
    recall = total_true_positive / (total_true_positive + total_false_negative) if (total_true_positive + total_false_negative) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Imprimir métricas
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")

    # Imprimir conteos de TP, FP y FN totales
    print(f"Total Verdaderos Positivos (TP): {total_true_positive}")
    print(f"Total Falsos Positivos (FP): {total_false_positive}")
    print(f"Total Falsos Negativos (FN): {total_false_negative}")

    



#adaptarlo a nuestro modelo
def test_iowa_model(iowa_model, iowa_validation_data):
    x, y = iowa_validation_data

    val_predictions = iowa_model.predict(x)

    # Compute the MAE and MSE values for the model
    assert mean_absolute_error(y, val_predictions) == pytest.approx(0.0, rel=0.1)
    assert mean_squared_error(y, val_predictions) == pytest.approx(0.0, rel=0.1)

"""