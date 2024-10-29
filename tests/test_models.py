#POR HACER  

import pickle 

import pytest 
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODELS_DIR = " "
PROCESSED_DATA_DIR = 'data/processed' 

#Cambiar
from src.models.evaluate import load_validation_data
#Cambiar
from some_yolo_library import YOLO  # Importa la clase YOLO de la biblioteca que estés utilizando



#Cambiar yolo_model.pth por el nombre de nuestro modelo    
@pytest.fixture
def yolo_model():
    #Cargar el modelo
    model_path = MODELS_DIR / "yolo_model.pth"
    model = YOLO(model_path)
    return model

#Asegurarnos que usa el formato correcto
def load_image(path):
    #Función para cargar la imagen
    from PIL import Image 
    return Image.open(path)

#Cambiar val_data_path por la ruta al archivo .txt con los nombres de las imagenes de validación    
@pytest.fixture
def yolo_validation_data():
    #Cargar las imagenes
    val_data_path = "path_imagenes_val.txt"
    with open(val_data_path, 'r') as file: 
        image_paths = [line.strip() for line in file.readlines()]
    images = [load_image(path) for path in image_paths] 
    return load_validation_data(PROCESSED_DATA_DIR)


# Ajusta los nombres de archivo y las salidas esperadas
@pytest.mark.parametrize(
    "image, expected",
    [
        ("image1.jpg", "expected_output1"),
        ("image1.jpg", "expected_output1")
    ],
)
def test_yolo_model(yolo_model, test_images, image, expected):
    #test_image contiene una lista de imágenes
    input_image = test_images[image]
    predictions = yolo_model.predict(input_image)
    assert predictions == expected


#adaptarlo a nuestro modelo
def test_iowa_model(iowa_model, iowa_validation_data):
    x, y = iowa_validation_data

    val_predictions = iowa_model.predict(x)

    # Compute the MAE and MSE values for the model
    assert mean_absolute_error(y, val_predictions) == pytest.approx(0.0, rel=0.1)
    assert mean_squared_error(y, val_predictions) == pytest.approx(0.0, rel=0.1)
