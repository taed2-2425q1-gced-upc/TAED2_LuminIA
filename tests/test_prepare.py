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


def test_load_parameters():
    """Test para cargar los parámetros correctamente."""
    params = load_parameters()
    assert "test_size" in params, "Falta el parámetro 'test_size'"
    assert params["test_size"] == 0.2, "El tamaño de prueba no es el esperado"
    assert "train" in params, "Falta el parámetro 'train'"
    assert "test" in params, "Falta el parámetro 'test'"

def test_load_parameters_with_invalid_yaml():
    """Test para cargar parámetros desde un archivo YAML inválido."""
    # Simula un archivo YAML malformado
    mock_file = mock_open(read_data="invalid_yaml: [")
    
    with patch("builtins.open", mock_file):
        params = load_parameters()
        
        # Verifica que se devuelva un diccionario vacío en caso de error
        assert params == {}, "Se esperaba un diccionario vacío al cargar un YAML inválido"

def test_get_image_files(tmp_path):
    """Test para la obtención de archivos de imagen del directorio."""
    
    # Crea un directorio temporal y añade algunos archivos de imagen
    mock_image_dir = tmp_path / "data" / "raw"
    mock_image_dir.mkdir(parents=True, exist_ok=True)

    # Crea 3 archivos de imagen de prueba
    expected_images = {f"{i:05}.jpg" for i in range(3)}  # Nombres esperados
    for i in range(3):
        mock_image_file = mock_image_dir / f"{i:05}.jpg"
        mock_image_file.touch()  # Crea un archivo vacío

    # Crea un archivo que no es una imagen
    non_image_file = mock_image_dir / "not_an_image.txt"
    non_image_file.touch()  # Crea un archivo de texto

    # Parchea la variable RAW_DATA_DIR_TS para que apunte al directorio temporal
    with patch("src.config.config.RAW_DATA_DIR_TS", mock_image_dir):
        images = get_image_files()

        # Verifica que todos los archivos devueltos son imágenes .jpg
        assert all(img.endswith('.jpg') for img in images), "No todos los archivos devueltos son imágenes .jpg"
        
        # Verifica que no se devuelvan archivos no deseados (por ejemplo, el archivo de texto)
        assert "not_an_image.txt" not in images, "Se ha devuelto un archivo que no es una imagen .jpg"

        # Verifica que las imágenes esperadas están en los resultados devueltos
        for expected in expected_images:
            assert expected in images, f"{expected} no se ha encontrado en la lista de imágenes devueltas."


def test_split_data():
    """Test para dividir los archivos de imagen en conjuntos de entrenamiento y prueba."""
    image_files = ["image1.jpg", "image2.jpg", "image3.jpg"]
    params = {"test_size": 0.2, "random_state": 42}
    train_files, test_files = split_data(image_files, params)

    assert len(train_files) == 2, "Se esperaba que el conjunto de entrenamiento tuviera 2 archivos"
    assert len(test_files) == 1, "Se esperaba que el conjunto de prueba tuviera 1 archivo"

def test_write_file_list(tmp_path):
    """Test para escribir la lista de archivos en un archivo de salida."""
    file_list = ["image1.jpg", "image2.jpg"]
    output_file = tmp_path / "output.txt"

    write_file_list(file_list, output_file)

    # Verifica que el archivo se haya creado y contenga el contenido esperado
    with open(output_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 2, "El número de líneas en el archivo de salida no es el esperado"
        assert lines[0].strip() == "image1.jpg", "La primera línea no es la esperada"
        assert lines[1].strip() == "image2.jpg", "La segunda línea no es la esperada"

def test_prepare_data(mocker):
    """Test para preparar los datos."""
    # Simula la llamada a `load_parameters`
    mocker.patch('src.features.prepare.load_parameters', return_value={
        "test_size": 0.2,
        "random_state": 42,
        "train": "train_images.txt",  # Debe coincidir con el nombre correcto
        "test": "test_images.txt"      # Debe coincidir con el nombre correcto
    })

    # Simula la llamada a `get_image_files`
    mocker.patch('src.features.prepare.get_image_files', return_value=["image1.jpg", "image2.jpg", "image3.jpg"])

    # Ejecuta la función que estamos probando
    prepare_data()

    # Verifica que los archivos de salida se hayan creado
    assert os.path.exists(PROCESSED_DATA_DIR / "train_images.txt"), "El archivo de entrenamiento no se ha creado"
    assert os.path.exists(PROCESSED_DATA_DIR / "test_images.txt"), "El archivo de prueba no se ha creado"
