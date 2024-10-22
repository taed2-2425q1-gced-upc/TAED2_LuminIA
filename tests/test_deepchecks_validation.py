import pytest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from src.validation.deepchecks_validation import (
    load_image_paths,
    load_yolo_annotations,
    load_images,
    create_batches,
)
from src.config.config import TEST_IMAGES_PATH
import tempfile
import os

@pytest.fixture(scope="module")
def setup_files():
    # Crea un directorio temporal para las imágenes de prueba
    with tempfile.TemporaryDirectory() as temp_dir:
        # Prepara un archivo temporal de imágenes para pruebas
        test_image_paths = [
            Path(temp_dir) / "image1.jpg",
            Path(temp_dir) / "image2.jpg",
            Path(temp_dir) / "image3.jpg"
        ]

        # Crea un archivo temporal para las rutas de las imágenes de prueba
        test_image_paths_file = Path(temp_dir) / "test_images.txt"
        
        # Crear el archivo con rutas de imágenes de prueba
        with open(test_image_paths_file, 'w') as f:
            for path in test_image_paths:
                f.write(str(path) + "\n")

        # Crea imágenes de prueba vacías
        for path in test_image_paths:
            img = Image.new('RGB', (100, 100))  # Crea una imagen negra de 100x100
            img.save(path)

        yield test_image_paths_file, test_image_paths  # Devuelve el archivo y las rutas para usarlas en las pruebas

def test_load_image_paths(setup_files):
    """ Testea la función load_image_paths """
    test_image_paths_file, setup_files = setup_files
    image_paths = load_image_paths(test_image_paths_file)
    assert len(image_paths) == 3  # Asegura que cargue las 3 imágenes
    for original_path, loaded_path in zip(setup_files, image_paths):
        assert loaded_path.name == Path(original_path).name

def test_load_yolo_annotations(setup_files):
    """ Testea la función load_yolo_annotations """
    test_image_paths_file, setup_files = setup_files
    # Crea archivos de anotaciones YOLO de prueba
    annotations_content = "0 0.5 0.5 1 1\n"  # Un ejemplo simple de anotación
    for path in setup_files:  # Usa las rutas generadas en setup_files
        annotation_file = Path(path).with_suffix('.txt')
        with open(annotation_file, 'w') as f:
            f.write(annotations_content)

    image_paths = [Path(p) for p in setup_files]
    annotations = load_yolo_annotations(image_paths)

    assert len(annotations) == len(setup_files)
    for annotation in annotations:
        assert isinstance(annotation, torch.Tensor)
        assert annotation.size(1) == 5  # Debería tener 5 columnas (class_id, x_min, y_min, w, h)

def test_load_images(setup_files):
    """ Testea la función load_images """
    test_image_paths_file, setup_files = setup_files
    image_paths = [Path(p) for p in setup_files]
    images = load_images(image_paths)

    assert len(images) == len(setup_files)
    for img in images:
        assert isinstance(img, np.ndarray)
        assert img.shape == (100, 100, 3)  # Debe ser una imagen RGB de 100x100

def test_create_batches(setup_files):
    """ Testea la función create_batches """
    test_image_paths_file, setup_files = setup_files
    image_paths = [Path(p) for p in setup_files]
    dummy_annotations = [torch.tensor([[0, 0, 0, 100, 100]]) for _ in setup_files]
    
    batches = list(create_batches(image_paths, dummy_annotations, batch_size=2))
    
    assert len(batches) == 2  # Debería haber 2 batches para 3 imágenes (2 + 1 en el último)
    assert len(batches[0]['images']) == 2  # El primer batch debe contener 2 imágenes
    assert len(batches[1]['images']) == 1  # El segundo batch debe contener 1 imagen

# Asegúrate de que todas las pruebas se ejecuten
if __name__ == "__main__":
    pytest.main()
