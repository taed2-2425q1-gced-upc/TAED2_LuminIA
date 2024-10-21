import os
import pandas as pd
import great_expectations as gx
from src.config import PROCESSED_DATA_DIR

# Inicializamos el DataContext existente
context = gx.get_context(mode="file")

# Obtener el checkpoint existente para validación
checkpoint = context.checkpoints.get("image_labels_checkpoint")

# Directorios de entrada
images_dir = PROCESSED_DATA_DIR / "ts"  # Las imágenes están en la carpeta "ts"
input_dir = PROCESSED_DATA_DIR / "predicted"  # Archivos .txt con los nombres de imágenes

# Leer los archivos .txt que contienen los nombres de las imágenes para train y test
train_images_file = input_dir / "train_images.txt"
test_images_file = input_dir / "test_images.txt"

# Leer los nombres de las imágenes de entrenamiento y prueba
with open(train_images_file, 'r') as f:
    train_images = [line.strip() for line in f.readlines()]

with open(test_images_file, 'r') as f:
    test_images = [line.strip() for line in f.readlines()]

# Lista para almacenar los datos de todos los archivos .txt de etiquetas (ground truth)
all_data = []

# Función para cargar las etiquetas de cada imagen
def load_ground_truth_for_images(image_list, split_name):
    for image_name in image_list:
        # Asumimos que los archivos .txt de etiquetas tienen el mismo nombre que la imagen pero con extensión .txt
        label_file = os.path.join(input_dir, f"{os.path.splitext(image_name)[0]}.txt")
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) == 5:  # Aseguramos que tenga 5 columnas: class, x_center, y_center, width, height
                        # Agregar una columna para identificar si es train o test
                        all_data.append(data + [split_name])

# Cargar los datos de las imágenes de entrenamiento y prueba
load_ground_truth_for_images(train_images, "train")
load_ground_truth_for_images(test_images, "test")

# Convertir los datos en un DataFrame de pandas
columns = ["class", "x_center", "y_center", "width", "height", "split"]
dataframe = pd.DataFrame(all_data, columns=columns)

# Asegurar que las columnas tengan los tipos de datos correctos
dataframe = dataframe.astype({
    "class": int,
    "x_center": float,
    "y_center": float,
    "width": float,
    "height": float,
    "split": str
})

# Definir los parámetros del batch para el checkpoint
batch_parameters = {"dataframe": dataframe}

# Ejecutar el checkpoint con los datos cargados
results = checkpoint.run(batch_parameters=batch_parameters)

# Construir los Data Docs para visualizar los resultados de la validación
context.build_data_docs()

# Obtener la URL para revisar los Data Docs
validation_result_identifier = results.list_validation_result_identifiers()[0]
context.open_data_docs(validation_result_identifier)
