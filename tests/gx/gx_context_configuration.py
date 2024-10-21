import os
import great_expectations as gx
from great_expectations.exceptions import DataContextError
import pandas as pd

# Inicializamos el DataContext
context = gx.get_context(mode="file")

# Configuración del datasource basado en archivos
datasource = context.data_sources.add_or_update_file(
    name="images_and_labels",
    base_directory="data/raw/ts",  # Directorio de las imágenes y archivos .txt
    glob_directive="*.txt",  # Solo archivos .txt con la ground truth
)

# Cargamos los archivos .txt que contienen el ground truth
ground_truth_files = [f for f in os.listdir('data/raw/ts') if f.endswith('.txt')]

# Crear un asset para representar los datos del ground truth
data_asset = datasource.add_file_asset(name="image_labels_ground_truth")

# Lista para almacenar los datos de todos los archivos .txt
all_data = []

# Procesamos cada archivo .txt para extraer las líneas de ground truth
for file in ground_truth_files:
    file_path = os.path.join('data/raw/ts', file)
    with open(file_path, 'r') as f:
        # Cada archivo puede tener múltiples líneas (una por objeto)
        for line in f:
            # Descomponer cada línea en sus valores
            data = line.strip().split()
            if len(data) == 5:  # Aseguramos que tenga 5 columnas
                all_data.append(data)

# Convertir los datos en un DataFrame de pandas
df = pd.DataFrame(all_data, columns=["class", "x_center", "y_center", "width", "height"])

# Asegurarse de que las columnas sean numéricas para las validaciones
df = df.astype({
    "class": int,
    "x_center": float,
    "y_center": float,
    "width": float,
    "height": float,
})

# Definir batch para el DataFrame cargado
batch_definition = context.add_batch_definition(
    dataframe=df,
    asset_name="image_labels_ground_truth"
)

# Creamos una suite de expectativas para validar el ground truth
expectations_suite = gx.ExpectationSuite("image_labels_validation")

try:
    context.suites.add(expectations_suite)
except DataContextError:
    expectations_suite = context.suites.get("image_labels_validation")

# Validar las columnas del ground truth, que deben ser: class, x_center, y_center, width, height
expectations_suite.add_expectation(
    gx.expectations.ExpectTableColumnsToMatchOrderedList(
        column_list=["class", "x_center", "y_center", "width", "height"]
    )
)

# Validar que los valores de clase no sean nulos y que estén dentro de un rango específico de clases válidas
expectations_suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="class"))
expectations_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="class", min_value=0, max_value=10))

# Validar que los valores de la bounding box (coordenadas y dimensiones) no sean nulos y estén dentro de un rango válido
expectations_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="x_center", min_value=0, max_value=1))
expectations_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="y_center", min_value=0, max_value=1))
expectations_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="width", min_value=0, max_value=1))
expectations_suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="height", min_value=0, max_value=1))

# Guardamos los cambios en la suite de expectativas
expectations_suite.save()

# Definir un validador para ejecutar la suite
validator = gx.ValidationDefinition(
    data=batch_definition, suite=expectations_suite, name="image_labels_validator"
)

try:
    context.validation_definitions.add(validator)
except DataContextError:
    context.validation_definitions.delete("image_labels_validator")
    validator = context.validation_definitions.add(validator)

# Crear un checkpoint para correr las expectativas y generar Data Docs
action_list = [
    gx.checkpoint.UpdateDataDocsAction(name="update_data_docs"),
]

checkpoint = gx.Checkpoint(
    name="image_labels_checkpoint",
    validation_definitions=[validator],
    actions=action_list,
    result_format={"result_format": "SUMMARY"},
)

try:
    context.checkpoints.add(checkpoint)
except DataContextError:
    context.checkpoints.delete("image_labels_checkpoint")
    checkpoint = context.checkpoints.add(checkpoint)

# Finalmente, corremos el checkpoint
results = context.run_checkpoint(checkpoint_name="image_labels_checkpoint")
context.build_data_docs()
