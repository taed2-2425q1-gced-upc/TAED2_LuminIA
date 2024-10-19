import great_expectations as gx
import pandas as pd
from PIL import Image
import os

# 1. Configura tu contexto de Great Expectations
context = gx.get_context(mode="file")

# 2. Crea un Data Source para tus imágenes
datasource = context.data_sources.add_or_update_pandas(
    name="image_dataset",
)

# 3. Carga las imágenes desde el archivo de texto
def load_images_from_list(file_path):
    with open(file_path, 'r') as f:
        # Carga los nombres de los archivos desde el archivo
        image_files = [line.strip().split('/')[-1] for line in f.readlines()]
        
    image_data = []
    for image_file in image_files:
        # Cambia aquí a la ruta correcta donde están tus imágenes
        image_path = os.path.join('data/raw/ts', image_file)
        
        if not os.path.exists(image_path):  # Verifica si la imagen existe
            print(f"Warning: The image {image_path} does not exist.")
            continue
            
        with Image.open(image_path) as img:
            image_data.append({
                "filename": image_file,
                "width": img.width,
                "height": img.height,
                "size": os.path.getsize(image_path)  # tamaño del archivo en bytes
            })
    
    return pd.DataFrame(image_data)

# 4. Crea el dataframe de pandas a partir de tus imágenes
image_dataframe = load_images_from_list('data/raw/train.txt')

# 5. Añade un dataframe asset a tu datasource
data_asset = datasource.add_dataframe_asset(name="processed_images")

# 6. Crea o actualiza la Expectation Suite
expectations_suite_name = "image_validation_suite"

# Obtener las suites existentes
existing_suites = context.suites.list()  # Verifica las suites existentes

# Verifica si la Expectation Suite ya existe
expectation_suite_names = [suite.name for suite in existing_suites]

if expectations_suite_name in expectation_suite_names:
    expectations_suite = context.suites.get(expectations_suite_name)
    print(f"Expectation Suite '{expectations_suite_name}' already exists.")
else:
    expectations_suite = gx.ExpectationSuite(expectations_suite_name)
    context.suites.add(expectations_suite)
    print(f"Created new Expectation Suite '{expectations_suite_name}'.")

# 7. Añade expectativas a tu suite
expectations_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeUnique(column="filename")
)
expectations_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToNotBeNull(column="filename")
)
expectations_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeGreaterThan(column="width", threshold=0)
)
expectations_suite.add_expectation(
    gx.expectations.ExpectColumnValuesToBeGreaterThan(column="height", threshold=0)
)

# 8. Crea una validación
validator = gx.ValidationDefinition(
    data=data_asset.get_batch_definition("my_image_batch"), suite=expectations_suite, name="image_validator"
)
context.validation_definitions.add(validator)

# 9. Crea un Checkpoint
action_list = [
    gx.checkpoint.UpdateDataDocsAction(name="update_data_docs"),
]
checkpoint = gx.Checkpoint(
    name="image_validation_checkpoint",
    validation_definitions=[validator],
    actions=action_list,
    result_format={"result_format": "SUMMARY"},
)
context.checkpoints.add(checkpoint)

# 10. Valida los datos contra tus expectativas
batch_parameters = {"dataframe": image_dataframe}  # Proporciona el DataFrame aquí al ejecutar la validación
checkpoint.run(batch_parameters=batch_parameters)
