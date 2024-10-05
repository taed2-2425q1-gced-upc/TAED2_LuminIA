import great_expectations as gx
import pandas as pd
import os

#Inicializar el contexto
context = gx.get_context()
context.add_or_update_expectation_suite("yolo_training_suite")

#Directorios
raw_data_dir = "data/raw/ts"
processed_data_dir = "data/processed"

#Leer los archivos de los dos conjuntos
with open(os.path.join(processed_data_dir, "train_images.txt"), 'r') as f:
    train_image_paths = f.read().splitlines()

data_records = []
for image_path in train_image_paths:
    txt_file = image_path.replace(".jpg", ".txt")
    if os.path.exists(txt_file):
        df_bbox = pd.read_csv(txt_file, header = None, delim_whitespace= True, names=["Class", "x_center", "y_center", "width", "height"])
        for _,row in df_bbox.iterrows():
            data_records.append({
                "ImagePath": image_path, 
                "Class": row["Class"], 
                "x_center": row["x_center"], 
                "y_center": row["y_center"]
                "width": row["width"]
                "height": row["height"]
            })

# Crea un DataFrame a partir de las rutas de las imágenes
train_data = pd.DataFrame(data_records)

#Datasource
datasource = context.sources.add_or_update_pandas(name="dataset")
data_asset = datasource.add_dataframe_asset(name="training", dataframe=train_data)

batch_request = data_asset.build_batch_request()

validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="yolo_training_suite",
    datasource_name="dataset",
    data_asset_name="training",
)

# Expectativas sobre las rutas de las imágenes
validator.expect_table_columns_to_match_ordered_list(
    column_list=["ImagePath", "Class", "x_center", "y_center", "width", "height"]
)
validator.expect_column_values_to_not_be_null("ImagePath")
validator.expect_column_values_to_not_be_null("Class")
validator.expect_column_values_to_not_be_null("x_center")
validator.expect_column_values_to_not_be_null("y_center")
validator.expect_column_values_to_not_be_null("width")
validator.expect_column_values_to_not_be_null("height")

# Validas que los valores de la clase son enteros entre 1 y 4
validator.expect_column_values_to_be_in_set("Class", [1, 2, 3, 4])

# Verifica tipos de datos
validator.expect_column_values_to_be_of_type("ImagePath", "str")
validator.expect_column_values_to_be_of_type("Class", "int")  
validator.expect_column_values_to_be_of_type("x_center", "float")
validator.expect_column_values_to_be_of_type("y_center", "float")
validator.expect_column_values_to_be_of_type("width", "float")
validator.expect_column_values_to_be_of_type("height", "float")


# Guarda las expectativas
validator.save_expectation_suite(discard_failed_expectations=False)

# Crea un checkpoint
checkpoint = context.add_or_update_checkpoint(
    name="my_checkpoint",
    validator=validator,
)

# Ejecuta el checkpoint y visualiza los resultados
checkpoint_result = checkpoint.run()
context.view_validation_result(checkpoint_result)