from deepchecks.vision import classification_dataset_from_directory
from deepchecks.vision.suites import data_integrity, train_test_validation

DATA_DIR = 'data/raw'
REPORTS_DIR = 'reports'

#Loading the data
train_ds, test_ds = classification_dataset_from_directory(
    DATA_DIR / 'ts', object_type="VisionData", image_extension="jpg"
)

#Create the validation suite
custom_suite = data_integrity()

custom_suite.add(
    train_test_validation()
)

############### ACABAR ##########################
## AÑADIR SUIT DE VALIDACIÓN CON EL MÉTODO ADD ##
#################################################


#Run the suite and save the results
result = custom_suite.run(train_ds, test_ds)

result.save_as_html(str(REPORTS_DIR / "deepchecks_validation.html"))


