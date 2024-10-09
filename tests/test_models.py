import pytest
from src.utils import load_config  # Supongamos que tienes una función que carga la configuración


def test_load_config():
    config_path = 'models/yolov3_ts_train.cfg'
    config = load_config(config_path)

    assert config is not None, "La configuración no se cargó correctamente"

    # Depurar las secciones cargadas
    print("Secciones en el archivo de configuración:", config.sections())

    # Verificar que la sección 'net' existe
    assert 'net' in config, "La configuración debe contener la sección 'net'"

    # Asegurarse de que la opción 'batch' existe en la sección 'net'
    assert 'batch' in config['net'], "La configuración debe contener el parámetro 'batch' en la sección 'net'"
