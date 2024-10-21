import configparser
from PIL import Image
import os



def load_config(path):
    config = configparser.ConfigParser(allow_no_value=True, delimiters=('=', ':'), strict=False)
    config.optionxform = str  # Mantener la capitalización exacta de las opciones
    config.read(path, encoding='utf-8')
    return config

def load_image(path):
    # Verificar si la ruta existe
    if not os.path.exists(path):
        raise FileNotFoundError(f"La imagen no se encuentra en la ruta: {path}")

    try:
        # Cargar la imagen
        img = Image.open(path)

        # Convertir a RGB si es necesario
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except (IOError, Image.UnidentifiedImageError) as e:
        raise IOError(f"El archivo en {path} no se puede abrir o no es una imagen válida. Detalle: {e}")