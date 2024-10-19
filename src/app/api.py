"""Main script: it includes our API initialization and endpoints."""
from fastapi.responses import JSONResponse,  FileResponse
import logging
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics import YOLO
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Dict, Union
from PIL import Image
import io
from codecarbon import track_emissions
from fastapi import File, FastAPI, HTTPException, UploadFile
import numpy as np
from pathlib import Path
import os

from src.config import METRICS_DIR, MODELS_DIR

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
torch.serialization.add_safe_globals([DetectionModel])

model_wrappers_dict: Dict[str, Dict[str, dict]] = {"tabular": {}, "image": {}}


model_wrappers_dict = {"image": {}}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads all YOLO models found in `MODELS_DIR` and adds them to `model_wrappers_dict`"""

    model_paths = [
        filename
        for filename in Path("models").iterdir()  
        if filename.suffix == ".pt" and filename.stem.startswith("ts_model")
    ]

    if not model_paths:
        logging.info("No se encontraron modelos .pt en el directorio MODELS_DIR.")
    
    for path in model_paths:
        try:
            # Cargar el modelo YOLO
            model = YOLO(path)  # Utiliza directamente la clase YOLO para cargar el modelo
            logging.info(f"Modelo YOLO cargado correctamente desde {path}")
            print(model)

            # Guardamos el modelo cargado en el diccionario
            model_wrappers_dict["image"]["detection"] = model

        except Exception as e:
            logging.error(f"Error al cargar el modelo desde {path}: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to load model")

    yield

    # Limpiar los modelos al final del ciclo de vida de la aplicación
    model_wrappers_dict["image"].clear()

# Define application
app = FastAPI(
    title="German Traffic Signs",
    description="This API lets you make predictions on the German Traffic Signs Benchmark Dataset with a simple model.",
    version="0.1",
    lifespan=lifespan,
)


@app.get("/", tags=["General"])  # path operation decorator
async def _index():
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to German Traffic Signs classifier! Please, read the `/docs`!"},
    }
    return response


@app.get("/models/tabular", tags=["Prediction"])
def _get_tabular_models_list(model_type: str = None):
    """Return the list of available models"""

    logging.info(f"Se ha accedido a la función _get_tabular_models_list con model_type: {model_type}")

    if model_type is not None:
        model = model_wrappers_dict["image"].get(model_type, None)
        if model is None:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST, detail="Model type not found. Avaliable Model type: 'detection'"
            )
        return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": model_wrappers_dict["image"].get(model_type, None),
    }
    else:
        logging.error(f"Error al cargar el modelo desde {path}: {e}")
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Model not found")


@app.post("/predict/image/", tags=["Prediction"])
async def _predict_image(file: UploadFile = File(...)):
    """
    Realiza la predicción sobre una imagen usando el modelo de detección de señales de tráfico preentrenado.
    
    Parameters
    ----------
    file : UploadFile
        La imagen a clasificar.
    
    Returns
    -------
    dict
        Un diccionario con las predicciones (las clases detectadas y sus bounding boxes).
    """
    try:
        # Leer la imagen cargada
        image_stream = await file.read()

        # Convertir la imagen cargada a formato PIL
        image = Image.open(io.BytesIO(image_stream))

        # Convertir la imagen PIL a un array de NumPy (YOLOv8 acepta rutas o arrays de NumPy)
        image_np = np.array(image)

        # Guardar temporalmente la imagen en el disco
        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)

        # Obtener el modelo YOLO desde el diccionario cargado previamente
        model = model_wrappers_dict["image"].get("detection", None)

        if model is None:
            raise HTTPException(status_code=500, detail="Modelo no cargado")

        # Crear un directorio para guardar los resultados si no existe
        save_dir = Path("runs/detect/predict")
        save_dir.mkdir(parents=True, exist_ok=True)

        # Realizar la predicción usando el modelo YOLO y guardar las imágenes procesadas
        results = model.predict(source=temp_image_path, save=True, conf=0.25, project=str(save_dir), name="image")  
        
        # Recuperar la ruta de la imagen procesada
        processed_image_path = save_dir / "image" / "temp_image.jpg"  # Por defecto, YOLO guarda las imágenes como 'image0.jpg', 'image1.jpg', etc.
        
        if not processed_image_path.exists():
            raise HTTPException(status_code=500, detail="Imagen procesada no encontrada")

        # Devolver la imagen procesada como una respuesta para descargar
        return FileResponse(str(processed_image_path), media_type="image/jpeg", filename="temp_image.jpg")

    except Exception as e:
        logging.error(f"Error al realizar la predicción: {e}")
        raise HTTPException(status_code=500, detail="Error al realizar la predicción")
    finally:
        # Eliminar la imagen temporal después de la predicción
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)