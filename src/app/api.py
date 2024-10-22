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
import shutil 

from src.config import METRICS_DIR, MODELS_DIR
from fastapi.middleware.cors import CORSMiddleware


logging.basicConfig(level=logging.INFO)
torch.serialization.add_safe_globals([DetectionModel])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the YOLO model found in `MODELS_DIR` and adds them to `model` global variable """
    global model
    
    model_paths = [
        filename
        for filename in MODELS_DIR.iterdir()  
        if filename.suffix == ".pt" and filename.stem.startswith("ts_model")
    ]

    if not model_paths:
        logging.info("No .pt models found in the MODELS_DIR directory.")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="No models found.")
    
    try:
        model = YOLO(model_paths[0])
        logging.info(f"YOLO model loaded successfully from {model_paths[0]}")

    except Exception as e:
        logging.error(f"Error loading the model from {model_paths[0]}: {e}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to load model")

    yield 
    
    # Clean up the model at the end of the application lifecycle
    model = None

app = FastAPI(
    title="Traffic Signs Detection",
    description="This API lets you make predictions of the images that you can upload using the YOLOv8 model.",
    version="0.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.get("/", tags=["General"]) 
async def _index():
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to Traffic Signs classifier! Please, read the `/docs`!"},
    }
    return response


@app.get("/model", tags=["Prediction"])
def _get_model():
    """Return the YOLOv8 model"""

    global model
    
    logging.info(f"Accessed the _get_model function")

    if model is None:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST, detail="Model not found"
            )
        
    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": model,
    }

@app.post("/predict/image/", tags=["Prediction"])
async def _predict_image(file: UploadFile = File(...)):
    """
    Makes a prediction on an image using the pretrained traffic sign detection model.
    
    Parameters
    ----------
    file : UploadFile
        The image to classify.
    
    Returns
    -------
    FileResponse
        The processed image with the corresponding bounding boxes based on the prediction.
    """
    global model
    logging.info("Received request to /predict/image")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not found")

    try:
        image_stream = await file.read()

        image = Image.open(io.BytesIO(image_stream))

        temp_image_path = "temp_image.jpg"
        image.save(temp_image_path)

        save_dir = Path("runs/detect/predict")

        image_dir = save_dir / "image"
        if image_dir.exists() and image_dir.is_dir():
            shutil.rmtree(image_dir)  

        results = model.predict(source=temp_image_path, save=True, conf=0.25, project=str(save_dir), name="image")  
        
        processed_image_path = image_dir / "temp_image.jpg"
        
        if not processed_image_path.exists():
            raise HTTPException(status_code=500, detail="Processed Image not found")
        
        return FileResponse(str(processed_image_path), media_type="image/jpeg", filename="temp_image.jpg")

    except Exception as e:
        logging.error(f"Error while making the prediction: {e}")
        raise HTTPException(status_code=500, detail="Error while making the prediction")
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


@app.get("/message")
async def get_message():
    return {"message": "Traffic Signs Detection"}