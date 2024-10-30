"""
Main script: it includes our API initialization and endpoints.
"""

import io
import logging
import os
import shutil
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path

import torch  # Ensure torch is imported
from fastapi import File, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from src.config.config import MODELS_DIR  # Ensure MODELS_DIR is imported

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

# Global variable for the model
MODEL = None # pylint: disable=W0603

torch.serialization.add_safe_globals([DetectionModel])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads the YOLO model found in `MODELS_DIR` and adds it to the global MODEL variable."""
    global MODEL
    model_paths = [
        filename for filename in MODELS_DIR.iterdir()
        if filename.suffix == ".pt" and filename.stem.startswith("ts_model")
    ]

    if not model_paths:
        logging.info("No .pt models found in the MODELS_DIR directory.")
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="No models found."
        )

    try:
        MODEL = YOLO(model_paths[0])
        logging.info("YOLO model loaded successfully from %s", model_paths[0])
    except Exception as e:
        logging.error("Error loading the model from %s: %s", model_paths[0], e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="Failed to load model"
        ) from e  # Use from e for clarity

    yield

    # Clean up the model at the end of the application lifecycle
    MODEL = None


app = FastAPI(
    title="Traffic Signs Detection",
    description=(
        "This API lets you make predictions on the images that you can upload "
        "using the YOLOv8 model."
    ),
    version="0.1",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
        "data": {
            "message": "Welcome to Traffic Signs classifier! Please, read the `/docs`!"
        },
    }
    return response


@app.get("/model", tags=["Prediction"])
def _get_model():
    """Return the YOLOv8 model."""
    logging.info("Accessed the _get_model function")

    if MODEL is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Model not found"
        )

    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": MODEL,
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
    logging.info("Received request to /predict/image")

    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not found")

    temp_image_path = "temp_image.jpg"  # Temporary image path

    try:
        image_stream = await file.read()
        image = Image.open(io.BytesIO(image_stream))

        image.save(temp_image_path)

        save_dir = Path("runs/detect/predict")
        image_dir = save_dir / "image"

        # Remove previous image directory if it exists
        if image_dir.exists() and image_dir.is_dir():
            shutil.rmtree(image_dir)

        # Predict using the YOLO model
        MODEL.predict(
            source=temp_image_path,
            save=True,
            conf=0.25,
            project=str(save_dir),
            name="image"
        )

        processed_image_path = image_dir / "temp_image.jpg"

        if not processed_image_path.exists():
            raise HTTPException(status_code=500, detail="Processed image not found")

        return FileResponse(
            str(processed_image_path),
            media_type="image/jpeg",
            filename="temp_image.jpg"
        )
    except Exception as e:
        logging.error("Error while making the prediction: %s", e)
        raise HTTPException(status_code=500, detail="Error while making the prediction") from e
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)


@app.get("/message", tags=["Others"])
async def get_message():
    """Returns a message indicating the service."""
    return {"message": "Traffic Signs Detection"}