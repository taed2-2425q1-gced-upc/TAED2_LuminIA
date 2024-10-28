"""
Module for testing the Traffic Signs Detection API.
"""

from fastapi.testclient import TestClient
import pytest
from src.app.app import app

# Create a TestClient instance for testing
client = TestClient(app)

# Define a global variable for the model
MODEL = None

def test_connection():
    """Test the connection to the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "OK",
        "status-code": 200,
        "data": {"message": "Welcome to Traffic Signs classifier! Please, read the `/docs`!"}
    }

@pytest.fixture(autouse=True)
def setup_model():
    """Fixture to set up the model before tests."""
    global model
    model = {"name": "YOLOv8", "version": "1.0"}
    yield  # Esto permite que se ejecute la prueba
    model = None  # Limpia el modelo después de la prueba

def test_get_model_not_found():
    """Test the behavior when the model is not found."""
    global model
    model = None  # Simula que el modelo no está disponible
    response = client.get("/model")
    assert response.status_code == 400  # Esperando BAD REQUEST
    assert response.json() == {
        "detail": "Model not found"
    }
