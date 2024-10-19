import os
from ultralytics import YOLO
from PIL import Image

def load_image(image_path):
    """Carga una imagen desde la ruta proporcionada."""
    try:
        return Image.open(image_path)
    except Exception as e:
        print(f"Error al cargar la imagen: {e}")
        return None

def predict_image(image_path, model):
    """
    Realiza una predicción sobre una imagen dada con el modelo YOLO
    e imprime la predicción y la etiqueta real.

    Args:
        image_path (str): Ruta de la imagen.
        model (YOLO): Modelo YOLO cargado.
    """
    # Cargar la imagen
    input_image = load_image(image_path)
    if input_image is None:
        print("No se pudo cargar la imagen.")
        return
    
    # Realizar la predicción con el modelo
    predictions = model.predict(input_image)

    # Imprimir la estructura de las predicciones completa
    print(f"Estructura completa de las predicciones para {image_path}: {predictions}")
    
    # Formatear las predicciones
    formatted_predictions = []
    for pred in predictions:
        if hasattr(pred, 'boxes'):
            for box in pred.boxes:
                # Formatear cada predicción (clase, confianza, coordenadas)
                formatted_predictions.append(
                    f"Clase: {box.cls.item()}, Confianza: {box.conf.item():.8f}, Coordenadas: {box.xyxy}"
                )
    
    # Imprimir las predicciones
    print(f"\nPredicciones formateadas para {image_path}:")
    for prediction in formatted_predictions:
        print(prediction)
    
    # Obtener la etiqueta esperada
    expected_output_path = image_path.replace(".jpg", ".txt")
    
    if os.path.exists(expected_output_path):
        with open(expected_output_path, 'r') as file:
            expected_output = file.read().strip()
            print(f"Etiqueta real: {expected_output}")
    else:
        print(f"No se encontró la etiqueta esperada para {image_path}.")

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar el modelo YOLO
    WEIGHTS_PATH = 'models/best.pt'
    yolo_model = YOLO(WEIGHTS_PATH)

    # Predecir una imagen
    predict_image("data/raw/ts/00426.jpg", yolo_model)
