import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

def predict(model_path, image_path, output_dir='prediction_output'):
    # Initialize the YOLO model with the trained weights
    model = YOLO(model_path)
    results = model.predict(source=image_path, save=True)

    # Find the saved image path in the prediction directory
    prediction_image_path = os.path.join(output_dir, os.path.basename(image_path))
    
    img = plt.imread(prediction_image_path)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis(False)
    plt.show()

    return prediction_image_path