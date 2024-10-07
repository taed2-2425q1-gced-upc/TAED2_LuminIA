import yaml
import torch
import json
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from your_yolov3_model import YOLOv3  # Asegúrate de que la clase del modelo YOLOv3 esté definida
from your_dataset import TrafficSignDataset  # Importa tu clase de dataset

# Leer los parámetros desde el archivo params.yaml
params_path = Path("params.yaml")
with open(params_path, "r") as params_file:
    try:
        params = yaml.safe_load(params_file)["train"]
    except yaml.YAMLError as exc:
        print(exc)

# Definir los parámetros del entrenamiento
epochs = params["epochs"]
batch_size = params["batch_size"]
learning_rate = params["learning_rate"]

# Configuración del modelo YOLOv3
model = YOLOv3(num_classes=4)  # Asegúrate de que el número de clases sea el correcto
model.train()

# Configuración del optimizador
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Definir transformaciones y dataset
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Cargar el dataset
train_dataset = TrafficSignDataset("data/processed/train/train.txt", transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Entrenamiento
for epoch in range(epochs):
    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = model.compute_loss(outputs, targets)  # Asegúrate de que esta función esté definida
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")

# Guardar el modelo entrenado
output_model_path = Path("models/trained_model.pt")
torch.save(model.state_dict(), output_model_path)

# Guardar métricas de entrenamiento
metrics = {
    "epochs": epochs,
    "batch_size": batch_size,
    "learning_rate": learning_rate,
    # Puedes agregar métricas relevantes de tu entrenamiento aquí
}

# Guardar las métricas en un archivo JSON
metrics_path = Path("metrics/train_metrics.json")
metrics_dir = metrics_path.parent
metrics_dir.mkdir(parents=True, exist_ok=True)

with open(metrics_path, "w") as f:
    json.dump(metrics, f)

print("Entrenamiento completado. Las métricas se han guardado.")
