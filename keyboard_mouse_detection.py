import os
import torch
import torch.cuda as cuda
from roboflow import Roboflow
from ultralytics import YOLO

def get_gpu_properties():
    if cuda.is_available():
        print(f"Votre appareil dispose de {cuda.device_count()} GPU(s).")
        print(f"GPU : {cuda.get_device_name(0)}")
        print(f"Propriétés du GPU : {cuda.get_device_properties(0)}")
        return True
    else:
        print("Votre appareil ne dispose pas de GPU.")
        return False

def current_device():
    if get_gpu_properties():
        print("Vous travaillez actuellement avec un GPU.")
    else:
        print("Vous travaillez actuellement avec un CPU.")

current_device()

# Connexion à Roboflow pour récupérer le dataset
print("Connexion à Roboflow...")
rf = Roboflow(api_key="apikey")
project = rf.workspace("workspace-il5zj").project("object-detection-plm7l")
version = project.version(1)
dataset = version.download("yolov8")
print("Dataset téléchargé avec succès.") 

# Entrainer le model
print("Début de l'entraînement du modèle...")
model = YOLO("yolov8n.yaml") 
results = model.train(data="./Object-Detection-1/data.yaml", epochs=100)
print("Entraînement terminé.")

# Charger le modèle entraîné
model = YOLO('./yolo/yolov8n.pt')
result = model("./5.jpg", imgsz=640,show=True, save=True)