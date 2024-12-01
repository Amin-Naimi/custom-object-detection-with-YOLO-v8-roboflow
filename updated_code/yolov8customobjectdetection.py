from ultralytics import YOLO
import os
from IPython.display import display, Image
from IPython import display
from roboflow import Roboflow

def initialize_and_download_data_from_roboflow(api_key, workspace_name, project_name, version_number):
    display.clear_output()
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace_name).project(project_name)
    version = project.version(version_number)
    return version.download("yolov8")

def train_model(model_path, data_path, epochs, img_size):
    model = YOLO(model_path)
    model.train(data=data_path, epochs=epochs, imgsz=img_size)
    print("Entraînement terminé.")

def validate_model(model_path, data_path):
    model = YOLO(model_path)
    validation_results = model.val(data=data_path)
    print("Validation terminée. Métriques :", validation_results)

def predict_image(model_path, image_path):
    model = YOLO(model_path)
    results = model.predict(source=image_path, save=True, show=True)
    #results.show()
    print("Prédiction terminée pour :", image_path)

def main():
    api_key = "wKPgK90xGcfBdWp6f8tt"
    workspace_name = "workspace-il5zj"
    project_name = "object-detection-plm7l"
    version_number = 1

    #dataset = initialize_and_download_data_from_roboflow(api_key, workspace_name, project_name, version_number)
    #print("Téléchargement des données terminé :", dataset)
    
    data_path = "./Object-Detection-1/data.yaml"

    # Entraînement du modèle
    """train_model(
        model_path="yolov8m.pt",
        data_path=data_path,
        epochs=100,
        img_size=640
    )"""

    trained_model_path = "./runs/detect/train/weights/best.pt"

    # Validation du modèle
    """validate_model(
        model_path=trained_model_path,
        data_path=data_path
    )"""

    image_path_1 = 0
    image_path_2 = "./1.png"
    #image_path_2 = "/content/souris-usb-havit-ms753-tunisie.jpg"
    
    # Prédictions sur des images
    predict_image(
        model_path=trained_model_path,
        image_path=image_path_2
    )

if __name__ == "__main__":
    main()
