from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import torch
import torchvision
import PIL.Image
import torchvision.models as models
from io import BytesIO

app = FastAPI()

# Load models
besty11_model = YOLO("../models/besty11_car_demage_detection.pt")

bestcnn_model = models.resnet50(pretrained=False)
num_features = bestcnn_model.fc.in_features
bestcnn_model.fc = torch.nn.Linear(num_features, 2)  # (clean/dirty)
bestcnn_model.load_state_dict(torch.load("../models/resnet50_car_dirtiness.pth", map_location=torch.device('cpu')))
bestcnn_model.eval()

# Preprocessing functions
def preprocess_for_yolo(image: PIL.Image.Image) -> torch.Tensor:
    image = torchvision.transforms.functional.to_tensor(image)
    image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(640, 640), mode='bilinear', align_corners=False)
    return image  # Shape: (1, 3, 640, 640)

def preprocess_for_cnn(image: PIL.Image.Image) -> torch.Tensor:
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)  # Shape: (1, 3, 224, 224)

# Inference endpoint
@app.post("/inference/")
async def run_inference(file: UploadFile = File(...), 
                        damage_conf_threshold: float = 0.5, 
                        dirtiness_conf_threshold: float = 0.5):
    try:
        # Read and preprocess the image
        image = PIL.Image.open(BytesIO(await file.read())).convert("RGB")
        
        # Dirtiness classification
        cnn_image = preprocess_for_cnn(image)
        cnn_output = bestcnn_model(cnn_image)
        probabilities = torch.nn.functional.softmax(cnn_output, dim=1)
        dirtiness_confidence = probabilities[0][1].item()
        is_dirty = dirtiness_confidence >= dirtiness_conf_threshold

        # Damage detection
        yolo_image = preprocess_for_yolo(image)
        yolo_results = besty11_model.predict(source=yolo_image, conf=damage_conf_threshold, save=False)
        damages = [
            {
                "box": box.xyxy.tolist(),
                "confidence": box.conf.item()
            } for box in yolo_results[0].boxes
        ]

        # Construct the response
        if not is_dirty and len(damages) == 0:
            status = 1
            message = "The car is clean and has no visible damages."
        elif is_dirty and len(damages) > 0:
            status = 0
            message = "The car is dirty and has visible damages."
        elif is_dirty:
            status = 0
            message = "The car is dirty but has no visible damages."
        else:
            status = 0
            message = "The car has visible damages but is not classified as dirty."

        return JSONResponse(content={
            "status": status,
            "message": message,
            "dirtiness": {
                "is_dirty": is_dirty,
                "confidence": dirtiness_confidence
            },
            "damages": damages
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
