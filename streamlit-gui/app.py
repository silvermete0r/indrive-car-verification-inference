import streamlit as st
from ultralytics import YOLO
import torch
import torchvision 
import PIL.Image
import torchvision.models as models  

st.set_page_config(
    page_title="inDrive Car Verification System Demo",
    page_icon="🚗",
    layout="wide"
)

with st.spinner('Loading YOLO model...'):
    besty11_model = YOLO("../models/besty11_car_demage_detection.pt")

with st.spinner('Loading CNN model...'):
    bestcnn_model = models.resnet50(pretrained=False)
    num_features = bestcnn_model.fc.in_features
    bestcnn_model.fc = torch.nn.Linear(num_features, 2)  # (clean/dirty)
    bestcnn_model.load_state_dict(torch.load("../models/resnet50_car_dirtiness.pth", map_location=torch.device('cpu')))
    bestcnn_model.eval()  

st.title("InDrive Car Verification Service (Dirtiness Classification & Damages Detection)")

THRESHOLD_DEMAGE_DET_CONF = st.slider("Damage Detection Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
THRESHOLD_DIRTINESS_CONF = st.slider("Dirtiness Classification Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

img_file = st.file_uploader("Upload an image of a car", type=["jpg", "jpeg", "png"])

def preprocess_for_yolo(img_file: PIL.Image.Image) -> torch.Tensor:
    image = PIL.Image.open(img_file).convert("RGB")
    image = torchvision.transforms.functional.to_tensor(image)
    image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(640, 640), mode='bilinear', align_corners=False)
    return image  # Return tensor with shape (1, 3, 640, 640)

def preprocess_for_cnn(img_file: PIL.Image.Image) -> torch.Tensor:
    image = PIL.Image.open(img_file).convert("RGB")
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    return image.unsqueeze(0)

if st.button("Run Inference"):
    if img_file is not None:
        columnA, columnB = st.columns(2)
        with columnA:
            st.image(img_file, caption="Uploaded Image", width=640)
        with columnB:
            with st.spinner('Running inference...'):
                cnn_image = preprocess_for_cnn(img_file)
                cnn_output = bestcnn_model(cnn_image)
                probabilities = torch.nn.functional.softmax(cnn_output, dim=1)  
                dirtiness_confidence = probabilities[0][1].item()
                is_dirty = dirtiness_confidence >= THRESHOLD_DIRTINESS_CONF
                yolo_image = preprocess_for_yolo(img_file)
                yolo_results = besty11_model.predict(source=yolo_image, conf=THRESHOLD_DEMAGE_DET_CONF, save=False)
                annotated_frame = yolo_results[0].plot()
                st.image(annotated_frame, caption="Inference Result", width=640)
        if not is_dirty and len(yolo_results[0].boxes) == 0:
            st.success("The car is clean and has no visible damages.")
            st.json({
                "status": 1,
                "dirtiness": {
                    "is_dirty": False,
                    "confidence": float(dirtiness_confidence)
                },
                "damages": []
            }, expanded=False)
        elif is_dirty and len(yolo_results[0].boxes) > 0:
            st.json({
                "status": 0,
                "dirtiness": {
                    "is_dirty": True,
                    "confidence": float(dirtiness_confidence)
                },
                "damages": [
                    {
                        "box": box.xyxy,
                        "confidence": box.conf
                    } for box in yolo_results[0].boxes
                ]
            }, expanded=False)
        elif is_dirty:
            st.warning("The car is dirty but has no visible damages.")
            st.json({
                "status": 0,
                "dirtiness": {
                    "is_dirty": True,
                    "confidence": float(dirtiness_confidence)
                },
                "damages": []
            }, expanded=False)
        else:
            st.warning("The car has visible damages but is not classified as dirty.")
            st.json({
                "status": 0,
                "dirtiness": {
                    "is_dirty": False,
                    "confidence": float(dirtiness_confidence)
                },
                "damages": [
                    {
                        "box": box.xyxy,
                        "confidence": box.conf
                    } for box in yolo_results[0].boxes
                ]
            }, expanded=False)
    else:
        st.write("Please upload an image to run inference.")