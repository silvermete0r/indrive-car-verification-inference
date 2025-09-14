import streamlit as st
from ultralytics import YOLO
import torch
import torchvision 
import PIL.Image

st.set_page_config(
    page_title="inDrive Car Verification System Demo",
    page_icon="🚗",
    layout="wide"
)

with st.spinner('Loading model...'):
    besty11_model = YOLO("../models/besty11_car_demage_detection.pt")

st.title("InDrive Car Verification Service (Dirtiness Classification & Damages Detection)")

img_file = st.file_uploader("Upload an image of a car", type=["jpg", "jpeg", "png"])

if st.button("Run Inference"):
    if img_file is not None:
        columnA, columnB = st.columns(2)
        with columnA:
            st.image(img_file, caption="Uploaded Image", width=640)
        with columnB:
            with st.spinner('Running inference...'):
                image = PIL.Image.open(img_file).convert("RGB")
                image = torchvision.transforms.functional.to_tensor(image)
                image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(640, 640), mode='bilinear', align_corners=False).squeeze(0)
                results = besty11_model.predict(source=image, conf=0.5, save=False)
                annotated_frame = results[0].plot()
                st.image(annotated_frame, caption="Inference Result", width=640)
        st.success("Inference completed!")
    else:
        st.write("Please upload an image to run inference.")