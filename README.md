# InDrive Car Verification Service (Dirtiness Classification & Damages Detection)

Diagram ~ Architecture:
[]()

## 1. **Damages Detection**:

1.1. Comparing **Instance Segmentation** and **Object Detection** models for damage detection on car images (models were trained on [Roboflow](https://roboflow.com/) platform):

| Model | Method | mAP50 | Precision | Recall | Train Time | Dataset Size | Dataset |
|-------|--------|-------|-----------|--------|------------|--------------|---------|
| Roboflow 3.0 Instance Segmentation | Instance Segmentation | 0.49 | 0.67 | 0.43 | 5h | 1700 (external) | [car-damage-coco-dataset](https://universe.roboflow.com/dan-vmm5z/car-damage-coco-dataset) |
| **🌟 YOLOv11 Object Detection (Fast)** | **Object Detection** | **0.66** | **0.75** | **0.63** | **3h** | 2932 (custom) | [multi-label-car-damage-detection-hfvtf](https://universe.roboflow.com/computer-vision-projects-w1m15/multi-label-car-damage-detection-hfvtf/) |

1.2. Choosing the best multi-label object detection model for car damages detection from YOLOv11 family:

| Model     | mAP (val 50-95) | Speed (CPU ONNX ms) | Speed (A100 / TensorRT ms) | Parameters (M) | FLOPs (B) |
|-----------|------------------|----------------------|-----------------------------|----------------|-----------|
| YOLO11n   | 39.5             | ~56.1                | ~1.5                        | 2.6            | 6.5       |
| **🌟 YOLO11s**   | **47.0**             | **~90.0**                | **~2.5**                        | **9.4**            | **21.5**      |
| YOLO11m | 51.5         | ~183.2               | ~4.7                        | 20.1           | 68.0      |
| YOLO11l   | 53.4             | ~238.6               | ~6.2                        | 25.3           | 86.9      |
| YOLO11x   | 54.7             | ~462.8               | ~11.3                       | 56.9           | 194.9     |

[YOLO11 Performance Benchmarks on COCO Dataset – Ultralytics Docs](https://docs.ultralytics.com/models/yolo11/)

1.3. Custom Dataset for Damages Detection:
 - 4 main labels: 
    - scratch (1,617 labels)
    - deformation (1,397 labels)
    - rust (1,058 labels)	
    - broken-glass (907 labels)
 - Main sources:
    - [car-damage](https://universe.roboflow.com/skillfactory/car-damage-c1f0i)
    - [car-scratch-and-dent](https://universe.roboflow.com/carpro/car-scratch-and-dent)
    - [rust-and-scrach](https://universe.roboflow.com/seva-at1qy/rust-and-scrach)
    - [car-scratch-xgxzs](https://universe.roboflow.com/project-kmnth/car-scratch-xgxzs)
    - [corrosion-of-metal-od](https://universe.roboflow.com/sgga/corrosion-of-metal-od)
    - [car_detection_fast_rcnn](https://universe.roboflow.com/test-qssu6/car_detection_fast_rcnn)
    - [car-cr9cg](https://universe.roboflow.com/egor-6ctjq/car-cr9cg)
  - Total collected, filtered and annotated: 2932 images.
  - The following pre-processing was applied to each image:
    - Auto-orientation of pixel data (with EXIF-orientation stripping)
    - Resize to 512x512 (Stretch) 
    - Augmentation applied (to create 2 versions of each source image):
        * Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
        * Randomly crop between 0 and 20 percent of the image
  - Dataset split: 88% train (4594), 6% validation (333), 6% test (302)
  - Total pre-processed and augmented dataset size: **5229 images**.
    - Damages are annotated in YOLOv11 format.
    - Dataset is available on [Roboflow](https://universe.roboflow.com/computer-vision-projects-w1m15/multi-label-car-damage-detection-hfvtf/dataset/2)

1.4. YOLO11s model training parameters:
 - Epochs: 100
 - Batch size: 16
 - Image size: 640x640
 - Device: CUDA (Tesla T4 ~ Google Colab)
 - Pre-trained weights: [yolov11s.pt](https://docs.ultralytics.com/ru/models/yolo11/)

1.5. 


## 2. **Dirtiness Classification**:

2.1. Dataset for Dirtiness Classification:
 - 3 main labels (manually collected and annotated):
    - clean (300 images)
    - dirty (300 images)
 - Main source: [Stanford Cars Dataset](https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset)

2.2. Comparing **Image Classification** pre-trained models for dirtiness classification on car images (models were trained on this [Kaggle Notebook](https://www.kaggle.com/code/armanzhalgasbayev/car-dirtiness-classification)):

| Model | Type | Accuracy | Precision | Recall | F1-Score | Train Time |
|-------|------|----------|-----------|--------|----------|------------|
| 