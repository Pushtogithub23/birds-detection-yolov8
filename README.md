# Birds Detection using YOLOv8 🦅

A custom implementation of YOLOv8 for detecting birds in images and videos. This project provides a complete pipeline from dataset preparation to model training and inference.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![Ultralytics](https://img.shields.io/badge/Ultralytics-Latest-blue)
![Roboflow](https://img.shields.io/badge/Roboflow-Dataset-yellow)

![crows_detected](https://github.com/user-attachments/assets/b914acd3-4dfe-4bc8-b816-31871afbf48b)

![kingfishers_detected](https://github.com/user-attachments/assets/897ddae9-fd8d-4eb9-88c5-6d66dceefcda)

![sparrow_detected_2](https://github.com/user-attachments/assets/e956316f-b660-423f-8f2d-1fa3949e5bd7)



## 📋 Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Inference](#inference)
- [Best Practices](#best-practices)
- [GPU Requirements](#gpu-requirements)

## ✨ Features
- Custom YOLOv8 model training for bird detection
- Support for both image and video inference
- Real-time object tracking in videos
- Configurable confidence thresholds
- Automatic annotation visualization
- Support for both local files and URLs
- Save capabilities for annotated images and videos

## 🔧 Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Google Colab (optional, for free GPU access)

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/Pushtogithub23/birds-detection-yolov8.git
cd birds-detection-yolov8
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Image Detection
```python
from detection import display_prediction

# For local image
display_prediction("path/to/your/image.jpg", save_fig=True, filename="detected.jpg")

# For image URL
display_prediction("https://example.com/image.jpg", save_fig=True, filename="detected.jpg")
```
I have attached below few detections in images:

![crows_detected_2](https://github.com/user-attachments/assets/d507d7da-8a19-43dd-9c39-54d8a602e983)

![peacocks_detected](https://github.com/user-attachments/assets/49f57849-7f8c-4d43-b932-e9004ae506fd)

![owls_detected](https://github.com/user-attachments/assets/014a9b59-489b-4c4c-a9af-0b6ce803c128)


### Video Detection
```python
from detection import predict_in_videos

predict_in_videos(
    "path/to/your/video.mp4",
    save_video=True,
    filename="detected_video.mp4"
)
```
I have attached a video detection below.


https://github.com/user-attachments/assets/a8d0d160-8589-4879-ad34-8ae1a0cb3a0f

## 📊 Dataset Preparation

1. Create a Roboflow account and obtain your API key
2. Update the API key in the notebook:
```python
rf = Roboflow(api_key="YOUR_API_KEY")
```

3. The dataset structure should be:
```
BIRDS-DETECTION-1/
├── train/
├── valid/
├── test/
└── data.yaml
```

## 🏋️ Model Training

The model is trained using YOLOv8-large (yolov8l.pt) as the base model:

```python
model = YOLO('yolov8l.pt')
model.train(
    data="BIRDS-DETECTION-1/data.yaml",
    epochs=100,
    imgsz=640
)
```

Training parameters can be adjusted based on your needs:
- `epochs`: Number of training epochs
- `imgsz`: Input image size
- `batch`: Batch size (adjust based on GPU memory)

## 🎯 Inference

### Detection Function Features
- Automatic thickness calculation based on image resolution
- Dynamic text scaling
- Confidence threshold filtering (default: 0.5)
- Color-coded annotations using 'magma' colormap
- Support for both image and video processing
- Object tracking in videos using ByteTrack

### Video Processing Features
- Real-time object tracking
- FPS-synchronized processing
- Progress visualization
- Press 'p' to stop processing

## 💡 Best Practices

### Dataset Preparation
- Ensure dataset is well-balanced
- Include images with varying lighting conditions
- Use data augmentation for better generalization

### Training Tips
- Monitor training metrics in runs/detect/train directory
- Adjust confidence threshold based on application needs
- Consider model ensembling for better results

### Production Deployment
- Move API keys to environment variables
- Implement proper error handling
- Consider model quantization for faster inference

## 💻 GPU Requirements

This project was developed and tested on Google Colab with T4 GPU acceleration. When running on different hardware:
- Adjust batch size according to available GPU memory
- Modify image size if needed
- Consider using model quantization for faster inference on less powerful hardware

