# Autonomous Deep Learning Model for Real-Time Celestial Object Detection in Deep Space Missions

**Course:** MAE 551 / MAE 451 — Arizona State University  
**Authors:** **Vedang Verma (Team Lead & Primary Author)**, Nicholas Cagle, Mohammed Alkhamis  

## Overview
Deep space exploration missions suffer from severe communication latency with Earth, making real-time human teleoperation impossible. This repository implements an edge-optimized **YOLOv8-Nano** computer vision pipeline designed to run on onboard spacecraft processors for autonomous detection and classification of **Asteroids**, **Space Debris**, **Spacecraft**, and **Background Stars**.

## Key Features
* **Edge-Optimized Architecture:** Uses YOLOv8-Nano (`3.2M` parameters) achieving **88.9% mAP@0.5** at **7.4 ms latency (~135 FPS)**.
* **Physics-Aware Data Augmentation:** Disables unnatural hue shifts while simulating solar glare, sensor noise, and 360-degree orbital rotations.
* **Onboard Deployment Ready:** Includes automated export to quantized **ONNX (FP16)** format for low-power flight computers (e.g., NVIDIA Jetson Orin Nano).

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd Autonomous-DeepSpace-Object-Detection
   ```

2. **Install dependencies:**
   ```bash
   pip install ultralytics torch torchvision opencv-python onnx onnxslim
   ```

## How to Run

### 1. Train & Evaluate the Model
Ensure your dataset follows the YOLO directory structure defined in `space_data.yaml`, then run:
```bash
python train_space_yolo.py
```

### 2. Run Inference on Space Imagery
To run real-time bounding box detection on sample test images:
```bash
yolo task=detect mode=predict model=runs/space_detection/yolov8n_deepspace/weights/best.pt source=datasets/deepspace_objects/images/test
```

## Results Summary

| Model | Parameters | mAP@0.5 | mAP@0.5:0.95 | Latency | Edge FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline Custom CNN | 1.8M | 68.4% | 41.2% | 48.5 ms | 20.6 |
| YOLOv8-Small | 11.2M | 91.2% | 67.8% | 19.2 ms | 52.0 |
| **YOLOv8-Nano (Ours)** | **3.2M** | **88.9%** | **64.5%** | **7.4 ms** | **135.1** |

## Author Contributions
* **Vedang Verma (Team Lead):** Led project architecture design, dataset preprocessing, YOLOv8 model training, edge-latency benchmarking, experimental analysis, GitHub repository implementation, and primary manuscript preparation.
* **Nicholas Cagle & Mohammed Alkhamis:** Assisted with literature review, technical verification, and final report proofreading.
