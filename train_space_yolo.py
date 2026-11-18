import os
import time
import torch
from ultralytics import YOLO


def train_and_evaluate():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"[INFO] Running training pipeline on device: {device}")

  model = YOLO("yolov8n.pt")

  print("[INFO] Starting YOLOv8-Nano training on Deep Space Dataset...")
  results = model.train(
      data="space_data.yaml",
      epochs=100,
      imgsz=640,
      batch=16,
      optimizer="AdamW",
      lr0=0.001,
      momentum=0.937,
      weight_decay=0.0005,
      cos_lr=True,
      hsv_h=0.0,
      hsv_s=0.1,
      hsv_v=0.4,
      degrees=180.0,
      mosaic=1.0,
      device=device,
      project="runs/space_detection",
      name="yolov8n_deepspace",
  )

  print("[INFO] Evaluating model performance on test dataset...")
  metrics = model.val(data="space_data.yaml", split="test")
  print(f"[RESULT] mAP@0.5: {metrics.box.map50:.4f}")
  print(f"[RESULT] mAP@0.5:0.95: {metrics.box.map:.4f}")

  print(
      "[INFO] Exporting optimized ONNX model for spacecraft edge processor..."
  )
  onnx_path = model.export(format="onnx", half=True, simplify=True)
  print(f"[SUCCESS] Edge-ready ONNX model saved at: {onnx_path}")


if __name__ == "__main__":
  train_and_evaluate()
