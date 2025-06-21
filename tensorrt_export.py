import os
from ultralytics import YOLO

# Load the YOLO11 model
model_path = "./models/yolo_model/best_5_31.pt"

# モデルファイルの存在確認
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

try:
    model = YOLO(model_path)
    
    # Export the model to TensorRT format with optimization options
    model.export(
        format="engine",
        device=0,  # GPU device ID
        half=True,  # FP16 precision for better performance
        dynamic=False,  # Fixed input size (not dynamic)
        simplify=True,  # Simplify the model
        workspace=4,  # TensorRT workspace size (GB)
        imgsz=1920,  # Fixed input size 1920x1920
    )
    
    print(f"TensorRT export completed successfully!")
    
except Exception as e:
    print(f"Export failed: {e}")