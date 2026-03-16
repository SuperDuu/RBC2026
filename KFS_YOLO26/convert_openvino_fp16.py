from ultralytics import YOLO
import os
import sys

def convert_to_openvino_fp16():
    """
    Convert the trained yolo26n model to OpenVINO IR format (FP16).
    """
    # Path to the best trained model
    pt_path = '/home/du/Desktop/RBC2026/KFS_YOLO26/runs/detect/RBC2026/best.pt'
    output_dir = 'models/yolo26n_openvino_fp16'
    
    # Check if .pt file exists
    if not os.path.exists(pt_path):
        print(f"Error: Model weights not found at {pt_path}")
        print("Please ensure the training has completed successfully.")
        sys.exit(1)
    
    print(f"[2/2] Exporting to OpenVINO (FP16)...")
    try:
        model = YOLO(pt_path)
        
        # Export the model
        # half=True ensures FP16 precision
        temp_export_path = model.export(format='openvino', half=True, imgsz=512)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        
        # Move the exported directory to models/
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        shutil.move(temp_export_path, output_dir)
        
        print(f"\nConversion successful!")
        print(f"Export Path: {output_dir}")
        print("-" * 30)
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    convert_to_openvino_fp16()
