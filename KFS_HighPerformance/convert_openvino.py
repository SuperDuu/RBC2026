import openvino as ov
import os
import sys

def convert_to_openvino():
    """
    Convert the GhostNetV3-YOLOv8n ONNX model to OpenVINO IR format.
    """
    onnx_path = 'ghostnetv3_yolov8n.onnx'
    output_dir = 'models/ghostnetv3_yolov8n_openvino'
    
    # Check if ONNX file exists
    if not os.path.exists(onnx_path):
        print(f"Error: ONNX file not found at {onnx_path}")
        print("Please run 'python ghostnetv3_yolov8n.py' first to generate the ONNX model.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[1/3] Loading ONNX model: {onnx_path}")
    try:
        # Convert model
        ov_model = ov.convert_model(onnx_path)
        
        print(f"[2/3] Saving OpenVINO IR to: {output_dir}")
        # Save model (XML and BIN)
        ov.save_model(ov_model, os.path.join(output_dir, 'model.xml'), compress_to_fp16=True)
        
        print(f"[3/3] Conversion successful!")
        print(f"Location: {os.path.abspath(output_dir)}")
        print("-" * 30)
        print(f"Files:")
        print(f"  - {os.path.join(output_dir, 'model.xml')}")
        print(f"  - {os.path.join(output_dir, 'model.bin')}")
        print("-" * 30)
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    convert_to_openvino()
