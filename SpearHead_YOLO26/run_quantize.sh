#!/bin/bash

# Path to the exported FP16 model
MODEL_PATH="/home/du/Desktop/RBC2026/SpearHead_YOLO26/models/yolo26n_openvino_fp16/best.xml"

# Path to save the INT8 model
OUTPUT_PATH="/home/du/Desktop/RBC2026/SpearHead_YOLO26/models/yolo26n_openvino_int8/best_int8.xml"

# Path to the calibration dataset (using train images)
DATASET_PATH="/home/du/Desktop/RBC2026/SpearHead_YOLO26/datasets/train/images"

# Run the quantization script
echo "🚀 Starting INT8 Quantization for YOLO26n..."
python3 /home/du/Desktop/RBC2026/SpearHead_YOLO26/quantize_yolo_nncf_fixed.py \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH" \
    --dataset "$DATASET_PATH" \
    --input-size 512 \
    --max-samples 300

echo "✅ Quantization process finished."
