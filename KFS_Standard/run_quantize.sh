#!/bin/bash
# Script để quantize YOLO sang INT8

cd "$(dirname "$0")"  

echo "Current directory: $(pwd)"
echo "Running quantization with NNCF..."

python3 quantize_yolo_nncf_fixed.py \
    --model models/best_openvino_model/best.xml \
    --output models/best_openvino_model_int8/best_int8.xml \
    --dataset datasets/train \
    --input-size 512 \
    --max-samples 300
