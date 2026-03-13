import openvino as ov
import tensorflow as tf
import numpy as np
import os

h5_model_path = 'models/classifier_v2.3.h5'
ov_output_path = 'models/openvino_cnn_v2_3/classifier_v2_3.xml'
os.makedirs('models/openvino_cnn_v2_3', exist_ok=True)
model = tf.keras.models.load_model(h5_model_path)
example_input = np.zeros((1, 64, 64, 1), dtype=np.float32)
try:
    ov_model = ov.convert_model(model, example_input=example_input)
    ov.save_model(ov_model, ov_output_path, compress_to_fp16=True)
    print(f"\n--- OK! ---")
    print(f"File: {ov_output_path}")
except Exception as e:
    print(f"\nFAIL: {e}")