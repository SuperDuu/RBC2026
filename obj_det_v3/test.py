import timm
import torch
import torch.nn as nn

# Kiểm tra model có sẵn
model = timm.create_model('ghostnetv3_100', pretrained=True, features_only=True)
print(model.feature_info.channels())  # [16, 24, 40, 80, 160] tùy variant