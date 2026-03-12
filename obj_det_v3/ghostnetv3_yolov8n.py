import torch
import torch.nn as nn
import time
import sys

# Try to import Ultralytics and TIMM modules
try:
    import timm
    from ultralytics.nn.modules import Conv, C2f, SPPF, Detect, Concat
except ImportError as e:
    print(f"Error: Required library not found. Please run: pip install ultralytics timm")
    print(f"Details: {e}")
    sys.exit(1)

class GhostNetV3Backbone(nn.Module):
    """
    GhostNetV3 backbone implementation using timm.
    Outputs feature maps at indices (3, 4) which corresponds to P4, P5.
    Expected output channels for ghostnetv3_050: [40, 80] for P4, P5.
    """
    def __init__(self, pretrained=False):
        super().__init__()
        # Using feature_only=True to get intermediate feature maps
        # out_indices=(3, 4) -> P4, P5 features
        try:
            self.model = timm.create_model(
                'ghostnetv3_050', 
                pretrained=pretrained, 
                features_only=True, 
                out_indices=(3, 4)
            )
        except RuntimeError as e:
            if "No pretrained weights exist" in str(e):
                print(f"Warning: {e}. Falling back to random initialization.")
                self.model = timm.create_model(
                    'ghostnetv3_050', 
                    pretrained=False, 
                    features_only=True, 
                    out_indices=(3, 4)
                )
            else:
                raise e
        
    def forward(self, x):
        # Returns a list of tensors: [P4, P5]
        features = self.model(x)
        return tuple(features)

class YOLOv8GhostNetV3(nn.Module):
    """
    YOLOv8-like architecture using GhostNetV3-050 as backbone and 2-scale PANet neck.
    """
    def __init__(self, nc=1, pretrained_backbone=False, freeze_backbone=False):
        super().__init__()
        self.nc = nc
        
        # --- 1. Backbone ---
        self.backbone = GhostNetV3Backbone(pretrained=pretrained_backbone)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Output channels from GhostNetV3-050 P4, P5: [40, 80]
        c4, c5 = 40, 80
        d = 64  # Neck width
        
        # --- 2. Neck (2-scale simplified PANet) ---
        # SPPF on the deepest level (P5)
        self.sppf = SPPF(c5, d, k=5)  # SPPF(80, 64)
        
        # Top-down path
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.lat4 = Conv(c4, d, 1, 1) # lateral 1x1 to align P4 (40) -> 64
        self.td1 = C2f(d + d, d, n=2, shortcut=False) # 64+64 -> 64
        
        # Bottom-up path
        self.down1 = Conv(d, d, 3, 2) # Downsample P4_feat to stack on P5 path
        self.bu1 = C2f(d + d, d, n=2, shortcut=False) # 64+64 -> 64
        
        self.concat = Concat(dimension=1)
        
        # --- 3. Head ---
        # Detect head expects [P4_feat, P5_feat] with channels [64, 64]
        self.head = Detect(nc=nc, ch=(d, d))

    def forward(self, x):
        # Backbone: P4 (40, 40x40), P5 (80, 20x20)
        p4, p5 = self.backbone(x)
        
        # P5 (80) -> SPPF -> (64, 20x20)
        p5_feat = self.sppf(p5)
        
        # Top-down: p5_feat (64) -> Upsample -> (64, 40x40)
        p5_up = self.upsample(p5_feat)
        p4_lat = self.lat4(p4) # 40 -> 64
        td1 = self.td1(self.concat([p5_up, p4_lat])) # (64+64=128) -> 64
        
        # Bottom-up: td1 (64) -> Downsample -> (64, 20x20)
        td1_down = self.down1(td1)
        bu1 = self.bu1(self.concat([td1_down, p5_feat])) # (64+64=128) -> 64
        
        # Final outputs for Detect head: [P4, P5]
        return self.head([td1, bu1])

def build_model(nc=1, pretrained_backbone=False, freeze_backbone=False):
    """
    Build the YOLOv8-GhostNetV3 model and print summary.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOv8GhostNetV3(nc=nc, pretrained_backbone=pretrained_backbone, freeze_backbone=freeze_backbone).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("-" * 30)
    print(f"Model: YOLOv8-GhostNetV3-050 (2-scale)")
    print(f"Device: {device}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("-" * 30)
    
    return model

def test_forward():
    """
    Test forward pass, verify output shapes and measure latency.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(nc=1)
    model.eval()
    
    # Dummy input (B, C, H, W)
    img = torch.randn(2, 3, 640, 640).to(device)
    
    print("\n[Testing Forward Pass]")
    print("Expected anchors for 2-scale 640x640: 40*40 + 20*20 = 2000")
    
    with torch.no_grad():
        # First pass to warm up
        outputs = model(img)
        
        # Verify output shape [2, 5, 2000] (5 = 4 box + 1 nc)
        if isinstance(outputs, (list, tuple)):
            print(f"Output type: {type(outputs)}")
            for i, out in enumerate(outputs):
                if isinstance(out, torch.Tensor):
                    print(f"Output {i} shape: {out.shape}")
        else:
            print(f"Output shape: {outputs.shape}")
            expected_shape = (2, 5, 2000)
            if outputs.shape == expected_shape:
                print(f"SUCCESS: Output shape matches {expected_shape}")
            else:
                print(f"WARNING: Output shape {outputs.shape} mismatch with expected {expected_shape}")
            
    print("\n[Benchmarking Latency]")
    num_iters = 10
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iters):
            _ = model(img)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iters
    print(f"Average inference time over {num_iters} iterations: {avg_time:.4f}s ({1/avg_time:.2f} FPS)")
    
    return model, img

if __name__ == "__main__":
    model, dummy_input = test_forward()
    
    print("\n[Exporting to ONNX]")
    onnx_file = "ghostnetv3_yolov8n.onnx"
    try:
        # We need to set the model to eval mode and use a single batch for simple export
        model.eval()
        single_batch_input = torch.randn(1, 3, 640, 640).to(next(model.parameters()).device)
        
        torch.onnx.export(
            model,
            single_batch_input,
            onnx_file,
            input_names=['images'],
            output_names=['output0'],
            opset_version=12,
            dynamic_axes={'images': {0: 'batch'}, 'output0': {0: 'batch'}}
        )
        print(f"Model exported successfully to {onnx_file}")
    except Exception as e:
        print(f"Failed to export ONNX: {e}")
