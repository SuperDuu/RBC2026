import torch
import torch.nn as nn
import time
import sys

try:
    import timm
    from ultralytics.nn.modules import Conv, C2f, SPPF, Detect
    from ultralytics.utils.loss import v8DetectionLoss
except ImportError as e:
    print(f"Error: Required library not found. Please run: pip install ultralytics timm")
    print(f"Details: {e}")
    sys.exit(1)


class GhostNetV3Backbone(nn.Module):
    """
    GhostNetV3-050 backbone wrapper using timm.
    out_indices=(3, 4) -> P4 (stride 16), P5 (stride 32).
    Suitable for large objects only (no P3).
    """
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            'ghostnetv3_100',       # _050 không có pretrained weights trên timm
            pretrained=pretrained,
            features_only=True,
            out_indices=(3, 4)
        )
        self.channels = self.model.feature_info.channels()
        print(f"GhostNetV3-100 channels: P4={self.channels[0]}, P5={self.channels[1]}")

    def forward(self, x):
        features = self.model(x)
        return features[0], features[1]  # p4, p5


class YOLOv8GhostNetV3(nn.Module):
    """
    YOLOv8-style detector with GhostNetV3-050 backbone.
    2-scale neck (P4 + P5), optimized for large objects, single class.

    forward() handles two call conventions:
      - Training  : x is a dict {'img': Tensor, ...} from Ultralytics trainer
      - Inference : x is a Tensor [B, 3, H, W]
    """
    def __init__(self, nc=1, pretrained_backbone=True, freeze_backbone=False):
        super().__init__()
        self.nc = nc
        self.criterion = None

        # ── BACKBONE ──────────────────────────────────────────
        self.backbone = GhostNetV3Backbone(pretrained=pretrained_backbone)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        c4, c5 = self.backbone.channels
        d = 64  # neck width

        # ── NECK (PANet, 2-scale) ──────────────────────────────
        self.sppf     = SPPF(c5, d, k=5)
        self.lat4     = Conv(c4, d, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.td1      = C2f(d + d, d, n=2, shortcut=False)  # top-down P5->P4
        self.down1    = Conv(d, d, 3, 2)
        self.bu1      = C2f(d + d, d, n=2, shortcut=False)  # bottom-up P4->P5

        # ── HEAD ──────────────────────────────────────────────
        self.head = Detect(nc=nc, ch=(d, d))
        # model.model is required by v8DetectionLoss
        self.model = nn.ModuleList([self.head])

    def _extract_features(self, imgs):
        """Backbone + neck -> (td1, bu1)."""
        p4, p5 = self.backbone(imgs)

        p5_feat = self.sppf(p5)
        p4_lat  = self.lat4(p4)
        td1     = self.td1(torch.cat([self.upsample(p5_feat), p4_lat], dim=1))
        bu1     = self.bu1(torch.cat([self.down1(td1), p5_feat], dim=1))
        return td1, bu1

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            # Ultralytics trainer passes full batch dict during training
            td1, bu1 = self._extract_features(x['img'])
            preds = self.head([td1, bu1])
            if self.training:
                if self.criterion is None:
                    self.criterion = v8DetectionLoss(self)
                return self.criterion(preds, x)  # returns (loss, loss_items)
            return preds
        else:
            # Tensor input: inference or ONNX export or Validation (with augment=True/False)
            td1, bu1 = self._extract_features(x)
            return self.head([td1, bu1])       # returns predictions


def build_model(nc=1, pretrained_backbone=True, freeze_backbone=False):
    """Build model and print parameter summary."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = YOLOv8GhostNetV3(
        nc=nc,
        pretrained_backbone=pretrained_backbone,
        freeze_backbone=freeze_backbone
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone  = sum(p.numel() for p in model.backbone.parameters())

    print("─" * 40)
    print(f"  Model    : YOLOv8-GhostNetV3-050 (2-scale)")
    print(f"  Classes  : {nc}")
    print(f"  Device   : {device}")
    print(f"  Backbone : {backbone/1e6:.2f}M params")
    print(f"  Neck+Head: {(total-backbone)/1e6:.2f}M params")
    print(f"  Total    : {total/1e6:.2f}M params")
    print(f"  Trainable: {trainable/1e6:.2f}M params")
    print("─" * 40)

    return model


def test_forward():
    """
    Verify shapes and latency.
    Expected output: [B, 5, 2000]
    640x640 -> 40x40 + 20x20 = 2000 anchors
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = build_model(nc=1)
    model.eval()

    dummy = torch.randn(2, 3, 640, 640).to(device)

    print("\n[Testing Forward Pass]")
    print("Expected: 40x40 + 20x20 = 2000 anchors -> shape [2, 5, 2000]")
    with torch.no_grad():
        outputs = model(dummy)

    if isinstance(outputs, (list, tuple)):
        for i, o in enumerate(outputs):
            if isinstance(o, torch.Tensor):
                print(f"  Output[{i}]: {o.shape}")
    else:
        print(f"  Output: {outputs.shape}")

    print("\n[Benchmarking]")
    n = 10
    t = time.time()
    with torch.no_grad():
        for _ in range(n):
            _ = model(dummy)
    avg = (time.time() - t) / n * 1000
    print(f"  Avg: {avg:.1f}ms  ({1000/avg:.1f} FPS)")

    return model
def benchmark_vs_yolov8n():
    from ultralytics import YOLO
    import torch, time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy  = torch.randn(1, 3, 640, 640).to(device)
    n      = 100

    # ── GhostNetV3 ────────────────────────────────────────
    ghost = build_model(nc=1).to(device)
    ghost.eval()
    with torch.no_grad():
        for _ in range(10): ghost(dummy)          # warmup
        t = time.time()
        for _ in range(n): ghost(dummy)
    ghost_ms = (time.time() - t) / n * 1000

    # ── YOLOv8n gốc ───────────────────────────────────────
    yolo = YOLO('yolov8n.pt').model.to(device)
    yolo.eval()
    with torch.no_grad():
        for _ in range(10): yolo(dummy)           # warmup
        t = time.time()
        for _ in range(n): yolo(dummy)
    yolo_ms = (time.time() - t) / n * 1000

    print(f"\n{'─'*40}")
    print(f"  GhostNetV3 (bản của bạn): {ghost_ms:.1f}ms")
    print(f"  YOLOv8n gốc            : {yolo_ms:.1f}ms")
    print(f"  Nhanh hơn              : {yolo_ms/ghost_ms:.2f}x")
    print(f"{'─'*40}")

    

if __name__ == '__main__':
    model = test_forward()

    print("\n[Exporting to ONNX]")
    try:
        model.eval()
        device = next(model.parameters()).device
        dummy  = torch.randn(1, 3, 640, 640).to(device)
        torch.onnx.export(
            model, dummy, "ghostnetv3_yolov8n.onnx",
            input_names=['images'],
            output_names=['output0'],
            opset_version=12,
            dynamic_axes={'images': {0: 'batch'}, 'output0': {0: 'batch'}}
        )
        print("  ONNX exported: ghostnetv3_yolov8n.onnx")
    except Exception as e:
        print(f"  ONNX export failed: {e}")
    benchmark_vs_yolov8n()