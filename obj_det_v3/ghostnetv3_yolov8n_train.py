import os
import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ghostnetv3_yolov8n import YOLOv8GhostNetV3


class GhostNetV3Trainer(DetectionTrainer):
    """
    Custom DetectionTrainer tích hợp GhostNetV3-100 backbone.
    2-scale neck (P4+P5) tối ưu cho object to, 1 class.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Khởi tạo YOLOv8GhostNetV3, bỏ qua cfg/weights mặc định."""

        nc = self.data.get('nc', 1)

        # ── Tạo model ────────────────────────────────────────
        model = YOLOv8GhostNetV3(
            nc=nc,
            pretrained_backbone=True
        )

        # ── Stride: bắt buộc cho anchor generation ───────────
        strides           = torch.tensor([16.0, 32.0])
        model.stride      = strides
        model.head.stride = strides

        # ── Attributes bắt buộc cho Ultralytics trainer ──────
        model.end2end  = False
        model.args     = self.args
        model.task     = 'detect'
        model.pt_path  = None
        model.yaml     = None

        # ── Head attributes ───────────────────────────────────
        model.head.nc      = nc
        model.head.nl      = 2    # số scale (P4, P5)
        model.head.reg_max = 16   # mặc định YOLOv8

        # ── Names ─────────────────────────────────────────────
        model.names = self.data.get(
            'names', {i: f'class_{i}' for i in range(nc)}
        )

        if verbose:
            total     = sum(p.numel() for p in model.parameters()) / 1e6
            trainable = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ) / 1e6
            backbone  = sum(
                p.numel() for p in model.backbone.parameters()
            ) / 1e6
            print(f"\n{'─'*40}")
            print(f"  Model    : YOLOv8-GhostNetV3-100 (2-scale)")
            print(f"  Classes  : {nc}")
            print(f"  Backbone : {backbone:.2f}M params")
            print(f"  Total    : {total:.2f}M params")
            print(f"  Trainable: {trainable:.2f}M params")
            print(f"{'─'*40}\n")

        return model


def train():
    device = '0' if torch.cuda.is_available() else 'cpu'
    if device == '0':
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print(f"Device: {'GPU' if device == '0' else 'CPU'}")

    train_args = dict(
        model='yolov8n.pt',                    # placeholder, bị override bởi get_model
        data='datasets/data.yaml',
        epochs=120,
        imgsz=512,
        batch=8,
        device=device,
        workers=4,
        project='RBC2026',
        name='ghostnetv3_050_2scale_512px',
        patience=30,
        save=True,
        cache='ram',
        # ── Augmentation ──────────────────────────────────────
        mosaic=1.0,
        mixup=0.2,
        degrees=20.0,
        scale=0.6,
        translate=0.1,
        fliplr=0.5,
        close_mosaic=20,
        # ── Hyperparams ───────────────────────────────────────
        lr0=0.01,
        cos_lr=True,
        label_smoothing=0.05,
        dropout=0.1,
        overlap_mask=True,
    )

    trainer = GhostNetV3Trainer(overrides=train_args)
    trainer.train()


if __name__ == '__main__':
    train()