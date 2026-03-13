# prune.py - chạy SAU KHI train xong
import torch
import torch.nn.utils.prune as prune
from ghostnetv3_yolov8n import YOLOv8GhostNetV3

def prune_model(weights_path, sparsity=0.5):
    """
    sparsity=0.5 → cắt 50% weight nhỏ nhất
    sparsity=0.8 → cắt 80% → gần 0.3M params
    """
    # Load model đã train
    model = YOLOv8GhostNetV3(nc=1, pretrained_backbone=False)
    model.load_state_dict(torch.load(weights_path))
    model.eval()

    before = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Before pruning: {before:.2f}M params")

    # Prune tất cả Conv2d layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=sparsity)
            prune.remove(module, 'weight')  # make permanent

    after = sum(
        (p != 0).sum().item() for p in model.parameters()
    ) / 1e6
    print(f"After pruning : {after:.2f}M non-zero params")

    return model


def finetune_pruned(model, epochs=20):
    """Fine-tune sau pruning để recover accuracy."""
    from ghostnetv3_yolov8n_train import GhostNetV3Trainer

    # Gán lại model đã prune vào trainer
    train_args = dict(
        model='yolov8n.pt',
        data='datasets/data.yaml',
        epochs=epochs,          # chỉ cần 20 epochs
        imgsz=512,
        batch=20,
        lr0=0.001,              # LR nhỏ hơn lúc train gốc
        cos_lr=True,
        project='RBC2026',
        name='ghostnetv3_pruned_finetune',
    )

    trainer = GhostNetV3Trainer(overrides=train_args)
    trainer.model = model
    trainer.train()


if __name__ == '__main__':
    # Đường dẫn weights sau khi train xong
    weights = 'RBC2026/ghostnetv3_050_2scale_512px/weights/best.pt'

    # Thử các mức pruning
    for sparsity in [0.5, 0.7, 0.8]:
        print(f"\n── Sparsity {sparsity*100:.0f}% ──")
        model = prune_model(weights, sparsity=sparsity)