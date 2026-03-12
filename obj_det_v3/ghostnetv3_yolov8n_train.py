import os
import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ghostnetv3_yolov8n import YOLOv8GhostNetV3

class GhostNetV3Trainer(DetectionTrainer):
    """
    Custom DetectionTrainer for YOLOv8 with GhostNetV3 backbone.
    """
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Initialize the custom YOLOv8GhostNetV3 model.
        """
        # nc is obtained from the data.yaml via the trainer configuration
        nc = self.data['nc'] if 'nc' in self.data else 1
        model = YOLOv8GhostNetV3(nc=nc, pretrained_backbone=False) # Use our custom class
        
        # Ultralytics expects these attributes for certain logic
        model.stride = torch.tensor([16.0, 32.0]) 
        # Attach names to the head if available
        if hasattr(self, 'data') and 'names' in self.data:
            model.names = self.data['names']
        else:
            model.names = {i: f'class_{i}' for i in range(nc)}
            
        return model

def train_ghostnet_model():
    """
    Setup and run training for the GhostNetV3-YOLOv8n model.
    Inherits hyperparameters from original yolov8n_train.py.
    """
    if torch.cuda.is_available():
        device = '0'
        torch.cuda.empty_cache() 
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print(f"Using GPU device: {device}")
    else:
        device = 'cpu'
        print("No GPU found! Using CPU.")

    # Training configuration matching yolov8n_train.py
    train_args = dict(
        model='yolov8n.pt',       # Satisfy string requirement (ignored in get_model)
        data='datasets/data.yaml', 
        epochs=100,               
        imgsz=512,               
        batch=20,                  
        device=device,                    
        workers=4,                
        project='RBC2026',                
        name='ghostnetv3_050_2scale_512px', 
        patience=30,              
        save=True,
        cache=False,              
        overlap_mask=True,
        lr0=0.01,                 
        cos_lr=True,              
        label_smoothing=0.05,     
        dropout=0.1,              
        mosaic=1.0,               
        mixup=0.2,                
        degrees=20.0,             
        scale=0.6,                
        translate=0.1,            
        fliplr=0.5,               
        close_mosaic=10
    )

    # Initialize custom trainer
    trainer = GhostNetV3Trainer(overrides=train_args)
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    train_ghostnet_model()
