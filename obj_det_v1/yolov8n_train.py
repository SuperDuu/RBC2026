from ultralytics import YOLO
import torch
import os

def train_best_model():
    if torch.cuda.is_available():
        device = '0'
        torch.cuda.empty_cache() 
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    else:
        print("No GPU!")
        return

    model = YOLO('yolov8n.pt') 
    model.train(
        data='datasets/data.yaml', 
        epochs=100,               
        imgsz=512,               
        batch=20,                  
        device=device,                    
        workers=4,                
        project='RBC2026',                
        name='yolov8n_512px', 
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
        close_mosaic=20           
    )

if __name__ == '__main__':
    train_best_model()