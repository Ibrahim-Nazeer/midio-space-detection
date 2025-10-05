EPOCHS = 25  # Reduced for focused fine-tuning
MOSAIC = 0.7  # Reduced - less aggressive
OPTIMIZER = 'AdamW'
MOMENTUM = 0.937
LR0 = 0.0001  # Lower learning rate for stable fine-tuning
LRF = 0.001   # Slower decay
SINGLE_CLS = False
PATIENCE = 15  # More patience
CLS_GAIN = 0.6  # Reduced - was too high
DFL_GAIN = 1.2  # Increased - better balance
WEIGHT_DECAY = 0.0005  # Reduced regularization
COS_LR = True
WARMUP_EPOCHS = 2.0
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR = 0.1
CLOSE_MOSAIC = 5  # Increased - keep mosaic longer
MIXUP = 0.1       # Reduced
COPY_PASTE = 0.2  # Reduced
FLIPUD = 0.3

import argparse
from ultralytics import YOLO
import os
import sys

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')    
    parser.add_argument('--patience', type=int, default=PATIENCE, help='Early stopping patience')

    args = parser.parse_args()
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    # Load your LATEST pre-trained model (from the just-finished training)
    model = YOLO("C:/Hackaura/Hackathon2_scripts/runs/detect/train/weights/best.pt")
    
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"), 
        epochs=args.epochs,
        device='0', 
        workers=2,
        optimizer=args.optimizer, 
        lr0=args.lr0, 
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=WEIGHT_DECAY,
        patience=args.patience,
        
        # CONSERVATIVE AUGMENTATION
        mosaic=args.mosaic,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        flipud=FLIPUD,
        close_mosaic=CLOSE_MOSAIC,
        
        # STABLE SETTINGS
        batch=6,
        imgsz=896,      # Keep proven resolution
        amp=True,
        cos_lr=True,
        nbs=64,
        
        # BALANCED AUGMENTATIONS
        degrees=10.0,
        translate=0.1,     # Reduced for stability
        scale=0.3,
        shear=1.5,
        perspective=0.0003,
        fliplr=0.5,
        
        # BALANCED LOSS
        cls=CLS_GAIN,
        dfl=DFL_GAIN,
        
        # VALIDATION
        val=True,
        save=True,
        save_period=5,     # More frequent saves
        plots=True,
        exist_ok=True,
    )
    
    # ========== FINAL VALIDATION ==========
    print("\n🎯 RUNNING FINAL VALIDATION...")
    
    try:
        # Run comprehensive validation
        metrics = model.val(
            data="yolo_params.yaml", 
            split="val", 
            workers=0,
            imgsz=896,
            conf=0.25,      # Use optimized confidence threshold
            iou=0.45
        )
        
        print("\n✅ FINAL VALIDATION RESULTS:")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        
        # Safe precision/recall handling
        precision_value = metrics.box.p
        recall_value = metrics.box.r
        if hasattr(precision_value, 'mean'):
            precision_value = precision_value.mean()
        if hasattr(recall_value, 'mean'):
            recall_value = recall_value.mean()
            
        print(f"Precision: {float(precision_value):.4f}")
        print(f"Recall: {float(recall_value):.4f}")
        
        print("\n🎯 PER-CLASS PERFORMANCE:")
        if hasattr(metrics, 'results_dict'):
            results_dict = metrics.results_dict
            for class_name in metrics.names:
                class_key = f'{class_name}/mAP50'
                if class_key in results_dict:
                    print(f"  {class_name}: {float(results_dict[class_key]):.4f}")
        
        print("\n🚀 TRAINING COMPLETED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
'''
Mixup boost val pred but reduces test pred
Mosaic shouldn't be 1.0  
'''


'''
                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]
Model summary: 225 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs
'''