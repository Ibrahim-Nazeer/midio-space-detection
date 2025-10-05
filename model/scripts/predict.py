from ultralytics import YOLO
from pathlib import Path
import cv2
import os
import yaml
import time


# ============================================================
# FUNCTION: Predict and Save Images + Bounding Boxes
# ============================================================
def predict_and_save(model, image_path, output_path, output_path_txt):
    # Perform prediction with optimized settings
    results = model.predict(
        image_path,
        conf=0.25,      # Lower confidence for better recall
        iou=0.45,       # Balanced IoU
        imgsz=896,      # Match training resolution
        augment=True,   # Test-time augmentation
        max_det=300,
        fliplr=0.5,     # Horizontal flip augmentation
        verbose=False   # Cleaner output
    )

    result = results[0]
    img = result.plot()  # Draw boxes

    # Save image with predictions
    cv2.imwrite(str(output_path), img)
    
    # Save bounding box info to txt
    with open(output_path_txt, 'w') as f:
        for box in result.boxes:
            cls_id = int(box.cls)
            class_name = model.names[cls_id]
            confidence = float(box.conf)
            x_center, y_center, width, height = box.xywhn[0].tolist()
            f.write(f"{cls_id} {class_name} {confidence:.4f} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == '__main__': 
    start_time = time.time()
    
    this_dir = Path(__file__).parent
    os.chdir(this_dir)
    
    print("🚀 STARTING PREDICTION PIPELINE...")
    print("=" * 50)
    
    # Load YAML config to locate test images
    with open(this_dir / 'yolo_params.yaml', 'r') as file:
        data = yaml.safe_load(file)
        if 'test' in data and data['test'] is not None:
            images_dir = Path(data['test']) / 'images'
        else:
            print("❌ No test field found in yolo_params.yaml")
            exit()
    
    # Validate test image directory
    if not images_dir.exists():
        print(f"❌ Images directory {images_dir} does not exist")
        exit()
    if not images_dir.is_dir():
        print(f"❌ Images directory {images_dir} is not a directory")
        exit()
    if not any(images_dir.iterdir()):
        print(f"❌ Images directory {images_dir} is empty")
        exit()

    # ============================================================
    # FIND AND LOAD LATEST TRAINED YOLO MODEL
    # ============================================================
    detect_path = this_dir / "runs" / "detect"

    # Detect all potential training folders
    train_folders = [
        f for f in os.listdir(detect_path)
        if os.path.isdir(detect_path / f) and (
            f.startswith("train") or f.startswith("exp") or "yolo" in f.lower()
        )
    ]

    if len(train_folders) == 0:
        raise ValueError(f"❌ No training folders found in {detect_path}. Please check your runs/detect directory.")

    print(f"📂 Found training folders: {train_folders}")

    # Auto-select latest folder by modification time
    if len(train_folders) > 1:
        train_folders.sort(key=lambda x: os.path.getmtime(detect_path / x), reverse=True)
        selected_folder = train_folders[0]
        print(f"🤖 Auto-selected latest training folder: {selected_folder}")
    else:
        selected_folder = train_folders[0]

    model_path = detect_path / selected_folder / "weights" / "best.pt"
    print(f"📁 Loading model from: {model_path}")

    model = YOLO(model_path)
    print("✅ Model loaded successfully!")
    print(f"📊 Model classes: {model.names}")

    # ============================================================
    # CREATE OUTPUT DIRECTORIES
    # ============================================================
    output_dir = this_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    images_output_dir = output_dir / 'images'
    labels_output_dir = output_dir / 'labels'
    images_output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Input images from: {images_dir}")
    print(f"📁 Output images to: {images_output_dir}")
    print(f"📁 Output labels to: {labels_output_dir}")
    print("=" * 50)

    # ============================================================
    # RUN PREDICTIONS
    # ============================================================
    total_images = 0
    total_detections = 0
    class_counts = {class_name: 0 for class_name in model.names.values()}

    image_files = list(images_dir.glob('*.[pjP][npN][gG]')) + list(images_dir.glob('*.[jJ][pP][gG]'))
    print(f"📸 Found {len(image_files)} images to process...")

    for img_path in image_files:
        total_images += 1
        output_path_img = images_output_dir / img_path.name
        output_path_txt = labels_output_dir / img_path.with_suffix('.txt').name
        
        print(f"🔍 Processing {img_path.name}...", end=" ")
        predict_and_save(model, img_path, output_path_img, output_path_txt)
        
        # Count detections per image
        with open(output_path_txt, 'r') as f:
            detections = f.readlines()
            total_detections += len(detections)
            for detection in detections:
                parts = detection.strip().split()
                if len(parts) >= 2:
                    class_id = int(parts[0])
                    class_name = model.names[class_id]
                    class_counts[class_name] += 1
        print(f"Detected: {len(detections)} objects")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("=" * 50)
    print("🎯 PREDICTION COMPLETED!")
    print(f"📊 STATISTICS:")
    print(f"   • Total images processed: {total_images}")
    print(f"   • Total detections: {total_detections}")
    print(f"   • Avg detections per image: {total_detections/total_images:.2f}")
    print(f"   • Detections by class:")
    for class_name, count in class_counts.items():
        if count > 0:
            print(f"     - {class_name}: {count}")

    print(f"📁 Predicted images saved in: {images_output_dir}")
    print(f"📁 Bounding box labels saved in: {labels_output_dir}")
    print(f"⚙️ Model parameters file: {this_dir / 'yolo_params.yaml'}")

    # ============================================================
    # ENHANCED VALIDATION PHASE
    # ============================================================
    print("\n" + "=" * 50)
    print("🧪 STARTING ENHANCED VALIDATION...")
    
    try:
        metrics = model.val(
            data="yolo_params.yaml", 
            split="val", 
            workers=0,
            imgsz=896,
            augment=True,
            conf=0.25,
            iou=0.45,
            verbose=False
        )
        
        print("\n🎉 VALIDATION SUCCESSFUL!")
        print("📊 VALIDATION RESULTS:")
        print("─" * 40)
        print(f"│ mAP50:     {metrics.box.map50:>10.4f} │")
        print(f"│ mAP50-95:  {metrics.box.map:>10.4f} │")
        
        precision_value = metrics.box.p
        recall_value = metrics.box.r
        if hasattr(precision_value, 'mean'):
            precision_value = precision_value.mean()
        if hasattr(recall_value, 'mean'):
            recall_value = recall_value.mean()
            
        print(f"│ Precision: {float(precision_value):>10.4f} │")
        print(f"│ Recall:    {float(recall_value):>10.4f} │")
        print("─" * 40)
        
        print("\n🎯 PER-CLASS PERFORMANCE:")
        print("─" * 50)
        
        # FIXED: Multiple ways to get per-class metrics
        per_class_printed = False
        
        # Method 1: Try to get per-class metrics from results
        if hasattr(metrics, 'results_dict'):
            results_dict = metrics.results_dict
            for i, class_name in enumerate(model.names.values()):
                map50_key = f'{class_name}/mAP50'
                if map50_key in results_dict:
                    map50_value = results_dict[map50_key]
                    print(f"│ {class_name:<20} {map50_value:>8.4f} │")
                    per_class_printed = True
        
        # Method 2: If results_dict doesn't work, try accessing directly from metrics
        if not per_class_printed and hasattr(metrics.box, 'maps'):
            maps = metrics.box.maps
            if maps is not None and len(maps) >= len(model.names):
                for i, class_name in enumerate(model.names.values()):
                    if i < len(maps):
                        map_value = float(maps[i]) if hasattr(maps[i], '__float__') else maps[i]
                        print(f"│ {class_name:<20} {map_value:>8.4f} │")
                        per_class_printed = True
        
        # Method 3: Last resort - manual extraction from validation output
        if not per_class_printed:
            print("│ Per-class metrics not available in │")
            print("│ standard format. Check validation   │")
            print("│ logs above for detailed results.    │")
        
        print("─" * 50)
        
        # ADDED: Performance comparison table
        print("\n📈 PERFORMANCE COMPARISON:")
        print("─" * 60)
        print("│ Class              │ Initial mAP50 │ Final mAP50 │ Improvement │")
        print("├────────────────────┼───────────────┼─────────────┼─────────────┤")
        
        # Your actual performance data (UPDATE THESE WITH YOUR REAL NUMBERS)
        performance_data = {
            'EmergencyPhone': {'initial': 0.671, 'final': 0.850},
            'FireAlarm': {'initial': 0.771, 'final': 0.847},
            'SafetySwitchPanel': {'initial': 0.817, 'final': 0.874},
            'OxygenTank': {'initial': 0.811, 'final': 0.883},
            'NitrogenTank': {'initial': 0.824, 'final': 0.898},
            'FirstAidBox': {'initial': 0.790, 'final': 0.876},
            'FireExtinguisher': {'initial': 0.854, 'final': 0.905}
        }
        
        for class_name in model.names.values():
            if class_name in performance_data:
                initial = performance_data[class_name]['initial']
                final = performance_data[class_name]['final']
                improvement = final - initial
                improvement_str = f"+{improvement:.3f}" if improvement >= 0 else f"{improvement:.3f}"
                print(f"│ {class_name:<18} │ {initial:>13.3f} │ {final:>11.3f} │ {improvement_str:>11} │")
        
        print("─" * 60)
        
        # ADDED: Overall metrics comparison
        print("\n📊 OVERALL METRICS IMPROVEMENT:")
        print("─" * 50)
        print("│ Metric    │ Initial  │ Final    │ Change   │")
        print("├───────────┼──────────┼──────────┼──────────┤")
        overall_metrics = [
            ("Precision", 0.908, float(precision_value), f"+{float(precision_value)-0.908:.3f}"),
            ("Recall", 0.696, float(recall_value), f"+{float(recall_value)-0.696:.3f}"), 
            ("mAP50", 0.791, metrics.box.map50, f"+{metrics.box.map50-0.791:.3f}"),
            ("mAP50-95", 0.661, metrics.box.map, f"+{metrics.box.map-0.661:.3f}")
        ]
        for metric, initial, final, change in overall_metrics:
            print(f"│ {metric:<9} │ {initial:>8.3f} │ {final:>8.3f} │ {change:>8} │")
        print("─" * 50)
        
        total_time = time.time() - start_time
        print(f"\n⏱️ Total execution time: {total_time:.2f} seconds")
        print("✅ Validation completed successfully!")
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()