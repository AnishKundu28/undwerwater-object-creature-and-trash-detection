"""
Underwater Trash Detection Training (GPU-Optimized)
Compatible with underwater_plastics dataset structure
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import torch
import psutil


# ====================== CONFIGURATION ====================== #
class TrainingConfig:
    MODEL_SIZE = 'yolov8s.pt'  # Use yolov8n.pt (nano) or yolov8s.pt (small)
    DATASET_PATH = r"D:\Projects\UnderWater Creature+Trash Detection\underwater-trash-detection\Dataset - underwater_plastics"

    EPOCHS = 50
    IMG_SIZE = 720
    BATCH_SIZE = 4  # <-- FIXED: 4GB VRAM safe
    WORKERS = 2  # <-- WINDOWS/VRAM safe
    DEVICE = 0

    # Training settings
    PATIENCE = 20
    SAVE_PERIOD = 5

    # Augmentation
    AUGMENT = True
    MOSAIC = 1.0
    MIXUP = 0.0
    DEGREES = 0.0
    TRANSLATE = 0.1
    SCALE = 0.5
    FLIPUD = 0.0
    FLIPLR = 0.5
    HSV_H = 0.015
    HSV_S = 0.8
    HSV_V = 0.5


# ====================== YAML HANDLER ====================== #
def create_dataset_yaml(dataset_path, output_file='trash_data.yaml'):
    dataset_path = Path(dataset_path).absolute()
    existing_yaml = dataset_path / 'data.yaml'

    if existing_yaml.exists():
        print(f"\n✅ Found existing data.yaml")
        with open(existing_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)

        yaml_data['path'] = str(dataset_path)
        yaml_data['train'] = 'train/images'
        yaml_data['val'] = 'valid/images'
        yaml_data['test'] = 'test/images'

        with open(output_file, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

        print(f"✔ Using existing class configuration")
        print(f"✔ Classes: {yaml_data.get('names', [])}")
        return str(Path(output_file).absolute())

    else:
        raise FileNotFoundError("❌ No data.yaml found in dataset path!")


# ====================== DATASET INFO ====================== #
def print_dataset_info(dataset_path):
    dataset_path = Path(dataset_path)

    train_images = len(list(dataset_path.glob('train/images/*')))
    val_images = len(list(dataset_path.glob('valid/images/*')))
    test_images = len(list(dataset_path.glob('test/images/*')))

    print("\n" + "=" * 60)
    print("                 DATASET INFORMATION")
    print("=" * 60)
    print(f"📁 Dataset path: {dataset_path}")
    print(f"\n📊 Dataset splits:")
    print(f"   Train images: {train_images}")
    print(f"   Validation images: {val_images}")
    print(f"   Test images: {test_images}")
    print(f"   Total images: {train_images + val_images + test_images}")
    print("=" * 60)


# ====================== TRAINING FUNCTION ====================== #
def train_model(config):
    print("\n" + "=" * 60)
    print("     UNDERWATER TRASH DETECTION - TRAINING START")
    print("=" * 60)

    print(f"\n💾 Dataset: {config.DATASET_PATH}")

    # Print dataset info
    print_dataset_info(config.DATASET_PATH)

    # Create data.yaml
    data_yaml = create_dataset_yaml(config.DATASET_PATH)

    print(f"\n📦 Loading model: {config.MODEL_SIZE}")
    model = YOLO(config.MODEL_SIZE)

    print("\n🚀 Starting training...\n")
    results = model.train(
        data=data_yaml,
        epochs=config.EPOCHS,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        device=config.DEVICE,
        workers=config.WORKERS,
        patience=config.PATIENCE,
        save_period=config.SAVE_PERIOD,
        augment=config.AUGMENT,
        mosaic=config.MOSAIC,
        mixup=config.MIXUP,
        degrees=config.DEGREES,
        translate=config.TRANSLATE,
        scale=config.SCALE,
        flipud=config.FLIPUD,
        fliplr=config.FLIPLR,
        hsv_h=config.HSV_H,
        hsv_s=config.HSV_S,
        hsv_v=config.HSV_V,
        pretrained=True,
        optimizer='AdamW',
        verbose=True,
        seed=42,
        deterministic=True,
        val=True,
        plots=True,
        save=True,
        project='underwater_trash_detection',
        name='training_run',
        exist_ok=True,
        amp=True,  # Automatic Mixed Precision
        cos_lr=True,
        close_mosaic=10
    )

    print("\n✅ Training complete! Validating model...\n")
    metrics = model.val()

    # Calculate accuracy metrics
    precision = metrics.box.mp
    recall = metrics.box.mr
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 60)
    print("                VALIDATION METRICS")
    print("=" * 60)
    print(f"📊 mAP50 (IoU=0.5): {metrics.box.map50:.4f} ({metrics.box.map50 * 100:.2f}%)")
    print(f"📊 mAP50-95 (IoU=0.5:0.95): {metrics.box.map:.4f} ({metrics.box.map * 100:.2f}%)")
    print(f"📊 Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"📊 Recall: {recall:.4f} ({recall * 100:.2f}%)")
    print(f"📊 F1 Score (Accuracy): {f1_score:.4f} ({f1_score * 100:.2f}%)")
    print("=" * 60)

    # Per-class metrics
    try:
        class_names = ['Mask', 'can', 'cellphone', 'electronics', 'gbottle',
                       'glove', 'metal', 'misc', 'net', 'pbag', 'pbottle',
                       'plastic', 'rod', 'sunglasses', 'tire']

        print("\n" + "-" * 60)
        print("              PER-CLASS METRICS")
        print("-" * 60)

        if hasattr(metrics.box, 'ap_class_index'):
            for i, class_idx in enumerate(metrics.box.ap_class_index):
                class_name = class_names[int(class_idx)]
                ap = metrics.box.ap[i]
                print(f"{class_name:15s}: AP50-95 = {ap:.4f} ({ap * 100:.2f}%)")
    except Exception as e:
        print(f"Per-class metrics not available: {e}")

    print("-" * 60)

    best_model = Path('underwater_trash_detection/training_run/weights/best.pt').absolute()
    print(f"\n📁 Best model saved at: {best_model}")
    print(f"📁 Last model saved at: underwater_trash_detection/training_run/weights/last.pt")
    print("=" * 60)

    print("\n✨ Training pipeline completed successfully!")
    print("\n📖 To use the trained model for inference:")
    print(">>> from ultralytics import YOLO")
    print(">>> model = YOLO('underwater_trash_detection/training_run/weights/best.pt')")
    print(">>> results = model('path/to/image.jpg')")
    print(">>> results[0].show()  # Display results")

    return model


# ====================== MAIN ====================== #
if __name__ == "__main__":
    config = TrainingConfig()

    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"\n💻 System RAM: {ram_gb:.1f} GB")

    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"🎮 GPU: {gpu_name}")
        print(f"🎮 GPU Memory: {gpu_memory:.1f} GB")
    else:
        print("⚠️ No GPU detected. Training will use CPU (slower).")

    if ram_gb < 8:
        print("⚠️ Warning: Low RAM (<8GB). Expect slow training on CPU.")

    input("\nPress ENTER to start training... ")

    train_model(config)
