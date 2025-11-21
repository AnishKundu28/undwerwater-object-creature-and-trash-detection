"""
Underwater Trash Detection Training (GPU-Optimized for GTX 1650 4GB)
"""

from ultralytics import YOLO
import yaml
from pathlib import Path
import torch
import psutil
import os

# Enable expandable CUDA memory segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ====================== CONFIGURATION ====================== #
class TrainingConfig:
    MODEL_SIZE = 'yolov8n.pt'  # Nano model for 4GB GPU
    DATASET_PATH = r"D:\Projects\UnderWater Creature+Trash Detection\underwater-trash-detection\Dataset - underwater_plastics"



    # Training settings
    PATIENCE = 15
    SAVE_PERIOD = 10

    # Light augmentation (heavy aug causes OOM)
    AUGMENT = True
    MOSAIC = 0.5
    MIXUP = 0.0
    DEGREES = 0.0
    TRANSLATE = 0.1
    SCALE = 0.3
    FLIPUD = 0.0
    FLIPLR = 0.5
    HSV_H = 0.015
    HSV_S = 0.6
    HSV_V = 0.4


# ====================== YAML HANDLER ====================== #
def create_dataset_yaml(dataset_path, output_file='trash_data.yaml'):
    dataset_path = Path(dataset_path).absolute()
    existing_yaml = dataset_path / 'data.yaml'

    if existing_yaml.exists():
        print("\n✅ Found existing data.yaml")
        with open(existing_yaml, 'r') as f:
            yaml_data = yaml.safe_load(f)

        yaml_data['path'] = str(dataset_path)
        yaml_data['train'] = 'train/images'
        yaml_data['val'] = 'valid/images'
        yaml_data['test'] = 'test/images'

        with open(output_file, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

        print("✔ Using existing class configuration")
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

    print_dataset_info(config.DATASET_PATH)

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
        mosaic=config.MOSAIC,
        mixup=config.MIXUP,
        translate=config.TRANSLATE,
        scale=config.SCALE,
        fliplr=config.FLIPLR,
        hsv_h=config.HSV_H,
        hsv_s=config.HSV_S,
        hsv_v=config.HSV_V,
        pretrained=True,
        optimizer='AdamW',
        cos_lr=True,
        val=False,  # <-- Do not validate every epoch to reduce VRAM usage
        close_mosaic=5,
        amp=True,
        project='underwater_trash_detection',
        name='training_run',
        exist_ok=True
    )

    print("\n✅ Training complete! Running final validation...\n")
    metrics = model.val()

    print("\n📊 Final Evaluation Complete")


# ====================== MAIN ====================== #
if __name__ == "__main__":
    config = TrainingConfig()

    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"\n💻 System RAM: {ram_gb:.1f} GB")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"🎮 GPU: {gpu_name}")
        print(f"🎮 GPU Memory: {gpu_memory:.1f} GB")
    else:
        print("⚠️ No GPU detected. Training will use CPU")

    input("\nPress ENTER to start training... ")
    train_model(config)
