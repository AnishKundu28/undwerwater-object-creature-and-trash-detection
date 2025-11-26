# 🌊 Underwater Object Detection System

## Marine Life & Ocean Trash Detection using Dual YOLOv8 Models

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ADD8.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A dual-model deep learning system for real-time detection of marine creatures and underwater trash using YOLOv8. The system employs two specialized models trained separately for optimal performance in underwater environments.


---

## ✨ Features

- **Dual Model Architecture**: Separate specialized models for marine creatures and trash detection
- **Real-time Detection**: Process webcam feeds and video files with high FPS
- **High Accuracy**: 87.5% mAP@50 for creatures, optimized trash detection
- **GPU Optimized**: Efficient training on GTX 1650 (4GB VRAM) and higher
- **Flexible Input**: Supports webcam, video files, and image batches
- **Visual Distinction**: Color-coded bounding boxes (Green: Creatures, Red: Trash)
- **REST API**: Flask-based API for integration with web/mobile applications
- **Easy Deployment**: Comprehensive training and inference scripts

---

## 🏗️ Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Input Source                            │
│         (Webcam / Video / Image)                         │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
    ┌────▼────┐             ┌───▼────┐
    │ Model 1 │             │Model 2 │
    │Creatures│             │ Trash  │
    │(YOLOv8s)│             │(YOLOv8n)│
    └────┬────┘             └───┬────┘
         │                       │
         │    ┌─────────────┐   │
         └───►│   Fusion    │◄──┘
              │   Engine    │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  Annotated  │
              │   Output    │
              │ (Color-coded)│
              └─────────────┘
```

### Model Specifications

| Component | Model | Parameters | Speed | Accuracy |
|-----------|-------|------------|-------|----------|
| **Creature Detection** | YOLOv8s | 11.1M | ~5ms | 87.5% mAP@50 |
| **Trash Detection** | YOLOv8n | 3.2M | ~3ms | 70-75% mAP@50 |
| **Combined System** | Dual | 14.3M | ~8ms | Optimized |

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU acceleration)
- 8GB RAM minimum (16GB recommended)
- GPU: GTX 1650 or better (optional but recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/underwater-detection.git
cd underwater-detection
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA support (GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

**requirements.txt:**
```txt
ultralytics>=8.0.0
opencv-python>=4.7.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
pyyaml>=6.0
psutil>=5.9.0
flask>=2.3.0
flask-cors>=4.0.0
pillow>=10.0.0
requests>=2.31.0
```


## 📁 Dataset Structure

### Creature Detection Dataset

```
Dataset-4/Underwater/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

**Classes (10):**
- Holothurian (Sea Cucumber)
- Echinus (Sea Urchin)
- Scallop
- Starfish
- Fish
- Corals
- Diver
- Cuttlefish
- Turtle
- Jellyfish

### Trash Detection Dataset

```
Dataset-Underwater-Plastics/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

**Classes (Variable):**
- Plastic bags
- Bottles
- Cans
- Wrappers
- Other marine debris

### Label Format (YOLO)

Each `.txt` file contains annotations in YOLO format:

```
class_id x_center y_center width height
0 0.5 0.5 0.3 0.4
1 0.2 0.3 0.15 0.2
```

All coordinates are normalized (0-1).

---

## 🎓 Training

### Train Creature Detection Model

```bash
python model-dataset-4-nano.py
```

**Configuration** (edit in script):
```python
class TrainingConfig:
    MODEL_SIZE = 'yolov8s.pt'
    DATASET_PATH = 'path/to/Dataset-4/Underwater'
    EPOCHS = 50
    BATCH_SIZE = 8
    IMG_SIZE = 640
    DEVICE = 0  # GPU
```

**Training Time:**
- GPU (GTX 1650): ~3 hours (50 epochs)
- GPU (RTX 3060): ~1.5 hours
- CPU: Not recommended (30+ hours)

### Train Trash Detection Model

```bash
python model-3.py
```

**Configuration:**
```python
class TrainingConfig:
    MODEL_SIZE = 'yolov8n.pt'
    DATASET_PATH = 'path/to/underwater_plastics'
    EPOCHS = 100
    BATCH_SIZE = 4
    IMG_SIZE = 640
```

**Training Time:**
- GPU (GTX 1650): ~2 hours (100 epochs)
- GPU (RTX 3060): ~1 hour

### Training Tips

1. **GPU Memory Management:**
   - Reduce `BATCH_SIZE` if OOM errors occur
   - Use YOLOv8n for 4GB VRAM
   - Enable mixed precision (`amp=True`)

2. **Data Augmentation:**
   - Mosaic augmentation helps with crowded scenes
   - HSV augmentation crucial for underwater lighting
   - Adjust based on dataset characteristics

3. **Monitoring:**
   ```bash
   # View training progress
   tensorboard --logdir=underwater_detection/training_run
   ```

---

## 💻 Usage

### 1. Real-time Webcam Detection

```bash
python test.py
```

Uses dual models for real-time detection on webcam feed.

### 2. Video File Processing

Edit `test.py` and set:
```python
VIDEO_PATH = "path/to/your/video.mp4"
```

Then run:
```bash
python test.py
```

### 3. Batch Image Processing

```python
from ultralytics import YOLO

creature_model = YOLO('models/creature_model.pt')
trash_model = YOLO('models/trash_model.pt')

# Process images
results_creatures = creature_model.predict('images/', conf=0.25)
results_trash = trash_model.predict('images/', conf=0.25)
```

### 4. Python API Usage

```python
import cv2
from ultralytics import YOLO

# Load models
creature_model = YOLO('models/creature_model.pt')
trash_model = YOLO('models/trash_model.pt')

# Load image
image = cv2.imread('underwater.jpg')

# Run detection
creatures = creature_model.predict(image, conf=0.25)[0]
trash = trash_model.predict(image, conf=0.25)[0]

# Process results
for box in creatures.boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    class_name = creatures.names[int(box.cls[0])]
    confidence = float(box.conf[0])
    print(f"Creature: {class_name} ({confidence:.2f})")

for box in trash.boxes:
    class_name = trash.names[int(box.cls[0])]
    print(f"Trash: {class_name}")
```

### Configuration Options

```python
# Confidence threshold (0.0 - 1.0)
CONF_THRESHOLD = 0.25  # Default: 0.25

# IoU threshold for NMS (0.0 - 1.0)
IOU_THRESHOLD = 0.7    # Default: 0.7

# Device selection
DEVICE = 0             # 0 for GPU, 'cpu' for CPU
```

### Keyboard Controls (test.py)

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `s` | Save current frame |
| `p` | Pause/Resume |
| `+` | Increase confidence threshold |
| `-` | Decrease confidence threshold |

---

#### 2. Predict (Multipart)
```bash
curl -X POST -F "file=@underwater.jpg" \
  http://localhost:5000/predict
```

#### 3. Batch Processing
```bash
curl -X POST -F "files=@img1.jpg" -F "files=@img2.jpg" \
  http://localhost:5000/batch_predict
```

---

## 🛠️ Advanced Usage

### Fine-tuning Models

```python
from ultralytics import YOLO

# Load existing model
model = YOLO('models/creature_model.pt')

# Continue training with new data
model.train(
    data='new_dataset/data.yaml',
    epochs=20,
    resume=True  # Resume from checkpoint
)
```

### Custom Post-processing

```python
def filter_detections(results, min_confidence=0.5):
    """Filter low-confidence detections"""
    filtered = []
    for box in results.boxes:
        if float(box.conf[0]) >= min_confidence:
            filtered.append(box)
    return filtered

# Use with dual detection
creatures = filter_detections(creature_results[0], 0.6)
trash = filter_detections(trash_results[0], 0.4)
```

### Export Models

```python
# Export to ONNX (faster inference)
model.export(format='onnx')

# Export to TensorFlow Lite (mobile)
model.export(format='tflite')

# Export to TensorRT (NVIDIA GPUs)
model.export(format='engine')
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m "Add amazing feature"
   ```
4. **Push to branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Ensure all tests pass
- Add clear commit messages

---

## 🐛 Known Issues

1. **GPU Memory:**
   - GTX 1650 (4GB) may struggle with batch_size > 8
   - Solution: Reduce batch size or use YOLOv8n

2. **Jellyfish vs Plastic Bags:**
   - Models can confuse transparent objects
   - Solution: Increase training epochs or use post-processing

3. **Low Light Conditions:**
   - Performance degrades in very dark water
   - Solution: Preprocessing with CLAHE enhancement

4. **Real-time Processing:**
   - CPU inference is slow (<10 FPS)
   - Solution: Use GPU or reduce image resolution

---

## 📝 TODO

- [ ] Add tracking algorithm (DeepSORT/ByteTrack)
- [ ] Implement data augmentation pipeline
- [ ] Create Docker container for deployment
- [ ] Add model quantization for edge devices
- [ ] Develop mobile app (Flutter/React Native)
- [ ] Add support for underwater videos dataset
- [ ] Implement active learning for continuous improvement
- [ ] Create web dashboard for monitoring
- [ ] Add export to cloud storage (AWS S3/GCS)
- [ ] Multilingual support for web interface

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   Copyright 2025 [Your Name]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

### Key Points of Apache 2.0:
- ✅ **Commercial use allowed**
- ✅ **Modification allowed**
- ✅ **Distribution allowed**
- ✅ **Patent use granted**
- ✅ **Private use allowed**
- ⚠️ **Must include license and copyright notice**
- ⚠️ **Must state significant changes**
- ⚠️ **No trademark use**
- ⚠️ **No warranty provided**

---

## 🙏 Acknowledgments

- **Ultralytics** for the excellent YOLOv8 framework
- **RUOD Dataset** for underwater creature images
- **TrashNet** for marine debris dataset
- **OpenCV** community for computer vision tools
- **PyTorch** team for the deep learning framework

### Datasets Used

1. **UOD (Underwater Object Detection)**
   - https://www.kaggle.com/datasets/banuprasadb/underwater-objects

2. **Underwater Trash Dataset**
   - https://www.kaggle.com/datasets/ankitkrsrivastava/underwater-trash-detection-yolov9?select=data.yaml


**Made with ❤️ for Ocean Conservation** 🌊

*Help protect our oceans by detecting and removing marine debris!*
