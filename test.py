# ================================
# Underwater Dual Model Detection
# Creatures + Trash Detection
# Video / Webcam Inference
# ================================

import os
import cv2
import time
from ultralytics import YOLO

# === CONFIGURATION ===
CREATURE_MODEL_PATH = r"D:\Projects\UnderWater Creature+Trash Detection\undwerwater-object-detection\Final-Project\best-50.pt"
TRASH_MODEL_PATH = r"D:\Projects\UnderWater Creature+Trash Detection\underwater-trash-detection\underwater_trash_detection\training_run\weights\best.pt"
VIDEO_PATH = r"D:\Projects\UnderWater Creature+Trash Detection\underwater-trash-detection\1.mp4"  # or None for webcam
CONF_THRESHOLD = 0.25
# ======================


def load_model(path, model_name):
    """Load a pre-trained YOLO model."""
    if os.path.exists(path):
        print(f"[INFO] Loading {model_name} model from: {path}")
        return YOLO(path)
    else:
        print(f"[ERROR] {model_name} model not found at: {path}")
        exit(1)


def run_dual_detection(creature_model, trash_model, conf=0.25, video_path=None):
    """Run both YOLO models on a video file or webcam feed."""
    cap = cv2.VideoCapture(0 if video_path is None else video_path)
    source_name = "webcam" if video_path is None else video_path
    print(f"[INFO] Starting dual detection from {source_name}...")

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return

    prev_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video or failed to read frame.")
                break

            # Run creature detection
            creature_results = creature_model.predict(frame, conf=conf, verbose=False)

            # Run trash detection
            trash_results = trash_model.predict(frame, conf=conf, verbose=False)

            # Draw creature detections (green boxes)
            annotated = frame.copy()
            for box in creature_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                cls_name = creature_results[0].names[cls_id]

                # Green box for creatures
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"[CREATURE] {cls_name}: {conf_val:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw trash detections (red boxes)
            for box in trash_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                cls_name = trash_results[0].names[cls_id]

                # Red box for trash
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"[TRASH] {cls_name}: {conf_val:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # FPS counter
            frame_count += 1
            if frame_count % 5 == 0:
                now = time.time()
                fps = 5 / (now - prev_time)
                prev_time = now
            else:
                fps = None

            if fps:
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Dual Detection: Creatures (Green) + Trash (Red) - Press Q to Quit", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Quit requested. Exiting...")
                break
            elif key == ord('s'):
                filename = f"frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"[INFO] Saved snapshot: {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Detection ended.")


if __name__ == "__main__":
    creature_model = load_model(CREATURE_MODEL_PATH, "Creature")
    trash_model = load_model(TRASH_MODEL_PATH, "Trash")
    run_dual_detection(creature_model, trash_model, conf=CONF_THRESHOLD, video_path=VIDEO_PATH)