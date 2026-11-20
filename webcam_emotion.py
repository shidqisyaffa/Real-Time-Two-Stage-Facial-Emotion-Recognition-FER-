"""
Deteksi Emosi Wajah (Desktop Standalone OpenCV)
Model: runs/classify/emotion_cls_run/weights/best.pt
"""

import sys
import time
from pathlib import Path

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV belum terpasang. Jalankan: pip install opencv-python")
    sys.exit(1)

try:
    import torch
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics belum terpasang. Jalankan: pip install ultralytics")
    sys.exit(1)

WEIGHTS_PATH = Path("runs/classify/emotion_cls_run/weights/best.pt")
DETECTOR_PATH = Path("yolov8n.pt")
MARGIN = 15

EMOTION_LABEL_MAP = {
    'angry': 'Marah',
    'disgusted': 'Jijik',
    'fearful': 'Takut',
    'happy': 'Senang',
    'neutral': 'Netral',
    'sad': 'Sedih',
    'surprised': 'Terkejut'
}

EMOTION_COLORS = {
    'angry': (60, 60, 220),       # Merah
    'disgusted': (150, 50, 150),  # Ungu
    'fearful': (30, 140, 230),    # Oranye
    'happy': (50, 200, 50),       # Hijau
    'neutral': (180, 180, 180),   # Abu-abu
    'sad': (200, 120, 50),        # Biru
    'surprised': (40, 215, 240)   # Kuning
}


def open_webcam():
    for idx in [0, 1, 2]:
        c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if c.isOpened():
            ret, test_frame = c.read()
            if ret and test_frame is not None and test_frame.size > 0:
                print(f"[INFO] Webcam aktif pada index {idx} (DirectShow)")
                return c
            c.release()

    for idx in [0, 1, 2]:
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            ret, test_frame = c.read()
            if ret and test_frame is not None and test_frame.size > 0:
                print(f"[INFO] Webcam aktif pada index {idx} (Default)")
                return c
            c.release()

    return None


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    if not WEIGHTS_PATH.exists():
        print(f"[ERROR] Bobot '{WEIGHTS_PATH}' belum ditemukan. Jalankan: python train_cls.py")
        sys.exit(1)

    print(f"[INFO] Memuat model ke {device}...")
    model = YOLO(str(WEIGHTS_PATH))
    person_detector = YOLO(str(DETECTOR_PATH)) if DETECTOR_PATH.exists() else None

    cap = open_webcam()
    if cap is None:
        print("[ERROR] Kamera tidak dapat dibuka. Gunakan web dashboard: python app.py")
        sys.exit(1)

    print("[INFO] Webcam aktif. Tekan tombol 'q' untuk keluar.")
    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (curr_time - prev_time + 1e-6))
        prev_time = curr_time

        h, w = frame.shape[:2]
        boxes = []

        if person_detector:
            res = person_detector.predict(source=frame, classes=[0], device=device, verbose=False, conf=0.3)
            for b in res[0].boxes:
                px1, py1, px2, py2 = map(int, b.xyxy[0].tolist())
                pw = px2 - px1
                ph = py2 - py1
                head_h = max(50, int(ph * 0.38))
                boxes.append((px1, py1, pw, head_h))

        if not boxes and w > 100:
            cw, ch = int(w * 0.5), int(h * 0.6)
            boxes.append((int((w - cw)/2), int((h - ch)/3), cw, ch))

        for (x, y, bw, bh) in boxes:
            x1 = max(0, x - MARGIN)
            y1 = max(0, y - MARGIN)
            x2 = min(frame.shape[1], x + bw + MARGIN)
            y2 = min(frame.shape[0], y + bh + MARGIN)

            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            res = model.predict(source=face_roi, device=device, verbose=False)
            probs = res[0].probs
            raw_emotion = res[0].names[probs.top1]
            conf = float(probs.top1conf) * 100

            color = EMOTION_COLORS.get(raw_emotion.lower(), (100, 200, 100))
            display_name = EMOTION_LABEL_MAP.get(raw_emotion.lower(), raw_emotion)

            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            label = f"{display_name} ({conf:.0f}%)"
            cv2.putText(frame, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        cv2.putText(frame, f"{gpu_name} | {fps:.1f} FPS", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)
        cv2.imshow("Deteksi Emosi Wajah (Tekan 'q' untuk keluar)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
