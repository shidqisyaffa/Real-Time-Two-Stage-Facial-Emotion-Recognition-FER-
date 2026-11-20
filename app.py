"""
Server Web Deteksi Emosi Wajah (Flask)
Model: runs/classify/emotion_cls_run/weights/best.pt
"""

import sys
import time
from pathlib import Path
import numpy as np

try:
    from flask import Flask, render_template, request, jsonify
except ImportError:
    print("[ERROR] Flask belum terpasang. Jalankan: pip install flask")
    sys.exit(1)

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

app = Flask(__name__)

# Konfigurasi Path & Model
WEIGHTS_PATH = Path("runs/classify/emotion_cls_run/weights/best.pt")
DETECTOR_PATH = Path("yolov8n.pt")

# Mapping label emosi ke Bahasa Indonesia
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
    'angry': '#ef4444',       # Merah
    'disgusted': '#a855f7',   # Ungu
    'fearful': '#f97316',     # Oranye
    'happy': '#22c55e',       # Hijau
    'neutral': '#94a3b8',     # Abu-abu
    'sad': '#3b82f6',         # Biru
    'surprised': '#eab308'    # Kuning
}

device = "cuda:0" if torch.cuda.is_available() else "cpu"
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

if not WEIGHTS_PATH.exists():
    print(f"[ERROR] Bobot '{WEIGHTS_PATH}' belum ditemukan! Jalankan: python train_cls.py")
    sys.exit(1)

print(f"[INFO] Memuat model klasifikasi emosi ke {gpu_name} ({device})...")
model = YOLO(str(WEIGHTS_PATH))

# Inisialisasi Haar Cascade Face Detector
face_cascade = None
if hasattr(cv2, 'CascadeClassifier'):
    cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_file)
    if not face_cascade.empty():
        print(f"[INFO] Haar Cascade aktif: {cascade_file}")
    else:
        face_cascade = None

# Detektor sekunder YOLOv8 jika Haar Cascade tidak tersedia di versi OpenCV ini
person_detector = None
if face_cascade is None and DETECTOR_PATH.exists():
    print(f"[INFO] Menggunakan detektor wajah berbasis YOLO ({DETECTOR_PATH}).")
    person_detector = YOLO(str(DETECTOR_PATH))


def detect_faces(frame, gray):
    """Mendeteksi wajah dan mengembalikan list bounding box (x, y, w, h)."""
    boxes = []

    # 1. Deteksi Utama: Haar Cascade dengan parameter stabil
    if face_cascade is not None:
        try:
            detected = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(80, 80)
            )
            for (x, y, w, h) in detected:
                boxes.append((int(x), int(y), int(w), int(h)))
            return boxes
        except Exception:
            pass

    # 2. Deteksi Alternatif: YOLOv8 person detector jika Haar Cascade tidak ada
    if person_detector is not None:
        try:
            res = person_detector.predict(source=frame, classes=[0], device=device, verbose=False, conf=0.4)
            for b in res[0].boxes:
                px1, py1, px2, py2 = map(int, b.xyxy[0].tolist())
                pw = px2 - px1
                ph = py2 - py1
                # Ambil area proporsional wajah di bagian atas badan
                face_w = int(pw * 0.7)
                face_h = int(ph * 0.35)
                face_x = px1 + int((pw - face_w) / 2)
                face_y = py1
                boxes.append((face_x, face_y, face_w, face_h))
        except Exception:
            pass

    return boxes


def analyze_frame(frame):
    """Mendeteksi wajah, crop ROI proporsional, dan inferensi emosi."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detect_faces(frame, gray)

    faces_result = []
    top_emotion_raw = "neutral"
    top_conf = 0.0

    for (x, y, w, h) in faces:
        # Padding proporsional 5% dari ukuran wajah
        pad_w = int(0.05 * w)
        pad_h = int(0.05 * h)
        y1 = max(0, y - pad_h)
        y2 = min(frame.shape[0], y + h + pad_h)
        x1 = max(0, x - pad_w)
        x2 = min(frame.shape[1], x + w + pad_w)

        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0 or face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
            continue

        # Inferensi Klasifikasi Emosi
        res = model.predict(source=face_roi, device=device, verbose=False)
        probs = res[0].probs
        raw_emotion = res[0].names[probs.top1]
        conf = float(probs.top1conf) * 100

        if conf > top_conf:
            top_conf = conf
            top_emotion_raw = raw_emotion

        display_name = EMOTION_LABEL_MAP.get(raw_emotion.lower(), raw_emotion)
        color = EMOTION_COLORS.get(raw_emotion.lower(), '#22c55e')

        faces_result.append({
            'box': [int(x), int(y), int(w), int(h)],
            'emotion': display_name,
            'raw_emotion': raw_emotion,
            'confidence': round(conf, 1),
            'color': color,
            'label_pos': [int(x), int(max(20, y - 10))]
        })

    vram_str = "0.0 GB"
    if torch.cuda.is_available():
        vram_str = f"{torch.cuda.memory_allocated(0) / (1024**3):.2f} GB"

    dominant_name = EMOTION_LABEL_MAP.get(top_emotion_raw.lower(), top_emotion_raw) if faces_result else "Tidak Ada Wajah"

    return {
        'faces': faces_result,
        'dominant_emotion': dominant_name,
        'confidence': round(top_conf, 1),
        'faces_count': len(faces_result),
        'gpu_name': gpu_name,
        'vram_used': vram_str
    }


# ==========================================
# ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict_frame', methods=['POST'])
def predict_frame():
    """Endpoint inferensi frame dari webcam."""
    file = request.files.get('frame')
    if not file:
        return jsonify({'error': 'Frame tidak ditemukan'}), 400

    start_time = time.time()

    file_bytes = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Gagal decode gambar'}), 400

    results = analyze_frame(frame)

    elapsed = time.time() - start_time
    fps = round(1.0 / (elapsed + 1e-6), 1)
    results['fps'] = fps

    return jsonify(results)


if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("      SERVER DETEKSI EMOSI WAJAH AKTIF")
    print("=" * 55)
    print("  Buka browser di: http://127.0.0.1:5000")
    print("  Tekan CTRL + C untuk berhenti.")
    print("=" * 55 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
