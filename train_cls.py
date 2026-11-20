"""
Script Training YOLOv8 Klasifikasi Emosi Wajah (FER-2013)
Hardware Target: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM)
Workspace: C:/shidqi_lab/projek1
"""

import sys
import os
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics belum terpasang di virtual environment ini.")
    print("[PETUNJUK] Jalankan perintah: pip install ultralytics")
    sys.exit(1)

# ==========================================
# KONFIGURASI TRAINING KLASIFIKASI
# ==========================================
MODEL_WEIGHTS = 'yolov8n-cls.pt'
DATASET_DIR = Path(__file__).resolve().parent / 'dataset' / 'image'


def main():
    root_dir = Path(__file__).resolve().parent
    dataset_path = DATASET_DIR
    weight_path = root_dir / MODEL_WEIGHTS

    print("=" * 65)
    print("      YOLOV8 EMOTION CLASSIFICATION TRAINING (TWO-STAGE)")
    print("=" * 65)
    print(f"Task             : CLASSIFY")
    print(f"Pretrained Model : {weight_path}")
    print(f"Dataset Path     : {dataset_path}")
    print(f"Device           : GPU (device=0 / RTX 3060)")
    print(f"Epochs           : 15 (Optimal untuk Transfer Learning)")
    print(f"Image Size       : 64 (Optimal untuk resolusi FER-2013 48x48)")
    print(f"Batch Size       : 128 (Memaksimalkan throughput 6GB VRAM)")
    print(f"Workers          : 4")
    print(f"Run Name         : emotion_cls_run")
    print("=" * 65)

    # Validasi dataset lokal
    if not dataset_path.exists():
        print(f"[ERROR] Direktori dataset tidak ditemukan di: {dataset_path}")
        sys.exit(1)

    train_dir = dataset_path / 'train'
    test_dir = dataset_path / 'test'
    if not train_dir.exists() or not test_dir.exists():
        print(f"[ERROR] Subdirektori 'train' atau 'test' tidak ditemukan di {dataset_path}")
        sys.exit(1)

    classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    print(f"[INFO] Kelas emosi ditemukan ({len(classes)} kelas): {sorted(classes)}")

    # Inisialisasi model: Gunakan bobot pretrained jika ada dan valid (>100KB), atau arsitektur lokal yaml secara offline
    weight_file = root_dir / MODEL_WEIGHTS
    if weight_file.exists() and weight_file.stat().st_size < 100_000:
        print(f"[PERINGATAN] Menghapus file corrupt '{MODEL_WEIGHTS}' ({weight_file.stat().st_size} bytes)...")
        weight_file.unlink(missing_ok=True)

    if weight_file.exists():
        print(f"[INFO] Memuat bobot pretrained lokal valid '{MODEL_WEIGHTS}' ({weight_file.stat().st_size / (1024*1024):.2f} MB)...")
        model = YOLO(str(weight_file))
    else:
        print(f"[INFO] File bobot .pt tidak ada di folder. Menggunakan arsitektur lokal 'yolov8n-cls.yaml' (100% Offline Mode, Tanpa Internet)...")
        model = YOLO('yolov8n-cls.yaml')

    # Eksekusi training klasifikasi
    print("[INFO] Memulai training klasifikasi...")
    results = model.train(
        data=str(dataset_path),   # Langsung mengarah ke folder yang berisi train/ & test/
        epochs=15,               # 15 epoch cepat untuk transfer learning
        imgsz=64,                # 64px sangat efisien untuk data wajah 48x48
        batch=128,               # Batch besar memaksimalkan GPU RTX 3060
        workers=4,               # DataLoader subprocess
        device=0,                # Lock ke GPU NVIDIA RTX 3060 (device=0)
        name='emotion_cls_run',  # Output di runs/classify/emotion_cls_run
        save=True,               # Simpan checkpoint bobot terbaik & terakhir
        plots=True,              # Buat kurva akurasi, loss, & confusion matrix
        val=True,                # Evaluasi akurasi pada test set tiap epoch
        exist_ok=True            # Overwrite jika run dengan nama sama sudah ada
    )

    print("\n[INFO] Training klasifikasi selesai!")
    print(f"[INFO] Bobot terbaik tersimpan di: runs/classify/emotion_cls_run/weights/best.pt")


if __name__ == '__main__':
    main()

