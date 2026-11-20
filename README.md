# Real-Time Two-Stage Facial Emotion Recognition (FER)

Sistem pengenalan ekspresi wajah waktu nyata (*real-time facial emotion recognition*) menggunakan arsitektur **Two-Stage Vision** yang memadukan deteksi wajah dan klasifikasi emosi berbasis **YOLOv8 Nano Classification**, dilengkapi antarmuka web dashboard interaktif menggunakan **Flask** dan akselerasi GPU **NVIDIA GeForce RTX 3060 Laptop GPU**.

---

## Daftar Isi
1. [Latar Belakang & Arsitektur Sistem](#1-latar-belakang--arsitektur-sistem)
2. [Alasan Pemilihan YOLOv8 Nano Classification](#2-alasan-pemilihan-yolov8-nano-classification)
3. [Eksplorasi Dataset (FER-2013)](#3-eksplorasi-dataset-fer-2013)
4. [Konfigurasi & Proses Training](#4-konfigurasi--proses-training)
5. [Hasil Evaluasi & Metrik Training](#5-hasil-evaluasi--metrik-training)
6. [Struktur Direktori Proyek](#6-struktur-direktori-proyek)
7. [Panduan Instalasi & Eksekusi](#7-panduan-instalasi--eksekusi)

---

## 1. Latar Belakang & Arsitektur Sistem

Dataset emosi wajah standar seperti FER-2013 hanya berisi potongan gambar wajah (*cropped face*), bukan gambar satu ruangan dengan kotak koordinat deteksi. Jika kita langsung menerapkan model Object Detection satu tahap (*Single-Stage Detection*) pada feed webcam:
- Model akan kesulitan karena gambar webcam berisi badan, latar belakang ruangan, dan objek lain di sekitar wajah.
- Model deteksi membutuhkan anotasi bounding box (`.txt`) untuk setiap gambar saat training, yang tidak tersedia secara alami pada dataset klasifikasi ekspresi.

Oleh karena itu, proyek ini menerapkan pendekatan **Two-Stage Vision Pipeline**:

```
Webcam Feed (1280x720 / 640x480)
               │
               ▼
   [ Stage 1: Face Detection ]
  (Haar Cascade / YOLOv8 Detection)
  Mendeteksi lokasi wajah & membuat koordinat bounding box (x, y, w, h)
               │
               ▼
     [ Cropping ROI Wajah ]
  Memotong area wajah dengan padding proporsional (5%)
               │
               ▼
 [ Stage 2: Emotion Classification ]
  (YOLOv8n-cls pada GPU RTX 3060)
  Memprediksi probabilitas 7 kelas emosi
               │
               ▼
[ Output: Bounding Box & Label Emosi Real-Time ]
```

---

## 2. Alasan Pemilihan YOLOv8 Nano Classification

Pada proyek ini digunakan model **YOLOv8n-cls** (*YOLOv8 Nano Classification*) dari Ultralytics dengan pertimbangan:

1. **Efisiensi Komputasi Super Ringan (*Ultra-Lightweight*)**:
   - Bobot model hasil training hanya berukuran **2.97 MB**.
   - Memiliki sekitar **2.719.288 parameter** dan kebutuhan komputasi hanya **3.4 GFLOPs**.
   - Latensi inferensi per wajah di GPU RTX 3060 hanya berkisar **2–4 milidetik**, memungkinkan streaming video berjalan mulus di **30+ FPS**.

2. **Konsumsi Memori Sangat Rendah**:
   - Alokasi VRAM GPU saat inferensi aktif hanya memakan **~0.2 – 0.4 GB** dari total 6 GB VRAM RTX 3060.
   - Sisa resource GPU dan RAM laptop tetap longgar untuk menjalankan browser dan server Flask secara bersamaan.

3. **Kesesuaian dengan Karakteristik Data**:
   - Arsitektur Convolutional Backbone (C2f module) pada YOLOv8 memiliki receptive field bertingkat yang sangat handal menangkap fitur mikro wajah (garis senyum di bibir, kerutan alis, mata membelalak) pada resolusi rendah.

---

## 3. Eksplorasi Dataset (FER-2013)

Dataset yang digunakan adalah **FER-2013 (Facial Expression Recognition 2013)**:

### Karakteristik Citra:
- **Total Gambar**: 35.887 citra wajah.
- **Resolusi Asli**: 48 × 48 piksel.
- **Format Warna**: Grayscale (1 kanal) / RGB.
- **Pembagian Data**:
  - `train`: **28.709 citra** (~80%)
  - `test`: **7.178 citra** (~20%)

### Distribusi 7 Kelas Emosi:
| ID | Nama Kelas | Terjemahan | Karakteristik Visual Utama |
|:--:|:-----------|:-----------|:---------------------------|
| 0 | `angry` | Marah | Alis menyempit ke tengah, bibir mengatup rapat / menegang |
| 1 | `disgusted` | Jijik | Hidung berkerut, bibir atas terangkat |
| 2 | `fearful` | Takut | Mata terbuka lebar, alis terangkat, mulut sedikit terbuka |
| 3 | `happy` | Senang | Sudut bibir terangkat ke atas, pipi terangkat (senyuman) |
| 4 | `neutral` | Netral | Otot wajah dalam keadaan relaks tanpa tegangan |
| 5 | `sad` | Sedih | Sudut bibir turun ke bawah, kelopak mata layu |
| 6 | `surprised` | Terkejut | Mulut membentuk huruf O, mata membelalak, dahi berkerut |

Struktur penyimpanan data pada proyek ini menggunakan format standar *ImageFolder*:
```
dataset/
└── image/
    ├── train/
    │   ├── angry/ (3.995 gambar)
    │   ├── disgusted/ (436 gambar)
    │   ├── fearful/ (4.097 gambar)
    │   ├── happy/ (7.215 gambar)
    │   ├── neutral/ (4.965 gambar)
    │   ├── sad/ (4.830 gambar)
    │   └── surprised/ (3.171 gambar)
    └── test/
        ├── angry/ (958 gambar)
        ├── disgusted/ (111 gambar)
        ├── fearful/ (1.024 gambar)
        ├── happy/ (1.774 gambar)
        ├── neutral/ (1.233 gambar)
        ├── sad/ (1.247 gambar)
        └── surprised/ (831 gambar)
```

---

## 4. Konfigurasi & Proses Training

Pelatihan dijalankan secara lokal menggunakan skrip [`train_cls.py`](train_cls.py) dengan spesifikasi lingkungan:
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM, CUDA 12.1).
- **Framework**: PyTorch 2.5.1+cu121 & Ultralytics 8.4.

### Hyperparameter Training:
```python
model.train(
    data='dataset/image',     # Root dataset klasifikasi
    epochs=15,                # 15 epoch
    imgsz=64,                 # Disesuaikan mendekati resolusi asli 48x48
    batch=128,                # Batch besar memaksimalkan throughput CUDA cores
    workers=4,                # Multi-process DataLoader
    device=0,                 # Lock ke GPU NVIDIA RTX 3060
    name='emotion_cls_run',   # Output folder
    save=True,                # Simpan bobot best.pt dan last.pt
    plots=True                # Generate kurva metrik & confusion matrix
)
```

**Alasan Pemilihan Parameter**:
- `imgsz=64`: Karena data aslinya 48×48, menggunakan resolusi 64px menghemat waktu komputasi hingga 4x lebih cepat dibanding resolusi standar YOLO (224px / 640px) tanpa kehilangan detail fitur ekspresi.
- `batch=128`: Memanfaatkan kapasitas 6GB VRAM secara optimal untuk stabilitas gradien batch normalization.

---

## 5. Hasil Evaluasi & Metrik Training

Training selesai dalam waktu **175 detik (~2.9 menit)** untuk seluruh 15 epoch.

### Perkembangan Metrik per Epoch:
| Epoch | Waktu (dtk) | Train Loss | Val Loss | Top-1 Accuracy | Top-5 Accuracy |
|:-----:|:-----------:|:----------:|:--------:|:--------------:|:--------------:|
| 1 | 34.8s | 1.859 | 1.783 | 25.1% | 89.6% |
| 3 | 50.0s | 1.738 | 1.607 | 36.3% | 93.6% |
| 6 | 89.9s | 1.510 | 1.343 | 48.5% | 96.5% |
| 9 | 118.1s | 1.383 | 1.238 | 52.8% | 97.4% |
| 12 | 146.7s | 1.315 | 1.195 | 54.1% | 97.5% |
| **15** | **175.4s** | **1.248** | **1.151** | **56.0%** | **97.8%** |

### Analisis Hasil:
1. **Top-1 Accuracy (56.0%)**:
   Pada dataset FER-2013, tebakan acak (*random guess*) adalah $1/7 \approx 14.3\%$, dan rata-rata batas kemampuan konsensus manusia (*human baseline accuracy*) pada dataset ini berada di kisaran ~65% karena banyak citra yang ambigu (misalnya ekspresi *fearful* yang mirip dengan *surprised*). Hasil akurasi 56.0% dalam 15 epoch menunjukkan model telah mempelajari pola fitur emosi dengan sangat baik.
2. **Top-5 Accuracy (97.8%)**:
   Menunjukkan bahwa pada 97.8% data uji, emosi yang benar hampir selalu masuk ke dalam 5 kandidat prediksi teratas model.
3. **Grafik & Visualisasi**:
   Seluruh grafik training, confusion matrix, dan sampel visualisasi prediksi tersimpan di folder:
   - `runs/classify/emotion_cls_run/results.png`
   - `runs/classify/emotion_cls_run/confusion_matrix_normalized.png`

---

## 6. Struktur Direktori Proyek

```
projek1/
│
├── dataset/                               <- Direktori dataset FER-2013
│   └── image/
│       ├── train/                         <- 28.709 citra train (7 kelas)
│       └── test/                          <- 7.178 citra uji (7 kelas)
│
├── runs/
│   └── classify/
│       └── emotion_cls_run/
│           ├── weights/
│           │   ├── best.pt                <- Bobot model terbaik (2.97 MB)
│           │   └── last.pt                <- Bobot epoch terakhir
│           ├── confusion_matrix.png       <- Matriks evaluasi prediksi
│           ├── results.png                <- Grafik kurva loss & akurasi
│           └── results.csv                <- Log metrik kuantitatif per epoch
│
├── templates/
│   └── index.html                         <- Antarmuka Web Dashboard (Tailwind CSS)
│
├── app.py                                 <- Server Web Flask & API Deteksi GPU
├── train_cls.py                           <- Skrip Training YOLOv8 Klasifikasi
├── webcam_emotion.py                      <- Skrip Uji Inferensi Standalone (OpenCV)
├── test_cam.py                            <- Utilitas Diagnostik Hardware Kamera
├── yolov8n.pt                             <- Bobot dasar YOLOv8 Nano
└── README.md                              <- Dokumentasi lengkap proyek
```

---

## 7. Panduan Instalasi & Eksekusi

### 1. Prasyarat Sistem
- Python 3.10 atau 3.11 (disarankan dalam virtual environment).
- GPU NVIDIA dengan driver CUDA yang sesuai (misal RTX 3060).

### 2. Instalasi Dependensi
Pastikan virtual environment telah aktif, lalu instal paket yang dibutuhkan:
```powershell
# Aktivasi venv
projek1\Scripts\activate

# Install PyTorch dengan dukungan CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Ultralytics, OpenCV, dan Flask
pip install ultralytics opencv-python flask
```

---

### 3. Menjalankan Ulang Training (Opsional)
Jika ingin melatih ulang model:
```powershell
python train_cls.py
```
Model terbaik otomatis disimpan ke `runs/classify/emotion_cls_run/weights/best.pt`.

---

### 4. Menjalankan Web Dashboard Streaming (Rekomendasi)
Jalankan server Flask:
```powershell
python app.py
```
Buka browser favorit Anda (Chrome / Edge) di alamat:
👉 **`http://127.0.0.1:5000`**

- Browser akan meminta izin kamera laptop. Klik **"Allow" / "Izinkan"**.
- Feed webcam langsung tampil dengan deteksi bounding box wajah, persentase keyakinan emosi, FPS, serta pemantauan GPU RTX 3060.
- Tekan **`Ctrl + C`** pada terminal untuk menghentikan server.

---

### 5. Menjalankan Inferensi Standalone Desktop (OpenCV)
Jika ingin menguji langsung lewat jendela aplikasi desktop tanpa browser:
```powershell
python webcam_emotion.py
```
- Jendela OpenCV akan terbuka menampilkan bounding box wajah berwarna dinamis sesuai emosi.
- Tekan tombol **`q`** pada keyboard untuk menutup jendela dan merilis kamera.

