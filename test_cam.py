"""
Skrip Diagnostik Kamera Laptop
Jalankan: python test_cam.py
"""
import sys
import time

print("=" * 50)
print("     DIAGNOSTIK KAMERA LAPTOP OPENCV")
print("=" * 50)

try:
    import cv2
    print(f"[OK] OpenCV Version: {cv2.__version__}")
except ImportError:
    print("[ERROR] OpenCV belum terpasang!")
    sys.exit(1)

def test_index(idx, backend_name, backend_flag):
    print(f"\n[MENGUJI] Index {idx} dengan backend {backend_name}...")
    start = time.time()
    try:
        if backend_flag is not None:
            cap = cv2.VideoCapture(idx, backend_flag)
        else:
            cap = cv2.VideoCapture(idx)
    except Exception as e:
        print(f"  -> Exception saat inisialisasi: {e}")
        return False

    elapsed = time.time() - start
    print(f"  -> Waktu inisialisasi: {elapsed:.2f} detik")

    if not cap.isOpened():
        print(f"  -> Gagal: cap.isOpened() bernilai False")
        cap.release()
        return False

    print(f"  -> cap.isOpened() True! Mencoba membaca frame...")
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"  -> [SUKSES!] Berhasil membaca frame: Resolusi {frame.shape[1]}x{frame.shape[0]}")
        cap.release()
        return True
    else:
        print(f"  -> Gagal membaca frame (ret={ret})")
        cap.release()
        return False

success = False
for idx in [0, 1]:
    for name, flag in [("CAP_MSMF", cv2.CAP_MSMF), ("CAP_DSHOW", cv2.CAP_DSHOW), ("DEFAULT", None)]:
        if test_index(idx, name, flag):
            print(f"\n>>> SOLUSI DITEMUKAN: Gunakan index {idx} dengan backend {name} <<<")
            success = True
            break
    if success:
        break

if not success:
    print("\n" + "!" * 50)
    print("[PERHATIAN] Tidak ada backend kamera yang berhasil.")
    print("Kemungkinan penyebab:")
    print("1. Izin Kamera Windows: Buka Windows Settings -> Privacy & security -> Camera -> Aktifkan 'Let desktop apps access your camera'.")
    print("2. Tombol Privacy / Fn Key di laptop (misal Fn+F10 / Fn+F6 atau tuas fisik penutup kamera).")
    print("3. Kamera sedang dibuka oleh aplikasi lain (Zoom, Teams, Discord, atau tab browser lain).")
    print("!" * 50)

