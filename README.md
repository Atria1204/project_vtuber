# 🎭 VTuber 2D Python (MediaPipe & OpenCV)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Motion%20Tracking-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Project VTuber 2D Sederhana namun Powerful.**
Aplikasi ini memungkinkan Anda mengendalikan avatar 2D secara *real-time* hanya dengan menggunakan webcam, tanpa peralatan VR/mocap khusus. Dibangun dengan **Python**, **MediaPipe** untuk pelacakan kerangka, dan **OpenCV** untuk pemrosesan visual.

---

## ✨ Fitur Utama

### 1. 🧍‍♂️ Full Body Tracking
Melacak pergerakan tubuh bagian atas hingga lutut secara responsif:
* **Kepala:** Mengikuti orientasi wajah (menoleh, miring kiri/kanan).
* **Torso (Badan):** Mengikuti pergerakan bahu dan pinggang.
* **Lengan:** Melacak sendi bahu, siku, dan pergelangan tangan dengan presisi.
* **Kaki:** Memisahkan bagian paha (celana atas) dan betis (celana bawah).

### 2. 👁️ Ekspresi Wajah & Mata Independen
* **Winking Support:** Mata kiri dan kanan dapat berkedip secara terpisah.
* **Mulut Dinamis:** Mendeteksi bukaan mulut pengguna:
    * *Diam* (Mulut tertutup)
    * *Senyum* (Gigi terlihat)
    * *Bicara/Tertawa* (Mulut terbuka lebar)

### 3. ✋ Hand Tracking (Gestur Tangan)
Mendeteksi kondisi jari untuk mengubah sprite tangan secara otomatis:
* **Open Palm:** Saat jari-jari terbuka lebar.
* **Relax:** Posisi tangan santai/natural.
* **Rotasi Dinamis:** Gambar tangan berputar mengikuti sudut pergelangan tangan asli Anda.

### 4. 🖥️ Dual Window Output
* **Tracking Debug:** Menampilkan feed webcam asli dengan garis skeleton (tulang) dari MediaPipe untuk memantau akurasi.
* **VTuber Final:** Hasil akhir avatar dengan background kustom (siap untuk di-capture ke OBS/Streaming Software).

### 5. 🎨 Seamless Layering
Sistem rendering cerdas yang menumpuk gambar (z-ordering) agar persendian terlihat menyatu secara alami:
> `Celana (Belakang)` -> `Lengan Atas` -> `Torso` -> `Lengan Bawah` -> `Kepala` -> `Wajah`

---

## 📂 Struktur Folder

Pastikan struktur folder proyek Anda terlihat seperti ini agar program dapat memuat aset dengan benar:

```text
MyVTuberProject/
│
├── assets/                 # Folder tempat semua gambar disimpan
│   ├── head.png
│   ├── torso.png
│   ├── lengan_atas.png
│   ├── lengan_bawah.png
│   ├── celana_atas1.png    # Paha
│   ├── celana_bawah1.png   # Betis
│   ├── mata_buka_kiri.png
│   ├── mata_buka_kanan.png
│   ├── mata_tutup_kiri.png
│   ├── mata_tutup_kanan.png
│   ├── mulut_buka.png
│   ├── gigi.png
│   ├── open_palm1.jpg      # Tangan terbuka
│   ├── relax_palm.jpg      # Tangan santai
│   └── bg.jpg              # Background (Bisa diganti Green Screen)
│
├── main.py      # File kode utama
└── README.md               # Dokumentasi ini
```

## ⚙️ Konfigurasi & Tuning

Agar avatar terlihat proporsional dan pas dengan gerakan, diperlukan untuk menyesuaikan beberapa parameter. Pengaturan ini terdapat di dalam file `main.py` pada bagian **SETTING PARAMETER**.

Silakan ubah nilai *float* pada variabel-variabel berikut sesuai kebutuhan:

| Variabel | Fungsi |
| :--- | :--- |
| `SET_SCALE_KEPALA` | Mengatur faktor skala (seberapa besar/kecil) kepala avatar terhadap tubuh. |
| `SET_OFFSET_HIDUNG` | Mengatur posisi vertikal (naik/turun) kepala agar pas menempel di leher. |
| `SET_SCALE_MATA_...` | Mengatur ukuran gambar mata (baik saat mata terbuka maupun tertutup). |
| `SET_OFFSET_MATA_X` | Mengatur jarak horizontal antar kedua mata (agar tidak terlalu dekat/jauh). |
| `SET_SCALE_TANGAN` | Mengatur ukuran gambar telapak tangan. |
| `SET_OFFSET_PINGGUL` | Mengatur posisi vertikal celana/baju bagian bawah (ditarik ke atas/bawah). |

> **Tips:** Lakukan perubahan nilai sedikit demi sedikit (misal: dari `1.0` ke `1.1`) sambil menjalankan program untuk melihat hasilnya secara langsung.

## Video Demo

* https://drive.google.com/file/d/1Wy8s-3_3fLv6iMir11m1_YV79lLsVsZl/view?usp=sharing
