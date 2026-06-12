# 📄 Panduan Layanan API Rekomendasi Global Bite

Dokumen ini menjelaskan struktur file di dalam folder `global_bite_api` untuk membantu integrasi dan deployment oleh tim **Cloud Computing (CC)** dan **Mobile Developer (MD)**.

---

## 📂 Daftar File & Penjelasannya

### 1. `recommender.py` (Engine / Otak AI)
*   **Fungsi:** Berisi logika perhitungan matematis sistem rekomendasi.
*   **Cara Kerja:**
    *   Membaca file model biner `global_bite_model.pkl` saat backend dinyalakan.
    *   Berisi fungsi `recommend_cross_city` yang menghitung skor gabungan: **70% dari kemiripan kata deskripsi (TF-IDF)** dan **30% dari kesamaan tag rasa (Jaccard Similarity)**, lalu digabungkan dengan **40% popularitas restoran** di kota tujuan.
    *   Berisi fungsi filter rasa (`recommend_by_tags`) dan fallback terpopuler per kota (`get_top_by_city`).

### 2. `main.py` (Web Server & Router API)
*   **Fungsi:** Mengatur jalannya server web menggunakan framework **FastAPI** dan mendefinisikan pintu gerbang URL (endpoint API) yang dapat dipanggil dari luar.
*   **Cara Kerja:**
    *   Menerima request HTTP GET dari frontend (misalnya list tag rasa atau list riwayat likes).
    *   Memanggil fungsi yang sesuai di `recommender.py`.
    *   Mengembalikan hasil rekomendasi dalam format **JSON** yang terstruktur dan rapi.

### 3. `requirements.txt` (Daftar Dependensi)
*   **Fungsi:** Mencatat semua library Python pihak ketiga beserta versinya yang wajib diinstal agar backend ini dapat berjalan.
*   **Daftar Library:**
    *   `fastapi` & `uvicorn` (untuk server web API).
    *   `scikit-learn` (untuk perhitungan cosine similarity & TF-IDF).
    *   `joblib` (untuk membaca file model `.pkl`).
    *   `pandas` & `numpy` (untuk manipulasi struktur data).

### 4. `Dockerfile` (Konfigurasi Container Cloud)
*   **Fungsi:** Berisi instruksi otomatisasi pembuatan container virtual (Docker Image) agar aplikasi ini bisa berjalan di server cloud manapun tanpa masalah perbedaan sistem operasi.
*   **Target Cloud:** Sangat direkomendasikan untuk dideploy ke **Google Cloud Run** karena Cloud Run mendukung Docker container secara bawaan dan memiliki fitur auto-scaling (murah dan hemat resource).

---

## 🚀 Cara Menjalankan Layanan (Lokal)

Sebelum menjalankan, pastikan Anda sudah menyalin file **`global_bite_model.pkl`** ke dalam folder ini.

1.  **Instalasi Dependensi:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Jalankan Server Lokal:**
    ```bash
    uvicorn main:app --reload
    ```
3.  **Akses Dokumentasi API:**
    Buka browser Anda dan akses `http://127.0.0.1:8000/docs` untuk melihat Swagger UI interaktif untuk menguji semua endpoint.

---

## 🌐 Panduan Deployment untuk Cloud Computing (GCP)

Tim CC dapat mendeploy folder ini langsung ke **Google Cloud Run** menggunakan Google Cloud SDK dengan perintah berikut di dalam direktori ini:

```bash
gcloud run deploy global-bite-api \
  --source . \
  --platform managed \
  --region asia-southeast2 \
  --allow-unauthenticated
```
*(GCP akan otomatis membaca `Dockerfile`, membuild container-nya, dan mengekspos port dinamis untuk API Anda).*
