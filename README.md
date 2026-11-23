# 🏥 Analisis & Prediksi Risiko Infertilitas (Studi Kasus 08)

Sistem Kecerdasan Buatan berbasis Web interaktif untuk memprediksi tingkat risiko infertilitas (kondisi sukar hamil) pasien berdasarkan riwayat medis dan kehamilan menggunakan algoritma **Machine Learning (Logistic Regression)**.

---

## 👤 Identitas Pengembang
* **Nama:** [Ipan]
* **NIM:** 23552011090
* **Kelas:** TIF RM23A
* **Mata Kuliah:** UTS Intelegensi Buatan
* **Program Studi:** Teknik Informatika

---

## 📝 Penjelasan Detail Komponen Proyek

Proyek ini dikembangkan secara modular untuk menjamin keamanan dari kebocoran data (*data leakage*), kemudahan pemeliharaan (*maintainability*), serta skalabilitas aplikasi saat dideploy.

### 1. Eksplorasi & Penilaian Kualitas Data (`latih.py`)
Sebelum melakukan pemodelan, dilakukan audit kualitas data otomatis terhadap dataset `kasus08_infert.csv`:
* **Jumlah Sampel:** 248 baris dan 8 kolom.
* **Pembersihan Kebocoran Data (*Anti-Leakage*):** Kolom desainer eksperimen seperti `stratum` dan `pooled.stratum` **dibuang** karena tidak memiliki nilai klinis nyata bagi pasien baru dan dapat menyebabkan bias (*data leakage*).
* **Pemisahan Data (Train-Test Split):** Pembagian data menjadi **80% Data Latih (198 sampel)** dan **20% Data Uji (50 sampel)** dilakukan di awal sebelum imputasi atau penskalaan nilai guna mencegah informasi dari data uji bocor ke dalam proses pelatihan.

### 2. Rekayasa Fitur / Feature Engineering (`fitur.py`)
Untuk meningkatkan akurasi dan pemahaman kontekstual model terhadap domain klinis, fitur-fitur baru diturunkan secara otomatis dari data mentah:
* **`total_kehamilan`** = Jumlah kelahiran hidup (`parity`) + Aborsi disengaja (`induced`) + Keguguran alami (`spontaneous`).
* **`total_kehilangan`** = Akumulasi dari riwayat keguguran baik disengaja maupun alami (`induced` + `spontaneous`).
* **`rasio_kehilangan`** = Proporsi total keguguran dibanding total kehamilan pasien.
* **`rasio_spontan`** = Proporsi keguguran alami dibanding total kehamilan pasien.

### 3. Eksperimen Pemodelan & Cross Validation (`latih.py`)
Evaluasi ketat dilakukan menggunakan **Stratified 5-Fold Cross Validation** pada data latih untuk membandingkan 4 algoritma utama:
1. **Logistic Regression (Terpilih)**
   * **Kelebihan:** Performa terbaik pada dataset tabular klinis berukuran kecil. Memiliki kemampuan generalisasi yang sangat stabil.
   * **Metrik Latih:** Akurasi CV: **68.64%**, **Recall (Sensitivitas) CV: 64.95%**, **ROC-AUC: 0.7657**.
   * **Alasan Pemilihan:** Logistic Regression menghasilkan nilai **Recall tertinggi**. Dalam dunia medis, nilai Recall yang tinggi sangat penting karena meminimalkan kesalahan deteksi negatif palsu (*False Negative* / pasien berisiko tinggi yang terlewat).
2. **Random Forest** (Akurasi: 70.19%, Recall: 54.29%)
3. **Gradient Boosting** (Akurasi: 70.67%, Recall: 42.09%)
4. **K-Nearest Neighbors (KNN, K=7)** (Akurasi: 70.21%, Recall: 40.77%)

### 4. Hasil Evaluasi Akhir pada Data Uji (Unseen Data)
Evaluasi akhir tunggal menggunakan 50 sampel data uji independen membuktikan model stabil dan bebas dari overfitting:
* **Akurasi Akhir:** `66.0%`
* **Sensitivitas / Recall:** `64.71%` (Model berhasil mengenali ~65% pasien risiko tinggi secara tepat)
* **Skor ROC-AUC:** `0.7629`
* **Confusion Matrix:**
  * **True Negative (TN):** 22 (Pasien sehat diprediksi sehat dengan benar)
  * **True Positive (TP):** 11 (Pasien berisiko diprediksi berisiko dengan benar)
  * **False Positive (FP):** 11
  * **False Negative (FN):** 6

### 5. Antarmuka Web Interaktif (`app.py`)
Aplikasi web dibangun menggunakan **Streamlit** dengan pendekatan berpusat pada pengguna (*User-Centric Design*) menggunakan Bahasa Indonesia:
* **Tab 1: Form Prediksi Pasien Baru**
  * Berisi formulir interaktif: Pendidikan terakhir (dengan deskripsi penjelas), usia pasien, jumlah kelahiran hidup, aborsi medis, dan keguguran spontan.
  * Menghitung visualisasi ringkasan klinis secara otomatis (*real-time*).
  * Menampilkan keputusan akhir klasifikasi apakah pasien masuk ke dalam **Kasus Risiko Tinggi** (disertai rekomendasi medis ke spesialis Obgyn) atau **Kontrol/Normal** lengkap dengan persentase keyakinan model.
* **Tab 2: Evaluasi & Performa Model**
  * Transparansi performa AI dengan menyajikan visualisasi tabel perbandingan model (*Cross Validation*), skor akurasi pengujian, sensitivitas, serta Confusion Matrix.

### 6. Integrasi Eksternal Cloudflare Tunnel (`Mjienfdfcz8_`)
Aplikasi lokal Streamlit (Port 8501) dihubungkan ke internet global menggunakan **Cloudflare Tunnel** secara aman tanpa melakukan port-forwarding berbahaya pada jaringan local, memungkinkan aplikasi diakses dari perangkat manapun melalui URL publik unik berakhiran `.trycloudflare.com`.

---

## 🛠️ Cara Menjalankan Proyek

### Prasyarat System
Pastikan Anda telah menginstal Python versi 3.8 - 3.11 dan library berikut:
```bash
pip install pandas scikit-learn joblib streamlit
Langkah-Langkah Running
Clone atau unduh repositori ini.
Jalankan skrip pelatihan untuk melatih model dan menghasilkan berkas binary artifact:
python latih.py
Jalankan server Streamlit lokal:
streamlit run app.py
Buka peramban (browser) Anda ke alamat: http://localhost:8501
