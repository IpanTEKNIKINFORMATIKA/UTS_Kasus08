
"""Aplikasi Web Streamlit untuk Prediksi Risiko Infertilitas (Kasus 8)."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import fitur  # Wajib di-import agar FunctionTransformer di pipeline terbaca saat load

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model_infertilitas_kasus08.joblib"

@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

st.set_page_config(page_title="Prediksi Risiko Infertilitas", layout="wide")
st.title("Sistem Prediksi Risiko Infertilitas (Sukar Hamil)")
st.caption("Aplikasi Deteksi Dini Berbasis Kecerdasan Buatan · UTS Kasus 08 · NIM 23552011090")

bundle = load_bundle()
if bundle is None:
    st.error("Berkas model tidak ditemukan! Silakan jalankan latih.py terlebih dahulu.")
    st.stop()

tab_prediksi, tab_evaluasi = st.tabs(["╡╠ Form Prediksi Pasien Baru", "═║ Evaluasi & Performa Model"])

# Pemetaan pendidikan agar lebih ramah dibaca
map_pendidikan = {
    "0-5yrs": "SD / Tidak Sekolah (0-5 tahun masa sekolah)",
    "6-11yrs": "SMP / SMA (6-11 tahun masa sekolah)",
    "12+ yrs": "Diploma / Sarjana / Lebih Tinggi (12+ tahun masa sekolah)"
}

with tab_prediksi:
    st.subheader("Isi Riwayat Kesehatan & Kehamilan Pasien")
    st.markdown("Silakan isi formulir di bawah ini dengan data pasien yang ingin dianalisis:")

    col1, col2 = st.columns(2)
    with col1:
        edu_selected = st.selectbox(
            "1. Tingkat Pendidikan Terakhir",
            options=bundle["opsi_education"],
            format_func=lambda x: map_pendidikan.get(x, x)
        )
        age = st.number_input("2. Usia Pasien Saat Ini (Tahun)", min_value=15, max_value=60, value=30, step=1)
        parity = st.number_input("3. Jumlah Kelahiran Hidup Sebelumnya (Jumlah Anak)", min_value=0, max_value=15, value=1, step=1)

    with col2:
        induced = st.number_input("4. Jumlah Riwayat Keguguran Sengaja (Medis/Kuret)", min_value=0, max_value=10, value=0, step=1)
        spontaneous = st.number_input("5. Jumlah Riwayat Keguguran Alami (Tanpa Sengaja)", min_value=0, max_value=10, value=0, step=1)

    # Menghitung ringkasan secara otomatis
    total_kehamilan = parity + induced + spontaneous
    total_kehilangan = induced + spontaneous

    st.info(
        f"**Ringkasan Riwayat:** Total Pernah Hamil = **{total_kehamilan} kali** | "
        f"Total Riwayat Kehilangan (Keguguran) = **{total_kehilangan} kali**"
    )

    if st.button("Mulai Analisis Risiko", type="primary"):
        input_df = pd.DataFrame([{
            "education": edu_selected,
            "age": int(age),
            "parity": int(parity),
            "induced": int(induced),
            "spontaneous": int(spontaneous)
        }])[bundle["fitur_mentah"]]

        # Prediksi menggunakan pipeline model terbaik
        proba_kasus = float(bundle["pipeline"].predict_proba(input_df)[0, 1])
        proba_kontrol = 1.0 - proba_kasus
        hasil_prediksi = 1 if proba_kasus >= bundle["threshold"] else 0

        st.markdown("---")
        st.subheader("Hasil Analisis AI:")

        c1, c2 = st.columns(2)
        with c1:
            if hasil_prediksi == 1:
                st.error("★═ **HASIL: TERDETEKSI RISIKO TINGGI INFERTILITAS (KASUS)**")
                st.write("Pasien disarankan untuk melakukan konsultasi lebih lanjut dengan dokter spesialis kandungan (Sp.OG).")
            else:
                st.success("╔═ **HASIL: RISIKO RENDAH / NORMAL (KONTROL)**")
                st.write("Pasien saat ini tergolong dalam kategori risiko rendah terkait infertilitas berdasarkan riwayat kehamilannya.")

        with c2:
            st.metric("Tingkat Keyakinan Risiko Tinggi", f"{proba_kasus:.1%}")
            st.metric("Tingkat Keyakinan Normal", f"{proba_kontrol:.1%}")

with tab_evaluasi:
    st.subheader(f"Informasi Performa AI ({bundle['nama_model']})")
    st.write("Model ini dievaluasi menggunakan data uji independen yang tidak pernah dilihat sebelumnya untuk menjamin akurasi sesungguhnya.")

    metrik = bundle["metrik_test"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Skor Akurasi Prediksi", f"{metrik['accuracy']:.1%}", help="Seberapa sering model memprediksi dengan benar keseluruhan data.")
    m2.metric("Sensitivitas (Recall)", f"{metrik['recall']:.1%}", help="Kemampuan model menemukan semua pasien yang benar-benar berisiko.")
    m3.metric("Skor ROC-AUC", f"{metrik['roc_auc']:.3f}", help="Kemampuan model membedakan pasien sehat vs berisiko tinggi.")

    st.markdown("**Tabel Perbandingan Akurasi Model (Hasil Evaluasi Lintas-Validasi):**")
    df_cv = pd.DataFrame(bundle["perbandingan_cv"])
    df_cv.columns = ["Nama Model", "Akurasi", "Presisi", "Recall", "F1-Score", "ROC-AUC"]
    st.dataframe(df_cv, use_container_width=True)
