"""Modul feature engineering untuk Kasus 8: Studi Infertilitas (NIM 23552011090)."""
import numpy as np
import pandas as pd

KOLOM_DIBUANG = ["stratum", "pooled.stratum"]
FITUR_MENTAH = ["education", "age", "parity", "induced", "spontaneous"]
TARGET_KOLOM = "case"

URUTAN_PENDIDIKAN = ["0-5yrs", "6-11yrs", "12+ yrs"]

FITUR_NUMERIK = [
    "age",
    "parity",
    "induced",
    "spontaneous",
    "total_kehamilan",
    "total_kehilangan",
    "rasio_kehilangan",
    "rasio_spontan",
]
FITUR_ORDINAL = ["education"]


def bersihkan_teks(data: pd.DataFrame) -> pd.DataFrame:
    """Membersihkan spasi berlebih pada kolom kategorik."""
    out = data.copy()
    if "education" in out.columns:
        out["education"] = out["education"].astype(str).str.strip()
    return out


def buat_fitur_turunan(data: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan fitur turunan dari riwayat kehamilan (stateless, bebas data leakage)."""
    out = bersihkan_teks(data)
    out["total_kehamilan"] = out["parity"] + out["induced"] + out["spontaneous"]
    out["total_kehilangan"] = out["induced"] + out["spontaneous"]

    denom = out["total_kehamilan"].replace(0, np.nan)
    out["rasio_kehilangan"] = (out["total_kehilangan"] / denom).fillna(0.0)
    out["rasio_spontan"] = (out["spontaneous"] / denom).fillna(0.0)
    return out
