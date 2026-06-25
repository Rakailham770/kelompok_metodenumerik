"""
regresi_core.py
================
Modul inti perhitungan REGRESI LINEAR SEDERHANA.
Mengikuti notasi dan metode dari materi kuliah Metode Numerik
(Sumber: Metode Numerik, Rinaldi Munir) - Tema 4: Regresi.

Notasi mengikuti slide dosen:
    f(x) = a + b.x      (a = intersep, b = slope)

Metode kuadrat terkecil (least squares), rumus langsung (sesuai slide 11):
    b = (n.SUM(xi.yi) - SUM(xi).SUM(yi)) / (n.SUM(xi^2) - (SUM(xi))^2)
    a = ybar - b.xbar
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. INTI: Regresi linear sederhana
# ---------------------------------------------------------------------------
def regresi_linear_dasar(x, y):
    """
    Menghitung koefisien a, b dari persamaan normal:
        n.a + b.SUM(x)        = SUM(y)
        a.SUM(x) + b.SUM(x^2) = SUM(x.y)

    Mengembalikan dict berisi a, b, dan seluruh komponen sigma
    untuk ditampilkan sebagai "tabel kerja" transparan di UI.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if n < 2:
        raise ValueError("Minimal 2 titik data diperlukan untuk regresi.")

    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x ** 2)
    sum_xy = np.sum(x * y)

    penyebut = n * sum_x2 - sum_x ** 2
    if abs(penyebut) < 1e-15:
        raise ValueError(
            "Penyebut nol (semua nilai x identik) - regresi tidak bisa dihitung."
        )

    b = (n * sum_xy - sum_x * sum_y) / penyebut
    a = (sum_y / n) - b * (sum_x / n)

    return {
        "a": a,
        "b": b,
        "n": n,
        "sum_x": sum_x,
        "sum_y": sum_y,
        "sum_x2": sum_x2,
        "sum_xy": sum_xy,
        "x_bar": sum_x / n,
        "y_bar": sum_y / n,
    }


def tabel_kerja(x, y):
    """
    Membangun DataFrame "tabel kerja" seperti contoh di slide:
    i | x | y | x^2 | x.y   (+ baris TOTAL di akhir)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    df = pd.DataFrame({
        "i": list(range(1, n + 1)),
        "x": list(x),
        "y": list(y),
        "x^2": list(x ** 2),
        "x.y": list(x * y),
    })

    total = {
        "i": "TOTAL",
        "x": np.sum(x),
        "y": np.sum(y),
        "x^2": np.sum(x ** 2),
        "x.y": np.sum(x * y),
    }
    df_total = pd.DataFrame([total])
    return pd.concat([df, df_total], ignore_index=True)


def metode_linear(x, y):
    """y = a + b.x"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    hasil = regresi_linear_dasar(x, y)
    a, b = hasil["a"], hasil["b"]

    df = tabel_kerja(x, y)

    def f_pred(x_baru):
        return a + b * x_baru

    persamaan = f"f(x) = {a:.6f} + {b:.6f}.x"
    return {
        "jenis": "Linear Sederhana",
        "param": {"a": a, "b": b},
        "persamaan": persamaan,
        "detail": hasil,
        "tabel": df,
        "predict": f_pred,
    }


# ---------------------------------------------------------------------------
# 2. ANALISIS GALAT (mengikuti slide 35: E_RMS dihitung di domain y asli)
# ---------------------------------------------------------------------------
def hitung_rms(y_asli, y_prediksi):
    y_asli = np.asarray(y_asli, dtype=float)
    y_prediksi = np.asarray(y_prediksi, dtype=float)
    deviasi_kuadrat = (y_asli - y_prediksi) ** 2
    total_deviasi_kuadrat = np.sum(deviasi_kuadrat)
    n = len(y_asli)
    e_rms = np.sqrt(total_deviasi_kuadrat / n)
    return {
        "deviasi_kuadrat": deviasi_kuadrat,
        "total_deviasi_kuadrat": total_deviasi_kuadrat,
        "e_rms": e_rms,
    }


# ---------------------------------------------------------------------------
# 3. DATASET CONTOH DARI MODUL (untuk verifikasi & demo cepat)
# ---------------------------------------------------------------------------
CONTOH_DATASET = {
    "Contoh slide 13/14 (hasil regresi linear: a=0.2862, b=1.7645)": {
        "x": [0.1, 0.4, 0.5, 0.7, 0.7, 0.9],
        "y": [0.61, 0.92, 0.99, 1.52, 1.47, 2.03],
    },
    "Soal Latihan slide 36": {
        "x": [0.01, 0.03, 0.09, 0.14, 0.2, 0.3, 0.4, 0.5, 0.62, 0.8],
        "y": [1, 2, 3.2, 4.5, 6, 7.1, 8.5, 9, 9.9, 10.1],
    },
}
