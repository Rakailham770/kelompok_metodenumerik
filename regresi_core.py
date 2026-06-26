"""
regresi_core.py
================
Modul inti perhitungan regresi linear & pelinearan model nirlanjar.
Mengikuti notasi dan metode dari materi kuliah Metode Numerik
(Sumber: Metode Numerik, Rinaldi Munir) - Tema 4: Regresi.

Notasi mengikuti slide dosen:
    f(x) = a + b.x      (a = intersep, b = slope)

Empat metode yang didukung:
    1. Linear sederhana   : y = a + b.x
    2. Pangkat sederhana  : y = C.x^b      (transform: X=log10(x), Y=log10(y))
    3. Eksponensial       : y = C.e^(b.x)  (transform: X=x,         Y=ln(y))
    4. Laju tumbuh jenuh  : y = C.x/(d+x)  (transform: X=1/x,       Y=1/y)

PENTING soal basis logaritma (mengikuti contoh persis di slide 30-34):
    - Model pangkat sederhana -> pakai log basis 10  (C = 10^a)
    - Model eksponensial      -> pakai ln (logaritma natural) (C = e^a)
Ini BUKAN kesalahan; dua model ini memang memakai basis berbeda di
materi aslinya, dan kode ini mengikutinya persis agar hasil cocok.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. INTI: Regresi linear sederhana pada data (X, Y) hasil transformasi
# ---------------------------------------------------------------------------
def regresi_linear_dasar(X, Y):
    """
    Menghitung koefisien a, b dari persamaan normal:
        n.a + b.SUM(X)        = SUM(Y)
        a.SUM(X) + b.SUM(X^2) = SUM(X.Y)

    Rumus langsung (sesuai slide 11):
        b = (n.SUM(Xi.Yi) - SUM(Xi).SUM(Yi)) / (n.SUM(Xi^2) - (SUM(Xi))^2)
        a = Ybar - b.Xbar
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n = len(X)

    if n < 2:
        raise ValueError("Minimal 2 titik data diperlukan untuk regresi.")

    sum_x = np.sum(X)
    sum_y = np.sum(Y)
    sum_x2 = np.sum(X ** 2)
    sum_xy = np.sum(X * Y)

    penyebut = n * sum_x2 - sum_x ** 2
    if abs(penyebut) < 1e-15:
        raise ValueError(
            "Penyebut nol (semua nilai X identik) - regresi tidak bisa dihitung."
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


def tabel_kerja(x_asli, y_asli, X, Y, label_X="X", label_Y="Y"):
    """
    Membangun DataFrame "tabel kerja" seperti contoh di slide:
    i | x | y | X^2 | X.Y              (kalau tidak ada transformasi)
    i | x | y | X | Y | X^2 | X.Y       (kalau ada transformasi)

    Untuk regresi linear murni, X==x dan Y==y (tidak ada transformasi),
    sehingga kolom X dan Y tidak digandakan - ditentukan secara EKSPLISIT
    di sini. Kolom kuadrat & hasil kali selalu diberi nama generik "X^2"
    dan "X.Y" supaya tidak ter-mangle saat label_X berupa teks panjang
    (misal "X=log x").
    """
    X_arr = np.asarray(X, dtype=float)
    Y_arr = np.asarray(Y, dtype=float)
    tanpa_transformasi = np.allclose(X_arr, np.asarray(x_asli, dtype=float)) and \
        np.allclose(Y_arr, np.asarray(y_asli, dtype=float))

    n = len(x_asli)
    data = {
        "i": list(range(1, n + 1)),
        "x": list(x_asli),
        "y": list(y_asli),
    }
    if not tanpa_transformasi:
        data[label_X] = list(X_arr)
        data[label_Y] = list(Y_arr)
    data["X^2"] = list(X_arr ** 2)
    data["X.Y"] = list(X_arr * Y_arr)

    df = pd.DataFrame(data)

    total = {
        "i": "TOTAL",
        "x": np.sum(x_asli),
        "y": np.sum(y_asli),
    }
    if not tanpa_transformasi:
        total[label_X] = np.sum(X_arr)
        total[label_Y] = np.sum(Y_arr)
    total["X^2"] = np.sum(X_arr ** 2)
    total["X.Y"] = np.sum(X_arr * Y_arr)

    df_total = pd.DataFrame([total])
    return pd.concat([df, df_total], ignore_index=True)


# ---------------------------------------------------------------------------
# 2. EMPAT METODE
# ---------------------------------------------------------------------------
def metode_linear(x, y):
    """y = a + b.x  (tanpa transformasi)"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    hasil = regresi_linear_dasar(x, y)
    a, b = hasil["a"], hasil["b"]

    df = tabel_kerja(x, y, x, y, label_X="x", label_Y="y")

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


def metode_pangkat(x, y):
    """
    y = C.x^b
    Transform: X = log10(x), Y = log10(y)
    a = log10(C)  ->  C = 10^a
    (mengikuti basis log10 sesuai Contoh 1 di slide 30-34)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError(
            "Model pangkat sederhana butuh semua nilai x dan y > 0 "
            "(karena memakai logaritma)."
        )

    X = np.log10(x)
    Y = np.log10(y)

    hasil = regresi_linear_dasar(X, Y)
    a, b = hasil["a"], hasil["b"]
    C = 10 ** a

    df = tabel_kerja(x, y, X, Y, label_X="X=log x", label_Y="Y=log y")

    def f_pred(x_baru):
        return C * np.power(x_baru, b)

    persamaan = f"y = {C:.6f}.x^{b:.6f}"
    return {
        "jenis": "Pangkat Sederhana",
        "param": {"C": C, "b": b, "a (=log C)": a},
        "persamaan": persamaan,
        "detail": hasil,
        "tabel": df,
        "predict": f_pred,
    }


def metode_eksponensial(x, y):
    """
    y = C.e^(b.x)
    Transform: X = x, Y = ln(y)
    a = ln(C)  ->  C = e^a
    (mengikuti basis ln sesuai Contoh 1 di slide 30-34)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.any(y <= 0):
        raise ValueError(
            "Model eksponensial butuh semua nilai y > 0 (karena memakai ln)."
        )

    X = x.copy()
    Y = np.log(y)

    hasil = regresi_linear_dasar(X, Y)
    a, b = hasil["a"], hasil["b"]
    C = np.exp(a)

    df = tabel_kerja(x, y, X, Y, label_X="X=x", label_Y="Y=ln y")

    def f_pred(x_baru):
        return C * np.exp(b * x_baru)

    persamaan = f"y = {C:.6f}.e^({b:.6f}.x)"
    return {
        "jenis": "Eksponensial",
        "param": {"C": C, "b": b, "a (=ln C)": a},
        "persamaan": persamaan,
        "detail": hasil,
        "tabel": df,
        "predict": f_pred,
    }


def metode_jenuh(x, y):
    """
    y = C.x / (d + x)   (saturation growth-rate)
    Transform: X = 1/x, Y = 1/y
    a = 1/C  ->  C = 1/a
    b = d/C  ->  d = b.C
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.any(x == 0) or np.any(y == 0):
        raise ValueError(
            "Model laju tumbuh jenuh butuh semua nilai x dan y != 0 "
            "(karena memakai 1/x dan 1/y)."
        )

    X = 1.0 / x
    Y = 1.0 / y

    hasil = regresi_linear_dasar(X, Y)
    a, b = hasil["a"], hasil["b"]

    if abs(a) < 1e-15:
        raise ValueError("Nilai a mendekati 0, C = 1/a tidak terdefinisi untuk data ini.")

    C = 1.0 / a
    d = b * C

    df = tabel_kerja(x, y, X, Y, label_X="X=1/x", label_Y="Y=1/y")

    def f_pred(x_baru):
        x_baru = np.asarray(x_baru, dtype=float)
        penyebut = d + x_baru
        with np.errstate(divide="ignore", invalid="ignore"):
            hasil_pred = np.where(
                np.abs(penyebut) < 1e-12,
                np.nan,
                C * x_baru / penyebut,
            )
        return hasil_pred

    persamaan = f"y = {C:.6f}.x / ({d:.6f} + x)"
    return {
        "jenis": "Laju Tumbuh Jenuh",
        "param": {"C": C, "d": d, "a (=1/C)": a, "b (=d/C)": b},
        "persamaan": persamaan,
        "detail": hasil,
        "tabel": df,
        "predict": f_pred,
    }


METODE_MAP = {
    "Linear Sederhana": metode_linear,
    "Pangkat Sederhana": metode_pangkat,
    "Eksponensial": metode_eksponensial,
    "Laju Tumbuh Jenuh": metode_jenuh,
}


# ---------------------------------------------------------------------------
# 3. ANALISIS GALAT (mengikuti slide 35: E_RMS dihitung di domain y ASLI)
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
# 4. DATASET CONTOH DARI MODUL (untuk verifikasi & demo cepat)
# ---------------------------------------------------------------------------
CONTOH_DATASET = {
    "Linear - Contoh slide 13/14 (hasil: a=0.2862, b=1.7645)": {
        "x": [0.1, 0.4, 0.5, 0.7, 0.7, 0.9],
        "y": [0.61, 0.92, 0.99, 1.52, 1.47, 2.03],
    },
    "Pangkat/Eksponensial - Contoh 1 slide 30-34 (pangkat: C=0.5,b=1.752 | eksp: C=0.343,b=0.6852)": {
        "x": [1, 2, 3, 4, 5],
        "y": [0.5, 1.7, 3.4, 5.7, 8.4],
    },
    "Soal Latihan slide 36": {
        "x": [0.01, 0.03, 0.09, 0.14, 0.2, 0.3, 0.4, 0.5, 0.62, 0.8],
        "y": [1, 2, 3.2, 4.5, 6, 7.1, 8.5, 9, 9.9, 10.1],
    },
}
