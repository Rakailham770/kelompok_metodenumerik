"""
app.py
======
Aplikasi Streamlit - Proyek Regresi Linear Sederhana (Metode Numerik)
Mendukung: input manual / upload Excel, tabel kerja detail,
grafik, analisis galat (E_RMS), dan prediksi.

CATATAN: Aplikasi ini KHUSUS regresi linear (y = a + b.x).
Struktur halaman lurus dari atas ke bawah (tidak memakai tab/kartu khusus) -
hanya tampilan warna/font yang disesuaikan dari versi default Streamlit.
"""

import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from regresi_core import metode_linear, hitung_rms, CONTOH_DATASET

st.set_page_config(page_title="Regresi Linear - Metode Numerik", layout="wide")

# ---------------------------------------------------------------------------
# IDENTITAS VISUAL (palet warna resmi IT PLN)
# ---------------------------------------------------------------------------
NAVY_DARK = "#00296B"
NAVY = "#003E87"
GOLD = "#FDC500"
BG = "#0B1220"
BG_CARD = "#121A2B"
TEXT = "#E8ECF4"
TEXT_DIM = "#8C97AC"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    .stApp {{
        background: {BG};
        color: {TEXT};
    }}
    [data-testid="stSidebar"] {{
        background: {BG_CARD};
        border-right: 1px solid {NAVY}55;
    }}
    h1, h2, h3 {{
        font-family: 'Sora', sans-serif;
        color: {TEXT};
    }}
    [data-testid="stMetric"] {{
        background: {BG_CARD};
        border: 1px solid {NAVY}66;
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_DIM};
    }}
    [data-testid="stMetricValue"] {{
        color: {GOLD};
    }}
    .stButton > button, .stDownloadButton > button {{
        background: {GOLD};
        color: {NAVY_DARK};
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.4rem;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: #FFD93D;
        color: {NAVY_DARK};
    }}
    [data-testid="stDataFrame"] {{
        border: 1px solid {NAVY}55;
        border-radius: 8px;
    }}

    /* Sembunyikan toolbar bawaan Streamlit (Deploy, menu titik tiga, dst) */
    [data-testid="stToolbar"] {{
        visibility: hidden;
    }}
    #MainMenu {{
        visibility: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("Regresi Linear Sederhana - Metode Numerik")
st.caption(
    "f(x) = a + b·x — dihitung dengan metode kuadrat terkecil (least squares). "
    "Sumber materi: Metode Numerik, Rinaldi Munir."
)

metode_terpilih = "Linear Sederhana"  # tetap (tidak ada pilihan lain)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.header("Pengaturan")
st.sidebar.info("**Bentuk model:**\n\nf(x) = a + b·x")

st.sidebar.divider()
sumber_data = st.sidebar.radio(
    "Sumber data",
    ["Input Manual", "Upload File Excel", "Pakai Contoh dari Modul"],
)

# ---------------------------------------------------------------------------
# AMBIL DATA x, y SESUAI SUMBER YANG DIPILIH
# ---------------------------------------------------------------------------
x_data, y_data = None, None

if sumber_data == "Input Manual":
    st.subheader("Input Data Manual")
    st.caption("Tambah/hapus baris langsung di tabel. Minimal 2 baris data.")

    default_df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [0.5, 1.7, 3.4, 5.7, 8.4]})
    edited_df = st.data_editor(
        default_df,
        num_rows="dynamic",
        use_container_width=True,
        key="manual_editor",
        column_config={
            "x": st.column_config.NumberColumn("x", format="%.4f", step=0.1),
            "y": st.column_config.NumberColumn("y", format="%.4f", step=0.1),
        },
    )

    edited_df = edited_df.dropna()
    if len(edited_df) >= 2:
        x_data = edited_df["x"].astype(float).values
        y_data = edited_df["y"].astype(float).values

elif sumber_data == "Upload File Excel":
    st.subheader("Upload File Excel")
    file = st.file_uploader("Pilih file .xlsx atau .xls", type=["xlsx", "xls"])

    if file is not None:
        try:
            df_excel = pd.read_excel(file)
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
            df_excel = None

        if df_excel is not None:
            st.write("Pratinjau data:")
            st.dataframe(df_excel.head(15), use_container_width=True)

            kolom = list(df_excel.columns)
            col1, col2 = st.columns(2)
            with col1:
                kolom_x = st.selectbox("Pilih kolom untuk X", kolom, index=0)
            with col2:
                default_y_idx = 1 if len(kolom) > 1 else 0
                kolom_y = st.selectbox("Pilih kolom untuk Y", kolom, index=default_y_idx)

            if kolom_x == kolom_y:
                st.warning("Kolom X dan Y tidak boleh sama. Pilih kolom yang berbeda.")
            else:
                try:
                    subset = df_excel[[kolom_x, kolom_y]].dropna()
                    x_data = pd.to_numeric(subset[kolom_x], errors="coerce").dropna().values
                    y_data = pd.to_numeric(subset[kolom_y], errors="coerce").dropna().values
                    if len(x_data) != len(y_data) or len(x_data) < 2:
                        st.warning("Data tidak valid atau kurang dari 2 baris setelah dibersihkan.")
                        x_data, y_data = None, None
                except Exception as e:
                    st.error(f"Kolom yang dipilih tidak bisa dikonversi ke angka: {e}")

else:  # Pakai Contoh dari Modul
    st.subheader("Contoh Dataset dari Modul")
    pilihan_contoh = st.selectbox("Pilih dataset contoh", list(CONTOH_DATASET.keys()))
    contoh = CONTOH_DATASET[pilihan_contoh]
    x_data = np.array(contoh["x"], dtype=float)
    y_data = np.array(contoh["y"], dtype=float)
    st.dataframe(pd.DataFrame({"x": x_data, "y": y_data}), use_container_width=True)

# ---------------------------------------------------------------------------
# PROSES PERHITUNGAN
# ---------------------------------------------------------------------------
if x_data is None or y_data is None or len(x_data) < 2:
    st.warning("Masukkan minimal 2 titik data (x, y) yang valid untuk melanjutkan.")
    st.stop()


def _data_sama(a, b):
    """Bandingkan dua array data secara aman (termasuk kalau salah satunya None)."""
    try:
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        return a_arr.shape == b_arr.shape and np.allclose(a_arr, b_arr, equal_nan=True)
    except Exception:
        return False


if st.button("Hitung Regresi", type="primary"):
    st.session_state["hitung"] = True
    st.session_state["x_data"] = x_data
    st.session_state["y_data"] = y_data
    st.session_state["metode"] = metode_terpilih

if not st.session_state.get("hitung"):
    st.info("Klik tombol **Hitung Regresi** untuk memproses data di atas.")
    st.stop()

data_berubah = (
    not _data_sama(x_data, st.session_state["x_data"])
    or not _data_sama(y_data, st.session_state["y_data"])
    or metode_terpilih != st.session_state["metode"]
)
if data_berubah:
    st.warning(
        "Data di atas sudah berubah sejak terakhir dihitung. "
        "Hasil di bawah ini masih dari **data sebelumnya** — klik **Hitung Regresi** lagi untuk memperbarui."
    )

x_data = st.session_state["x_data"]
y_data = st.session_state["y_data"]
metode_terpilih = st.session_state["metode"]

try:
    hasil = metode_linear(x_data, y_data)
except ValueError as e:
    st.error(str(e))
    st.stop()

a, b = hasil["param"]["a"], hasil["param"]["b"]

st.divider()
st.header(f"Hasil: {hasil['jenis']}")

# --- Tabel kerja detail ---
st.subheader("Tabel Kerja (Detail Perhitungan)")
st.caption(
    "Tabel ini menunjukkan langkah perhitungan Σx, Σy, Σx², Σxy "
    "seperti pada contoh di materi kuliah."
)
st.dataframe(
    hasil["tabel"].style.format(precision=6, na_rep=""),
    use_container_width=True,
)

# --- Ringkasan sigma & koefisien ---
d = hasil["detail"]
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("n (jumlah data)", d["n"])
    st.metric("Σx", f"{d['sum_x']:.6f}")
with col2:
    st.metric("Σy", f"{d['sum_y']:.6f}")
    st.metric("Σx²", f"{d['sum_x2']:.6f}")
with col3:
    st.metric("Σxy", f"{d['sum_xy']:.6f}")

st.subheader("Koefisien & Persamaan Akhir")
param_col1, param_col2 = st.columns(2)
param_col1.metric("a (intersep)", f"{a:.6f}")
param_col2.metric("b (slope)", f"{b:.6f}")

st.success(f"**Persamaan hasil regresi:**  f(x) = {a:.6f} + {b:.6f}.x")

# --- Grafik ---
st.subheader("Grafik")
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.scatter(x_data, y_data, color=GOLD, zorder=3, label="Data asli", s=45)

x_min, x_max = float(np.min(x_data)), float(np.max(x_data))
x_smooth = np.linspace(x_min, x_max, 200)
y_smooth = hasil["predict"](x_smooth)
ax.plot(x_smooth, y_smooth, color="#5AA9E6", linewidth=2.2, label="Garis regresi", zorder=2)

ax.set_xlabel("x", color=TEXT)
ax.set_ylabel("y", color=TEXT)
ax.set_title(f"f(x) = {a:.4f} + {b:.4f}.x", color=TEXT, fontsize=12)
ax.tick_params(colors=TEXT_DIM)
for spine in ax.spines.values():
    spine.set_color(f"{NAVY}88")
legend = ax.legend(facecolor=BG_CARD, edgecolor=f"{NAVY}88")
for text in legend.get_texts():
    text.set_color(TEXT)
ax.grid(alpha=0.15, color=TEXT_DIM)
st.pyplot(fig)

# --- Analisis galat ---
st.subheader("Analisis Galat (E_RMS)")
y_prediksi_di_titik_data = hasil["predict"](x_data)
galat = hitung_rms(y_data, y_prediksi_di_titik_data)

galat_df = pd.DataFrame({
    "x": x_data,
    "y (asli)": y_data,
    "y (prediksi)": y_prediksi_di_titik_data,
    "deviasi": y_data - y_prediksi_di_titik_data,
    "deviasi\u00b2": galat["deviasi_kuadrat"],
})
st.dataframe(galat_df.style.format(precision=6), use_container_width=True)

c1, c2 = st.columns(2)
c1.metric("Total Deviasi Kuadrat (Σ(y-ŷ)²)", f"{galat['total_deviasi_kuadrat']:.6f}")
c2.metric("E_RMS", f"{galat['e_rms']:.6f}")

st.caption(
    "E_RMS = √(Σ(y - ŷ)² / n), dihitung pada domain y asli, "
    "sesuai contoh analisis galat di materi kuliah."
)

# --- Prediksi nilai baru ---
st.subheader("Prediksi Nilai Baru")
x_baru = st.number_input("Masukkan nilai x untuk diprediksi", value=float(x_data[0]), format="%.6f")
if st.button("Prediksi y"):
    try:
        y_hasil = hasil["predict"](np.array([x_baru]))[0]
        st.success(f"**Prediksi:** untuk x = {x_baru}, maka y ≈ **{y_hasil:.6f}**")
    except Exception as e:
        st.error(f"Gagal menghitung prediksi: {e}")

# --- Unduh hasil ---
st.divider()
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    hasil["tabel"].to_excel(writer, sheet_name="Tabel Kerja", index=False)
    galat_df.to_excel(writer, sheet_name="Analisis Galat", index=False)
    pd.DataFrame([hasil["param"]]).to_excel(writer, sheet_name="Koefisien", index=False)
buffer.seek(0)

st.download_button(
    "Unduh Hasil Lengkap (Excel)",
    data=buffer,
    file_name="hasil_regresi_linear.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
