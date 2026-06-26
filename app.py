"""
app.py
======
Aplikasi Streamlit - Proyek Regresi (Metode Numerik)
Mendukung 4 metode: Linear Sederhana, Pangkat Sederhana, Eksponensial,
Laju Tumbuh Jenuh. Input manual / upload Excel, tabel kerja detail,
grafik, analisis galat (E_RMS), dan prediksi.

Desain: konsep "kertas kerja praktikum" (graph paper) - navy sebagai
warna tinta, kuning sebagai warna stabilo penanda hasil akhir. Dipilih
supaya tidak terlihat seperti dashboard SaaS generik bertema gelap.
"""

import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from regresi_core import METODE_MAP, hitung_rms, CONTOH_DATASET

st.set_page_config(page_title="Regresi - Metode Numerik", layout="wide")

# ---------------------------------------------------------------------------
# IDENTITAS VISUAL — konsep "kertas kerja praktikum"
# ---------------------------------------------------------------------------
PAPER = "#F7F4EC"        # warna kertas
PAPER_LINE = "rgba(0,41,107,0.07)"   # garis kotak-kotak seperti graph paper
CARD = "#FFFFFF"
INK = "#1A2333"          # warna teks utama, seperti tinta
NAVY = "#00296B"
BLUE = "#01509D"
GOLD = "#FDC500"
MUTED = "#6B7280"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
    }}

    .stApp {{
        background-color: {PAPER};
        background-image:
            linear-gradient({PAPER_LINE} 1px, transparent 1px),
            linear-gradient(90deg, {PAPER_LINE} 1px, transparent 1px);
        background-size: 28px 28px;
        color: {INK};
    }}

    [data-testid="stSidebar"] {{
        background-color: {CARD};
        border-right: 1px solid {NAVY}33;
    }}

    [data-testid="stSidebar"] .eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {BLUE};
    }}

    .eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        font-weight: 500;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {BLUE};
        margin-bottom: 0.25rem;
    }}

    .page-title {{
        font-family: 'Newsreader', serif;
        font-weight: 700;
        font-size: clamp(1.7rem, 4.5vw, 2.6rem);
        color: {NAVY};
        margin: 0 0 0.3rem 0;
        line-height: 1.15;
    }}

    h2, h3 {{
        font-family: 'Newsreader', serif;
        color: {NAVY};
        font-weight: 600;
    }}

    /* angka selalu monospace - identitas alat hitung */
    [data-testid="stMetricValue"], [data-testid="stDataFrame"], code {{
        font-family: 'IBM Plex Mono', monospace !important;
    }}

    [data-testid="stMetric"] {{
        background: {CARD};
        border: 1px solid {NAVY}30;
        border-radius: 3px;
        padding: 0.7rem 0.9rem;
    }}
    [data-testid="stMetricLabel"] {{
        color: {MUTED};
        font-family: 'IBM Plex Sans', sans-serif;
    }}
    [data-testid="stMetricValue"] {{
        color: {NAVY};
    }}

    /* elemen tanda tangan: hasil akhir digaris-bawah seperti stabilo */
    .highlight {{
        background: linear-gradient(100deg,
            rgba(253,197,0,0) 0%, rgba(253,197,0,0.62) 6%,
            rgba(253,197,0,0.62) 94%, rgba(253,197,0,0) 100%);
        padding: 0.2em 0.35em;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        color: {INK};
    }}

    .stButton > button, .stDownloadButton > button {{
        background: {NAVY};
        color: {PAPER};
        border: none;
        border-radius: 3px;
        font-weight: 500;
        padding: 0.5rem 1.3rem;
        min-height: 2.6rem;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: {BLUE};
        color: #FFFFFF;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid {NAVY}30;
        border-radius: 3px;
    }}

    [data-testid="stToolbar"] {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}

    /* responsif HP: rapatkan padding & ukuran font di layar kecil */
    @media (max-width: 640px) {{
        .page-title {{ font-size: 1.6rem; }}
        [data-testid="stMetric"] {{ padding: 0.5rem 0.6rem; }}
        .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Metode Numerik &middot; Tema 4 Regresi</div>', unsafe_allow_html=True)
st.markdown('<div class="page-title">Lembar Kerja Regresi</div>', unsafe_allow_html=True)
st.caption("Dihitung dengan metode kuadrat terkecil (least squares). Sumber materi: Metode Numerik, Rinaldi Munir.")

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="eyebrow">Pengaturan</div>', unsafe_allow_html=True)

metode_terpilih = st.sidebar.selectbox(
    "Pilih metode regresi",
    list(METODE_MAP.keys()),
    help="Pilih bentuk model yang ingin dicocokkan dengan data.",
)

rumus_info = {
    "Linear Sederhana": "f(x) = a + b\u00b7x",
    "Pangkat Sederhana": "y = C\u00b7x\u1d47   (X=log\u2081\u2080x, Y=log\u2081\u2080y)",
    "Eksponensial": "y = C\u00b7e^(b\u00b7x)   (X=x, Y=ln y)",
    "Laju Tumbuh Jenuh": "y = C\u00b7x/(d+x)   (X=1/x, Y=1/y)",
}
st.sidebar.markdown(
    f"<div style='font-family:\"IBM Plex Mono\",monospace; font-size:0.95rem; "
    f"color:{NAVY}; background:{PAPER}; border:1px solid {NAVY}30; "
    f"border-radius:3px; padding:0.6rem 0.7rem;'>{rumus_info[metode_terpilih]}</div>",
    unsafe_allow_html=True,
)

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
        "Data atau metode di atas sudah berubah sejak terakhir dihitung. "
        "Hasil di bawah ini masih dari **data/metode sebelumnya** — klik **Hitung Regresi** lagi untuk memperbarui."
    )

x_data = st.session_state["x_data"]
y_data = st.session_state["y_data"]
metode_terpilih = st.session_state["metode"]

try:
    hasil = METODE_MAP[metode_terpilih](x_data, y_data)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.divider()
st.header(f"Hasil: {hasil['jenis']}")

# --- Tabel kerja detail ---
st.subheader("Tabel Kerja (Detail Perhitungan)")
st.caption("Langkah perhitungan \u03a3x, \u03a3y, \u03a3x\u00b2, \u03a3xy seperti pada contoh di materi kuliah.")
st.dataframe(
    hasil["tabel"].style.format(precision=6, na_rep=""),
    use_container_width=True,
)

d = hasil["detail"]
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("n (jumlah data)", d["n"])
    st.metric("\u03a3x", f"{d['sum_x']:.6f}")
with col2:
    st.metric("\u03a3y", f"{d['sum_y']:.6f}")
    st.metric("\u03a3x\u00b2", f"{d['sum_x2']:.6f}")
with col3:
    st.metric("\u03a3xy", f"{d['sum_xy']:.6f}")

st.subheader("Koefisien & Persamaan Akhir")
param_cols = st.columns(len(hasil["param"]))
for col, (k, v) in zip(param_cols, hasil["param"].items()):
    col.metric(k, f"{v:.6f}")

st.markdown(
    f"<p style='margin-top:0.6rem;'><span class='highlight'>{hasil['persamaan']}</span></p>",
    unsafe_allow_html=True,
)

# --- Grafik ---
st.subheader("Grafik")
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(CARD)
ax.set_facecolor(CARD)

ax.scatter(
    x_data, y_data, facecolor=GOLD, edgecolor=NAVY, linewidth=1.1,
    s=55, zorder=3, label="Data asli",
)

x_min, x_max = float(np.min(x_data)), float(np.max(x_data))
if metode_terpilih in ("Pangkat Sederhana", "Laju Tumbuh Jenuh") and x_min <= 0:
    x_min = min(x_data[x_data > 0]) if np.any(x_data > 0) else x_min
x_smooth = np.linspace(x_min, x_max, 200)
try:
    y_smooth = hasil["predict"](x_smooth)
    ax.plot(x_smooth, y_smooth, color=BLUE, linewidth=2, label="Kurva regresi", zorder=2)
except Exception:
    pass

ax.set_xlabel("x", color=INK)
ax.set_ylabel("y", color=INK)
ax.set_title(hasil["persamaan"], color=NAVY, fontsize=11, fontweight="bold")
ax.tick_params(colors=MUTED)
for spine in ax.spines.values():
    spine.set_color(f"{NAVY}40")
ax.minorticks_on()
ax.grid(which="major", alpha=0.18, color=NAVY)
ax.grid(which="minor", alpha=0.08, color=NAVY)
legend = ax.legend(facecolor=CARD, edgecolor=f"{NAVY}40")
for text in legend.get_texts():
    text.set_color(INK)
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
c1.metric("Total Deviasi Kuadrat", f"{galat['total_deviasi_kuadrat']:.6f}")
c2.metric("E_RMS", f"{galat['e_rms']:.6f}")

st.caption("E_RMS = \u221a(\u03a3(y - \u0177)\u00b2 / n), dihitung pada domain y asli, sesuai contoh analisis galat di materi kuliah.")

# --- Prediksi nilai baru ---
st.subheader("Prediksi Nilai Baru")
x_baru = st.number_input("Masukkan nilai x untuk diprediksi", value=float(x_data[0]), format="%.6f")
if st.button("Prediksi y"):
    try:
        y_hasil = hasil["predict"](np.array([x_baru]))[0]
        if np.isnan(y_hasil):
            st.error("Prediksi tidak terdefinisi untuk nilai x ini (pada model Laju Tumbuh Jenuh, ini terjadi saat x = -d).")
        else:
            st.markdown(
                f"Untuk x = {x_baru}, maka <span class='highlight'>y \u2248 {y_hasil:.6f}</span>",
                unsafe_allow_html=True,
            )
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
    file_name=f"hasil_regresi_{hasil['jenis'].replace(' ', '_').lower()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
