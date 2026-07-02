"""
serving/streamlit_app.py
========================
Antarmuka web lokal untuk memprediksi credit score satu nasabah.

Tampilannya sengaja dibuat lugas: kotak angka dan pilihan dropdown. Yang dinilai
bukan keindahan tampilan melainkan kegunaannya. Pengguna nyata aplikasi ini
adalah petugas kredit yang butuh form jelas dan tahan salah ketik, bukan
halaman yang cantik. Maka kotak angka diberi batas bawah supaya tidak bisa diisi
nilai negatif yang konyol, dan kategori dipilih dari daftar supaya mustahil
salah eja.

Model dimuat dengan st.cache_resource. Tanpa itu, model akan dibaca ulang dari
disk setiap kali pengguna menekan tombol, yang membuat aplikasi berat. Dengan
cache, model cukup dimuat sekali lalu dipakai berulang.

Menjalankan dari folder project:
    python -m streamlit run serving/streamlit_app.py
"""

import sys
from pathlib import Path

# Pastikan folder project (induk dari serving/) ada di jalur impor, supaya
# `import src...` tetap ketemu walau Streamlit menjalankan skrip dari subfolder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import LOAN_KINDS, SCHEMA, loan_column_name
from serving.predictor import classify, load_pipeline

# Pilihan kategori yang dikenal model, sesuai data latihan.
CATEGORY_CHOICES = {
    "Credit_Mix": ["Good", "Standard", "Bad", "Unknown"],
    "Payment_of_Min_Amount": ["Yes", "No", "NM"],
    "Payment_Behaviour": [
        "Low_spent_Small_value_payments",
        "Low_spent_Medium_value_payments",
        "Low_spent_Large_value_payments",
        "High_spent_Small_value_payments",
        "High_spent_Medium_value_payments",
        "High_spent_Large_value_payments",
    ],
}


@st.cache_resource
def get_pipeline():
    return load_pipeline()


def collect_inputs() -> dict:
    """Kumpulkan semua isian pengguna jadi satu dict fitur yang siap diprediksi."""
    record = {}

    st.subheader("Data Angka")
    left, right = st.columns(2)
    for i, col in enumerate(SCHEMA.numeric_all):
        target = left if i % 2 == 0 else right
        label = col.replace("_", " ")
        record[col] = target.number_input(label, min_value=0.0, value=0.0, step=1.0)

    st.subheader("Data Kategori")
    for col, choices in CATEGORY_CHOICES.items():
        record[col] = st.selectbox(col.replace("_", " "), choices)

    st.subheader("Jenis Pinjaman yang Dimiliki")
    boxes = st.columns(2)
    for i, kind in enumerate(LOAN_KINDS):
        checked = boxes[i % 2].checkbox(kind)
        record[loan_column_name(kind)] = int(checked)

    return record


def main():
    st.set_page_config(page_title="Prediksi Credit Score", layout="centered")
    st.title("Prediksi Credit Score Nasabah")
    st.write(
        "Isi data nasabah di bawah ini, lalu tekan tombol untuk melihat prediksi "
        "kelas credit score (Poor, Standard, atau Good) beserta keyakinannya."
    )

    pipeline = get_pipeline()
    record = collect_inputs()

    if st.button("Jalankan Prediksi", type="primary"):
        result = classify(pipeline, record)
        st.success(f"Prediksi credit score: {result['label']}")
        st.write("Probabilitas tiap kelas:")
        st.bar_chart({k: [v] for k, v in result["proba"].items()})


if __name__ == "__main__":
    main()
