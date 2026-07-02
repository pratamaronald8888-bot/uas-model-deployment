"""
features.py
===========
Pekerjaan file ini ada dua, dan keduanya sengaja dipisah karena waktunya beda.

BAGIAN PERTAMA: pembersihan yang dilakukan SEBELUM data dipecah jadi train dan
test. Pembersihan ini wajib jalan ke seluruh data, sebab kalau formatnya belum
benar (masih ada underscore di angka, masih ada teks campuran di satu sel) kita
bahkan tidak bisa melakukan scaling atau encoding. Ini ditangani class FrameCleaner.

BAGIAN KEDUA: tiga operasi yang sengaja DITUNDA sampai SETELAH split, yaitu
imputasi, scaling, dan encoding. Ketiganya menghitung statistik dari data
(rata-rata, median, daftar kategori). Kalau statistik itu dihitung dari seluruh
data termasuk data test, maka informasi data test bocor ke proses training. Ini
yang disebut data leakage, dan akibatnya model terlihat bagus saat uji tapi
payah saat menemui data sungguhan. Maka ketiganya dibungkus jadi satu
ColumnTransformer yang baru menghitung statistik saat .fit dipanggil di data train.

Untuk encoding dipakai handle_unknown='ignore'. Alasannya, di produksi bisa
muncul kategori yang belum pernah dilihat model. Dengan setelan itu kategori
asing diterjemahkan jadi nol semua, bukan bikin error, jadi pipeline tetap hidup.
"""

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.config import (
    LABEL_TO_CODE,
    LOAN_COLUMNS,
    LOAN_KINDS,
    SCHEMA,
    loan_column_name,
)

# Pola untuk membaca teks "X Years and Y Months".
_HISTORY_PATTERN = re.compile(r"(\d+)\s*years?.*?(\d+)\s*months?", re.IGNORECASE)


class FrameCleaner:
    """
    Membersihkan data mentah menjadi bentuk yang konsisten, semuanya sebelum split.

    Tiap jenis pembersihan dipecah jadi method kecil supaya alurnya enak dibaca
    dan tiap keputusan bisa ditelusuri satu per satu.
    """

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out = self._expand_loan_types(out)
        out = self._repair_numeric_text(out)
        out = self._convert_history_age(out)
        out = self._nullify_impossible_numbers(out)
        out = self._mark_missing_categories(out)
        out = out.drop(columns=SCHEMA.drop_cols, errors="ignore")
        return out

    def _expand_loan_types(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Kolom Type_of_Loan bisa berisi banyak jenis pinjaman dalam satu sel,
        contohnya "Personal Loan, Student Loan, and Mortgage Loan". Model tidak
        bisa mencerna teks campuran begitu. Solusinya, setiap jenis pinjaman
        dijadikan kolomnya sendiri yang isinya 1 kalau dimiliki dan 0 kalau tidak.

        Sel berisi "Not Specified" atau kosong otomatis menjadi nol di semua
        kolom loan, yang artinya tidak ada informasi pinjaman. Kolom aslinya
        dibuang karena sudah tidak diperlukan.
        """
        raw = frame.get("Type_of_Loan", pd.Series([""] * len(frame))).fillna("").astype(str)
        for kind in LOAN_KINDS:
            col = loan_column_name(kind)
            frame[col] = raw.str.contains(re.escape(kind), case=False).astype(int)
        return frame.drop(columns=["Type_of_Loan"], errors="ignore")

    def _repair_numeric_text(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Beberapa kolom seharusnya angka tapi terbaca teks karena ada underscore
        nyangkut, contohnya "20364.57_". Ada juga sel berisi teks sampah seperti
        deretan angka raksasa berpagar underscore. Underscore dibuang lalu sisanya
        dipaksa jadi angka. Apa pun yang gagal jadi angka otomatis menjadi NaN dan
        diurus belakangan oleh imputer.

        Pembersihan ini disapukan ke seluruh kolom numerik, bukan hanya yang
        sudah ketahuan kotor, supaya tidak ada satu pun teks yang lolos ke tahap
        scaling. Kolom Credit_History_Months belum ada di tahap ini, jadi diurus
        terpisah oleh _convert_history_age.
        """
        numeric_cols = SCHEMA.numeric_normal + SCHEMA.numeric_skewed
        for col in numeric_cols:
            if col not in frame.columns:
                continue
            cleaned = (
                frame[col].astype(str).str.replace("_", "", regex=False).str.strip()
            )
            frame[col] = pd.to_numeric(cleaned, errors="coerce")
        return frame

    def _convert_history_age(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Credit_History_Age tertulis sebagai "9 Years and 8 Months". Diubah jadi
        total bulan (9 kali 12 ditambah 8). Dipilih satuan bulan, bukan tahun,
        supaya tidak kehilangan ketelitian karena pembulatan desimal.
        """
        frame["Credit_History_Months"] = frame["Credit_History_Age"].map(self._history_to_months)
        return frame.drop(columns=["Credit_History_Age"], errors="ignore")

    @staticmethod
    def _history_to_months(value) -> float:
        """Fungsi bantu murni, tidak butuh self. Mengubah teks usia kredit jadi jumlah bulan."""
        if pd.isna(value):
            return np.nan
        found = _HISTORY_PATTERN.search(str(value))
        if not found:
            return np.nan
        years, months = int(found.group(1)), int(found.group(2))
        return years * 12 + months

    def _nullify_impossible_numbers(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Nilai negatif yang mustahil secara logika diubah jadi NaN untuk diimputasi nanti."""
        for col in SCHEMA.negative_is_invalid:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame.loc[frame[col] < 0, col] = np.nan
        return frame

    def _mark_missing_categories(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Penanda kosong pada kolom kategorikal diperlakukan berbeda-beda,
        tergantung apa makna kekosongannya setelah dibandingkan dengan target.

        Credit_Mix yang berisi underscore diubah jadi kategori 'Unknown'.
        Jumlahnya cukup banyak dan polanya khas, jadi memaksanya ke modus malah
        merusak sebaran. Lebih jujur dianggap kategori tersendiri.

        Payment_of_Min_Amount yang berisi 'NM' dibiarkan apa adanya. Sebaran 'NM'
        berbeda dari Yes maupun No, artinya kekosongan ini justru membawa
        informasi (missing not at random) sehingga punya daya prediksi.

        Payment_Behaviour yang berisi string sampah '!@9#%8' diubah jadi NaN,
        sebab string itu tidak bermakna dan sebarannya mirip kategori lain
        sehingga tidak memberi daya prediksi. Nanti diimputasi pakai modus.
        """
        frame["Credit_Mix"] = frame["Credit_Mix"].replace("_", "Unknown")
        frame["Payment_Behaviour"] = frame["Payment_Behaviour"].replace("!@9#%8", np.nan)
        return frame


def encode_target(series: pd.Series) -> pd.Series:
    """Ubah label target dari teks ke angka. Wajib numerik karena XGBoost menolak label teks."""
    return series.map(LABEL_TO_CODE).astype(int)


def build_feature_pipeline() -> ColumnTransformer:
    """
    Rangkai imputasi, scaling, dan encoding jadi satu objek. Ketiganya baru
    belajar statistik saat .fit dipanggil pada data train, jadi data test tidak
    ikut bocor.

    Empat jalur:
      jalur normal     : isi kosong pakai mean, lalu StandardScaler
      jalur skewed     : isi kosong pakai median, lalu RobustScaler
      jalur kategorikal: isi kosong pakai modus, lalu OneHotEncoder
      jalur loan       : dibiarkan apa adanya karena sudah berupa 0 dan 1
    """
    normal_branch = Pipeline([
        ("isi", SimpleImputer(strategy="mean")),
        ("skala", StandardScaler()),
    ])
    skewed_branch = Pipeline([
        ("isi", SimpleImputer(strategy="median")),
        ("skala", RobustScaler()),
    ])
    categorical_branch = Pipeline([
        ("isi", SimpleImputer(strategy="most_frequent")),
        ("kode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("normal", normal_branch, SCHEMA.numeric_normal),
            ("skewed", skewed_branch, SCHEMA.numeric_skewed),
            ("kategori", categorical_branch, SCHEMA.categorical),
            ("loan", "passthrough", LOAN_COLUMNS),
        ],
        remainder="drop",
    )
