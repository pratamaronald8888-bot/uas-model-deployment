"""
config.py
=========
Tempat tunggal untuk semua keputusan tentang kolom, path, dan target.

Semua angka dan daftar kolom di sini bukan tebakan. Semuanya adalah kesimpulan
yang sudah dibuktikan lewat eksplorasi di notebook analisis_dan_eksperimen.ipynb.
Pipeline tinggal memakai kesimpulan itu supaya proses retraining cepat dan tidak
perlu mengulang analisis dari nol setiap kali data baru masuk.

Alasan dipusatkan di satu file: kalau suatu saat ada kolom baru atau ada kolom
yang perlu diperlakukan beda, kita cukup ubah di sini, dan seluruh modul lain
(loader, features, estimators, serving) otomatis ikut menyesuaikan.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Folder dasar dihitung relatif terhadap posisi file ini, jadi pipeline tetap
# jalan walau dipanggil dari direktori kerja mana pun.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data_A.csv"
MODEL_DIR = PROJECT_ROOT / "model_store"
STAGING_DIR = PROJECT_ROOT / "model_store" / "staging"
MODEL_FILE = MODEL_DIR / "credit_pipeline_A.pkl"

# Nama eksperimen yang muncul di MLflow UI.
MLFLOW_EXPERIMENT = "credit_score_dataset_A"

# Gerbang kualitas. Model hanya dianggap layak deploy bila F1-macro tembus angka
# ini. Di eksperimen notebook untuk dataset A, tiga model teratas berada di
# kisaran 0.66 sampai 0.69. Maka ambang dipasang 0.65 sebagai batas minimal
# kewajaran: cukup tinggi untuk menolak model yang jelas rusak, tapi tetap
# realistis dengan performa yang memang bisa dicapai di data ini.
APPROVAL_F1 = 0.65

RANDOM_SEED = 42
TEST_FRACTION = 0.2

# Kolom target dan urutan kelasnya. Urutan sengaja dibuat ordinal dari kondisi
# kredit paling buruk ke paling baik. Wajib diubah ke angka karena XGBoost
# menolak label berupa teks.
TARGET = "Credit_Score"
CLASS_ORDER = ["Poor", "Standard", "Good"]
LABEL_TO_CODE = {name: i for i, name in enumerate(CLASS_ORDER)}
CODE_TO_LABEL = {i: name for name, i in LABEL_TO_CODE.items()}


@dataclass
class ColumnSchema:
    """
    Pengelompokan kolom hasil eksplorasi. Dibungkus dataclass supaya gampang
    dipanggil sebagai satu objek (schema) dan tidak tercecer sebagai variabel
    global yang lepas-lepas.
    """

    # Kolom yang dibuang total. Tiga alasan, sesuai analisis fitur di notebook:
    # 1. Hanya identitas, tidak punya daya prediksi: Unnamed: 0, ID, Customer_ID, Name, SSN
    # 2. Hanya penanda waktu, tidak logis menentukan skor kredit: Month
    # 3. Berpotensi membuat model bias ke demografi tertentu: Occupation, Age
    drop_cols: list = field(default_factory=lambda: [
        "Unnamed: 0", "ID", "Customer_ID", "Name", "SSN",
        "Month", "Occupation", "Age",
    ])

    # Kolom yang seharusnya angka tapi terbaca teks karena ada underscore atau
    # karakter sampah. Akan dibersihkan lalu dipaksa jadi numerik.
    dirty_numeric: list = field(default_factory=lambda: [
        "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
    ])

    # Kolom numerik yang nilai negatifnya tidak masuk akal, jadi negatif diubah
    # ke NaN. Tidak mungkin punya minus satu rekening bank atau minus pinjaman.
    negative_is_invalid: list = field(default_factory=lambda: [
        "Num_Bank_Accounts", "Num_of_Loan",
        "Num_of_Delayed_Payment", "Num_Credit_Inquiries",
    ])
    # Catatan: Delay_from_due_date dan Changed_Credit_Limit justru boleh negatif
    # (bayar sebelum jatuh tempo, atau limit kredit turun), jadi tidak diutak-atik.

    # Kolom numerik dengan distribusi mendekati normal (skew kecil, sedikit
    # outlier). Cocok di-imputasi pakai mean lalu di-scale pakai StandardScaler.
    numeric_normal: list = field(default_factory=lambda: [
        "Credit_Utilization_Ratio", "Credit_History_Months",
    ])

    # Kolom numerik yang miring atau banyak outlier. Lebih aman pakai median
    # untuk imputasi dan RobustScaler untuk scaling, karena keduanya tahan
    # terhadap nilai ekstrem.
    numeric_skewed: list = field(default_factory=lambda: [
        "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
        "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
        "Outstanding_Debt", "Total_EMI_per_month", "Amount_invested_monthly",
        "Monthly_Balance",
    ])

    # Kolom kategorikal yang akan di-OneHotEncode.
    categorical: list = field(default_factory=lambda: [
        "Credit_Mix", "Payment_of_Min_Amount", "Payment_Behaviour",
    ])

    @property
    def numeric_all(self) -> list:
        """Gabungan kolom numerik, dipakai antarmuka serving untuk membuat form."""
        return self.numeric_normal + self.numeric_skewed


# Daftar jenis pinjaman pada kolom Type_of_Loan. Tiap jenis nanti dijadikan satu
# kolom biner sendiri (multi-hot), karena satu sel bisa memuat banyak jenis loan
# sekaligus dan model tidak bisa mencerna teks campuran seperti itu.
LOAN_KINDS = [
    "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan",
    "Home Equity Loan", "Mortgage Loan", "Payday Loan",
    "Personal Loan", "Student Loan",
]


def loan_column_name(kind: str) -> str:
    """Ubah nama jenis loan menjadi nama kolom yang aman, contoh 'Auto Loan' jadi 'has_auto_loan'."""
    slug = kind.lower().replace("-", " ").replace("  ", " ").strip().replace(" ", "_")
    return f"has_{slug}"


LOAN_COLUMNS = [loan_column_name(k) for k in LOAN_KINDS]

# Satu objek schema yang dipakai bersama oleh seluruh modul.
SCHEMA = ColumnSchema()
