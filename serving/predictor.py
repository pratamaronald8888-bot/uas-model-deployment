"""
serving/predictor.py
====================
Kode inferencing untuk pemakaian lokal.

Yang perlu dipahami soal kontrak masukan: seluruh urusan imputasi, scaling, dan
encoding sudah terbungkus rapi di dalam file pkl (preprocessor digabung model
jadi satu pipeline). Jadi fungsi di sini tidak perlu lagi membersihkan apa pun.
Ia cukup menerima satu baris data yang kolom-kolomnya sudah dalam bentuk hasil
cleaning, yaitu kolom numerik, kolom kategorikal, dan kolom loan biner. Tugas
mengumpulkan masukan dari pengguna ke dalam bentuk itu ada di streamlit_app.py.

File ini sengaja tidak tahu apa-apa soal Streamlit. Tujuannya supaya logika
prediksi bisa dites sendiri dan dipakai ulang di tempat lain (misalnya skrip
batch) tanpa harus menyalakan antarmuka web.
"""

import joblib
import pandas as pd

from src.config import CODE_TO_LABEL, MODEL_FILE


def load_pipeline(path=MODEL_FILE):
    """Muat pipeline pkl. joblib mengembalikan objek pipeline sklearn yang utuh."""
    return joblib.load(path)


def classify(pipeline, record: dict) -> dict:
    """
    Prediksi satu nasabah. Masukan berupa dict satu baris fitur.
    Keluaran berisi label teks, kode kelas, dan probabilitas tiap kelas.
    """
    frame = pd.DataFrame([record])
    code = int(pipeline.predict(frame)[0])
    proba = pipeline.predict_proba(frame)[0]
    return {
        "label": CODE_TO_LABEL[code],
        "code": code,
        "proba": {CODE_TO_LABEL[i]: float(p) for i, p in enumerate(proba)},
    }


if __name__ == "__main__":
    # Uji cepat dengan satu contoh, untuk memastikan model bisa dimuat dan menebak.
    pipe = load_pipeline()
    contoh = {
        "Annual_Income": 18500.0, "Monthly_Inhand_Salary": 1500.0,
        "Num_Bank_Accounts": 7, "Num_Credit_Card": 7, "Interest_Rate": 28,
        "Num_of_Loan": 4, "Delay_from_due_date": 25, "Num_of_Delayed_Payment": 12,
        "Changed_Credit_Limit": 9.4, "Num_Credit_Inquiries": 9.0,
        "Outstanding_Debt": 2300.0, "Credit_Utilization_Ratio": 29.0,
        "Total_EMI_per_month": 88.0, "Amount_invested_monthly": 70.0,
        "Monthly_Balance": 290.0, "Credit_History_Months": 120,
        "Credit_Mix": "Standard", "Payment_of_Min_Amount": "Yes",
        "Payment_Behaviour": "Low_spent_Small_value_payments",
        "has_auto_loan": 0, "has_credit_builder_loan": 0,
        "has_debt_consolidation_loan": 1, "has_home_equity_loan": 0,
        "has_mortgage_loan": 1, "has_payday_loan": 0,
        "has_personal_loan": 1, "has_student_loan": 0,
    }
    print(classify(pipe, contoh))
