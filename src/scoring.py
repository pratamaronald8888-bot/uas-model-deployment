"""
scoring.py
==========
Menghitung performa tiap model dan mencatatnya ke MLflow.

Metrik utama yang dipakai adalah F1 dengan rata-rata macro, bukan akurasi.
Alasannya, target kita tidak seimbang. Kelas Standard jauh lebih banyak
daripada Poor dan Good. Kalau dinilai pakai akurasi, model yang asal menebak
Standard untuk semua orang sudah kelihatan benar separuh lebih, padahal sama
sekali tidak berguna untuk mengenali nasabah Good atau Poor. F1 menggabungkan
precision dan recall, dan versi macro merata-ratakan F1 tiap kelas dengan bobot
sama, jadi kelas yang sedikit tetap dihitung adil.

MLflow dipakai karena soal meminta eksperimen yang tercatat dan terpantau.
Setiap kali sebuah model dievaluasi, parameter, metrik, dan modelnya disimpan
sebagai satu run. Hasilnya bisa dibandingkan lewat MLflow UI dengan perintah
`mlflow ui`.
"""

import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


class ModelScorer:
    """Menghitung empat metrik klasifikasi dan menuliskan hasilnya ke MLflow."""

    def __init__(self, averaging="macro"):
        self.averaging = averaging

    def measure(self, y_true, y_pred) -> dict:
        """Kembalikan dict empat metrik supaya gampang dibandingkan dan dicatat."""
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(
                precision_score(y_true, y_pred, average=self.averaging, zero_division=0), 4
            ),
            "recall": round(
                recall_score(y_true, y_pred, average=self.averaging, zero_division=0), 4
            ),
            "f1": round(f1_score(y_true, y_pred, average=self.averaging, zero_division=0), 4),
        }

    def record(self, trainer, metrics: dict) -> str:
        """
        Catat satu run ke MLflow: jenis model, hyperparameter aktual, metrik, dan
        modelnya. Kembalikan run id supaya bisa dilacak balik.
        """
        with mlflow.start_run(run_name=trainer.label) as active:
            mlflow.log_param("model", trainer.label)
            mlflow.log_params(trainer.pipeline.named_steps["model"].get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(trainer.pipeline, name="pipeline")
            trainer.run_id = active.info.run_id

        print(
            f"[scoring] {trainer.label:<14} "
            f"acc {metrics['accuracy']:.4f}  prec {metrics['precision']:.4f}  "
            f"rec {metrics['recall']:.4f}  f1 {metrics['f1']:.4f}"
        )
        return trainer.run_id
