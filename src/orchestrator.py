"""
orchestrator.py
===============
Penyatu semua langkah jadi satu alur yang bisa dijalankan sekali perintah.

Urutan yang dijalankan method run():
  1. Baca data lewat DatasetLoader
  2. Bersihkan lewat FrameCleaner (semua sebelum split)
  3. Pecah jadi train dan test, lalu ubah target jadi angka
  4. Latih tiga model lewat trainer yang mewarisi BaseTrainer
  5. Nilai tiap model dan catat ke MLflow
  6. Pilih yang F1 macro-nya tertinggi
  7. Lewatkan gerbang kualitas, kalau lolos simpan modelnya ke file pkl

Hasil akhirnya adalah satu file pipeline pkl di model_store yang nanti dipakai
oleh inferencing lokal maupun deployment.
"""

import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    APPROVAL_F1,
    MLFLOW_EXPERIMENT,
    MODEL_DIR,
    MODEL_FILE,
    RANDOM_SEED,
    TARGET,
    TEST_FRACTION,
)
from src.estimators import TRAINER_REGISTRY
from src.features import FrameCleaner, build_feature_pipeline, encode_target
from src.loader import DatasetLoader
from src.scoring import ModelScorer


class TrainingFlow:
    """Menjalankan pipeline training menyeluruh dan menyimpan model terbaik."""

    def __init__(self, test_fraction=TEST_FRACTION, seed=RANDOM_SEED):
        self.test_fraction = test_fraction
        self.seed = seed
        self.cleaner = FrameCleaner()
        self.scorer = ModelScorer(averaging="macro")
        self.board = {}
        self.winner = None

    def run(self) -> pd.DataFrame:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        self._banner("PIPELINE TRAINING CREDIT SCORE - DATASET A")

        # Langkah 1 dan 2
        frame = DatasetLoader().read()
        frame = self.cleaner.transform(frame)
        print(f"[flow] data bersih: {frame.shape[0]} baris, {frame.shape[1]} kolom")

        # Langkah 3
        X = frame.drop(columns=[TARGET])
        y = encode_target(frame[TARGET])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_fraction, random_state=self.seed, stratify=y
        )
        print(f"[flow] train {len(X_train)} baris, test {len(X_test)} baris")

        # Langkah 4 dan 5
        for trainer_cls in TRAINER_REGISTRY:
            # Preprocessor dibuat baru untuk tiap model supaya tiap pipeline punya
            # transformer sendiri yang hanya belajar dari data train model itu.
            trainer = trainer_cls(build_feature_pipeline())
            print(f"\n[flow] melatih {trainer.label}")
            trainer.fit(X_train, y_train)
            preds = trainer.predict(X_test)
            metrics = self.scorer.measure(y_test, preds)
            self.scorer.record(trainer, metrics)
            self.board[trainer.label] = {"trainer": trainer, **metrics}

        # Langkah 6
        self.winner = max(self.board.values(), key=lambda row: row["f1"])["trainer"]
        best_f1 = self.board[self.winner.label]["f1"]
        self._banner(f"MODEL TERPILIH: {self.winner.label} dengan F1 {best_f1:.4f}")

        # Langkah 7
        self._gate_and_save(best_f1)
        return self._leaderboard()

    def _gate_and_save(self, best_f1: float) -> None:
        """Simpan model hanya bila lolos ambang F1. Kalau tidak lolos, model tidak disimpan."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if best_f1 >= APPROVAL_F1:
            # compress=3 menekan ukuran file model secara signifikan. Penting karena
            # model hutan acak besar, dan file ini harus ikut diunggah ke GitHub
            # (batas 100 MB) untuk dipakai deployment Streamlit.
            joblib.dump(self.winner.pipeline, MODEL_FILE, compress=3)
            print(f"[flow] lolos gerbang ({best_f1:.4f} >= {APPROVAL_F1}). Model disimpan ke {MODEL_FILE.name}")
        else:
            print(f"[flow] gagal gerbang ({best_f1:.4f} < {APPROVAL_F1}). Model tidak disimpan.")

    def _leaderboard(self) -> pd.DataFrame:
        rows = [
            {
                "model": name,
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
            }
            for name, row in self.board.items()
        ]
        table = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
        print("\nPERBANDINGAN MODEL:")
        print(table.to_string(index=False))
        return table

    @staticmethod
    def _banner(text: str) -> None:
        line = "=" * 60
        print(f"\n{line}\n{text}\n{line}")
