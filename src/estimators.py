"""
estimators.py
=============
Bagian yang paling kental konsep OOP-nya.

Idenya begini. Semua model pohon yang kita pakai punya perilaku yang sama
persis: digabung dengan preprocessor jadi satu pipeline, dilatih, lalu dipakai
memprediksi. Yang beda cuma satu, yaitu jenis model dan setelan hyperparameter
khasnya. Maka kita tulis sekali perilaku yang sama itu di sebuah base class
abstrak (BaseTrainer), dan tiap model cukup mengisi satu method yang membedakan
mereka, yaitu _make_estimator().

Keuntungannya:
  Tidak ada kode kembar. Logika rangkai pipeline, fit, dan predict ditulis sekali.
  Kontrak antar pengembang terkunci. _make_estimator() ditandai sebagai abstract
    method, jadi kalau ada yang menambah model baru tapi lupa mengisinya, Python
    langsung menolak saat objek dibuat, bukan diam-diam salah di belakang.
  Menambah model cukup membuat satu child baru, base class tidak perlu disentuh.

Hyperparameter di sini sengaja ditulis tetap, bukan dituning lagi. Tuning sudah
dilakukan di notebook eksperimen. Pipeline mengutamakan kecepatan retraining,
jadi langsung memakai angka yang sudah terbukti bagus.
"""

from abc import ABC, abstractmethod

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import CLASS_ORDER, RANDOM_SEED


class BaseTrainer(ABC):
    """Induk untuk semua trainer. Memuat perilaku yang sama bagi seluruh model."""

    label = "base"

    def __init__(self, preprocessor):
        # Atribut yang dimiliki semua trainer.
        self.preprocessor = preprocessor
        self.pipeline = None
        self.run_id = None

    @abstractmethod
    def _make_estimator(self):
        """Tiap child wajib mengembalikan objek model sklearn miliknya sendiri."""
        raise NotImplementedError

    def assemble(self) -> Pipeline:
        """Gabungkan preprocessor dan model jadi satu pipeline. Sama untuk semua model."""
        self.pipeline = Pipeline([
            ("prep", self.preprocessor),
            ("model", self._make_estimator()),
        ])
        return self.pipeline

    def fit(self, X, y) -> Pipeline:
        if self.pipeline is None:
            self.assemble()
        self.pipeline.fit(X, y)
        return self.pipeline

    def predict(self, X):
        return self.pipeline.predict(X)


class RandomForestTrainer(BaseTrainer):
    """Hutan acak. Tahan terhadap outlier, jadi cocok untuk data yang penuh nilai ekstrem ini."""

    label = "random_forest"

    def _make_estimator(self):
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )


class ExtraTreesTrainer(BaseTrainer):
    """Pohon ekstra acak. Mirip hutan acak tapi titik pisahnya lebih acak, jadi latihannya cepat."""

    label = "extra_trees"

    def _make_estimator(self):
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )


class XGBoostTrainer(BaseTrainer):
    """Gradient boosting versi XGBoost. Butuh target berupa angka, yang sudah kita siapkan."""

    def _make_estimator(self):
        return XGBClassifier(
            n_estimators=400,
            max_depth=7,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.2,
            objective="multi:softprob",
            num_class=len(CLASS_ORDER),
            eval_metric="mlogloss",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

    label = "xgboost"


# Daftar trainer yang dipakai pipeline. Ini tiga model teratas hasil eksperimen
# notebook. Decision Tree dan Gradient Boosting sengaja tidak dibawa ke pipeline
# karena di eksperimen kalah, dan demi kecepatan cukup tiga terbaik yang diadu.
TRAINER_REGISTRY = [RandomForestTrainer, ExtraTreesTrainer, XGBoostTrainer]
