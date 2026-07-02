"""
loader.py
=========
Langkah pertama pipeline: memasukkan data mentah dengan aman.

Kenapa langkah ini dibuat terpisah dan dibungkus class? Di dunia nyata data
tidak datang sekali lalu selesai. Data terus mengalir, dan setiap batch baru
harus dicek dulu sebelum dipakai melatih model. Kalau filenya tidak ada, atau
isinya kosong, atau kolom targetnya hilang, pipeline harus berhenti sejak awal,
bukan baru ketahuan error jauh di belakang saat training.

Class DatasetLoader memegang dua hal sebagai atribut: dari mana data dibaca dan
ke mana salinan tervalidasinya disimpan. Method read() yang menjalankan
pekerjaannya dan mengembalikan DataFrame siap pakai untuk langkah berikutnya.
"""

import pandas as pd

from src.config import DATA_FILE, STAGING_DIR, TARGET


class DatasetLoader:
    """Membaca CSV mentah, menjalankan pemeriksaan dasar, lalu menyimpan salinan tervalidasi."""

    def __init__(self, source=DATA_FILE, staged_name="data_A_validated.csv"):
        self.source = source
        self.staged_path = STAGING_DIR / staged_name

    def read(self) -> pd.DataFrame:
        """Baca data, validasi, simpan salinan, kembalikan DataFrame."""
        if not self.source.exists():
            raise FileNotFoundError(
                f"File data tidak ditemukan di {self.source}. "
                "Pastikan data_A.csv ada di folder project."
            )

        frame = pd.read_csv(self.source)
        self._run_checks(frame)

        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.staged_path, index=False)

        rows, cols = frame.shape
        print(f"[loader] data dibaca dari {self.source.name}: {rows} baris, {cols} kolom")
        print(f"[loader] salinan tervalidasi disimpan ke {self.staged_path}")
        return frame

    @staticmethod
    def _run_checks(frame: pd.DataFrame) -> None:
        """
        Pemeriksaan minimum sebelum data boleh lanjut. Kalau salah satu gagal,
        pipeline langsung berhenti karena melatih model di atas data rusak
        hanya membuang waktu.
        """
        if frame.empty:
            raise ValueError("Dataset kosong, tidak ada yang bisa dilatih.")
        if TARGET not in frame.columns:
            raise ValueError(f"Kolom target '{TARGET}' tidak ada di dataset.")
        if frame[TARGET].isna().all():
            raise ValueError(f"Kolom target '{TARGET}' kosong semua.")


if __name__ == "__main__":
    # Bisa dijalankan sendiri untuk uji cepat: python -m src.loader
    DatasetLoader().read()
