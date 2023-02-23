import os
from pathlib import Path

from pydantic import BaseSettings


class Paths(BaseSettings):
    BASE = Path(os.getcwd())
    DATA = BASE / "data"
    RAW = DATA / "raw"
    EXTERNAL = DATA / "external"
    FONTS = DATA / "fonts"
    LOGOS = DATA / "logos"
    MODELS = BASE / "models"


paths = Paths()


class WatermarkSettings(BaseSettings):
    WATERMARK_SIZE: int = 1000
    RANDOM_HORIZONTAL_MOVEMENT: int = 300
    RANDOM_VERTICAL_MOVEMENT: int = 300


watermark_settings = WatermarkSettings()


class TrainingSettings(BaseSettings):
    SIZE_OF_IMAGE: tuple[int, int] = 200, 200
    SIZE_OF_BATCH: int = 16
    NUM_OF_EPOCHS: int = 5
    DEVICE: str = "gpu"


training_settings = TrainingSettings()
