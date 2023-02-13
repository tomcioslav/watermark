from pydantic import BaseSettings
from pathlib import Path
import os


class Paths(BaseSettings):
    BASE = Path(os.getcwd())
    DATA = BASE / "data"
    RAW = DATA / "raw"
    EXTERNAL = DATA / "external"
    FONTS = DATA / "fonts"
    LOGOS = DATA / "logos"


paths = Paths()


class WatermarkCreationSettings(BaseSettings):
    WATERMARK_SIZE: int = 1000
    RANDOM_HORIZONTAL_MOVEMENT: int = 300
    RANDOM_VERTICAL_MOVEMENT: int = 300
