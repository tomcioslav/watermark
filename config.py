from pydantic import BaseSettings
from pathlib import Path
import os

class Paths(BaseSettings):
    base = Path(os.getcwd())
    data = base / "data"
    raw = data / "raw"
    external = data / "external"
    fonts = data / "fonts"
    logos = data / "logos"

paths = Paths()