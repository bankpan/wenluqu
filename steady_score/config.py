from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class OCRConfig:
    image_dir: Path
    output_dir: Path
    raw_dir: Path
    log_dir: Path
    error_file: Path
    lang: Literal["ch", "en", "chinese_cht"] = "ch"
    min_confidence: float = 0.6
    crop_margin: int = 10
    binarize_threshold: int = 240


@dataclass(slots=True)
class PAVAConfig:
    min_bin_size: int = 8
    confidence: float = 0.95
    target_ratio: float = 0.75
    interpolation_digits: int = 0  # nearest whole score segment
    allow_fallback: bool = True
