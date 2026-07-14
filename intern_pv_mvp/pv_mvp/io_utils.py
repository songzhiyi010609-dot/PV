from __future__ import annotations

import re
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(value: str, fallback: str = "item") -> str:
    value = str(value or "").strip()
    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    value = value.strip("._ ")
    return value[:90] or fallback


def to_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def relative_to_base(path: str | Path, base: str | Path) -> str:
    path = Path(path)
    base = Path(base)
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()
