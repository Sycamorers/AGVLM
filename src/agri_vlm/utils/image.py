"""Image helpers."""

from pathlib import Path
from typing import Iterable, List

from PIL import Image


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_SUFFIXES


def collect_image_paths(root: Path) -> List[Path]:
    return sorted([path for path in root.rglob("*") if is_image_path(path)])


def open_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def save_solid_image(path: Path, color: Iterable[int]) -> None:
    image = Image.new("RGB", (32, 32), tuple(color))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
