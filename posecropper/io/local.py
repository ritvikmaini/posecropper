"""Local-filesystem ImageSource / OutputSink (core deps only)."""
import os

import cv2
import numpy as np

from posecropper.encode import encode_jpeg_adobergb
from posecropper.io.base import ImageSource, OutputSink


class LocalImageSource(ImageSource):
    def read(self, path: str) -> np.ndarray:
        image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found or unreadable: {path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


class LocalOutputSink(OutputSink):
    def __init__(self, root: str = "."):
        self.root = root

    def write(self, image_rgb: np.ndarray, path: str) -> None:
        full = os.path.join(self.root, path)
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        encode_jpeg_adobergb(image_rgb, full)
