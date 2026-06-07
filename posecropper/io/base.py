"""Abstract image source / output sink so the core pipeline is storage-agnostic.
Local implementations power the offline demo; GCS implementations (io/gcs.py) power
the production Cloud Function."""
from abc import ABC, abstractmethod

import numpy as np


class ImageSource(ABC):
    @abstractmethod
    def read(self, path: str) -> np.ndarray:
        """Return the image at `path` as an RGB uint8 numpy array."""


class OutputSink(ABC):
    @abstractmethod
    def write(self, image_rgb: np.ndarray, path: str) -> None:
        """Persist an RGB image to `path` (Adobe-RGB JPEG)."""
