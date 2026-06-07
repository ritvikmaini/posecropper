"""Google Cloud Storage ImageSource / OutputSink (cloud path only).
Imported lazily by posecropper.cloud — never by the offline demo."""
import os
import tempfile

import cv2
import numpy as np
from google.cloud import storage

from posecropper.encode import encode_jpeg_adobergb
from posecropper.io.base import ImageSource, OutputSink


class GcsImageSource(ImageSource):
    def __init__(self, bucket: storage.Bucket):
        self.bucket = bucket

    def read(self, path: str) -> np.ndarray:
        blob = self.bucket.blob(path)
        data = blob.download_as_bytes()
        arr = np.frombuffer(data, np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found at path: {path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


class GcsOutputSink(OutputSink):
    def __init__(self, bucket: storage.Bucket):
        self.bucket = bucket

    def write(self, image_rgb: np.ndarray, path: str) -> None:
        tmp = os.path.join(tempfile.gettempdir(), os.path.basename(path))
        encode_jpeg_adobergb(image_rgb, tmp)
        self.bucket.blob(path).upload_from_filename(tmp)
