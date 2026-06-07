"""Adobe-RGB color-managed JPEG encoding — the catalog-fidelity output step,
split from any upload concern so it is reusable by every sink."""
from pathlib import Path

from PIL import Image

_ICC_PATH = Path(__file__).parent / "profiles" / "AdobeRGB1998.icc"


def encode_jpeg_adobergb(image_rgb, out_path: str) -> None:
    """Save an RGB numpy array as a quality-100 JPEG with the AdobeRGB1998 ICC profile embedded."""
    Image.fromarray(image_rgb).save(out_path, "JPEG", quality=100, icc_profile=_ICC_PATH.read_bytes())
