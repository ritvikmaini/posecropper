"""Filename metadata extraction and garment-category classification.

These functions are pure (no GCP). The catalog lookup that turns a style code into
an ItemGroupName lives behind the CategoryProvider interface (posecropper.catalog).
"""
import os
import re

from posecropper import taxonomy

_FILENAME_PATTERN = re.compile(r"^(?P<model>[^_]+)_(?P<pose>[^-]+)-")


def parse_filename(image_path: str) -> dict:
    """Extract the style/model code and pose code from a filename of the form
    `<model>_<pose>-...jpg`. Returns reading_error=True when it does not match."""
    filename = os.path.basename(image_path)
    match = _FILENAME_PATTERN.match(filename)
    if match:
        return {
            "model": match.group("model"),
            "pose": match.group("pose"),
            "original_file_image": filename,
            "reading_error": False,
        }
    return {
        "model": None,
        "pose": None,
        "original_file_image": filename,
        "reading_error": True,
    }


def classify_category(item_group: str | None) -> str:
    """Map a catalog ItemGroupName to a routing bucket: 'top', 'bottom', or 'review'.
    Unknown / accessory groups fall back to 'review' (both crops, human-checked)."""
    if item_group in taxonomy.UPPER_HALF_ARTICLES:
        return "top"
    if item_group in taxonomy.LOWER_HALF_ARTICLES:
        return "bottom"
    if item_group in taxonomy.REVIEW_ARTICLES:
        return "review"
    return "review"
