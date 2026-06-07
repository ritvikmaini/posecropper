"""Pure crop geometry. No GCP, no image I/O, no pose model — operates on a numpy
RGB array plus landmark objects (each exposing normalized .x / .y). All strategies
target height:width = taxonomy.ASPECT_RATIO (1.33).

Each strategy is split into a coordinate-returning `*_box()` function and a thin
`crop_*()` slicing wrapper. The box functions let the visualizer draw the actual crop
regions (and the aspect-ratio guard's behaviour) rather than approximations."""
import numpy as np

from posecropper.taxonomy import (
    ASPECT_RATIO,
    EXTRA_ABOVE_CENTER_RATIO,
    EXTRA_BELOW_CENTER_RATIO,
    VERTICAL_BOTTOM_PADDING,
    VERTICAL_TOP_PADDING,
)

# --- Coordinate-returning box functions: return (x1, y1, x2, y2) ---

def default_box(image_height, image_width, aspect=ASPECT_RATIO):
    """Metadata-only full-width center crop region; needs no landmarks."""
    full_image_height = int(image_width * aspect)
    center_y = image_height // 2
    top_y = max(0, center_y - full_image_height // 2)
    bottom_y = min(image_height, top_y + full_image_height)
    return 0, top_y, image_width, bottom_y


def full_box_raw(image_height, image_width, landmarks):
    """Pre-guard full-body box derived from the landmark vertical extent + padding,
    centered on the landmark mean-x. Returns ((x1, y1, x2, y2), realized_ratio)."""
    y_coords = [lm.y * image_height for lm in landmarks]
    min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
    model_height = max_y - min_y

    top_y = max(0, min_y - int(VERTICAL_TOP_PADDING * model_height))
    bottom_y = min(image_height, max_y + int(VERTICAL_BOTTOM_PADDING * model_height))

    full_image_height = bottom_y - top_y
    full_image_width = int(full_image_height / ASPECT_RATIO)

    center_x = int(np.mean([lm.x * image_width for lm in landmarks]))
    left_x = max(0, center_x - full_image_width // 2)
    right_x = min(image_width, left_x + full_image_width)

    realized_ratio = full_image_height / (right_x - left_x)
    return (left_x, top_y, right_x, bottom_y), realized_ratio


def full_box(image_height, image_width, landmarks):
    """Guarded full-body box: returns the raw geometry when its realized ratio is within
    1.33 ± 0.01, otherwise falls back to `default_box` coords (the aspect-ratio guard)."""
    (x1, y1, x2, y2), realized_ratio = full_box_raw(image_height, image_width, landmarks)
    if not ((ASPECT_RATIO - 0.01) <= realized_ratio <= (ASPECT_RATIO + 0.01)):
        return default_box(image_height, image_width, ASPECT_RATIO)
    return x1, y1, x2, y2


def top_box(image_height, image_width, landmarks, center_x, center_y,
            extra_vertical_padding_ratio=VERTICAL_TOP_PADDING,
            extra_below_center_ratio=EXTRA_BELOW_CENTER_RATIO):
    """Upper-body box: anchor at the body center, extend toward the head."""
    y_coords = [lm.y * image_height for lm in landmarks]
    min_y = int(np.min(y_coords))
    model_height = int(np.max(y_coords)) - min_y

    top_y = max(0, min_y - int(extra_vertical_padding_ratio * model_height))
    top_half_extra_space = int(extra_below_center_ratio * (center_y - top_y))
    bottom = center_y + top_half_extra_space
    width = int((bottom - top_y) / ASPECT_RATIO)

    left_x = max(0, center_x - width // 2)
    right_x = min(image_width, center_x + width // 2)
    return left_x, top_y, right_x, bottom


def bottom_box(image_height, image_width, landmarks, center_x, center_y,
               extra_vertical_padding_ratio=VERTICAL_BOTTOM_PADDING,
               extra_above_center_ratio=EXTRA_ABOVE_CENTER_RATIO):
    """Lower-body box: anchor at the body center, extend toward the feet."""
    y_coords = [lm.y * image_height for lm in landmarks]
    max_y = int(np.max(y_coords))
    model_height = max_y - int(np.min(y_coords))

    bottom_y = min(image_height, max_y + int(extra_vertical_padding_ratio * model_height))
    extra = int(extra_above_center_ratio * (bottom_y - center_y))
    top = center_y - extra
    width = int((bottom_y - top) / ASPECT_RATIO)

    left_x = max(0, center_x - width // 2)
    right_x = min(image_width, center_x + width // 2)
    return left_x, top, right_x, bottom_y


# --- Thin slicing wrappers: return the cropped RGB array ---

def crop_default(image_rgb, image_height, image_width, aspect_ratio=ASPECT_RATIO):
    x1, y1, x2, y2 = default_box(image_height, image_width, aspect_ratio)
    return image_rgb[y1:y2, x1:x2]


def crop_full(image_rgb, image_height, image_width, landmarks):
    try:
        x1, y1, x2, y2 = full_box(image_height, image_width, landmarks)
        return image_rgb[y1:y2, x1:x2]
    except Exception as e:  # noqa: BLE001 - surfaced as ValueError at the boundary
        raise ValueError(f"Error cropping full image: {e}") from e


def crop_top(image_rgb, image_height, image_width, landmarks, center_x, center_y):
    try:
        x1, y1, x2, y2 = top_box(image_height, image_width, landmarks, center_x, center_y)
        return image_rgb[y1:y2, x1:x2]
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Error cropping top: {e}") from e


def crop_bottom(image_rgb, image_height, image_width, landmarks, center_x, center_y):
    try:
        x1, y1, x2, y2 = bottom_box(image_height, image_width, landmarks, center_x, center_y)
        return image_rgb[y1:y2, x1:x2]
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Error cropping bottom: {e}") from e
