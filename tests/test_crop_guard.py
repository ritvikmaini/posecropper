import numpy as np

from posecropper.crop import crop_default, crop_full, default_box, full_box, full_box_raw
from posecropper.taxonomy import ASPECT_RATIO
from tests.stubs import make_landmarks


def _aspect(img):
    return img.shape[0] / img.shape[1]


def test_full_box_returns_default_box_coords_on_frame_edge():
    # Tall, narrow image: a subject spanning the full height forces the derived
    # full-body width to exceed the image width, breaking the 1.33 ratio after
    # clamping -> the guard must return the default_box coordinates.
    landmarks = make_landmarks([(0.5, 0.0), (0.5, 1.0)])  # y spans 0..400

    (_, _, _, _), realized = full_box_raw(400, 100, landmarks)
    assert not (ASPECT_RATIO - 0.01 <= realized <= ASPECT_RATIO + 0.01)  # raw geometry is off-spec

    assert full_box(400, 100, landmarks) == default_box(400, 100, ASPECT_RATIO)  # guard engaged


def test_crop_full_falls_back_on_frame_edge():
    image = np.zeros((400, 100, 3), dtype=np.uint8)
    landmarks = make_landmarks([(0.5, 0.0), (0.5, 1.0)])

    result = crop_full(image, 400, 100, landmarks)
    expected = crop_default(image, 400, 100, ASPECT_RATIO)

    assert result.shape == expected.shape           # fell back to default
    assert result.shape[1] == 100                   # full image width
    assert abs(_aspect(result) - ASPECT_RATIO) < 0.02


def test_crop_full_keeps_geometry_when_ratio_holds():
    # Wide image with a centered subject: the derived width fits, so the guard
    # does NOT fire and the landmark-driven crop is returned (not full width).
    image = np.zeros((800, 600, 3), dtype=np.uint8)
    landmarks = make_landmarks([(0.4, 0.25), (0.6, 0.75)])  # y 200..600, x ~300

    assert full_box(800, 600, landmarks) != default_box(800, 600, ASPECT_RATIO)  # guard passive
    result = crop_full(image, 800, 600, landmarks)

    assert result.shape[1] < 600                    # narrower than full width
    assert abs(_aspect(result) - ASPECT_RATIO) < 0.02
