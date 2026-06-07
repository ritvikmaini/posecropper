"""Render the four diagnostic gallery panels per sample image plus an outcome-led
before/after hero, saved to assets/. Reuses the production pose overlay and the
luminance-adaptive labeling from the segmentation diagnostic. Core deps only (opencv, numpy)."""
import cv2
import numpy as np

from posecropper import crop, pipeline, pose, router
from posecropper.taxonomy import ASPECT_RATIO

# Color per strategy (RGB tuples; arrays are kept in RGB throughout)
STRATEGY_COLORS = {
    "full": (0, 200, 0),
    "top": (0, 120, 255),
    "bottom": (255, 120, 0),
    "default": (180, 180, 180),
}


def _adaptive_text_color(image_rgb):
    """Rec. 601 luminance -> black on light backgrounds, white on dark (legible labels)."""
    avg = np.mean(image_rgb, axis=(0, 1))
    brightness = np.sqrt(0.299 * avg[0] ** 2 + 0.587 * avg[1] ** 2 + 0.114 * avg[2] ** 2)
    return (0, 0, 0) if brightness > 127 else (255, 255, 255)


def _label(img, lines, origin=(10, 30), color=None):
    color = color or _adaptive_text_color(img)
    y = origin[1]
    for line in lines:
        cv2.putText(img, line, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 32
    return img


def _resize_to_height(img, height):
    scale = height / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * scale)), height))


def panel_pose_overlay(image_rgb, pose_detector):
    """Panel 1: 33 landmarks + skeleton + computed center anchor."""
    pose_landmarks, _, cx, cy = pose.detect(pose_detector, image_rgb)
    return pose.annotate_overlay(image_rgb, pose_landmarks, cx, cy)


def _strategy_box(name, h, w, landmarks, cx, cy):
    """Return the real (x1,y1,x2,y2) the given strategy would crop, from crop.*_box."""
    if name == "full":
        return crop.full_box(h, w, landmarks)
    if name == "top":
        return crop.top_box(h, w, landmarks, cx, cy)
    if name == "bottom":
        return crop.bottom_box(h, w, landmarks, cx, cy)
    return crop.default_box(h, w)


def panel_candidate_boxes(image_rgb, pose_detector, strategies=("full", "top", "bottom", "default")):
    """Panel 2: each candidate crop box drawn at its REAL position in its strategy color,
    labeled with the realized aspect ratio (height/width) computed from that box."""
    vis = image_rgb.copy()
    h, w = image_rgb.shape[:2]
    _, landmarks, cx, cy = pose.detect(pose_detector, image_rgb)
    labels = []
    for name in strategies:
        x1, y1, x2, y2 = _strategy_box(name, h, w, landmarks, cx, cy)
        realized = (y2 - y1) / (x2 - x1)
        color = STRATEGY_COLORS.get(name, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
        labels.append(f"{name}: {realized:.2f} (target {ASPECT_RATIO})")
    return _label(vis, labels)


def panel_router_decision(image_rgb, pose_code, category):
    """Panel 3: annotate the routing inputs and the chosen strategy."""
    vis = image_rgb.copy()
    decision = router.decide(pose_code, category)
    lines = [
        f"pose_code = {pose_code}",
        f"category  = {category}",
        f"-> {', '.join(decision.strategies)}" + ("  [REVIEW]" if decision.review else ""),
        decision.reason,
    ]
    return _label(vis, lines)


def panel_aspect_guard(image_rgb, pose_detector):
    """Panel 4 (two-up):
    LEFT  = the real sample with its true guarded full_box and realized ratio (guard passive).
    RIGHT = a forced frame-edge case (a tall-narrow vertical sliver of the sample): the
            off-spec raw geometry box (red, with its broken ratio) and — only when the guard
            actually engages — the full-width fallback box (green) the guard substitutes."""
    h, w = image_rgb.shape[:2]
    _, landmarks, cx, cy = pose.detect(pose_detector, image_rgb)

    # LEFT: real image, true guarded full box.
    left = image_rgb.copy()
    _, realized = crop.full_box_raw(h, w, landmarks)
    fx1, fy1, fx2, fy2 = crop.full_box(h, w, landmarks)
    cv2.rectangle(left, (fx1, fy1), (fx2, fy2), (0, 200, 0), 3)
    _label(left, [f"full_box realized ratio: {realized:.2f}",
                  f"guard passive (target {ASPECT_RATIO} +/- 0.01)"])

    # RIGHT: forced frame-edge via a tall-narrow vertical sliver of the sample.
    sliver_w = max(1, h // 4)
    s_x1 = max(0, min(cx - sliver_w // 2, w - sliver_w))
    sliver = image_rgb[:, s_x1:s_x1 + sliver_w].copy()
    sh, sw = sliver.shape[:2]
    (bx1, by1, bx2, by2), raw_ratio = crop.full_box_raw(sh, sw, landmarks)
    engaged = crop.full_box(sh, sw, landmarks) == crop.default_box(sh, sw, ASPECT_RATIO)
    cv2.rectangle(sliver, (bx1, by1), (bx2, by2), (255, 0, 0), 3)          # off-spec raw geometry (red)
    if engaged:
        gx1, gy1, gx2, gy2 = crop.default_box(sh, sw, ASPECT_RATIO)
        cv2.rectangle(sliver, (gx1, gy1), (gx2, gy2), (0, 200, 0), 3)      # full-width fallback (green)
    _label(sliver, [f"raw ratio: {raw_ratio:.2f}",
                    "guard engaged -> full-width fallback" if engaged else "within tolerance"])

    # Stack the two sub-panels side by side at a common height.
    return np.hstack([_resize_to_height(left, h), _resize_to_height(sliver, h)])


def build_hero(image_rgb, pose_code, category, pose_detector=None, include_overlay=True):
    """Outcome-led hero: ORIGINAL (left) -> [pose overlay (middle, optional)] ->
    FINAL delivered crop (right). The final crop is the actual pipeline output for these
    inputs (the first strategy; for a review case that is the top crop)."""
    if pose_detector is None:
        pose_detector = pose.load_pose_detector()
    _, crops, overlay = pipeline.process_image(image_rgb, pose_code, category, pose_detector)
    final = next(iter(crops.values()))

    before = _label(image_rgb.copy(), ["BEFORE"])
    after = _label(final.copy(), ["AFTER"])
    parts = [before]
    if include_overlay and overlay is not None:
        parts.append(overlay)
    parts.append(after)

    height = max(p.shape[0] for p in parts)
    return np.hstack([_resize_to_height(p, height) for p in parts])


def panel_process_to_product(image_rgb, strategy, pose_detector):
    """Process -> product strip for one strategy:
    LEFT  = the source with the 33-point pose overlay (keypoints + skeleton + center) AND
            this strategy's real crop box drawn in its strategy color, labeled with the box's
            realized aspect ratio.
    RIGHT = the FINAL delivered crop that strategy produces.
    Directly shows "the bounding box of the processing, then the final product"."""
    h, w = image_rgb.shape[:2]
    pose_landmarks, landmarks, cx, cy = pose.detect(pose_detector, image_rgb)

    left = pose.annotate_overlay(image_rgb, pose_landmarks, cx, cy)
    x1, y1, x2, y2 = _strategy_box(strategy, h, w, landmarks, cx, cy)
    color = STRATEGY_COLORS.get(strategy, (255, 255, 255))
    cv2.rectangle(left, (x1, y1), (x2, y2), color, 4)
    realized = (y2 - y1) / (x2 - x1)
    _label(left, [f"{strategy} crop box -> ratio {realized:.2f} (target {ASPECT_RATIO})"])

    product = image_rgb[y1:y2, x1:x2].copy()
    _label(product, [f"FINAL {strategy}"])

    return np.hstack([_resize_to_height(left, h), _resize_to_height(product, h)])


def render_all(image_rgb, pose_code, category, out_dir, pose_detector=None):
    """Render the full gallery to out_dir as compact JPEGs and return the saved paths:
    the outcome-led hero, the four diagnostic panels (pose overlay, candidate boxes, router
    decision, aspect guard), and a process->product strip for each of top/bottom/full."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    panels = {
        "hero": build_hero(image_rgb, pose_code, category, pose.load_pose_detector()),
        "pose_overlay": panel_pose_overlay(image_rgb, pose.load_pose_detector()),
        "candidate_boxes": panel_candidate_boxes(image_rgb, pose.load_pose_detector()),
        "router_decision": panel_router_decision(image_rgb, pose_code, category),
        "aspect_guard": panel_aspect_guard(image_rgb, pose.load_pose_detector()),
    }
    for strategy in ("top", "bottom", "full"):
        panels[strategy] = panel_process_to_product(image_rgb, strategy, pose.load_pose_detector())

    paths = {}
    for name, img in panels.items():
        path = os.path.join(out_dir, f"{name}.jpg")
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        paths[name] = path
    return paths
