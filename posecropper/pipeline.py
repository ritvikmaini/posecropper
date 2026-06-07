"""Shared orchestration: router decision -> pose (if needed) -> crop strategies.
Pure of storage and catalog concerns (those are injected). Used by both the offline
demo and the Cloud Function so the routing/crop logic exists exactly once."""
from posecropper import crop, pose, router


def _run_strategy(name, image_rgb, h, w, landmarks, center_x, center_y):
    if name == "full":
        return crop.crop_full(image_rgb, h, w, landmarks)
    if name == "default":
        return crop.crop_default(image_rgb, h, w)
    if name == "top":
        return crop.crop_top(image_rgb, h, w, landmarks, center_x, center_y)
    if name == "bottom":
        return crop.crop_bottom(image_rgb, h, w, landmarks, center_x, center_y)
    raise ValueError(f"Unknown strategy: {name}")


def process_image(image_rgb, pose_code, category, pose_detector=None):
    """Return (decision, crops, overlay).
    crops: dict strategy_name -> RGB array. overlay: pose overlay RGB or None."""
    decision = router.decide(pose_code, category)
    h, w = image_rgb.shape[:2]

    needs_pose = bool(set(decision.strategies) & {"full", "top", "bottom"})
    landmarks = center_x = center_y = None
    overlay = None
    if needs_pose:
        if pose_detector is None:
            pose_detector = pose.load_pose_detector()
        pose_landmarks, landmarks, center_x, center_y = pose.detect(pose_detector, image_rgb)
        overlay = pose.annotate_overlay(image_rgb, pose_landmarks, center_x, center_y)

    crops = {
        name: _run_strategy(name, image_rgb, h, w, landmarks, center_x, center_y)
        for name in decision.strategies
    }
    return decision, crops, overlay
