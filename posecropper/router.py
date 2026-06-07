"""Metadata-driven crop router.

Selects a crop strategy from the filename pose code and the catalog-derived garment
category, with explicit precedence. An explicit, known pose code always wins; if the
pose code is unrecognized the catalog category decides; if that is ambiguous the
router hedges by producing both top and bottom crops for human review.
"""
from dataclasses import dataclass

# Pose codes that always map to a full-body crop.
FULL_BODY_POSES = {"M12", "M13", "M14", "M15", "M16", "M32", "M35", "M36"}
# Pose codes that map to a metadata-only default crop (pose detection skipped).
DEFAULT_POSES = {f"M{n}" for n in range(17, 32)}  # M17..M31 inclusive


@dataclass
class CropDecision:
    strategies: list[str]   # one or more of: full, default, top, bottom
    review: bool            # True -> outputs routed to the /review/ folder
    reason: str             # human-readable explanation (used by the visualizer)


def decide(pose_code: str | None, category: str | None) -> CropDecision:
    if pose_code in FULL_BODY_POSES:
        return CropDecision(["full"], False, f"pose {pose_code} -> full-body crop")
    if pose_code in DEFAULT_POSES:
        return CropDecision(["default"], False, f"pose {pose_code} -> default crop (pose detection skipped)")
    if pose_code is not None and str(pose_code).startswith("P"):
        return CropDecision(["default"], False, f"product/flat-lay pose {pose_code} -> default crop")
    if category == "top":
        return CropDecision(["top"], False, "category 'top' -> top crop")
    if category == "bottom":
        return CropDecision(["bottom"], False, "category 'bottom' -> bottom crop")
    return CropDecision(["top", "bottom"], True,
                        f"ambiguous (pose={pose_code}, category={category}) -> review: both top & bottom")
