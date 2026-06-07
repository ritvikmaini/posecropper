"""Synthetic stand-ins for MediaPipe landmarks so crop geometry can be tested
without a real image or model. Each landmark exposes normalized .x / .y in [0,1]."""


class Landmark:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def make_landmarks(points):
    """points: iterable of (x, y) normalized tuples -> list[Landmark]."""
    return [Landmark(x, y) for x, y in points]
