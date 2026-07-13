import cv2
import numpy as np
from skimage.morphology import skeletonize


def _binarize_ridges(enhanced: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        4,
    )
    return binary


def extract_minutiae(enhanced: np.ndarray, border: int = 12, max_points: int = 60) -> list[dict]:
    """Classical crossing-number minutiae extraction on a skeletonized ridge map.

    Skeletonizes the binarized ridge structure, then classifies each skeleton pixel
    by its 8-neighbour crossing number: 1 => ridge ending, 3 => bifurcation.
    This is deliberately the standard, well-understood minutiae method (not a novel
    detector) -- the novelty in this project lives in what is *done* with the
    minutiae (see causal_attribution.py), not in how they are found.

    Returns a list of {"x", "y", "type"} dicts, "type" in {"ending", "bifurcation"}.
    Coordinates are in the same pixel space as the input `enhanced` image.
    """
    binary = _binarize_ridges(enhanced)
    skeleton = skeletonize(binary > 0).astype(np.uint8)

    h, w = skeleton.shape
    padded = np.pad(skeleton, 1, mode="constant")

    minutiae: list[dict] = []
    ys, xs = np.nonzero(skeleton)

    for y, x in zip(ys.tolist(), xs.tolist()):
        if y < border or x < border or y > h - border or x > w - border:
            continue

        py, px = y + 1, x + 1
        ring = [
            padded[py - 1, px], padded[py - 1, px + 1], padded[py, px + 1],
            padded[py + 1, px + 1], padded[py + 1, px], padded[py + 1, px - 1],
            padded[py, px - 1], padded[py - 1, px - 1],
        ]

        crossing_number = sum(
            abs(int(ring[i]) - int(ring[(i + 1) % 8])) for i in range(8)
        ) // 2

        if crossing_number == 1:
            minutiae.append({"x": int(x), "y": int(y), "type": "ending"})
        elif crossing_number == 3:
            minutiae.append({"x": int(x), "y": int(y), "type": "bifurcation"})

    if len(minutiae) > max_points:
        rng = np.random.default_rng(42)
        keep = np.sort(rng.choice(len(minutiae), size=max_points, replace=False))
        minutiae = [minutiae[i] for i in keep.tolist()]

    return minutiae
