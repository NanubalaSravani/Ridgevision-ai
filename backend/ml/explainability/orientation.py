import cv2
import numpy as np


def compute_orientation_field(gray: np.ndarray, block_size: int = 16) -> dict[str, np.ndarray]:
    """Block-wise ridge orientation field via the standard gradient least-squares method.

    Returns:
        orientation: dominant ridge angle per block (radians), shape (H//block, W//block)
        coherence:   per-block orientation consistency in [0, 1]; 1 = ridges flow in a
                     single clean direction, low values mark cores/deltas/high-curvature
                     or noisy regions.
    """
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    h, w = gray.shape
    blocks_y = max(h // block_size, 1)
    blocks_x = max(w // block_size, 1)

    orientation = np.zeros((blocks_y, blocks_x), dtype=np.float32)
    coherence = np.zeros((blocks_y, blocks_x), dtype=np.float32)

    for by in range(blocks_y):
        for bx in range(blocks_x):
            y0, y1 = by * block_size, min((by + 1) * block_size, h)
            x0, x1 = bx * block_size, min((bx + 1) * block_size, w)
            gxx = gx[y0:y1, x0:x1]
            gyy = gy[y0:y1, x0:x1]

            vx = float(np.sum(2 * gxx * gyy))
            vy = float(np.sum(gxx ** 2 - gyy ** 2))
            orientation[by, bx] = 0.5 * np.arctan2(vx, vy)

            energy = np.sqrt(vx ** 2 + vy ** 2)
            norm = float(np.sum(gxx ** 2 + gyy ** 2)) + 1e-6
            coherence[by, bx] = energy / norm

    return {"orientation": orientation, "coherence": coherence}


def singularity_map(coherence: np.ndarray) -> np.ndarray:
    """Converts a coherence field into a 0-1 'ridge curvature / singularity likelihood'
    map. Low coherence (inconsistent local ridge direction) marks cores, deltas, and
    other high-curvature dermatoglyphic structure -- the biologically interesting
    regions, as opposed to flat, uniform ridge flow.
    """
    inverted = 1.0 - coherence
    if inverted.size > 1:
        inverted = cv2.GaussianBlur(inverted.astype(np.float32), (0, 0), 1.0)
    lo, hi = float(inverted.min()), float(inverted.max())
    if hi - lo < 1e-6:
        return np.zeros_like(inverted)
    return (inverted - lo) / (hi - lo)
