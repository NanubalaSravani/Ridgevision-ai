import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def compute_attention_intensity(enhanced: np.ndarray) -> np.ndarray:
    """Raw grayscale attention/saliency intensity map (0-255), before colorization.

    Exposed separately so other explainability modules (e.g. the orientation-attention
    alignment score) can consume the same attention signal that gets visualized.
    """
    edges_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=5)
    edges_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=5)
    attention = cv2.magnitude(edges_x, edges_y)
    attention = cv2.GaussianBlur(attention, (0, 0), 3)
    attention = cv2.normalize(attention, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return attention


def heuristic_grad_cam_b64(original_bgr: np.ndarray, enhanced: np.ndarray) -> str:
    attention = compute_attention_intensity(enhanced)
    heatmap = cv2.applyColorMap(attention, cv2.COLORMAP_TURBO)
    base = cv2.resize(original_bgr, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv2.INTER_AREA)
    overlay = cv2.addWeighted(base, 0.58, heatmap, 0.42, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    buffer = BytesIO()
    Image.fromarray(overlay_rgb).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
