import cv2
import numpy as np

from backend.ml.explainability.orientation import compute_orientation_field, singularity_map


def orientation_attention_alignment(
    enhanced_gray: np.ndarray,
    attention_map_gray: np.ndarray,
    block_size: int = 16,
) -> dict:
    """Orientation-Attention Alignment Score (OAAS).

    Most explainability sections just show a saliency/attention map and stop there.
    OAAS instead correlates *where the model attends* against an independently
    computed dermatoglyphic quantity -- ridge curvature / singularity likelihood
    derived from the orientation field (cores, deltas, high-curvature ridge zones).

    A high alignment score is evidence the network has learned to attend to
    biologically meaningful ridge structure rather than incidental texture or
    sensor artifacts -- turning the heatmap into a testable claim instead of a
    picture.
    """
    fields = compute_orientation_field(enhanced_gray, block_size=block_size)
    singularity = singularity_map(fields["coherence"])

    attention_resized = cv2.resize(
        attention_map_gray.astype(np.float32),
        (singularity.shape[1], singularity.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    a_lo, a_hi = float(attention_resized.min()), float(attention_resized.max())
    attention_norm = (attention_resized - a_lo) / (a_hi - a_lo + 1e-6)

    flat_a = attention_norm.flatten()
    flat_s = singularity.flatten()

    if flat_a.std() < 1e-6 or flat_s.std() < 1e-6:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(flat_a, flat_s)[0, 1])

    if attention_norm.size:
        threshold = np.percentile(attention_norm, 75)
        high_attention_mask = attention_norm > threshold
        overlap_ratio = float(
            singularity[high_attention_mask].mean() if high_attention_mask.any() else 0.0
        )
    else:
        overlap_ratio = 0.0

    return {
        "alignment_correlation": round(correlation, 3),
        "high_attention_singularity_overlap": round(overlap_ratio, 3),
    }
