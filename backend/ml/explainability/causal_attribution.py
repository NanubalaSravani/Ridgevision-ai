import cv2
import numpy as np

from backend.ml.explainability.minutiae import extract_minutiae


def _inpaint_out(image_bgr: np.ndarray, x: int, y: int, radius: int) -> np.ndarray:
    """Locally removes a small circular region around (x, y) via inpainting -- this
    erases a *specific biological structure* rather than an arbitrary square patch,
    which is what tells apart minutiae-causal ablation from generic occlusion.
    """
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 255, -1)
    return cv2.inpaint(image_bgr, mask, radius + 2, cv2.INPAINT_TELEA)


def minutiae_causal_attribution(
    predictor,
    image_bgr: np.ndarray,
    enhanced_gray: np.ndarray,
    predicted_class: str,
    baseline_confidence: float,
    max_points: int = 40,
) -> dict:
    """Minutiae-Causal Attribution (MCA).

    Standard explainability (GradCAM/SHAP/attention) answers "where does the model
    look?" -- a correlational question. MCA instead asks a causal one: "what happens
    to the prediction if this specific ridge ending or bifurcation did not exist?"

    For each detected minutia, the region is locally in-painted out (removing that
    structure while keeping surrounding ridge context intact) and inference is
    re-run. The resulting confidence drop is attributed to that exact minutia, then
    aggregated by structure type (ending vs. bifurcation) to produce a per-class,
    per-structure-type causal importance breakdown -- e.g. "bifurcations account for
    61% of this prediction's causal support, ridge endings 39%".

    This ties the explanation directly to dermatoglyphic anatomy instead of a
    generic saliency map, and doubles as a sanity check: if ablating minutiae barely
    moves the prediction, that's a red flag the model is relying on something other
    than genuine ridge structure (e.g. sensor artifacts or background).
    """
    minutiae = extract_minutiae(enhanced_gray, max_points=max_points)
    if not minutiae:
        return {
            "minutiae_count": 0,
            "ending_count": 0,
            "bifurcation_count": 0,
            "ending_attribution_pct": 0.0,
            "bifurcation_attribution_pct": 0.0,
            "top_minutiae": [],
        }

    scale_x = image_bgr.shape[1] / enhanced_gray.shape[1]
    scale_y = image_bgr.shape[0] / enhanced_gray.shape[0]
    radius = max(4, int(round(6 * (scale_x + scale_y) / 2)))

    results = []
    for point in minutiae:
        x = int(round(point["x"] * scale_x))
        y = int(round(point["y"] * scale_y))
        x = min(max(x, 0), image_bgr.shape[1] - 1)
        y = min(max(y, 0), image_bgr.shape[0] - 1)

        perturbed = _inpaint_out(image_bgr, x, y, radius)
        probs = predictor.probabilities_for_bgr_image(perturbed)
        drop = baseline_confidence - probs.get(predicted_class, 0.0)

        results.append({"x": point["x"], "y": point["y"], "type": point["type"], "confidence_drop": round(float(drop), 3)})

    endings = [r["confidence_drop"] for r in results if r["type"] == "ending"]
    bifurcations = [r["confidence_drop"] for r in results if r["type"] == "bifurcation"]

    ending_sum = sum(endings) if endings else 0.0
    bifurcation_sum = sum(bifurcations) if bifurcations else 0.0
    total = ending_sum + bifurcation_sum
    total = total if abs(total) > 1e-9 else 1.0

    top_minutiae = sorted(results, key=lambda r: abs(r["confidence_drop"]), reverse=True)[:10]

    return {
        "minutiae_count": len(results),
        "ending_count": len(endings),
        "bifurcation_count": len(bifurcations),
        "ending_attribution_pct": round(100 * ending_sum / total, 1),
        "bifurcation_attribution_pct": round(100 * bifurcation_sum / total, 1),
        "top_minutiae": top_minutiae,
    }
