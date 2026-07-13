import hashlib
import math
from pathlib import Path

import cv2
import numpy as np

from backend.core.config import CLASS_LABELS, settings
from backend.ml.explainability.attention_alignment import orientation_attention_alignment
from backend.ml.explainability.causal_attribution import minutiae_causal_attribution
from backend.ml.explainability.grad_cam import compute_attention_intensity, heuristic_grad_cam_b64
from backend.ml.feature_engineering.texture import (
    extract_fusion_texture_vector,
    extract_fusion_texture_vector_multilbp,
    extract_texture_features,
)
from backend.ml.preprocessing.fingerprint import decode_image, enhance_fingerprint


MODEL_OUTPUT_LABELS = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O-", "O+"]


class PredictorError(ValueError):
    pass


class ChannelAttention:
    pass


class SpatialAttention:
    pass


class CBAM:
    pass


class RidgeVisionPredictor:
    def __init__(self) -> None:
        self.model_path = Path("models/ridgevision_model.keras")
        self.model_88_path = Path("models/ridgevision_b0_88_style_best.keras")
        self.model_91_path = Path("models/ridgevision_b3_91_style_best.keras")
        self.model = None
        self.model_88 = None
        self.model_91 = None
        self.model_image_size = (224, 224)
        self.tensorflow = None
        self.is_fusion_model = False
        self.is_ensemble_model = False

    def _load_trained_model(self) -> bool:
        if not self.model_path.exists():
            return False

        if self.model is None:
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise PredictorError("TensorFlow is required to use models/ridgevision_model.keras.") from exc

            self.tensorflow = tf

            custom_objects = self._custom_objects(tf)

            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects=custom_objects,
                compile=False,
            )

            input_shape = self.model.input_shape
            self.is_fusion_model = isinstance(input_shape, list) and len(input_shape) >= 2

            image_shape = input_shape[0] if self.is_fusion_model else input_shape

            if image_shape[1] and image_shape[2]:
                self.model_image_size = (int(image_shape[2]), int(image_shape[1]))

            self.model_texture_dim = None
            if self.is_fusion_model and input_shape[1][-1]:
                self.model_texture_dim = int(input_shape[1][-1])

            print("Loaded model:", self.model_path)
            print("Model input shape:", self.model.input_shape)
            print("Fusion model:", self.is_fusion_model)
            print("Image size:", self.model_image_size)
            print("Texture dim:", self.model_texture_dim)

        return True

    def _load_ensemble_models(self) -> bool:
        if not self.model_88_path.exists() or not self.model_91_path.exists():
            return False

        if self.model_88 is None or self.model_91 is None:
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise PredictorError(
                    "TensorFlow is required to use the trained ensemble .keras models."
                ) from exc

            self.tensorflow = tf
            custom_objects = self._custom_objects(tf)

            self.model_88 = tf.keras.models.load_model(
                self.model_88_path,
                custom_objects=custom_objects,
                compile=False,
            )
            self.model_91 = tf.keras.models.load_model(
                self.model_91_path,
                custom_objects=custom_objects,
                compile=False,
            )
            self.is_ensemble_model = True
            self.is_fusion_model = True

            print("Loaded ensemble model 88:", self.model_88_path)
            print("Loaded ensemble model 91:", self.model_91_path)

        return True

    def _custom_objects(self, tf):
        class ChannelAttention(tf.keras.layers.Layer):
            def __init__(self, ratio=8, **kwargs):
                super().__init__(**kwargs)
                self.ratio = ratio

            def build(self, input_shape):
                channels = input_shape[-1]
                reduced = max(channels // self.ratio, 1)
                self.shared_dense_1 = tf.keras.layers.Dense(reduced, activation="relu")
                self.shared_dense_2 = tf.keras.layers.Dense(channels)

            def call(self, inputs):
                avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
                max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
                avg_out = self.shared_dense_2(self.shared_dense_1(avg_pool))
                max_out = self.shared_dense_2(self.shared_dense_1(max_pool))
                return inputs * tf.nn.sigmoid(avg_out + max_out)

        class SpatialAttention(tf.keras.layers.Layer):
            def __init__(self, kernel_size=7, **kwargs):
                super().__init__(**kwargs)
                self.conv = tf.keras.layers.Conv2D(
                    1,
                    kernel_size=kernel_size,
                    padding="same",
                    activation="sigmoid",
                )

            def call(self, inputs):
                avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
                max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
                concat = tf.concat([avg_pool, max_pool], axis=-1)
                return inputs * self.conv(concat)

        class CBAM(tf.keras.layers.Layer):
            def __init__(self, ratio=8, kernel_size=7, **kwargs):
                super().__init__(**kwargs)
                self.channel_attention = ChannelAttention(ratio)
                self.spatial_attention = SpatialAttention(kernel_size)

            def call(self, inputs):
                x = self.channel_attention(inputs)
                return self.spatial_attention(x)

        return {
            "CBAM": CBAM,
            "ChannelAttention": ChannelAttention,
            "SpatialAttention": SpatialAttention,
        }

    def _probabilities_for_processed(self, image_bgr: np.ndarray, enhanced_gray: np.ndarray) -> dict[str, float]:
        """Shared branch logic (ensemble / trained / research mode) given an already
        preprocessed image. Used both by predict() and by explainability methods that
        need to re-run inference on perturbed copies of the image (e.g. minutiae
        causal ablation)."""
        if self._load_ensemble_models():
            return self._ensemble_model_probabilities(image_bgr=image_bgr, enhanced_gray=enhanced_gray)
        if self._load_trained_model():
            return self._trained_model_probabilities(image_bgr=image_bgr, enhanced_gray=enhanced_gray)

        features = extract_texture_features(enhanced_gray)
        # research mode scores from features + a hash of pixel bytes; re-derive bytes
        # from the (possibly perturbed) image so the ablation actually changes the score
        image_bytes = cv2.imencode(".png", image_bgr)[1].tobytes()
        return self._research_mode_probabilities(features, image_bytes)

    def probabilities_for_bgr_image(self, image_bgr: np.ndarray) -> dict[str, float]:
        """Public entry point: full preprocessing + inference for a raw BGR image
        array (as opposed to `predict`, which takes raw upload bytes and also builds
        the user-facing response payload). Used by explainability modules that need
        to re-run inference on perturbed/ablated copies of the fingerprint."""
        processed = enhance_fingerprint(image_bgr)
        return self._probabilities_for_processed(image_bgr, processed["enhanced"])

    def predict(self, image_bytes: bytes, explain: bool = False) -> dict:
        try:
            image = decode_image(image_bytes)
            processed = enhance_fingerprint(image)
            features = extract_texture_features(processed["enhanced"])
        except ValueError as exc:
            raise PredictorError(str(exc)) from exc

        probabilities = self._probabilities_for_processed(image, processed["enhanced"])

        if self.is_ensemble_model:
            inference_mode = "trained_ensemble"
        elif self.model is not None:
            inference_mode = "trained_model"
        else:
            inference_mode = "research_mode"

        predicted_class = max(probabilities, key=probabilities.get)
        attention_intensity = compute_attention_intensity(processed["enhanced"])
        heatmap = heuristic_grad_cam_b64(processed["original_bgr"], processed["enhanced"])

        result = {
            "predicted_class": predicted_class,
            "confidence": round(probabilities[predicted_class], 2),
            "all_probabilities": probabilities,
            "grad_cam_b64": heatmap,
            "features": {key: round(value, 4) for key, value in features.items()},
            "inference_mode": inference_mode,
            "model_type": "ensemble_fusion_cbam_texture"
            if self.is_ensemble_model
            else "fusion_cbam_texture"
            if self.is_fusion_model
            else "single_image_cnn",
            "disclaimer": settings.disclaimer,
        }

        if explain:
            oaas = orientation_attention_alignment(processed["enhanced"], attention_intensity)
            mca = minutiae_causal_attribution(
                self,
                image_bgr=processed["original_bgr"],
                enhanced_gray=processed["enhanced"],
                predicted_class=predicted_class,
                baseline_confidence=probabilities[predicted_class],
            )
            result["explainability"] = {
                "orientation_attention_alignment": oaas,
                "minutiae_causal_attribution": mca,
            }

        return result

    def _predict_fusion_model(
        self,
        model,
        image_bgr: np.ndarray,
        image_size: tuple[int, int],
        texture_vector: np.ndarray,
    ) -> np.ndarray:
        resized = cv2.resize(image_bgr, image_size, interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image_batch = np.expand_dims(image_rgb.astype("float32"), axis=0)
        texture_batch = np.expand_dims(texture_vector, axis=0)

        return model.predict(
            {
                "image_input": image_batch,
                "texture_input": texture_batch,
            },
            verbose=0,
        )[0]

    def _enhance_for_trained_texture(self, image_bgr: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, image_size, interpolation=cv2.INTER_AREA)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        responses = []
        for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=4.0,
                theta=theta,
                lambd=10.0,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F,
            )
            responses.append(cv2.filter2D(gray, cv2.CV_32F, kernel))

        enhanced = np.max(np.stack(responses, axis=0), axis=0)
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        return enhanced.astype(np.uint8)

    def _ensemble_model_probabilities(self, image_bgr: np.ndarray, enhanced_gray: np.ndarray) -> dict[str, float]:
        enhanced_224 = self._enhance_for_trained_texture(image_bgr, (224, 224))
        enhanced_300 = self._enhance_for_trained_texture(image_bgr, (300, 300))

        predictions_88 = self._predict_fusion_model(
            self.model_88,
            image_bgr=image_bgr,
            image_size=(224, 224),
            texture_vector=extract_fusion_texture_vector(enhanced_224),
        )
        predictions_91 = self._predict_fusion_model(
            self.model_91,
            image_bgr=image_bgr,
            image_size=(300, 300),
            texture_vector=extract_fusion_texture_vector_multilbp(enhanced_300),
        )

        predictions = (0.5 * predictions_88) + (0.5 * predictions_91)

        return self._probability_dict(predictions, MODEL_OUTPUT_LABELS)

    def _trained_model_probabilities(self, image_bgr: np.ndarray, enhanced_gray: np.ndarray) -> dict[str, float]:
        resized = cv2.resize(image_bgr, self.model_image_size, interpolation=cv2.INTER_AREA)
        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image_array = image_rgb.astype("float32")
        image_batch = np.expand_dims(image_array, axis=0)

        if self.is_fusion_model:
            if self.model_texture_dim == 74:
                trained_enhanced = self._enhance_for_trained_texture(image_bgr, (300, 300))
                texture_vector = extract_fusion_texture_vector_multilbp(trained_enhanced)
            else:
                trained_enhanced = self._enhance_for_trained_texture(image_bgr, (224, 224))
                texture_vector = extract_fusion_texture_vector(trained_enhanced)

            texture_batch = np.expand_dims(texture_vector, axis=0)

            predictions = self.model.predict(
                {
                    "image_input": image_batch,
                    "texture_input": texture_batch,
                },
                verbose=0,
            )[0]
        else:
            predictions = self.model.predict(image_batch, verbose=0)[0]

        return self._probability_dict(predictions, MODEL_OUTPUT_LABELS)

    def _research_mode_probabilities(self, features: dict[str, float], image_bytes: bytes) -> dict[str, float]:
        feature_vector = np.array(
            [
                features["lbp_uniformity"],
                features["lbp_peak"],
                features["glcm_contrast"],
                features["glcm_homogeneity"],
                features["glcm_energy"],
                features["ridge_density"],
                features["ridge_strength"],
                features["intensity_entropy"],
            ],
            dtype=np.float64,
        )

        seed = int(hashlib.sha256(image_bytes[:4096]).hexdigest()[:12], 16)

        scores = []
        for index, label in enumerate(CLASS_LABELS):
            phase = (seed % (997 + index * 13)) / 997.0
            weights = np.sin(np.arange(1, feature_vector.size + 1) * (index + 1.7 + phase))
            score = float(np.dot(feature_vector, weights))
            score += math.cos((index + 1) * features["ridge_density"] * 6.0)
            score += 0.18 if "+" in label and features["ridge_strength"] > 0.18 else 0
            scores.append(score)

        scores_array = np.array(scores, dtype=np.float64)
        scores_array = scores_array - scores_array.max()
        exp_scores = np.exp(scores_array)
        normalized = exp_scores / exp_scores.sum()

        return self._probability_dict(normalized, CLASS_LABELS)

    def _probability_dict(self, predictions: np.ndarray, labels: list[str]) -> dict[str, float]:
        return {
            label: round(float(probability * 100), 2)
            for label, probability in zip(labels, predictions)
        }


ridgevision_predictor = RidgeVisionPredictor()
