"""Loads the trained letter classifier and predicts from normalised landmarks."""

import json
import os

import numpy as np
import tensorflow as tf

from core.features import FEATURE_DIM, normalize_landmarks

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "gesture_model.keras")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "model", "label_map.json")


class GestureClassifier:
    def __init__(self, model_path=MODEL_PATH, label_map_path=LABEL_MAP_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model at {model_path}. Run training/train_model.py first."
            )
        self.model = tf.keras.models.load_model(model_path, compile=False)
        with open(label_map_path) as f:
            self.label_map = {int(k): v for k, v in json.load(f).items()}

    def predict(self, feature_vector):
        """feature_vector: 63 normalised numbers. Returns (letter, confidence 0-1)."""
        if feature_vector is None:
            return None, 0.0
        x = np.asarray(feature_vector, dtype=np.float32)
        if x.shape != (FEATURE_DIM,) or not np.isfinite(x).all():
            raise ValueError(f"expected {FEATURE_DIM} finite numbers")
        probs = self.model(x.reshape(1, -1), training=False).numpy()[0]
        best = int(np.argmax(probs))
        return self.label_map[best], float(probs[best])

    def predict_landmarks(self, landmarks, mirror=False):
        """landmarks: 21 rows of (x, y, z) as MediaPipe reports them."""
        return self.predict(normalize_landmarks(landmarks, mirror=mirror))
