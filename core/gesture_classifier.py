"""
gesture_classifier.py
-----------------------
Loads the trained model + label map and turns a landmark feature vector
into a predicted letter/gesture + confidence score.
"""

import json
import os

import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "gesture_model.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "model", "label_map.json")


class GestureClassifier:
    def __init__(self, model_path=MODEL_PATH, label_map_path=LABEL_MAP_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                f"Run training/train_model.py first."
            )
        self.model = tf.keras.models.load_model(model_path)

        with open(label_map_path, "r") as f:
            raw_map = json.load(f)
        # JSON keys are strings; convert back to int -> label
        self.label_map = {int(k): v for k, v in raw_map.items()}

    def predict(self, feature_vector):
        """
        feature_vector: np.array shape (63,)
        Returns: (predicted_label: str, confidence: float 0-1)
        """
        if feature_vector is None:
            return None, 0.0

        x = np.expand_dims(feature_vector, axis=0)
        probs = self.model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        label = self.label_map.get(idx, "?")
        return label, confidence
