"""Fit the shipped classifier on all recorded samples.

Run: python training/train_model.py
Held-out numbers come from training/evaluate.py, which fits the same
recipe on the train part of a split. This script uses every sample, so
the saved model has seen the test part too and cannot be scored again.

Writes model/gesture_model.keras and model/label_map.json.
"""

import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features import FEATURE_DIM
from training.dataset import BASE_DIR, load

MODEL_PATH = os.path.join(BASE_DIR, "model", "gesture_model.keras")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "model", "label_map.json")
EPOCHS = 60  # fixed: there is no clean validation set to stop early on
SEED = 42


def build_model(num_classes, seed=SEED):
    tf.keras.utils.set_random_seed(seed)
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Input(shape=(FEATURE_DIM,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    return model


def main():
    X, labels = load()
    classes = sorted(set(labels))
    y = np.array([classes.index(l) for l in labels])
    print(f"{len(y)} samples, classes: {classes}")

    model = build_model(len(classes))
    model.fit(X, y, epochs=EPOCHS, batch_size=32, verbose=0)
    model.save(MODEL_PATH)
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump({str(i): c for i, c in enumerate(classes)}, f, indent=2)
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
