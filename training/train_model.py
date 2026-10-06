"""
train_model.py
----------------
RUN THIS ON YOUR OWN MACHINE after collecting data with
data_collection/collect_landmarks.py.

Trains a small feed-forward neural network (deep learning classifier)
on the 63-number hand-landmark vectors to recognize gesture classes.

Why not a CNN on raw images here?
Because our input is already a compact, meaningful feature vector
(landmark coordinates), not pixels. A CNN is designed to learn spatial
filters from images; feeding it 63 numbers wastes its strengths. A
dense (fully connected) network is the correct deep learning tool for
this kind of tabular/coordinate input, trains in seconds/minutes
instead of hours, and typically reaches higher accuracy with far less
data than an image-based CNN for this task.

Outputs:
    model/gesture_model.h5      - trained Keras model
    model/label_map.json        - maps class index -> letter/gesture name
"""

import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models, callbacks

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(BASE_DIR, "training", "dataset", "landmarks.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.h5")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.json")


def build_model(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if not os.path.exists(DATA_CSV):
        print(f"ERROR: Dataset not found at {DATA_CSV}")
        print("Run data_collection/collect_landmarks.py first.")
        return

    df = pd.read_csv(DATA_CSV)
    print(f"Loaded {len(df)} samples across classes: {sorted(df['label'].unique())}")

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y_raw = df["label"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(input_dim=X.shape[1], num_classes=num_classes)
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=15, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal validation accuracy: {val_acc * 100:.2f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)

    label_map = {int(i): label for i, label in enumerate(encoder.classes_)}
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Label map saved to: {LABEL_MAP_PATH}")

    if val_acc < 0.90:
        print("\nTip: accuracy below 90%. Try collecting more samples per class,")
        print("varying hand angle/lighting, or double-checking mislabeled samples.")


if __name__ == "__main__":
    main()
