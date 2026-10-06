"""Compare classifiers on held-out data and write model/metrics.json.

Run: python training/evaluate.py
"""

import importlib.util
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset import BASE_DIR, load, recording_blocks, temporal_split

METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
SEED = 42


def make_candidates():
    return {
        "logistic_regression": lambda: LogisticRegression(C=10, max_iter=2000),
        "knn_k5": lambda: KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "mlp_keras": make_keras,
    }


class KerasWrapper:
    def __init__(self, num_classes, seed=SEED):
        from training.train_model import build_model
        self.model = build_model(num_classes, seed)

    def fit(self, X, y):
        from training.train_model import EPOCHS
        self.model.fit(X, y, epochs=EPOCHS, batch_size=32, verbose=0)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X, verbose=0), axis=1)


def make_keras(seed=SEED):
    return KerasWrapper(num_classes=10, seed=seed)


def report(y_true, y_pred, classes):
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)), zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "per_letter": {
            c: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                "f1": round(float(f[i]), 4), "test_samples": int(support[i])}
            for i, c in enumerate(classes)
        },
        "confusion_matrix": {"labels": list(classes),
                             "rows_are_true_columns_are_predicted": confusion_matrix(y_true, y_pred, labels=range(len(classes))).tolist()},
    }


def main():
    X, labels = load()
    classes = sorted(set(labels))
    y = np.array([classes.index(l) for l in labels])

    tr, te = temporal_split(labels)
    rtr, rte = train_test_split(np.arange(len(y)), test_size=0.2, random_state=SEED, stratify=y)

    out = {
        "dataset": {
            "file": "training/dataset/landmarks.csv",
            "samples": int(len(y)),
            "per_letter": {c: int((y == i).sum()) for i, c in enumerate(classes)},
            "recording_blocks": int(len(set(recording_blocks(labels)))),
            "recorded_by": "one person, one hand, one camera",
        },
        "split": {
            "main": f"temporal: last 20% of each recording block held out, 10 samples dropped before it. "
                    f"{len(tr)} train, {len(te)} test.",
            "optimistic_for_comparison": f"random stratified 80/20. {len(rtr)} train, {len(rte)} test. Near-duplicate "
                                         f"neighbours land on both sides, so this number is inflated.",
            "caveat": "Every recording block is one continuous hold by the same person in the same session, "
                      "so even the temporal split does not test new people, hands, lighting or cameras.",
        },
        "models": {},
    }
    for name, make in make_candidates().items():
        if name == "mlp_keras" and importlib.util.find_spec("tensorflow") is None:
            print("tensorflow not installed, skipping mlp_keras")
            continue
        temporal = report(y[te], make().fit(X[tr], y[tr]).predict(X[te]), classes)
        rand = make().fit(X[rtr], y[rtr]).predict(X[rte])
        out["models"][name] = {
            "temporal_split": temporal,
            "random_split_accuracy": round(float(accuracy_score(y[rte], rand)), 4),
        }
        if name == "mlp_keras":
            accs = [accuracy_score(y[te], make(seed).fit(X[tr], y[tr]).predict(X[te])) for seed in range(5)]
            out["models"][name]["temporal_accuracy_over_5_seeds"] = {
                "mean": round(float(np.mean(accs)), 4), "min": round(float(min(accs)), 4), "max": round(float(max(accs)), 4)}
        print(f"{name}: temporal {temporal['accuracy']:.4f}  random {out['models'][name]['random_split_accuracy']:.4f}")

    with open(METRICS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", METRICS_PATH)


if __name__ == "__main__":
    main()
