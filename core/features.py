"""Landmark normalisation shared by training, data collection and inference.

Every place that turns MediaPipe hand landmarks into the 63-number model
input must go through normalize_landmarks, so they cannot drift apart.
"""

import numpy as np

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3
MIDDLE_MCP = 9

# The training samples were recorded from a horizontally flipped (selfie)
# frame. Landmarks taken from an unflipped camera frame (the browser sends
# these) must be mirrored before they match that data.
BROWSER_MIRRORED = True


def normalize_landmarks(points, mirror=False):
    """Return a flat float32 vector of length 63.

    points: array-like of 21 (x, y, z) rows, MediaPipe image coordinates.
    The wrist becomes the origin and the wrist to middle-finger-MCP
    distance becomes 1. With mirror=True, x is negated after centring,
    which is the same as flipping the image left to right.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected {NUM_LANDMARKS} landmarks with x, y, z; got shape {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError("landmarks contain NaN or infinity")
    pts = pts - pts[0]
    if mirror:
        pts[:, 0] = -pts[:, 0]
    scale = max(float(np.linalg.norm(pts[MIDDLE_MCP])), 1e-6)
    return (pts / scale).astype(np.float32).flatten()
