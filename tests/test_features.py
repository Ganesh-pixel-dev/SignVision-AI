import numpy as np
import pytest

from core.features import FEATURE_DIM, normalize_landmarks
from training.dataset import load, recording_blocks, temporal_split


@pytest.fixture
def hand():
    rng = np.random.default_rng(0)
    return rng.uniform(0.2, 0.8, size=(21, 3)).astype(np.float32)


def test_shape_and_origin(hand):
    v = normalize_landmarks(hand)
    assert v.shape == (FEATURE_DIM,)
    assert np.allclose(v[:3], 0)
    assert np.isclose(np.linalg.norm(v.reshape(21, 3)[9]), 1.0, atol=1e-5)


def test_translation_invariant(hand):
    assert np.allclose(normalize_landmarks(hand), normalize_landmarks(hand + [0.2, -0.1, 0.05]), atol=1e-5)


def test_scale_invariant(hand):
    assert np.allclose(normalize_landmarks(hand), normalize_landmarks(hand * 0.4), atol=1e-5)


def test_mirror_matches_a_flipped_image(hand):
    # Flipping the camera frame maps x to 1 - x. The browser sends the
    # unflipped frame, the training data came from the flipped one.
    flipped = hand.copy()
    flipped[:, 0] = 1.0 - flipped[:, 0]
    assert np.allclose(normalize_landmarks(flipped), normalize_landmarks(hand, mirror=True), atol=1e-5)
    assert not np.allclose(normalize_landmarks(flipped), normalize_landmarks(hand), atol=1e-3)


def test_bad_input_rejected(hand):
    with pytest.raises(ValueError):
        normalize_landmarks(hand[:20])
    bad = hand.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError):
        normalize_landmarks(bad)


def test_degenerate_hand_does_not_divide_by_zero():
    v = normalize_landmarks(np.zeros((21, 3)))
    assert np.isfinite(v).all()


def test_training_csv_uses_the_same_normalisation():
    X, _ = load()
    pts = X.reshape(-1, 21, 3)
    assert np.allclose(pts[:, 0], 0, atol=1e-6)
    assert np.allclose(np.linalg.norm(pts[:, 9], axis=1), 1.0, atol=1e-4)


def test_temporal_split_has_gap_and_no_overlap():
    _, labels = load()
    tr, te = temporal_split(labels)
    assert not set(tr) & set(te)
    blocks = recording_blocks(labels)
    for b in set(blocks[te]):
        last_train = max(i for i in tr if blocks[i] == b)
        first_test = min(i for i in te if blocks[i] == b)
        assert first_test - last_train > 10
