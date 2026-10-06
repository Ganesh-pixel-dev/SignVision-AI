import numpy as np
import pytest

from core.features import normalize_landmarks
from core.gesture_classifier import GestureClassifier
from training.dataset import load


@pytest.fixture(scope="module")
def clf():
    return GestureClassifier()


def test_returns_known_letter_and_probability(clf):
    X, _ = load()
    letter, conf = clf.predict(X[500])
    assert letter in "ABCDEFGHIJ" and 0 < conf <= 1


def test_none_gives_no_prediction(clf):
    assert clf.predict(None) == (None, 0.0)


def test_rejects_wrong_shape_and_nan(clf):
    with pytest.raises(ValueError):
        clf.predict(np.zeros(62))
    with pytest.raises(ValueError):
        clf.predict(np.full(63, np.nan))


def test_shipped_model_fits_its_own_training_data(clf):
    # Not an accuracy claim: this only catches a model file that does not
    # match the label map or the feature layout.
    X, y = load()
    idx = np.arange(0, len(y), 20)
    correct = sum(clf.predict(X[i])[0] == y[i] for i in idx)
    assert correct / len(idx) > 0.9


def test_browser_landmarks_give_the_same_answer_as_training_vector(clf):
    # Rebuild raw browser landmarks from a training vector: undo the
    # normalisation with an arbitrary shift and scale, and un-flip x.
    X, _ = load()
    v = X[1000].reshape(21, 3)
    selfie = v * 0.25 + np.array([0.5, 0.6, 0.0])
    browser = selfie.copy()
    browser[:, 0] = 1.0 - browser[:, 0]
    assert np.allclose(normalize_landmarks(browser, mirror=True), X[1000], atol=1e-4)
    assert clf.predict_landmarks(browser, mirror=True) == clf.predict(normalize_landmarks(selfie))
