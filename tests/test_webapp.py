import numpy as np

from webapp.app import Recognizer, create_app


class FakeClassifier:
    def predict(self, features):
        return "B", 0.97


class FakeTTS:
    def __init__(self, ok=True):
        self.ok = ok
        self.said = []

    def speak_async(self, text):
        self.said.append(text)
        return self.ok


def make(tts=None, classifier=FakeClassifier()):
    rec = Recognizer(classifier)
    rec.words.hold_frames = 3
    return rec, create_app(rec, tts or FakeTTS()).test_client()


def test_frames_build_a_word_and_speak_clears_it():
    tts = FakeTTS()
    rec, client = make(tts)
    for _ in range(3):
        rec.step(np.zeros(63))
    assert client.get("/state").get_json()["word"] == "B"
    assert client.post("/speak").get_json() == {"spoken": "B", "ok": True, "error": None}
    assert tts.said == ["B"]
    state = client.get("/state").get_json()
    assert state["word"] == "" and state["history"] == ["B"]


def test_speak_with_nothing_typed_does_not_call_tts():
    tts = FakeTTS()
    _, client = make(tts)
    assert client.post("/speak").get_json()["spoken"] == ""
    assert tts.said == []


def test_tts_failure_is_reported_not_raised():
    rec, client = make(FakeTTS(ok=False))
    for _ in range(3):
        rec.step(np.zeros(63))
    body = client.post("/speak").get_json()
    assert body["ok"] is False and "Text to speech" in body["error"]


def test_no_hand_frames_release_the_letter():
    rec, client = make()
    rec.words.release_frames = 2
    for _ in range(3):
        rec.step(np.zeros(63))
    rec.step(None), rec.step(None)
    for _ in range(3):
        rec.step(np.zeros(63))
    assert client.get("/state").get_json()["word"] == "BB"


def test_backspace_and_reset():
    rec, client = make()
    for _ in range(3):
        rec.step(np.zeros(63))
    assert client.post("/backspace").get_json() == {"word": ""}
    for _ in range(3):
        rec.step(None)
    client.post("/reset")
    assert client.get("/state").get_json()["word"] == ""


def test_state_reports_missing_model():
    _, client = make(classifier=None)
    assert client.get("/state").get_json()["model_ready"] is False
