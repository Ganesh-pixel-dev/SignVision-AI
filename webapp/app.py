"""Single-user sign to text demo that reads the server's own webcam.

Run: python webapp/app.py   (http://127.0.0.1:5000)

One background thread reads the camera, finds the hand, classifies the
letter and feeds the word builder. The browser polls /state and shows
the camera as an MJPEG stream. Settings come from environment variables,
see .env.example.
"""

import os
import sys
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.word_builder import WordBuilder

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))


class Recognizer(threading.Thread):
    """Reads frames, updates the shared state, keeps the latest JPEG."""

    def __init__(self, classifier, camera_index=CAMERA_INDEX):
        super().__init__(daemon=True)
        self.classifier = classifier
        self.camera_index = camera_index
        self.lock = threading.Lock()
        self.words = WordBuilder()
        self.letter, self.confidence = None, 0.0
        self.camera_ok = None
        self.jpeg = None
        self.halt = threading.Event()

    def run(self):
        from core.hand_tracker import HandTracker
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            self.camera_ok = False
            return
        self.camera_ok = True
        tracker = HandTracker(max_hands=1)
        while not self.halt.is_set():
            ok, frame = capture.read()
            if not ok:
                self.camera_ok = False
                break
            frame = cv2.flip(frame, 1)
            results = tracker.process(frame)
            frame = tracker.draw_landmarks(frame, results)
            self.step(HandTracker.extract_feature_vector(results))
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                with self.lock:
                    self.jpeg = buf.tobytes()
        capture.release()

    def step(self, features):
        """Handle one frame's features. None means no hand was found."""
        letter, confidence = None, 0.0
        if features is not None and self.classifier is not None:
            letter, confidence = self.classifier.predict(features)
        with self.lock:
            self.letter, self.confidence = letter, confidence
            self.words.update(letter, confidence)

    def snapshot(self):
        with self.lock:
            return {
                "letter": self.letter,
                "confidence": round(self.confidence * 100, 1),
                "word": self.words.current_word,
                "history": list(self.words.history),
                "model_ready": self.classifier is not None,
                "camera_ok": self.camera_ok,
            }


def create_app(recognizer, tts):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        def frames():
            while recognizer.camera_ok is not False:
                with recognizer.lock:
                    jpeg = recognizer.jpeg
                if jpeg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(0.04)
        return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/state")
    def state():
        return jsonify(recognizer.snapshot())

    @app.route("/speak", methods=["POST"])
    def speak():
        with recognizer.lock:
            word = recognizer.words.commit_word()
        if not word:
            return jsonify({"spoken": "", "ok": True})
        ok = tts.speak_async(word)
        error = None if ok else "Text to speech is not available on this machine."
        return jsonify({"spoken": word, "ok": ok, "error": error})

    @app.route("/reset", methods=["POST"])
    def reset():
        with recognizer.lock:
            recognizer.words.reset()
        return jsonify({"status": "reset"})

    @app.route("/backspace", methods=["POST"])
    def backspace():
        with recognizer.lock:
            recognizer.words.backspace()
            return jsonify({"word": recognizer.words.current_word})

    return app


if __name__ == "__main__":
    from speech.text_to_speech import TextToSpeech

    try:
        from core.gesture_classifier import GestureClassifier
        classifier = GestureClassifier()
    except FileNotFoundError as exc:
        print(f"[warning] {exc} Letters will not be recognised.")
        classifier = None

    recognizer = Recognizer(classifier)
    recognizer.start()
    app = create_app(recognizer, TextToSpeech())
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("WEBAPP_PORT", "5000")),
            threaded=True, debug=False)
