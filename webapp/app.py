"""
app.py
-------
RUN THIS ON YOUR OWN MACHINE (needs a real webcam + trained model).

Flask web app for Stage 1: Sign -> Text -> Speech.

Before running:
    1. Collect data:  python data_collection/collect_landmarks.py
    2. Train model:   python training/train_model.py
    3. Run this app:  python webapp/app.py
    4. Open browser:  http://localhost:5000
"""

import os
import sys
import threading

import cv2
from flask import Flask, Response, jsonify, render_template, request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.hand_tracker import HandTracker
from core.gesture_classifier import GestureClassifier
from core.word_builder import WordBuilder
from speech.text_to_speech import TextToSpeech

app = Flask(__name__)

# ---- Shared state (single-user local demo; not multi-user safe) ----
tracker = HandTracker(max_hands=1)
word_builder = WordBuilder(hold_frames=15, confidence_threshold=0.85)
tts = TextToSpeech()

try:
    classifier = GestureClassifier()
    MODEL_READY = True
except FileNotFoundError as e:
    print(f"[WARNING] {e}")
    classifier = None
    MODEL_READY = False

state_lock = threading.Lock()
latest_state = {
    "letter": None,
    "confidence": 0.0,
    "word": "",
    "history": [],
    "model_ready": MODEL_READY,
}

camera = cv2.VideoCapture(0)


def generate_frames():
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        results = tracker.process(frame)
        frame = tracker.draw_landmarks(frame, results)

        letter, confidence = None, 0.0
        if MODEL_READY:
            feature_vector = HandTracker.extract_feature_vector(results)
            if feature_vector is not None:
                letter, confidence = classifier.predict(feature_vector)
                word_builder.update(letter, confidence)

        with state_lock:
            latest_state["letter"] = letter
            latest_state["confidence"] = round(confidence * 100, 1)
            latest_state["word"] = word_builder.current_word
            latest_state["history"] = list(word_builder.history)

        # overlay for the raw video feed too (nice for demos/screenshots)
        if letter:
            cv2.putText(frame, f"{letter} ({confidence*100:.1f}%)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/state")
def get_state():
    with state_lock:
        return jsonify(latest_state)


@app.route("/speak", methods=["POST"])
def speak():
    with state_lock:
        word = word_builder.commit_word()
    if word:
        threading.Thread(target=tts.speak, args=(word,), daemon=True).start()
    return jsonify({"spoken": word})


@app.route("/reset", methods=["POST"])
def reset():
    word_builder.reset()
    return jsonify({"status": "reset"})


@app.route("/backspace", methods=["POST"])
def backspace():
    word_builder.backspace()
    return jsonify({"word": word_builder.current_word})


if __name__ == "__main__":
    if not MODEL_READY:
        print("=" * 60)
        print("No trained model found yet - the app will run but won't")
        print("predict letters until you run training/train_model.py")
        print("=" * 60)
    app.run(debug=True, threaded=True)
