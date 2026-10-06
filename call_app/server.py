"""
server.py
----------
RUN THIS ON YOUR OWN MACHINE.

Signaling + translation-bridge server for the 2-person call.

What this server actually does:
1. WebRTC signaling relay (offer/answer/ICE candidates) so two browser
   tabs/devices can establish a direct peer-to-peer video call.
2. Relays the shared "sign mode" toggle to both participants at once.
3. Receives hand-landmark vectors from the deaf participant's browser
   (extracted client-side using MediaPipe Hands via CDN), runs them
   through the SAME trained classifier from core/, and sends the
   predicted letter/word back to both participants (captions on the
   hearing side).
4. Receives recognized speech text from the hearing participant's
   browser (Web Speech API) and matches it against the recorded
   skeleton phrase library, telling the deaf participant's browser
   which skeleton animation to play.

NOTE ON SCOPE: This uses Python's `socket.io` rooms for signaling,
which works great on localhost or the same network for a demo/college
project. A production version would add a TURN server for NAT
traversal across different networks - that's an infrastructure
concern, not an AI/CV one, so it's intentionally left out here.
"""

import json
import os
import sys

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit, join_room

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.gesture_classifier import GestureClassifier
from core.word_builder import WordBuilder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKELETON_LIB_DIR = os.path.join(BASE_DIR, "skeleton", "library")

app = Flask(__name__)
app.config["SECRET_KEY"] = "signvision-dev-secret"
socketio = SocketIO(app, cors_allowed_origins="*")

# One shared word-builder per room (simple demo scope: single active room state)
room_word_builders = {}

try:
    classifier = GestureClassifier()
    MODEL_READY = True
except FileNotFoundError:
    classifier = None
    MODEL_READY = False


def get_available_phrases():
    if not os.path.exists(SKELETON_LIB_DIR):
        return []
    return sorted(
        f[:-5] for f in os.listdir(SKELETON_LIB_DIR) if f.endswith(".json")
    )


def match_phrase(spoken_text):
    """
    Very simple matcher: checks if any recorded phrase appears as a
    whole word/phrase inside the spoken text. Swap this for something
    smarter (fuzzy matching, stemming) as your phrase library grows.
    """
    if not spoken_text:
        return None
    spoken_upper = spoken_text.upper()
    for phrase in get_available_phrases():
        if phrase in spoken_upper:
            return phrase
    return None


@app.route("/")
def index():
    return render_template("room.html", model_ready=MODEL_READY)


@app.route("/skeleton_data/<phrase>")
def skeleton_data(phrase):
    path = os.path.join(SKELETON_LIB_DIR, f"{phrase.upper()}.json")
    if not os.path.exists(path):
        return jsonify({"error": "not found"}), 404
    with open(path, "r") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/available_phrases")
def available_phrases():
    return jsonify(get_available_phrases())


# ---------------- Socket.IO signaling ----------------

@socketio.on("join")
def on_join(data):
    room = data["room"]
    join_room(room)
    room_word_builders.setdefault(room, WordBuilder())
    emit("peer_joined", {"sid": request.sid}, room=room, include_self=False)


@socketio.on("signal")
def on_signal(data):
    """Relay WebRTC offer/answer/ICE candidates to the other peer in the room."""
    room = data["room"]
    emit("signal", data, room=room, include_self=False)


@socketio.on("toggle_sign_mode")
def on_toggle_sign_mode(data):
    room = data["room"]
    enabled = data["enabled"]
    emit("sign_mode_changed", {"enabled": enabled}, room=room, include_self=False)
    emit("sign_mode_changed", {"enabled": enabled}, room=room, include_self=True)


@socketio.on("hand_landmarks")
def on_hand_landmarks(data):
    """
    data = {"room": ..., "features": [63 floats]}
    Sent continuously from the deaf participant's browser while in sign mode.
    """
    room = data["room"]
    if not MODEL_READY:
        return
    features = data.get("features")
    if not features or len(features) != 63:
        return

    label, confidence = classifier.predict(__import__("numpy").array(features, dtype="float32"))
    wb = room_word_builders.setdefault(room, WordBuilder())
    word = wb.update(label, confidence)

    emit("sign_prediction", {
        "letter": label,
        "confidence": round(confidence * 100, 1),
        "word": word,
    }, room=room)  # send to both, including the signer (so they see their own captions)


@socketio.on("commit_word")
def on_commit_word(data):
    room = data["room"]
    wb = room_word_builders.setdefault(room, WordBuilder())
    word = wb.commit_word()
    emit("word_committed", {"word": word}, room=room)


@socketio.on("speech_text")
def on_speech_text(data):
    """
    data = {"room": ..., "text": "hello how are you"}
    Sent from the hearing participant's browser via Web Speech API.
    """
    room = data["room"]
    text = data.get("text", "")
    matched_phrase = match_phrase(text)
    emit("speech_caption", {"text": text}, room=room)
    if matched_phrase:
        emit("play_sign", {"phrase": matched_phrase}, room=room)


if __name__ == "__main__":
    if not MODEL_READY:
        print("[WARNING] No trained model found - sign predictions disabled.")
        print("Run training/train_model.py first for full functionality.")
    socketio.run(app, debug=True, port=5001)
