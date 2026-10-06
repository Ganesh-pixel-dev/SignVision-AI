"""Signalling and translation server for the two-person call.

Run: python call_app/server.py   (http://127.0.0.1:5001)

The server relays WebRTC signalling between the two people in a room,
keeps the room's sign-mode state, classifies hand landmarks sent by the
signer, and turns the hearing person's speech into phrase names that the
deaf person's browser plays back as a skeleton animation.

Settings come from environment variables, see .env.example.
"""

import json
import mimetypes
import os
import re
import secrets
import sys
import time

from flask import Flask, abort, jsonify, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features import BROWSER_MIRRORED
from core.word_builder import WordBuilder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKELETON_LIB_DIR = os.path.join(BASE_DIR, "skeleton", "library")

ROOM_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
MAX_PEOPLE = 2
MAX_SPEECH_CHARS = 300
MAX_SIGNAL_BYTES = 20000
MIN_LANDMARK_INTERVAL = 0.04  # seconds; messages arriving faster are dropped
mimetypes.add_type("application/wasm", ".wasm")
DEFAULT_ICE_SERVERS = [{"urls": "stun:stun.l.google.com:19302"}]


def load_dotenv_if_present():
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(os.path.join(BASE_DIR, ".env"))


def ice_servers_from_env():
    """ICE_SERVERS is a JSON list. Without it, one public STUN server is used."""
    raw = os.environ.get("ICE_SERVERS", "").strip()
    if not raw:
        return DEFAULT_ICE_SERVERS
    try:
        servers = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ICE_SERVERS is not valid JSON: {exc}") from exc
    if not isinstance(servers, list) or not all(isinstance(s, dict) and "urls" in s for s in servers):
        raise RuntimeError('ICE_SERVERS must be a JSON list of objects with a "urls" key')
    return servers


def list_phrases(lib_dir):
    if not os.path.isdir(lib_dir):
        return []
    return sorted(f[:-5] for f in os.listdir(lib_dir) if f.endswith(".json"))


def match_phrases(spoken_text, phrases):
    """Phrases found as whole words in the text, in the order they were spoken."""
    found = []
    for phrase in phrases:
        m = re.search(r"\b" + re.escape(phrase) + r"\b", spoken_text, flags=re.IGNORECASE)
        if m:
            found.append((m.start(), phrase))
    return [p for _, p in sorted(found)]


def valid_landmarks(value):
    if not isinstance(value, list) or len(value) != 21:
        return False
    for point in value:
        if not isinstance(point, list) or len(point) != 3:
            return False
        if not all(isinstance(c, (int, float)) and not isinstance(c, bool) and abs(c) < 1e3 for c in point):
            return False
    return True


class Room:
    def __init__(self):
        self.members = []
        self.sign_mode = False
        self.signer = None
        self.words = WordBuilder()


def create_app(classifier=None, phrase_dir=SKELETON_LIB_DIR):
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
    origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
    socketio = SocketIO(app, cors_allowed_origins=origins or None)

    ice_servers = ice_servers_from_env()
    rooms = {}
    sid_room = {}
    last_landmark = {}

    def tell(message):
        emit("error_message", {"message": message})

    def room_of(sid):
        name = sid_room.get(sid)
        return (name, rooms[name]) if name in rooms else (None, None)

    def leave(sid):
        name, room = room_of(sid)
        sid_room.pop(sid, None)
        last_landmark.pop(sid, None)
        if room is None:
            return
        if sid in room.members:
            room.members.remove(sid)
        leave_room(name, sid=sid)
        if room.signer == sid:
            room.sign_mode, room.signer = False, None
            room.words.reset()
            emit("sign_mode_changed", {"enabled": False, "signer": None}, to=name)
        if room.members:
            emit("peer_left", {"sid": sid}, to=name)
        else:
            del rooms[name]

    @app.route("/")
    def index():
        return render_template("room.html")

    @app.route("/config")
    def config():
        return jsonify({"iceServers": ice_servers, "modelReady": classifier is not None})

    @app.route("/available_phrases")
    def available_phrases():
        return jsonify(list_phrases(phrase_dir))

    @app.route("/skeleton_data/<phrase>")
    def skeleton_data(phrase):
        name = phrase.upper()
        if name not in list_phrases(phrase_dir):
            abort(404)
        with open(os.path.join(phrase_dir, name + ".json")) as f:
            return jsonify(json.load(f))

    @socketio.on("join")
    def on_join(data):
        sid = request.sid
        name = data.get("room") if isinstance(data, dict) else None
        if not isinstance(name, str) or not ROOM_RE.match(name):
            return emit("join_error", {"reason": "bad_room",
                                       "message": "Room names use letters, digits, - and _ (up to 40 characters)."})
        if sid_room.get(sid) == name:
            return
        leave(sid)
        room = rooms.setdefault(name, Room())
        if len(room.members) >= MAX_PEOPLE:
            return emit("join_error", {"reason": "room_full",
                                       "message": f"Room {name} already has {MAX_PEOPLE} people."})
        others = list(room.members)
        room.members.append(sid)
        sid_room[sid] = name
        join_room(name)
        emit("joined", {"room": name, "sid": sid, "peers": others, "signMode": room.sign_mode,
                        "signer": room.signer, "modelReady": classifier is not None})
        emit("peer_joined", {"sid": sid}, to=name, include_self=False)

    @socketio.on("leave")
    def on_leave(data=None):
        leave(request.sid)

    @socketio.on("disconnect")
    def on_disconnect(*_):
        leave(request.sid)

    @socketio.on("signal")
    def on_signal(data):
        name, room = room_of(request.sid)
        if room is None or not isinstance(data, dict) or data.get("type") not in ("offer", "answer", "candidate"):
            return
        if len(json.dumps(data)) > MAX_SIGNAL_BYTES:
            return tell("Signalling message too large.")
        payload = {k: data[k] for k in ("type", "sdp", "candidate") if k in data}
        emit("signal", payload, to=name, include_self=False)

    @socketio.on("toggle_sign_mode")
    def on_toggle(data):
        name, room = room_of(request.sid)
        if room is None or not isinstance(data, dict) or not isinstance(data.get("enabled"), bool):
            return
        enabled = data["enabled"]
        room.sign_mode = enabled
        room.signer = request.sid if enabled else None
        room.words.reset()
        emit("sign_mode_changed", {"enabled": enabled, "signer": room.signer}, to=name)

    @socketio.on("hand_landmarks")
    def on_hand_landmarks(data):
        name, room = room_of(request.sid)
        if room is None or not room.sign_mode or room.signer != request.sid:
            return
        if classifier is None:
            return tell("No trained model found. Run training/train_model.py.")
        landmarks = data.get("landmarks") if isinstance(data, dict) else None
        if not valid_landmarks(landmarks):
            return tell("Landmarks must be 21 points of [x, y, z].")
        now = time.monotonic()
        if now - last_landmark.get(request.sid, 0) < MIN_LANDMARK_INTERVAL:
            return
        last_landmark[request.sid] = now

        letter, confidence = classifier.predict_landmarks(landmarks, mirror=BROWSER_MIRRORED)
        word = room.words.update(letter, confidence)
        emit("sign_prediction", {"letter": letter, "confidence": round(confidence * 100, 1), "word": word}, to=name)

    @socketio.on("hand_lost")
    def on_hand_lost(data=None):
        name, room = room_of(request.sid)
        if room is None or room.signer != request.sid:
            return
        word = room.words.update(None, 0.0)
        emit("sign_prediction", {"letter": None, "confidence": 0.0, "word": word}, to=name)

    @socketio.on("commit_word")
    def on_commit(data=None):
        name, room = room_of(request.sid)
        if room is None or room.signer != request.sid:
            return
        emit("word_committed", {"word": room.words.commit_word()}, to=name)

    @socketio.on("backspace")
    def on_backspace(data=None):
        name, room = room_of(request.sid)
        if room is None or room.signer != request.sid:
            return
        room.words.backspace()
        emit("sign_prediction", {"letter": None, "confidence": 0.0, "word": room.words.current_word}, to=name)

    @socketio.on("speech_text")
    def on_speech_text(data):
        name, room = room_of(request.sid)
        text = data.get("text") if isinstance(data, dict) else None
        if room is None or not isinstance(text, str):
            return
        text = text.strip()[:MAX_SPEECH_CHARS]
        if not text:
            return
        emit("speech_caption", {"text": text}, to=name)
        phrases = match_phrases(text, list_phrases(phrase_dir))
        if phrases:
            emit("play_sign", {"phrases": phrases}, to=name)

    app.rooms = rooms
    return app, socketio


def load_classifier():
    try:
        from core.gesture_classifier import GestureClassifier
        return GestureClassifier()
    except FileNotFoundError as exc:
        print(f"[warning] {exc} Sign recognition is off.")
        return None


if __name__ == "__main__":
    load_dotenv_if_present()
    app, socketio = create_app(load_classifier())
    socketio.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5001")),
                 debug=os.environ.get("FLASK_DEBUG") == "1", allow_unsafe_werkzeug=True)
