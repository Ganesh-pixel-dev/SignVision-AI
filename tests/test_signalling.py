import json
import time

import pytest

from call_app.server import create_app, match_phrases, valid_landmarks


class FakeClassifier:
    def predict_landmarks(self, landmarks, mirror=False):
        return "A", 0.99


POINTS = [[0.1 * i, 0.2, 0.0] for i in range(21)]


@pytest.fixture
def env(tmp_path):
    for name in ("HELLO", "THANK YOU", "NO"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"phrase": name, "fps": 20, "frames": []}))
    app, sio = create_app(FakeClassifier(), phrase_dir=str(tmp_path))

    def client():
        return sio.test_client(app)
    return app, client


def names(c):
    return [m["name"] for m in c.get_received()]


def join(c, room="r1"):
    c.emit("join", {"room": room})
    return c.get_received()


def test_join_and_peer_notified(env):
    app, new = env
    a, b = new(), new()
    first = join(a)
    assert first[0]["name"] == "joined" and first[0]["args"][0]["peers"] == []
    join(b)
    assert "peer_joined" in names(a)


def test_bad_room_names_rejected(env):
    _, new = env
    c = new()
    for bad in ["", "a b", "../x", "x" * 41, None, 5]:
        out = join(c, bad)
        assert out[0]["name"] == "join_error"


def test_third_person_refused(env):
    _, new = env
    a, b, c = new(), new(), new()
    join(a), join(b)
    out = join(c)
    assert out[0]["name"] == "join_error" and out[0]["args"][0]["reason"] == "room_full"


def test_leaving_frees_a_seat_and_notifies(env):
    app, new = env
    a, b, c = new(), new(), new()
    join(a), join(b)
    b.disconnect()
    assert "peer_left" in names(a)
    assert join(c)[0]["name"] == "joined"


def test_empty_room_is_deleted(env):
    app, new = env
    a = new()
    join(a)
    a.disconnect()
    assert app.rooms == {}


def test_reconnect_rejoins_cleanly(env):
    _, new = env
    a, b = new(), new()
    join(a), join(b)
    b.disconnect()
    a.get_received()
    b2 = new()
    assert join(b2)[0]["args"][0]["peers"] != []
    assert "peer_joined" in names(a)


def test_signal_relayed_only_to_the_other_peer(env):
    _, new = env
    a, b = new(), new()
    join(a), join(b), a.get_received()
    a.emit("signal", {"type": "offer", "sdp": {"sdp": "x"}, "room": "someone-elses"})
    got = b.get_received()
    assert got[0]["name"] == "signal" and "room" not in got[0]["args"][0]
    assert a.get_received() == []


def test_signal_from_outside_a_room_or_bad_type_ignored(env):
    _, new = env
    a, b = new(), new()
    join(b)
    a.emit("signal", {"type": "offer", "sdp": {}})
    join(a), b.get_received()
    a.emit("signal", {"type": "evil"})
    assert b.get_received() == []


def test_only_the_signer_gets_predictions(env):
    _, new = env
    a, b = new(), new()
    join(a), join(b)
    a.emit("toggle_sign_mode", {"enabled": True})
    a.get_received(), b.get_received()
    b.emit("hand_landmarks", {"landmarks": POINTS})
    assert b.get_received() == [] and a.get_received() == []
    a.emit("hand_landmarks", {"landmarks": POINTS})
    for client in (a, b):
        got = [m for m in client.get_received() if m["name"] == "sign_prediction"]
        assert got and got[0]["args"][0]["letter"] == "A"


def test_bad_landmarks_get_an_error_not_a_crash(env):
    _, new = env
    a = new()
    join(a)
    a.emit("toggle_sign_mode", {"enabled": True})
    a.get_received()
    for bad in [None, {"landmarks": "x"}, {"landmarks": POINTS[:5]}, {"landmarks": [[0, 0, "a"]] * 21}]:
        a.emit("hand_landmarks", bad)
        assert names(a) == ["error_message"]


def test_signer_leaving_turns_sign_mode_off(env):
    _, new = env
    a, b = new(), new()
    join(a), join(b)
    a.emit("toggle_sign_mode", {"enabled": True})
    b.get_received()
    a.disconnect()
    got = b.get_received()
    assert any(m["name"] == "sign_mode_changed" and m["args"][0]["enabled"] is False for m in got)


def test_toggle_requires_a_boolean(env):
    _, new = env
    a = new()
    join(a)
    a.emit("toggle_sign_mode", {"enabled": "yes"})
    assert a.get_received() == []


def test_commit_word_sends_the_typed_word(env):
    _, new = env
    a, b = new(), new()
    join(a), join(b)
    a.emit("toggle_sign_mode", {"enabled": True})
    a.get_received(), b.get_received()
    for _ in range(15):
        a.emit("hand_landmarks", {"landmarks": POINTS})
        time.sleep(0.045)
    a.get_received()
    a.emit("commit_word")
    got = [m for m in b.get_received() if m["name"] == "word_committed"]
    assert got[0]["args"][0]["word"] == "A"


def test_speech_matches_whole_phrases_in_order(env):
    _, new = env
    a, b = new(), new()
    join(a), join(b), a.get_received(), b.get_received()
    b.emit("speech_text", {"text": "well thank you and hello there"})
    got = {m["name"]: m["args"][0] for m in a.get_received()}
    assert got["play_sign"]["phrases"] == ["THANK YOU", "HELLO"]
    b.emit("speech_text", {"text": "I know nothing"})
    assert [m["name"] for m in a.get_received()] == ["speech_caption"]


def test_speech_input_is_validated(env):
    _, new = env
    a = new()
    join(a)
    for bad in [None, {"text": 5}, {"text": "   "}, "hello"]:
        a.emit("speech_text", bad)
        assert a.get_received() == []


def test_http_routes(env):
    app, _ = env
    c = app.test_client()
    assert c.get("/").status_code == 200
    assert c.get("/skeleton_data/hello").status_code == 200
    assert c.get("/skeleton_data/..%5C..%5Cmodel%5Clabel_map").status_code == 404
    assert c.get("/skeleton_data/NOPE").status_code == 404
    assert c.get("/available_phrases").get_json() == ["HELLO", "NO", "THANK YOU"]
    assert c.get("/config").get_json()["iceServers"][0]["urls"].startswith("stun:")


def test_ice_servers_from_env(monkeypatch):
    monkeypatch.setenv("ICE_SERVERS", '[{"urls": "turn:t.example:3478", "username": "u", "credential": "p"}]')
    app, _ = create_app()
    assert app.test_client().get("/config").get_json()["iceServers"][0]["username"] == "u"
    monkeypatch.setenv("ICE_SERVERS", "not json")
    with pytest.raises(RuntimeError):
        create_app()


def test_helpers():
    assert match_phrases("I know", ["NO"]) == []
    assert valid_landmarks(POINTS)
    assert not valid_landmarks(POINTS[:20])
