# Overnight fixes

Branch `overnight-fixes`. The snapshot of your original code is the first commit on `main`/this history (`Snapshot before overnight fixes`). The old Keras model `gesture_model.h5` is in that commit.

## What was broken

Recognition
1. **Mirroring bug in the call app.** Training data came from a flipped frame (`cv2.flip`), but the browser sent landmarks from the unflipped video. x was mirrored relative to training, so the call app was classifying mirrored hands.
2. **Normalisation was written twice** (Python and JavaScript). Only luck kept them equal. No test.
3. **The reported accuracy was a random split** of near-duplicate samples. Measured here: 97 to 99% random split, 72 to 82% when the end of each recording is held out.
4. **Word builder:** a one-frame flicker mid-hold typed a letter twice; and because "no hand" frames were never reported, you could not type a double letter (the builder never learned the hand had left).
5. `train_model.py` stopped on the validation set it also reported, so the number was tuned on itself.

Call app
6. Both people's browsers ran hand tracking and fed the same word builder when sign mode was on.
7. Room name came from every message (client trusted): anyone could emit into any room. No validation of any event; a malformed message crashed the handler.
8. No limit on room size, nothing on leave or disconnect, no reconnect handling. One shared state per room never cleaned up.
9. `/skeleton_data/<phrase>` joined the user string into a file path (path traversal to any `.json`).
10. Phrase matching was a substring test: "no" matched "know" and "nothing".
11. `SECRET_KEY` hard coded, CORS `*`, `debug=True`.
12. Camera or mic refused: unhandled promise rejection, page silently dead. Speech recognition errors only logged.
13. Skeleton playback: 468 face dots drawn per frame, canvas square (stretched the 4:3 recording), hands flickered on frames where MediaPipe lost them.
14. MediaPipe loaded from a CDN, so sign mode needed internet.
15. README said calls work "on the same network" by LAN IP; browsers refuse camera access on plain http off localhost.

Webapp
16. Every `/video_feed` client started its own camera read loop. Camera opened at import. `debug=True` with the reloader (opens the camera twice, exposes the Werkzeug debugger). TTS engine created once and used from many threads. Failures ignored.

Other
17. README claimed things not in the code ("fully working" without evidence, accuracy-adjacent language, unused `matplotlib`/`eventlet` pins).

## What I changed

(commit hashes: run `git log --oneline`)
- **Shared normalisation** `core/features.py` with a `mirror` flag; the server now receives raw landmarks and normalises them with the same code as training. Test proves browser-with-mirror equals flipped-image.
- **Evaluation** `training/evaluate.py`: random vs temporal split (blocks of consecutive same-letter rows, with a gap), logistic regression and kNN baselines against the Keras net over 5 seeds, writes `model/metrics.json`. `training/train_model.py` now trains for fixed epochs on all data. Model saved as `gesture_model.keras`.
- **Word builder** rewritten with a release counter. `update(None, 0)` reports no hand.
- **Server** rewritten: per-room state, two people max, server-side room membership, signer role, validated payloads, rate limit on landmarks, leave/disconnect/reconnect, whole-word phrase matching in spoken order, whitelist for skeleton files, `/config` with ICE servers from `ICE_SERVERS`, secret and CORS from env, debug off, bound to 127.0.0.1.
- **Webapp**: one camera thread, state under a lock, honest camera and model states, TTS failures reported.
- **UI** rewritten for both apps: plain high-contrast layout, one accent colour, text labels for every state (who signs, what was recognised, confidence number plus meter), large captions, aria-live regions, keyboard operable with visible focus, works at 390px, no autoplay of the sign animation under reduced motion. Camera/mic refused, no speech support, connection lost, room full and TURN failure each show a message. Screenshots in `docs/screenshots/` (before and after).
- MediaPipe Hands vendored into `call_app/static/vendor/` (about 22 MB).
- `.env.example`, `.gitignore`, requirements cleaned.

## Results

- `python -m pytest`: 46 passed (word builder, normalisation and mirror, dataset split, prediction wrapper, signalling events with the Flask-SocketIO test client, webapp routes).
- Held-out accuracy (temporal split, 392 test samples, same person and session): logistic regression 73.5%, kNN 71.7%, Keras net 74.5% at seed 42 and 81.8% mean over 5 seeds (79.6 to 83.7). Random split for comparison: 97.7 to 99.3%.
- Per letter and confusion matrix: `model/metrics.json`. Worst: B (0% recall: the end of the B recording reads as C), J precision 0.36, I recall 0.57.
- Browser flow checked with Edge and a fake camera: two pages join, sign role and speaker role show correctly, a spoken "hello and thank you" plays the HELLO skeleton, a third person is refused, leaving updates the other screen, no horizontal scroll at 390px.

## Still not done

- No real webcam test. Live recognition through the browser is untested; only the maths is.
- Only A to J and five phrases. No evidence about other people's hands.
- Your data is one recording block per letter, so I cannot measure session-to-session variation. Held-out numbers are still optimistic.
- The B recording drifts. Probably re-record B (and I, J).
- No TURN server. Cross-network calls are not verified.
- The calls over LAN need HTTPS for the camera. I did not set that up.
- I did not run the skeleton recorder or the data collector (webcam). They are unchanged except that normalisation now lives in `core/features.py`.
- `speech/speech_to_text.py` and `skeleton/player.py` untouched and unaudited.
- Browser speech recognition is Chrome/Edge only and needs internet.

## Check yourself

1. Real webcam: run the call app, sign each of A to J in the call, watch the letter and confidence. If letters come out mirrored-wrong (e.g. everything wrong in the same way), flip `BROWSER_MIRRORED` in `core/features.py` and tell me.
2. Do the five phrase animations look right (the hand frames are patched by copying the nearest detected hand frame)?
3. Say "hello" and "thank you" into the mic in Chrome.
4. TTS in the webapp (`pyttsx3`) on your machine.
5. Re-record B with a couple of separate sittings and rerun `training/evaluate.py`.
6. Note: I replaced the global TensorFlow install's broken package inside `.venv` only (the one under your user site-packages had no `__init__`).
