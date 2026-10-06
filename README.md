# SignVision AI

Real-time Indian Sign Language recognition + a two-person video call
where one participant can be deaf: their signs are translated to
text/speech for the hearing participant, and the hearing participant's
speech is translated into an animated sign-language skeleton for the
deaf participant.

## What's actually in this project (honest scope)

| Feature | Status |
|---|---|
| Hand landmark detection (MediaPipe) | ✅ Fully working |
| Letter/gesture classification (neural network on landmarks) | ✅ Fully working, you train it on your own recorded data |
| Word building + text-to-speech | ✅ Fully working |
| Standalone sign→text web app (Stage 1) | ✅ Fully working |
| Skeleton recording/playback for common phrases | ✅ Fully working |
| Two-person video call (WebRTC) | ✅ Working on localhost / same network |
| Sign-mode toggle (shared, both screens change) | ✅ Fully working |
| Speech → sign-skeleton translation | ✅ Working, but limited to a **library of phrases you record yourself** — this is a deliberate, honest design choice (see below) |
| "True" AI-generated sign language for arbitrary sentences | ❌ Not implemented — this is an unsolved research problem industry-wide, not something skipped due to laziness |

**Why speech→sign uses a recorded phrase library instead of generating
new signs:** Generating fluent, grammatically correct sign language
from arbitrary text is an open research problem — no one (including
well-funded companies) has fully solved it. Instead, this project
records a real person performing common words/phrases once via
MediaPipe, saves the landmark motion, and replays it as a skeleton
animation whenever that word is needed. This is the same practical
approach real sign-language-tech products use for their MVP stage.

## Project structure

```
SignVision_AI/
├── core/                     # Shared building blocks
│   ├── hand_tracker.py       # MediaPipe wrapper + landmark normalization
│   ├── gesture_classifier.py # Loads trained model, predicts letter/gesture
│   └── word_builder.py       # Smooths noisy predictions into clean words
│
├── data_collection/
│   └── collect_landmarks.py  # Webcam tool to record training samples
│
├── training/
│   ├── dataset/               # Your recorded landmarks.csv goes here
│   └── train_model.py         # Trains the neural network classifier
│
├── model/                     # Trained model + label map saved here
│
├── speech/
│   ├── text_to_speech.py     # Offline TTS (pyttsx3)
│   └── speech_to_text.py     # Mic -> text (used standalone/testing)
│
├── skeleton/
│   ├── recorder.py            # Record a phrase as a skeleton animation
│   ├── player.py               # Preview a recorded phrase (CLI)
│   └── library/                # Saved phrase recordings (.json)
│
├── webapp/                    # STAGE 1: standalone sign->text demo
│   ├── app.py
│   ├── templates/index.html
│   └── static/
│
└── call_app/                  # STAGE 4: full 2-person call demo
    ├── server.py               # Signaling + translation bridge server
    ├── templates/room.html
    └── static/
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note on PyAudio:** if `pip install pyaudio` fails on Windows, install
via `pip install pipwin && pipwin install pyaudio`. On Mac: `brew
install portaudio` first. On Linux: `sudo apt install portaudio19-dev`
first.

## Step-by-step: building your own working demo

### 1. Collect training data (needs your webcam)

```bash
python data_collection/collect_landmarks.py
```

By default this records 10 classes (A-J). Hold up each sign and press
its letter key to save a sample. Aim for **150-300 samples per class**,
varying hand angle and distance a bit. Edit the `CLASSES` list at the
top of the script to add more letters/gestures once the pipeline works.

### 2. Train the classifier

```bash
python training/train_model.py
```

This trains a small feed-forward neural network on the landmark
vectors (not raw images — faster and more accurate for this task, see
comments in `training/train_model.py` for why). Saves the model to
`model/gesture_model.h5`.

### 3. Try the standalone sign→text demo

```bash
python webapp/app.py
```

Open `http://localhost:5000`. Sign a letter, watch it build into a
word, click Speak.

### 4. Record some sign phrases (for the speech→sign side)

```bash
python skeleton/recorder.py "HELLO"
python skeleton/recorder.py "THANK YOU"
python skeleton/recorder.py "YES"
python skeleton/recorder.py "NO"
python skeleton/recorder.py "HELP"
```

Press SPACE to start, sign the phrase, press SPACE again to stop &
save. Preview any recording with:

```bash
python skeleton/player.py "HELLO"
```

### 5. Run the full two-person call demo

```bash
python call_app/server.py
```

Open `http://localhost:5001` in **two different browser tabs** (or two
devices on the same network, using your machine's local IP instead of
localhost). Use **Chrome** — the speech recognition piece relies on
the Web Speech API, which Chrome supports best.

1. Both tabs: enter the same room name, click Join.
2. Toggle **Sign Mode** on (either side) — both screens switch.
3. In one tab, hold up a trained hand sign — the other tab shows the
   caption + word building live.
4. In the other tab, speak one of your recorded phrases (e.g. say
   "hello") — the first tab shows an animated skeleton performing it.

## Known limitations (be upfront about these if presenting this)

- **Speech→sign is phrase-matching, not generation.** Only phrases
  you've recorded will animate; anything else just shows as a caption.
- **WebRTC here uses only a public STUN server, no TURN server.**
  Works reliably on localhost or the same network. Calling across
  different networks/firewalls may fail without a TURN server — that's
  a networking/infrastructure addition, not an AI limitation.
- **Single active room state** in `call_app/server.py` — fine for a
  demo, would need per-room isolation for many simultaneous calls.
- **Landmark-based classification works best for static hand shapes**
  (like fingerspelling letters). Signs that are primarily defined by
  *motion* (many real ISL words) would need a sequence model (e.g. an
  LSTM over landmark sequences) rather than a single-frame classifier
  — a natural next step once the letter-level pipeline is solid.

## Suggested next steps / future enhancements

- Expand from 10 letters to the full alphabet, then to common words
- Add a sequence model (LSTM/GRU on landmark sequences) for motion-based signs
- Smooth skeleton playback transitions between consecutive phrases
- Add a TURN server for cross-network calling
- Mobile app version
