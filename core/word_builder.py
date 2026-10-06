"""
word_builder.py
-----------------
Raw per-frame predictions are noisy (a sign held for 2 seconds might
flicker between "A" and "S" a few times, or briefly show "?" during
hand movement). This module smooths that into a clean, deliberate
letter-by-letter word builder.

Logic:
- Keep a short rolling buffer of recent predictions.
- Only "confirm" a letter once the SAME label has been the top
  prediction for `hold_frames` consecutive frames AND confidence
  stays above `confidence_threshold`.
- Ignore repeated confirmations of the same letter back-to-back
  (so holding "A" for 3 seconds doesn't spam "AAAAAA") - the user
  must drop below threshold (e.g. move their hand) before the same
  letter can be added again.
- A special "SPACE" or "NEXT" gesture class can be reserved to add a
  space between words, if you choose to train one.
"""

from collections import deque


class WordBuilder:
    def __init__(self, hold_frames=15, confidence_threshold=0.85, buffer_size=20):
        self.hold_frames = hold_frames
        self.confidence_threshold = confidence_threshold
        self.buffer = deque(maxlen=buffer_size)
        self.current_word = ""
        self.history = []
        self.last_confirmed_label = None
        self._armed = True  # becomes False right after confirming, until label changes

    def update(self, label, confidence):
        """
        Call this every frame with the latest (label, confidence).
        Returns the currently confirmed word (may be unchanged this frame).
        """
        if label is None or confidence < self.confidence_threshold:
            self.buffer.append(None)
            self._armed = True
            return self.current_word

        self.buffer.append(label)

        # Check if the last `hold_frames` predictions are all this same label
        recent = list(self.buffer)[-self.hold_frames:]
        if len(recent) == self.hold_frames and all(r == label for r in recent):
            if self._armed:
                self.current_word += label
                self._armed = False  # require the sign to change before re-triggering
        else:
            # if label changed from what's being held, re-arm
            if label != self.last_confirmed_label:
                self._armed = True

        self.last_confirmed_label = label
        return self.current_word

    def commit_word(self):
        """Push current word into history and clear it (e.g. after speaking it)."""
        if self.current_word:
            self.history.append(self.current_word)
        word = self.current_word
        self.current_word = ""
        self.buffer.clear()
        self._armed = True
        return word

    def reset(self):
        self.current_word = ""
        self.buffer.clear()
        self._armed = True

    def backspace(self):
        self.current_word = self.current_word[:-1]
