"""Turns noisy per-frame predictions into letters and words.

A letter is added once the same label has been the top prediction, above
the confidence threshold, for `hold_frames` frames in a row. After that
the builder is locked: the same letter is not added again until the sign
has been absent for `release_frames` frames in a row. Absent means no
hand, low confidence, or a different label. One flickering frame in the
middle of a long hold therefore does not type the letter twice, and
dropping the hand for a moment does let you type a double letter.

Call update(None, 0.0) on frames where no hand was found. If those frames
are never reported, the builder cannot tell the hand left.
"""


class WordBuilder:
    def __init__(self, hold_frames=15, confidence_threshold=0.85, release_frames=5):
        if hold_frames < 1 or release_frames < 1:
            raise ValueError("hold_frames and release_frames must be at least 1")
        self.hold_frames = hold_frames
        self.confidence_threshold = confidence_threshold
        self.release_frames = release_frames
        self.current_word = ""
        self.history = []
        self._candidate = None
        self._run = 0
        self._locked_label = None
        self._absent_run = 0

    def update(self, label, confidence):
        """Feed one frame. Returns the current word."""
        if label is None or confidence < self.confidence_threshold:
            self._candidate, self._run = None, 0
            self._note_absent()
            return self.current_word

        if label == self._candidate:
            self._run += 1
        else:
            self._candidate, self._run = label, 1

        if self._locked_label is not None:
            if label == self._locked_label:
                self._absent_run = 0
                return self.current_word
            self._note_absent()

        if self._run >= self.hold_frames and self._locked_label is None:
            self.current_word += label
            self._locked_label = label
            self._absent_run = 0
        return self.current_word

    def _note_absent(self):
        if self._locked_label is None:
            return
        self._absent_run += 1
        if self._absent_run >= self.release_frames:
            self._locked_label = None
            self._absent_run = 0

    def commit_word(self):
        """Move the current word into history and clear it. Returns the word."""
        word = self.current_word
        if word:
            self.history.append(word)
        self.reset()
        return word

    def reset(self):
        self.current_word = ""
        self._candidate, self._run = None, 0
        self._locked_label = None
        self._absent_run = 0

    def backspace(self):
        self.current_word = self.current_word[:-1]
