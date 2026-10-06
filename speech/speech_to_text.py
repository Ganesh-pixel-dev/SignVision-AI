"""
speech_to_text.py
--------------------
Converts microphone speech into text, used on the hearing person's
side of the call so their words can be matched against the sign
library and shown/animated to the deaf participant.

Uses the SpeechRecognition library. By default this calls Google's
free web speech API (needs internet). For a fully offline setup you
can swap in a local model (e.g. Vosk) later without changing the
rest of the pipeline - just keep the same `listen_once()` interface.
"""

import speech_recognition as sr


class SpeechToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

    def listen_once(self, timeout=5, phrase_time_limit=6):
        """
        Listens for a single utterance and returns the recognized text,
        or None if nothing understandable was captured.
        """
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )
            except sr.WaitTimeoutError:
                return None

        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Speech recognition service error: {e}")
            return None


if __name__ == "__main__":
    stt = SpeechToText()
    print("Say something...")
    result = stt.listen_once()
    print("You said:", result)
