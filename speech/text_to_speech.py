"""
text_to_speech.py
--------------------
Offline text-to-speech using pyttsx3 (works without internet).
"""

import pyttsx3


class TextToSpeech:
    def __init__(self, rate=160, volume=1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

    def speak(self, text):
        if not text:
            return
        self.engine.say(text)
        self.engine.runAndWait()


if __name__ == "__main__":
    tts = TextToSpeech()
    tts.speak("Sign Vision A I is ready")
