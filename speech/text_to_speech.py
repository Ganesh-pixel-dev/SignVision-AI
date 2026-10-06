"""Offline text to speech with pyttsx3.

pyttsx3 can fail to start (no speech engine installed) or fail while
speaking. Both are reported through the return value instead of raising,
so a missing voice never takes the app down.
"""

import threading

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class TextToSpeech:
    def __init__(self, rate=160, volume=1.0):
        self.rate = rate
        self.volume = volume
        self._lock = threading.Lock()  # one utterance at a time

    def speak(self, text):
        """Speak text and wait. Returns True if it was spoken."""
        if not text:
            return True
        if pyttsx3 is None:
            return False
        with self._lock:
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)
                engine.say(text)
                engine.runAndWait()
                return True
            except Exception as exc:  # pyttsx3 raises assorted driver errors
                print(f"[tts] could not speak: {exc}")
                return False

    def speak_async(self, text):
        """Speak in a background thread. Returns False if no engine library is installed."""
        if pyttsx3 is None:
            return False
        threading.Thread(target=self.speak, args=(text,), daemon=True).start()
        return True


if __name__ == "__main__":
    print("spoken" if TextToSpeech().speak("Sign Vision is ready") else "text to speech unavailable")
