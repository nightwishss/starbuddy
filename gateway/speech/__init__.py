"""Board-local speech processing for the phone gateway."""

from .engine import SpeechEngine, SpeechError, prepare_tts_text

__all__ = ["SpeechEngine", "SpeechError", "prepare_tts_text"]
