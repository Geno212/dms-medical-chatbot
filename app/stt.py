"""Local speech-to-text via faster-whisper.

Optional capability: faster-whisper is imported lazily so the rest of the
system runs even when it isn't installed. Whisper handles Arabic and English
(and code-switching) with automatic language detection.
"""

from __future__ import annotations

from functools import lru_cache

from .config import get_config
from .logging_setup import get_logger

log = get_logger()


class TranscriptionUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _get_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionUnavailable(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc
    config = get_config()
    compute = "int8" if config.whisper_device == "cpu" else "float16"
    return WhisperModel(config.whisper_model, device=config.whisper_device, compute_type=compute)


def transcribe(audio_path: str) -> dict:
    """Transcribe an audio file; returns {'transcribed_text', 'language'}."""
    model = _get_model()
    segments, info = model.transcribe(audio_path, vad_filter=True)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    log.info(
        "stt: file=%s detected_language=%s (p=%.2f) transcribed=%r",
        audio_path, info.language, info.language_probability, text[:120],
    )
    return {"transcribed_text": text, "language": info.language}
