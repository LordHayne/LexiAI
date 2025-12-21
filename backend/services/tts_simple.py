"""
Simple TTS Service using gTTS (Google Text-to-Speech)
Fallback-Lösung für schnelle Voice-Integration
"""

try:
    from gtts import gTTS
except ModuleNotFoundError:
    gTTS = None
import io
import logging
from typing import Optional

logger = logging.getLogger("lexi_tts_simple")


async def synthesize_speech_simple(text: str, language: str = "de") -> bytes:
    """
    Generiert Sprach-Audio mit gTTS.

    Args:
        text: Zu sprechender Text
        language: Sprache (de, en, etc.)

    Returns:
        MP3 Audio als bytes
    """
    try:
        if gTTS is None:
            raise RuntimeError("gTTS is not installed. Install with: pip install gtts")

        logger.info(f"🔊 Generiere TTS: {len(text)} Zeichen, Sprache: {language}")

        # gTTS generieren
        tts = gTTS(text=text, lang=language, slow=False)

        # In Memory Buffer speichern
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        audio_data = audio_buffer.read()
        logger.info(f"✅ TTS generiert: {len(audio_data)} Bytes")

        return audio_data

    except Exception as e:
        logger.error(f"TTS Fehler: {e}")
        raise
