"""
XTTS (Coqui TTS) Integration für LexiAI

Dieser Service stellt die Verbindung zum lokalen XTTS Docker Server her
und ermöglicht Voice Cloning mit Lexi's trainierter Stimme.

Author: LexiAI
Created: 2025-11-09
"""

import aiohttp
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any
from fastapi import HTTPException

logger = logging.getLogger("lexi_xtts")


class XTTSService:
    """
    XTTS Text-to-Speech Service für Voice Cloning mit Lexi's Stimme.

    Verbindet sich mit dem lokalen XTTS Docker Server und generiert
    Sprache mit der trainierten Lexi-Stimme.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8020",
        speaker_wav_path: str = "/app/voices/lexi.wav",  # Pfad im Container
        default_language: str = "de",
        timeout: int = 60
    ):
        """
        Initialisiert den XTTS Service.

        Args:
            server_url: URL des XTTS Docker Servers
            speaker_wav_path: Pfad zur Speaker-Datei (im Container)
            default_language: Standard-Sprache für TTS
            timeout: Timeout für API-Requests in Sekunden
        """
        self.server_url = server_url.rstrip('/')
        self.speaker_wav_path = speaker_wav_path
        self.default_language = default_language
        self.timeout = aiohttp.ClientTimeout(total=timeout)

        logger.info(f"🎤 XTTS Service initialisiert: {server_url}")
        logger.debug(f"Speaker-Datei: {speaker_wav_path}")
        logger.debug(f"Sprache: {default_language}")

    async def health_check(self) -> Dict[str, Any]:
        """
        Prüft ob der XTTS Server erreichbar ist.

        Returns:
            Dict mit Health-Informationen

        Raises:
            HTTPException: Wenn Server nicht erreichbar
        """
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{self.server_url}/health") as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"XTTS Health Check: OK")
                        return {
                            "status": "ok",
                            "server_url": self.server_url,
                            "response": data
                        }
                    else:
                        error_text = await response.text()
                        logger.warning(f"XTTS Health Check failed: {response.status}")
                        return {
                            "status": "error",
                            "server_url": self.server_url,
                            "error": f"HTTP {response.status}: {error_text}"
                        }

        except aiohttp.ClientError as e:
            logger.error(f"XTTS Server nicht erreichbar: {e}")
            return {
                "status": "unreachable",
                "server_url": self.server_url,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"XTTS Health Check Fehler: {e}")
            return {
                "status": "error",
                "server_url": self.server_url,
                "error": str(e)
            }

    async def synthesize_speech(
        self,
        text: str,
        language: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        temperature: float = 0.75,
        length_penalty: float = 1.0,
        repetition_penalty: float = 5.0,
        top_k: int = 50,
        top_p: float = 0.85,
        speed: float = 1.0,
        enable_text_splitting: bool = True
    ) -> bytes:
        """
        Generiert Sprach-Audio aus Text mit XTTS.

        Args:
            text: Zu sprechender Text
            language: Sprache (Standard: self.default_language)
            speaker_wav: Pfad zur Speaker-Datei (Standard: self.speaker_wav_path)
            temperature: Kreativität der Generierung (0.0-1.0, höher = expressiver)
            length_penalty: Längenpenalty für Generierung
            repetition_penalty: Verhindert Wiederholungen
            top_k: Top-K Sampling
            top_p: Nucleus Sampling
            speed: Sprechgeschwindigkeit (0.5-2.0)
            enable_text_splitting: Automatisches Text-Splitting für lange Texte

        Returns:
            WAV Audio-Daten als bytes

        Raises:
            HTTPException: Bei Fehler in der TTS-Generierung
        """
        start_time = time.time()

        # Parameter
        language = language or self.default_language
        speaker_wav = speaker_wav or self.speaker_wav_path

        logger.info(f"🔊 Generiere TTS: {len(text)} Zeichen, Sprache: {language}")
        logger.debug(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")

        # Request Payload
        payload = {
            "text": text,
            "language": language,
            "speaker_wav": speaker_wav,
            "temperature": temperature,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "top_k": top_k,
            "top_p": top_p,
            "speed": speed,
            "enable_text_splitting": enable_text_splitting
        }

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                # XTTS API Endpoint
                url = f"{self.server_url}/tts_to_audio/"

                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"XTTS TTS Fehler: {response.status} - {error_text}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"XTTS TTS fehlgeschlagen: {response.status} - {error_text[:200]}"
                        )

                    # Audio-Daten lesen
                    audio_data = await response.read()
                    processing_time = time.time() - start_time

                    logger.info(
                        f"✅ TTS generiert: {processing_time:.2f}s, "
                        f"{len(audio_data)} Bytes, "
                        f"{len(text)} Zeichen"
                    )

                    return audio_data

        except aiohttp.ClientError as e:
            logger.error(f"XTTS Server nicht erreichbar: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"XTTS Server nicht erreichbar. Ist der Docker Container gestartet? ({str(e)})"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Unerwarteter Fehler bei TTS-Generierung: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"TTS-Generierung fehlgeschlagen: {str(e)}"
            )

    async def stream_speech(
        self,
        text: str,
        language: Optional[str] = None,
        speaker_wav: Optional[str] = None,
        **kwargs
    ):
        """
        Generiert Sprach-Audio als Stream (für große Texte).

        Args:
            text: Zu sprechender Text
            language: Sprache
            speaker_wav: Speaker-Datei
            **kwargs: Zusätzliche Parameter für synthesize_speech

        Yields:
            Audio-Chunks als bytes
        """
        # XTTS Server unterstützt Streaming via /tts_stream
        language = language or self.default_language
        speaker_wav = speaker_wav or self.speaker_wav_path

        payload = {
            "text": text,
            "language": language,
            "speaker_wav": speaker_wav,
            **kwargs
        }

        logger.info(f"🔊 Starte TTS-Streaming: {len(text)} Zeichen")

        try:
            async with aiohttp.ClientSession(timeout=self.timeout) as session:
                url = f"{self.server_url}/tts_stream/"

                async with session.post(url, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"XTTS Streaming Fehler: {response.status}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"TTS Streaming fehlgeschlagen: {error_text[:200]}"
                        )

                    # Stream Audio-Chunks
                    chunk_count = 0
                    async for chunk in response.content.iter_chunked(8192):
                        chunk_count += 1
                        yield chunk

                    logger.info(f"✅ Streaming beendet: {chunk_count} Chunks")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Streaming-Fehler: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"TTS Streaming fehlgeschlagen: {str(e)}"
            )

    def get_info(self) -> Dict[str, Any]:
        """
        Gibt Informationen über den XTTS Service zurück.

        Returns:
            Dict mit Service-Informationen
        """
        return {
            "service": "XTTS (Coqui TTS)",
            "server_url": self.server_url,
            "speaker_wav": self.speaker_wav_path,
            "default_language": self.default_language,
            "timeout": self.timeout.total,
            "voice": "Lexi (Custom Trained)",
            "capabilities": {
                "voice_cloning": True,
                "streaming": True,
                "multilingual": True,
                "languages_supported": ["de", "en", "es", "fr", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "ko", "hi"]
            }
        }


# Singleton Instance (lazy initialization)
_xtts_service: Optional[XTTSService] = None


def get_xtts_service(
    server_url: str = "http://localhost:8020",
    speaker_wav_path: str = "/app/voices/lexi.wav",
    default_language: str = "de"
) -> XTTSService:
    """
    Gibt die Singleton-Instanz des XTTS Service zurück.

    Args:
        server_url: URL des XTTS Servers
        speaker_wav_path: Pfad zur Speaker-Datei
        default_language: Standard-Sprache

    Returns:
        XTTSService Instanz
    """
    global _xtts_service

    if _xtts_service is None:
        _xtts_service = XTTSService(
            server_url=server_url,
            speaker_wav_path=speaker_wav_path,
            default_language=default_language
        )
        logger.info("🎤 XTTS Service Singleton erstellt")

    return _xtts_service


# Convenience Functions für einfache Nutzung

async def synthesize_with_lexi_voice(text: str, language: str = "de") -> bytes:
    """
    Generiert Sprache mit Lexi's Stimme (Convenience-Funktion).

    Args:
        text: Zu sprechender Text
        language: Sprache (default: de)

    Returns:
        WAV Audio als bytes
    """
    service = get_xtts_service()
    return await service.synthesize_speech(text, language=language)


async def check_xtts_health() -> Dict[str, Any]:
    """
    Prüft XTTS Server Status (Convenience-Funktion).

    Returns:
        Health-Status als Dict
    """
    service = get_xtts_service()
    return await service.health_check()
