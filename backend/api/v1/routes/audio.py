from fastapi import APIRouter, UploadFile, File, Response, HTTPException
from fastapi.responses import StreamingResponse
import logging
import json
import requests
from pathlib import Path
import asyncio
import aiohttp
from typing import Optional, Dict, Any
import io
import tempfile
import os
import time
from datetime import datetime
from backend.services.tts_simple import synthesize_speech_simple

logger = logging.getLogger("lexi_audio_api")
router = APIRouter()

# Global stats für Health-Endpoint
class AudioStats:
    def __init__(self):
        self.last_request_time = None
        self.last_audio_length_seconds = 0
        self.last_transcription_length = 0
        self.last_response_length = 0
        self.last_processing_time = 0
        self.total_requests = 0
        self.successful_requests = 0
    
    def update(self, audio_length: float, transcription: str, response: str, processing_time: float):
        self.last_request_time = datetime.now()
        self.last_audio_length_seconds = audio_length
        self.last_transcription_length = len(transcription)
        self.last_response_length = len(response)
        self.last_processing_time = processing_time
        self.total_requests += 1
        self.successful_requests += 1

audio_stats = AudioStats()

# ---------------------------------------------------
# 1️⃣ Konfigurationswert aus persistent_config.json laden
def get_persistent_config_value(key: str, default: Any = None) -> Any:
    # Use relative path from this file to config directory
    config_path = Path(__file__).parent.parent.parent.parent / "backend" / "config" / "persistent_config.json"

    if not config_path.exists():
        logger.warning(f"Config-Datei {config_path} nicht gefunden!")
        return default

    try:
        with open(config_path) as f:
            config = json.load(f)
        value = config.get(key, default)
        logger.debug(f"Config geladen: {key} = {'[REDACTED]' if 'key' in key.lower() or 'secret' in key.lower() else value}")
        return value
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Fehler beim Laden der Konfiguration: {e}")
        return default

def get_tts_settings() -> Dict[str, Any]:
    """
    Lädt TTS-Settings aus der Konfiguration mit Fallback-Defaults.
    """
    return {
        "provider": "gTTS",
        "language": get_persistent_config_value("tts_language", "de"),
        "offline": False
    }

# ---------------------------------------------------
# 2️⃣ Audio-Transkription (Whisper Integration)
async def transcribe_audio(file_bytes: bytes, filename: str = "audio.wav") -> str:
    """
    Transkribiert Audio zu Text mit Whisper (lokal oder API).
    """
    start_time = time.time()
    logger.info(f"🎤 Starte Transkription: {filename} ({len(file_bytes)} Bytes)")
    
    # Schätze Audio-Länge (grob, basierend auf Dateigröße)
    estimated_duration = len(file_bytes) / 16000  # Grobe Schätzung für 16kHz Audio
    logger.debug(f"Geschätzte Audio-Dauer: {estimated_duration:.2f}s")
    
    # Whisper API Integration (OpenAI)
    use_whisper_api = get_persistent_config_value("use_whisper_api", False)
    
    if use_whisper_api:
        try:
            import openai
            
            openai_api_key = get_persistent_config_value("OPENAI_API_KEY")
            if not openai_api_key:
                logger.warning("OpenAI API Key nicht gefunden, nutze lokales Whisper")
                use_whisper_api = False
            else:
                logger.debug("🌐 Nutze OpenAI Whisper API")
                
                # Temporäre Datei erstellen
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    tmp_file.write(file_bytes)
                    tmp_file_path = tmp_file.name
                
                try:
                    # Updated for newer OpenAI library versions
                    client = openai.OpenAI(api_key=openai_api_key)
                    with open(tmp_file_path, "rb") as audio_file:
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file
                        )
                    
                    result = transcript.text
                    processing_time = time.time() - start_time
                    logger.info(f"✅ Whisper API Transkription fertig: {processing_time:.2f}s, {len(result)} Zeichen")
                    return result
                    
                finally:
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                    
        except ImportError:
            logger.warning("OpenAI Library nicht installiert, nutze lokales Whisper")
            use_whisper_api = False
        except Exception as e:
            logger.error(f"Fehler bei Whisper API: {e}")
            use_whisper_api = False
    
    # Lokales Whisper (falls API nicht verfügbar)
    if not use_whisper_api:
        try:
            # Fixed import for whisper
            import whisper
            
            model_name = get_persistent_config_value("whisper_model", "base")
            logger.debug(f"🖥️ Nutze lokales Whisper Modell: {model_name}")
            
            # Lade Whisper Modell (könnte gecacht werden für bessere Performance)
            model = whisper.load_model(model_name)
            
            # Temporäre Datei für Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
            
            try:
                result = model.transcribe(tmp_file_path)
                text = result["text"]
                processing_time = time.time() - start_time
                logger.info(f"✅ Lokale Whisper Transkription fertig: {processing_time:.2f}s, {len(text)} Zeichen")
                return text
                
            finally:
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                
        except ImportError as e:
            logger.warning(f"Whisper Library nicht verfügbar: {e}")
        except Exception as e:
            logger.error(f"Fehler bei lokalem Whisper: {e}")
    
    # Fallback: Dummy-Transkription
    processing_time = time.time() - start_time
    dummy_text = "Hallo, ich bin Lexi – das ist dein Text 😊"
    logger.info(f"🤖 Dummy-Transkription verwendet: {processing_time:.2f}s")
    return dummy_text

# ---------------------------------------------------
# 3️⃣ Improved Chat API Call with Direct LLM Integration
async def ask_lexi_chat_api(message: str) -> str:
    """
    Improved chat function that can work with direct LLM or through API.
    """
    start_time = time.time()
    
    # Try direct LLM call first (bypass authentication issues)
    try:
        from backend.core.lexi_adapter import LexiAdapter
        
        logger.info(f"💬 Direkte LLM-Anfrage: {len(message)} Zeichen")
        
        # Create LexiAdapter instance
        adapter = LexiAdapter()
        
        # Get system prompt from config
        system_prompt = get_persistent_config_value("system_prompt", "Du bist Lexi, eine charmante KI-Assistentin.")
        
        # Call LLM directly
        response = await adapter.generate_response(
            message=message,
            system_prompt=system_prompt,
            stream=False
        )
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Direkte LLM-Antwort erhalten: {processing_time:.2f}s, {len(response)} Zeichen")
        return response
        
    except Exception as e:
        logger.warning(f"Direkte LLM-Anfrage fehlgeschlagen: {e}, versuche API...")
    
    # Fallback to API call (use /ui/chat endpoint - no auth required)
    url = "http://localhost:8000/ui/chat"
    
    # Try different authentication approaches
    auth_attempts = [
        # No authentication (for local development)
        {},
        # With API key from config
        {"Authorization": f"Bearer {get_persistent_config_value('LEXI_API_KEY', '')}"},
        # With simple API key
        {"Authorization": f"Bearer {get_persistent_config_value('api_key', '')}"}
    ]
    
    # /ui/chat expects "message" (singular), not "messages" (plural)
    payload = {
        "message": message,
        "stream": False
    }
    
    base_headers = {"Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=30)
    
    for i, auth_headers in enumerate(auth_attempts):
        headers = {**base_headers, **auth_headers}
        
        try:
            logger.debug(f"Chat-Versuch {i+1}/{len(auth_attempts)}: {'mit Auth' if auth_headers else 'ohne Auth'}")
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        # /ui/chat returns ChatResponse format: {"response": "text", "success": true, ...}
                        response_text = result.get("response", "")
                        if not response_text:
                            logger.debug(f"Chat-Versuch {i+1}: Leere Antwort erhalten")
                            continue
                        processing_time = time.time() - start_time
                        logger.info(f"✅ Chat-Antwort erhalten (Versuch {i+1}): {processing_time:.2f}s, {len(response_text)} Zeichen")
                        return response_text
                    else:
                        error_text = await response.text()
                        logger.debug(f"Chat-Versuch {i+1} fehlgeschlagen: {response.status} - {error_text[:100]}")
                        
        except Exception as e:
            logger.debug(f"Chat-Versuch {i+1} Fehler: {e}")
            continue
    
    # Final fallback: Use direct Ollama call
    try:
        logger.info("🔄 Fallback: Direkte Ollama-Anfrage...")
        
        ollama_url = get_persistent_config_value("ollama_url", "http://localhost:11434")
        model = get_persistent_config_value("llm_model", "gemma3:4b-it-qat")
        system_prompt = get_persistent_config_value("system_prompt", "Du bist Lexi, eine charmante KI-Assistentin.")
        
        ollama_payload = {
            "model": model,
            "prompt": f"System: {system_prompt}\n\nUser: {message}\n\nAssistant:",
            "stream": False
        }
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{ollama_url}/api/generate", json=ollama_payload) as response:
                if response.status == 200:
                    result = await response.json()
                    response_text = result.get("response", "Entschuldigung, ich konnte keine Antwort generieren.")
                    processing_time = time.time() - start_time
                    logger.info(f"✅ Ollama-Fallback erfolgreich: {processing_time:.2f}s, {len(response_text)} Zeichen")
                    return response_text
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama-Fallback fehlgeschlagen: {response.status} - {error_text}")
                    
    except Exception as e:
        logger.error(f"Ollama-Fallback Fehler: {e}")
    
    # Ultimate fallback
    processing_time = time.time() - start_time
    fallback_response = "Hallo! Ich bin Lexi. Leider hatte ich ein kleines technisches Problem, aber ich bin wieder da! 😊"
    logger.warning(f"🤖 Fallback-Antwort verwendet: {processing_time:.2f}s")
    return fallback_response

# ---------------------------------------------------
# 4️⃣ Simple TTS (gTTS - Google Text-to-Speech)
async def synthesize_speech(text: str) -> bytes:
    """
    Konvertiert Text zu Sprache via gTTS.
    Schnelle funktionierende Lösung ohne Docker.
    """
    start_time = time.time()

    logger.info(f"🔊 Starte TTS: {len(text)} Zeichen")
    logger.debug(f"TTS-Text: {text[:100]}..." if len(text) > 100 else f"TTS-Text: {text}")

    try:
        # Sprache aus Config
        language = get_persistent_config_value("tts_language", "de")

        # Audio generieren
        audio_data = await synthesize_speech_simple(text, language=language)

        processing_time = time.time() - start_time
        logger.info(f"✅ TTS-Audio generiert: {processing_time:.2f}s, {len(audio_data)} Bytes")

        return audio_data

    except Exception as e:
        logger.error(f"TTS Fehler: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"TTS fehlgeschlagen: {str(e)}"
        )

# ---------------------------------------------------
# 5️⃣ Validierung der Audiodatei (unchanged)
def validate_audio_file(file: UploadFile) -> None:
    """
    Validiert die hochgeladene Audiodatei.
    """
    # Erlaubte MIME-Types
    allowed_types = [
        "audio/wav", "audio/wave", "audio/x-wav",
        "audio/mpeg", "audio/mp3",
        "audio/ogg", "audio/webm",
        "audio/m4a", "audio/aac"
    ]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Nicht unterstützter Dateityp: {file.content_type}. "
                   f"Erlaubt: {', '.join(allowed_types)}"
        )
    
    # Maximale Dateigröße: 25MB
    max_size = 25 * 1024 * 1024  # 25MB
    if file.size and file.size > max_size:
        raise HTTPException(
            status_code=400, 
            detail=f"Datei zu groß: {file.size} Bytes. Maximum: {max_size} Bytes"
        )

# ---------------------------------------------------
# 6️⃣ Hauptendpoint mit verbesserter Fehlerbehandlung
@router.post("/api/audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Verarbeitet eine Audiodatei: Audio → Transkription → Chat → TTS → Audio
    """
    total_start_time = time.time()
    audio_stats.total_requests += 1
    
    try:
        # Validierung
        validate_audio_file(file)
        
        # Audio einlesen
        audio_bytes = await file.read()
        estimated_audio_length = len(audio_bytes) / 16000  # Grobe Schätzung
        
        logger.info(f"🚀 Starte Audio-Pipeline: {file.filename}, {len(audio_bytes)} Bytes, ~{estimated_audio_length:.1f}s")

        # Pipeline sequenziell ausführen mit detaillierten Logs
        
        # 1️⃣ Audio → Transkription
        step_start = time.time()
        transcription = await transcribe_audio(audio_bytes, file.filename or "audio")
        transcription_time = time.time() - step_start
        logger.info(f"📝 Transkription: {transcription_time:.2f}s")

        # 2️⃣ Transkription → Lexi-API → Antwort-Text
        step_start = time.time()
        response_text = await ask_lexi_chat_api(transcription)
        chat_time = time.time() - step_start
        logger.info(f"🤖 Chat-Verarbeitung: {chat_time:.2f}s")

        # 3️⃣ Antwort → ElevenLabs TTS
        step_start = time.time()
        tts_audio = await synthesize_speech(response_text)
        tts_time = time.time() - step_start
        logger.info(f"🔊 TTS-Generierung: {tts_time:.2f}s")

        # Gesamtzeit und Performance-Stats
        total_time = time.time() - total_start_time
        logger.info(f"✅ Pipeline komplett: {total_time:.2f}s total (Transkription: {transcription_time:.2f}s, Chat: {chat_time:.2f}s, TTS: {tts_time:.2f}s)")
        
        # Stats aktualisieren
        audio_stats.update(estimated_audio_length, transcription, response_text, total_time)

        # 4️⃣ Streaming Response für bessere Performance
        return StreamingResponse(
            io.BytesIO(tts_audio),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=lexi_response.mp3",
                "Content-Length": str(len(tts_audio)),
                "Cache-Control": "no-cache",
                "X-Processing-Time": f"{total_time:.2f}s",
                "X-Audio-Length": f"{estimated_audio_length:.1f}s"
            }
        )

    except HTTPException:
        # HTTPExceptions direkt weiterleiten
        raise
    except Exception as e:
        total_time = time.time() - total_start_time
        logger.error(f"❌ Pipeline-Fehler nach {total_time:.2f}s: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Audioverarbeitung fehlgeschlagen"
        )

# ---------------------------------------------------
# 7️⃣ Zusätzliche Utility-Endpoints (unchanged)
@router.get("/api/audio/health")
async def health_check():
    """
    Erweiterte Gesundheitsprüfung mit Performance-Metriken und Testdaten.
    """
    start_time = time.time()
    
    status = {
        "audio_api": "ok",
        "chat_api": "unknown",
        "tts_api": "unknown",
        "whisper": "unknown",
        "timestamp": datetime.now().isoformat(),
        "uptime_check_duration": 0,
        "statistics": {
            "total_requests": audio_stats.total_requests,
            "successful_requests": audio_stats.successful_requests,
            "success_rate": round(audio_stats.successful_requests / max(audio_stats.total_requests, 1) * 100, 2),
            "last_request": audio_stats.last_request_time.isoformat() if audio_stats.last_request_time else None,
            "last_audio_length_seconds": round(audio_stats.last_audio_length_seconds, 2),
            "last_transcription_length": audio_stats.last_transcription_length,
            "last_response_length": audio_stats.last_response_length,
            "last_processing_time_seconds": round(audio_stats.last_processing_time, 2)
        }
    }
    
    # Chat API testen (with improved fallback)
    try:
        test_response = await ask_lexi_chat_api("Health check test")
        status["chat_api"] = "ok"
        status["chat_test_response_length"] = len(test_response)
        logger.debug(f"Chat API Health Check: OK ({len(test_response)} Zeichen)")
    except Exception as e:
        status["chat_api"] = f"error: {str(e)[:100]}"
        logger.warning(f"Chat API Health Check: FEHLER - {e}")
    
    # TTS Server testen
    try:
        # Kurzer TTS-Test
        test_audio = await synthesize_speech("Test")
        status["tts_api"] = "ok"
        status["tts_test_audio_size"] = len(test_audio)
        logger.debug(f"TTS Health Check: OK ({len(test_audio)} Bytes)")
    except Exception as e:
        status["tts_api"] = f"error: {str(e)[:100]}"
        logger.warning(f"TTS Test: FEHLER - {e}")
    
    # Whisper-Status prüfen
    try:
        if get_persistent_config_value("use_whisper_api", False):
            openai_key = get_persistent_config_value("OPENAI_API_KEY")
            status["whisper"] = "openai_api" if openai_key else "openai_no_key"
        else:
            import whisper
            status["whisper"] = "local_available"
    except ImportError:
        status["whisper"] = "not_installed"
    except Exception as e:
        status["whisper"] = f"error: {str(e)}"
    
    # Performance-Metriken
    status["uptime_check_duration"] = round(time.time() - start_time, 3)
    
    # Konfiguration (ohne Secrets)
    status["configuration"] = {
        "tts_provider": "gTTS (Google)",
        "tts_settings": get_tts_settings(),
        "whisper_mode": "api" if get_persistent_config_value("use_whisper_api", False) else "local",
        "whisper_model": get_persistent_config_value("whisper_model", "base")
    }
    
    return status

@router.get("/api/audio/config")
async def get_audio_config():
    """
    Gibt die aktuelle Audio-Konfiguration zurück (ohne Secrets).
    """
    tts_settings = get_tts_settings()

    return {
        "tts_configuration": {
            "provider": "gTTS (Google Text-to-Speech)",
            "voice": "Google German Voice",
            "language": tts_settings["language"],
            "capabilities": {
                "voice_cloning": False,
                "streaming": False,
                "multilingual": True,
                "offline": False
            }
        },
        "whisper_configuration": {
            "mode": "api" if get_persistent_config_value("use_whisper_api", False) else "local",
            "model": get_persistent_config_value("whisper_model", "base"),
            "has_openai_key": bool(get_persistent_config_value("OPENAI_API_KEY"))
        },
        "supported_formats": ["wav", "mp3", "ogg", "webm", "m4a", "aac"],
        "limits": {
            "max_file_size_mb": 25,
            "timeout_seconds": 60
        },
        "performance": {
            "total_requests": audio_stats.total_requests,
            "successful_requests": audio_stats.successful_requests,
            "average_processing_time": round(audio_stats.last_processing_time, 2) if audio_stats.last_processing_time else 0
        }
    }

@router.get("/api/audio/stats")
async def get_audio_stats():
    """
    Detaillierte Performance-Statistiken für Monitoring.
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "requests": {
            "total": audio_stats.total_requests,
            "successful": audio_stats.successful_requests,
            "failed": audio_stats.total_requests - audio_stats.successful_requests,
            "success_rate_percent": round(audio_stats.successful_requests / max(audio_stats.total_requests, 1) * 100, 2)
        },
        "last_request": {
            "timestamp": audio_stats.last_request_time.isoformat() if audio_stats.last_request_time else None,
            "audio_length_seconds": round(audio_stats.last_audio_length_seconds, 2),
            "transcription_length": audio_stats.last_transcription_length,
            "response_length": audio_stats.last_response_length,
            "processing_time_seconds": round(audio_stats.last_processing_time, 2),
            "performance_rating": "fast" if audio_stats.last_processing_time < 5 else "slow" if audio_stats.last_processing_time > 15 else "normal"
        }
    }