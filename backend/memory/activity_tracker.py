"""
Activity Tracker für LexiAI

Trackt User-Aktivität und erkennt Idle-Zeiten für Background-Learning.
Implementiert als Thread-safe Singleton.

Verwendung:
    from backend.memory.activity_tracker import ActivityTracker

    tracker = ActivityTracker()
    tracker.track_activity()  # Bei jeder User-Interaktion aufrufen

    if tracker.is_idle(minutes=30):
        # System ist idle - führe intensive Learning-Tasks aus
        pass
"""

from datetime import datetime, timedelta, UTC
import threading
from typing import Optional


class ActivityTracker:
    """
    Thread-safe Singleton Activity Tracker.

    Trackt die letzte User-Aktivität und ermöglicht Idle-Detection.
    Dies erlaubt es dem System, intensive Learning-Tasks nur auszuführen,
    wenn der User nicht aktiv mit dem System interagiert.
    """

    _instance: Optional['ActivityTracker'] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> 'ActivityTracker':
        """
        Singleton Pattern: Nur eine Instanz existiert.
        Thread-safe Implementation.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-checked locking
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initialisiert den Activity Tracker.
        Wird nur einmal ausgeführt dank Singleton Pattern.
        """
        # Verhindert Re-Initialisierung bei Singleton
        if self._initialized:
            return

        self._initialized = True
        self._last_activity: datetime = datetime.now(UTC)
        self._activity_lock: threading.Lock = threading.Lock()

    def track_activity(self) -> None:
        """
        Registriert eine User-Aktivität (z.B. Chat-Message).

        Diese Methode sollte bei jeder User-Interaktion aufgerufen werden:
        - Chat-Message gesendet
        - API-Request empfangen
        - Frontend-Interaktion

        Thread-safe.
        """
        with self._activity_lock:
            self._last_activity = datetime.now(UTC)

    def is_idle(self, minutes: int = 30) -> bool:
        """
        Prüft ob das System idle ist.

        Args:
            minutes: Anzahl Minuten ohne Aktivität um als "idle" zu gelten

        Returns:
            True wenn System länger als `minutes` idle ist, sonst False

        Thread-safe.

        Beispiel:
            if tracker.is_idle(30):
                # System ist seit 30+ Minuten idle
                # Starte intensive Background-Tasks
                run_memory_synthesis()
        """
        with self._activity_lock:
            time_since_last_activity = datetime.now(UTC) - self._last_activity
            idle_threshold = timedelta(minutes=minutes)
            return time_since_last_activity >= idle_threshold

    def get_last_activity(self) -> datetime:
        """
        Gibt den Zeitpunkt der letzten Aktivität zurück.

        Returns:
            datetime: Zeitpunkt der letzten User-Aktivität (UTC)

        Thread-safe.
        """
        with self._activity_lock:
            return self._last_activity

    def get_idle_duration(self) -> timedelta:
        """
        Berechnet wie lange das System bereits idle ist.

        Returns:
            timedelta: Dauer seit letzter Aktivität

        Thread-safe.

        Beispiel:
            duration = tracker.get_idle_duration()
            print(f"Idle seit {duration.total_seconds() / 60:.1f} Minuten")
        """
        with self._activity_lock:
            return datetime.now(UTC) - self._last_activity

    def reset(self) -> None:
        """
        Setzt den Activity Tracker zurück (für Tests).

        WARNUNG: Nur für Testing verwenden!
        """
        with self._activity_lock:
            self._last_activity = datetime.now(UTC)


# Convenience Function für einfachen Zugriff
def track_activity() -> None:
    """
    Convenience Function: Registriert eine Aktivität.

    Verwendung:
        from backend.memory.activity_tracker import track_activity
        track_activity()
    """
    tracker = ActivityTracker()
    tracker.track_activity()


def is_system_idle(minutes: int = 30) -> bool:
    """
    Convenience Function: Prüft ob System idle ist.

    Args:
        minutes: Idle-Schwellwert in Minuten

    Returns:
        True wenn idle, sonst False

    Verwendung:
        from backend.memory.activity_tracker import is_system_idle

        if is_system_idle(30):
            print("System ist idle - starte Background Learning")
    """
    tracker = ActivityTracker()
    return tracker.is_idle(minutes)
