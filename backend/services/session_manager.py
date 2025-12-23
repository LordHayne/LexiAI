"""
Session Manager für Conversation History

Speichert die letzten N Nachrichten pro User-Session für Context-Continuity.

Autor: LexiAI Development Team
Version: 1.0
Datum: 2025-01-24
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ConversationSession:
    """Einzelne Konversations-Session für einen User."""

    def __init__(self, user_id: str, max_turns: int = 10):
        """
        Initialisiert eine Konversations-Session.

        Args:
            user_id: User ID
            max_turns: Maximale Anzahl an Turns (user + assistant pairs) zu speichern
        """
        self.user_id = user_id
        self.max_turns = max_turns
        self.messages: deque = deque(maxlen=max_turns * 2)  # * 2 für user + assistant
        self.created_at = datetime.utcnow()
        self.last_active = datetime.utcnow()
        self.pending_automation: Optional[dict] = None
        self.pending_script: Optional[dict] = None

    def add_message(self, role: str, content: str):
        """
        Fügt eine Nachricht zur Session hinzu.

        Args:
            role: 'user' oder 'assistant'
            content: Nachrichteninhalt
        """
        self.messages.append({
            'role': role,
            'content': content,
            'timestamp': datetime.utcnow().isoformat()
        })
        self.last_active = datetime.utcnow()
        logger.debug(f"Message added to session {self.user_id}: {role}")

    def get_messages(self, include_timestamps: bool = False) -> List[Dict]:
        """
        Gibt alle Messages der Session zurück.

        Args:
            include_timestamps: Ob Timestamps inkludiert werden sollen

        Returns:
            Liste von Messages im OpenAI-Format
        """
        if include_timestamps:
            return list(self.messages)
        else:
            return [
                {'role': msg['role'], 'content': msg['content']}
                for msg in self.messages
            ]

    def clear(self):
        """Löscht alle Messages aus der Session."""
        self.messages.clear()
        logger.info(f"Session cleared for user {self.user_id}")

    def is_expired(self, timeout_minutes: int = 30) -> bool:
        """
        Prüft ob Session abgelaufen ist.

        Args:
            timeout_minutes: Timeout in Minuten

        Returns:
            True wenn Session abgelaufen
        """
        age = datetime.utcnow() - self.last_active
        return age > timedelta(minutes=timeout_minutes)

    def get_turn_count(self) -> int:
        """Gibt Anzahl der Turns (User-Nachrichten) zurück."""
        return sum(1 for msg in self.messages if msg['role'] == 'user')


class SessionManager:
    """
    Manager für User-Sessions und Conversation History.

    Features:
    - Speichert letzte N Turns pro User
    - Automatisches Cleanup abgelaufener Sessions
    - Thread-safe Operations
    """

    def __init__(
        self,
        max_turns: int = 10,
        session_timeout_minutes: int = 30,
        cleanup_interval_minutes: int = 10
    ):
        """
        Initialisiert den Session Manager.

        Args:
            max_turns: Maximale Turns pro Session
            session_timeout_minutes: Session Timeout
            cleanup_interval_minutes: Cleanup Intervall
        """
        self.max_turns = max_turns
        self.session_timeout_minutes = session_timeout_minutes
        self.cleanup_interval_minutes = cleanup_interval_minutes

        self.sessions: Dict[str, ConversationSession] = {}
        self.last_cleanup = datetime.utcnow()

        logger.info(
            f"✅ SessionManager initialized "
            f"(max_turns={max_turns}, timeout={session_timeout_minutes}min)"
        )

    def get_session(self, user_id: str) -> ConversationSession:
        """
        Holt oder erstellt Session für User.

        Args:
            user_id: User ID

        Returns:
            ConversationSession
        """
        # Cleanup check
        self._cleanup_expired_sessions()

        if user_id not in self.sessions:
            self.sessions[user_id] = ConversationSession(
                user_id=user_id,
                max_turns=self.max_turns
            )
            logger.info(f"📝 New session created for user: {user_id}")

        return self.sessions[user_id]

    def add_message(self, user_id: str, role: str, content: str):
        """
        Fügt Message zu User-Session hinzu.

        Args:
            user_id: User ID
            role: 'user' oder 'assistant'
            content: Nachrichteninhalt
        """
        session = self.get_session(user_id)
        session.add_message(role, content)

    def get_conversation_history(
        self,
        user_id: str,
        max_turns: Optional[int] = None
    ) -> List[Dict]:
        """
        Gibt Conversation History für User zurück.

        Args:
            user_id: User ID
            max_turns: Optional: Limitiere auf N Turns (überschreibt default)

        Returns:
            Liste von Messages im OpenAI-Format
        """
        if user_id not in self.sessions:
            return []

        session = self.sessions[user_id]
        messages = session.get_messages()

        if max_turns and max_turns < self.max_turns:
            # Limitiere auf letzte N Turns (N user + N assistant messages)
            return messages[-(max_turns * 2):]

        return messages

    def clear_session(self, user_id: str):
        """Löscht Session für User."""
        if user_id in self.sessions:
            self.sessions[user_id].clear()
            del self.sessions[user_id]
            logger.info(f"🗑️  Session deleted for user: {user_id}")

    def set_pending_automation(self, user_id: str, automation: dict):
        """Store pending automation for confirmation flow."""
        session = self.get_session(user_id)
        session.pending_automation = automation

    def get_pending_automation(self, user_id: str) -> Optional[dict]:
        """Get pending automation for confirmation flow."""
        session = self.get_session(user_id)
        return session.pending_automation

    def clear_pending_automation(self, user_id: str):
        """Clear pending automation for confirmation flow."""
        session = self.get_session(user_id)
        session.pending_automation = None

    def set_pending_script(self, user_id: str, script: dict):
        """Store pending script for confirmation flow."""
        session = self.get_session(user_id)
        session.pending_script = script

    def get_pending_script(self, user_id: str) -> Optional[dict]:
        """Get pending script for confirmation flow."""
        session = self.get_session(user_id)
        return session.pending_script

    def clear_pending_script(self, user_id: str):
        """Clear pending script for confirmation flow."""
        session = self.get_session(user_id)
        session.pending_script = None

    def _cleanup_expired_sessions(self):
        """Cleanup abgelaufener Sessions (automatisch)."""
        now = datetime.utcnow()

        # Prüfe ob Cleanup notwendig
        if now - self.last_cleanup < timedelta(minutes=self.cleanup_interval_minutes):
            return

        # Cleanup
        expired_users = [
            user_id
            for user_id, session in self.sessions.items()
            if session.is_expired(self.session_timeout_minutes)
        ]

        for user_id in expired_users:
            del self.sessions[user_id]
            logger.info(f"🧹 Expired session cleaned up: {user_id}")

        self.last_cleanup = now

        if expired_users:
            logger.info(f"🧹 Cleanup: {len(expired_users)} expired sessions removed")

    def get_stats(self) -> Dict:
        """Gibt Statistiken zurück."""
        return {
            "total_sessions": len(self.sessions),
            "active_users": list(self.sessions.keys()),
            "total_messages": sum(
                len(session.messages)
                for session in self.sessions.values()
            ),
            "last_cleanup": self.last_cleanup.isoformat()
        }


# Singleton Instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """
    Get or create singleton session manager.

    Returns:
        SessionManager
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(
            max_turns=10,  # Letzte 10 Turns
            session_timeout_minutes=30,  # 30 Minuten Timeout
            cleanup_interval_minutes=10  # Cleanup alle 10 Minuten
        )
    return _session_manager
