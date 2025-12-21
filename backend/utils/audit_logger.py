"""
Audit logging for the Lexi middleware API.
"""
import logging
import json
import time
import uuid
import socket
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from backend.config.feature_flags import FeatureFlags
from backend.config.persistence import ConfigPersistence

# Konfigurierbarer Logpfad aus Konfigdatei laden
_config = ConfigPersistence.load_config()
# Use absolute path relative to project root
default_log_path = Path(__file__).parent.parent.parent / "backend" / "logs" / "lexi_audit.log"
# Use default if audit_log_path is empty or not set
audit_log_path = _config.get("audit_log_path", "")
LOG_PATH = Path(audit_log_path) if audit_log_path and audit_log_path.strip() else default_log_path

# Setup logging
audit_logger = logging.getLogger("lexi_middleware.audit")
if not audit_logger.hasHandlers():
    audit_logger.setLevel(logging.INFO)
    # Create parent directories if they don't exist
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        # If we can't create the directory, log to the current directory
        LOG_PATH = Path("lexi_audit.log")
    audit_logger.addHandler(logging.FileHandler(LOG_PATH))
    audit_logger.addHandler(logging.StreamHandler())

class AuditLogger:
    """
    Audit logging utility for tracking important system events.
    """

    @staticmethod
    def current_timestamp_ms() -> int:
        return int(time.time() * 1000)

    @staticmethod
    def log_event(
        action: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        status: str = "success",
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
        category: Optional[str] = None
    ):
        if not FeatureFlags.is_enabled("audit_logging"):
            return

        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "status": status,
            "event_id": str(uuid.uuid4())
        }

        if user_id:
            entry["user_id"] = user_id
        if resource:
            entry["resource"] = resource
        if details:
            entry["details"] = details
        if ip_address:
            entry["ip_address"] = ip_address
        if session_id:
            entry["session_id"] = session_id
        if category:
            entry["category"] = category

        try:
            entry["hostname"] = socket.gethostname()
        except Exception as e:
            audit_logger.warning(f"Failed to get hostname: {e}")

        try:
            audit_logger.info("AUDIT:" + json.dumps(entry))
        except Exception as e:
            audit_logger.error(f"Failed to log audit entry: {e} | Original entry: {entry}")

    @staticmethod
    def log_chat(user_id: str, message_length: int, response_length: int, memory_used: bool, session_id: Optional[str] = None, ip_address: Optional[str] = None):
        details = {
            "message_length": message_length,
            "response_length": response_length,
            "memory_used": memory_used,
            "timestamp_ms": AuditLogger.current_timestamp_ms()
        }

        AuditLogger.log_event(
            action="CHAT",
            user_id=user_id,
            resource="chat",
            status="success",
            details=details,
            ip_address=ip_address,
            session_id=session_id,
            category="chat"
        )

    @staticmethod
    def log_memory_store(user_id: str, memory_id: str, content_length: int, tags: Optional[list] = None, session_id: Optional[str] = None, ip_address: Optional[str] = None):
        details = {
            "memory_id": memory_id,
            "content_length": content_length,
            "timestamp_ms": AuditLogger.current_timestamp_ms()
        }
        if tags:
            if not isinstance(tags, list):
                audit_logger.warning("Expected list for tags, got %s", type(tags).__name__)
            else:
                details["tags"] = tags

        AuditLogger.log_event(
            action="MEMORY_STORE",
            user_id=user_id,
            resource=f"memory/{memory_id}",
            status="success",
            details=details,
            ip_address=ip_address,
            session_id=session_id,
            category="memory"
        )

    @staticmethod
    def log_memory_query(user_id: str, query: Optional[str], result_count: int, session_id: Optional[str] = None, ip_address: Optional[str] = None):
        details = {
            "result_count": result_count,
            "timestamp_ms": AuditLogger.current_timestamp_ms()
        }
        if query:
            details["query"] = query

        AuditLogger.log_event(
            action="MEMORY_QUERY",
            user_id=user_id,
            resource="memory",
            status="success",
            details=details,
            ip_address=ip_address,
            session_id=session_id,
            category="memory"
        )

    @staticmethod
    def log_config_change(updated_keys: list, session_id: Optional[str] = None, ip_address: Optional[str] = None):
        details = {
            "updated_keys": updated_keys,
            "timestamp_ms": AuditLogger.current_timestamp_ms()
        }

        AuditLogger.log_event(
            action="CONFIG_CHANGE",
            resource="config",
            status="success",
            details=details,
            ip_address=ip_address,
            session_id=session_id,
            category="config"
        )

    @staticmethod
    def log_error(error_code: str, message: str, user_id: Optional[str] = None, resource: Optional[str] = None, session_id: Optional[str] = None, ip_address: Optional[str] = None):
        details = {
            "error_code": error_code,
            "message": message,
            "timestamp_ms": AuditLogger.current_timestamp_ms()
        }

        AuditLogger.log_event(
            action="ERROR",
            user_id=user_id,
            resource=resource,
            status="failure",
            details=details,
            ip_address=ip_address,
            session_id=session_id,
            category="error"
        )

    @staticmethod
    def log_system_event(event_type: str, details: Optional[Dict[str, Any]] = None, session_id: Optional[str] = None):
        AuditLogger.log_event(
            action=event_type,
            resource="system",
            details=details or {},
            session_id=session_id,
            category="system"
        )
