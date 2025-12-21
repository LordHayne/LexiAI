"""
Error handling middleware for the Lexi API.
"""
from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
import logging

# Setup logging
logger = logging.getLogger("lexi_middleware.error_handler")

# Standardized error codes
ERROR_CODES = {
    # General errors
    "INTERNAL_ERROR": {"code": 1000, "message": "Interner Serverfehler"},
    "INVALID_REQUEST": {"code": 1001, "message": "Ungültige Anfrage"},
    
    # Authentication errors
    "AUTH_FAILED": {"code": 2000, "message": "Authentifizierung fehlgeschlagen"},
    "TOKEN_EXPIRED": {"code": 2001, "message": "Token abgelaufen"},
    "INSUFFICIENT_PERMISSIONS": {"code": 2002, "message": "Unzureichende Berechtigungen"},
    
    # Memory errors
    "MEMORY_ERROR": {"code": 3000, "message": "Fehler im Gedächtnissystem"},
    "STORAGE_FAILED": {"code": 3001, "message": "Speichern fehlgeschlagen"},
    "RETRIEVAL_FAILED": {"code": 3002, "message": "Abrufen fehlgeschlagen"},
    
    # LLM errors
    "LLM_ERROR": {"code": 4000, "message": "Fehler im Sprachmodell"},
    "PROMPT_ERROR": {"code": 4001, "message": "Fehler bei der Prompt-Verarbeitung"},
    "RESPONSE_TIMEOUT": {"code": 4002, "message": "Zeitüberschreitung bei der Antwort"},
    
    # Configuration errors
    "CONFIG_ERROR": {"code": 5000, "message": "Konfigurationsfehler"},
    "INVALID_PARAMETER": {"code": 5001, "message": "Ungültiger Parameter"},
}

class ErrorHandler:
    @staticmethod
    def handle_exception(exc):
        """
        Handle exceptions and return standardized error responses.
        """
        # FastAPI HTTP exceptions
        if isinstance(exc, HTTPException):
            error_detail = getattr(exc, "detail", "Ein Fehler ist aufgetreten")
            headers = getattr(exc, "headers", None)
            
            if exc.status_code == status.HTTP_401_UNAUTHORIZED:
                error_code = "AUTH_FAILED"
            elif exc.status_code == status.HTTP_403_FORBIDDEN:
                error_code = "INSUFFICIENT_PERMISSIONS"
            elif exc.status_code == status.HTTP_404_NOT_FOUND:
                error_code = "INVALID_REQUEST"
            elif exc.status_code == status.HTTP_400_BAD_REQUEST:
                error_code = "INVALID_REQUEST"
            else:
                error_code = "INTERNAL_ERROR"
                
            response = {
                "success": False,
                "error_code": error_code,
                "message": error_detail,
            }
            
            logger.error(f"HTTP Exception: {error_code} - {error_detail}")
            return JSONResponse(
                status_code=exc.status_code,
                content=response,
                headers=headers,
            )
        
        # Custom application exceptions
        if hasattr(exc, "error_code") and exc.error_code in ERROR_CODES:
            error_code = exc.error_code
            error_detail = str(exc)
            status_code = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            # Default to internal error for unexpected exceptions
            error_code = "INTERNAL_ERROR"
            error_detail = str(exc)
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        
        response = {
            "success": False,
            "error_code": error_code,
            "message": ERROR_CODES[error_code]["message"],
            "details": error_detail,
        }
        
        logger.error(f"Application Exception: {error_code} - {error_detail}")
        return JSONResponse(
            status_code=status_code,
            content=response,
        )
    
    @staticmethod
    def create_error(error_code, details=None, status_code=status.HTTP_400_BAD_REQUEST):
        """
        Create a standardized error response.
        """
        if error_code not in ERROR_CODES:
            error_code = "INTERNAL_ERROR"
            
        response = {
            "success": False,
            "error_code": error_code,
            "message": ERROR_CODES[error_code]["message"],
        }
        
        if details:
            response["details"] = details
            
        return JSONResponse(
            status_code=status_code,
            content=response,
        )

# Custom exception classes
class LexiError(Exception):
    """Base exception for all Lexi API errors."""
    def __init__(self, message, error_code="INTERNAL_ERROR", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR):
        self.error_code = error_code
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)

class MemoryError(LexiError):
    """Exception for memory-related errors."""
    def __init__(self, message, error_code="MEMORY_ERROR", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message, error_code, status_code)

class LLMError(LexiError):
    """Exception for LLM-related errors."""
    def __init__(self, message, error_code="LLM_ERROR", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message, error_code, status_code)
        
class ConfigError(LexiError):
    """Exception for configuration-related errors."""
    def __init__(self, message, error_code="CONFIG_ERROR", status_code=status.HTTP_400_BAD_REQUEST):
        super().__init__(message, error_code, status_code)
