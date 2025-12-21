"""
Security Test Suite for LexiAI
==============================

Comprehensive security tests covering:
- Authentication and authorization
- Rate limiting and DoS protection
- Injection attack prevention (SQL, NoSQL, XSS, Command, etc.)
- Input validation and sanitization
- Session management

Run all security tests:
    pytest tests/security/ -v

Run specific test category:
    pytest tests/security/test_authentication.py -v
    pytest tests/security/test_rate_limiting.py -v
    pytest tests/security/test_injection_attacks.py -v
"""
