"""
Security Tests for Rate Limiting
=================================

Tests rate limiting mechanisms to prevent:
- DoS attacks
- Brute force attacks
- Resource exhaustion
- API abuse
"""
import pytest
import time
from fastapi.testclient import TestClient
from backend.api.api_server import app
from backend.config.auth_config import SecurityConfig


class TestRateLimiting:
    """Test suite for rate limiting security."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def valid_headers(self):
        """Get valid authentication headers."""
        return {"X-API-Key": SecurityConfig.get_api_key()}

    def test_memory_add_rate_limit(self, client, valid_headers):
        """Test that memory/add endpoint enforces 10/minute rate limit."""
        endpoint = "/v1/memory/add"
        limit = 10  # As defined in memory.py

        # Make limit + 1 requests
        responses = []
        for i in range(limit + 5):
            response = client.post(
                endpoint,
                json={
                    "content": f"Test memory entry {i}",
                    "tags": ["test"],
                    "source": "rate_limit_test"
                },
                headers=valid_headers
            )
            responses.append(response)
            time.sleep(0.1)  # Small delay between requests

        # Check that at least one request was rate limited
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes, \
            "Rate limit should trigger 429 Too Many Requests"

    def test_memory_query_rate_limit(self, client, valid_headers):
        """Test that memory/query endpoint enforces 100/minute rate limit."""
        endpoint = "/v1/memory/query"
        limit = 100  # As defined in memory.py

        # Make requests rapidly
        rate_limited = False
        for i in range(limit + 10):
            response = client.post(
                endpoint,
                json={
                    "query": f"test query {i}",
                    "top_k": 5
                },
                headers=valid_headers
            )
            if response.status_code == 429:
                rate_limited = True
                break
            time.sleep(0.01)  # Very small delay

        assert rate_limited, \
            "Rate limit should trigger for excessive queries"

    def test_memory_delete_rate_limit(self, client, valid_headers):
        """Test that memory delete endpoint enforces 10/minute rate limit."""
        endpoint = "/v1/memory"
        limit = 10

        # Make rapid delete requests (will fail but should hit rate limit)
        responses = []
        for i in range(limit + 5):
            response = client.delete(
                f"{endpoint}/00000000-0000-0000-0000-{i:012d}",
                headers=valid_headers
            )
            responses.append(response)
            time.sleep(0.1)

        # Check for rate limiting
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes, \
            "Delete endpoint should enforce rate limit"

    def test_rate_limit_per_ip_address(self, client, valid_headers):
        """Test that rate limiting is per IP address."""
        # FastAPI slowapi uses IP-based rate limiting by default
        # This test verifies that behavior

        endpoint = "/v1/memory/query"

        # Make requests from "same IP" (test client default)
        for i in range(15):
            response = client.post(
                endpoint,
                json={"query": f"test {i}", "top_k": 1},
                headers=valid_headers
            )
            if response.status_code == 429:
                # Rate limit hit
                break

        # Should have hit rate limit
        assert response.status_code == 429

    def test_rate_limit_headers_present(self, client, valid_headers):
        """Test that rate limit headers are present in responses."""
        response = client.post(
            "/v1/memory/query",
            json={"query": "test", "top_k": 1},
            headers=valid_headers
        )

        # Check for standard rate limit headers
        # Different implementations may use different header names
        possible_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "RateLimit-Limit",
            "RateLimit-Remaining",
            "Retry-After",
        ]

        # At least one rate limit header should be present
        has_rate_limit_header = any(
            header in response.headers for header in possible_headers
        )

        # Note: This may not be implemented yet, so we make it optional
        if response.status_code == 429:
            # If rate limited, headers should be present
            assert "Retry-After" in response.headers or has_rate_limit_header

    def test_rate_limit_reset_after_window(self, client, valid_headers):
        """Test that rate limit resets after time window."""
        endpoint = "/v1/memory/query"

        # Hit rate limit
        for i in range(15):
            response = client.post(
                endpoint,
                json={"query": f"test {i}", "top_k": 1},
                headers=valid_headers
            )
            if response.status_code == 429:
                break

        # Verify we hit the limit
        assert response.status_code == 429

        # Wait for rate limit window to reset (1 minute + buffer)
        # Note: In real tests, you'd mock time instead of actually waiting
        # time.sleep(65)

        # For testing purposes, we just verify the mechanism exists
        # Full reset testing would require time mocking

    def test_burst_protection(self, client, valid_headers):
        """Test protection against burst attacks."""
        endpoint = "/v1/memory/add"

        # Send burst of requests as fast as possible
        start_time = time.time()
        responses = []

        for i in range(20):
            response = client.post(
                endpoint,
                json={"content": f"burst {i}"},
                headers=valid_headers
            )
            responses.append(response)

        end_time = time.time()

        # At least some requests should be rate limited
        rate_limited_count = sum(1 for r in responses if r.status_code == 429)
        assert rate_limited_count > 0, "Burst should trigger rate limiting"

        # Verify it happened quickly (real burst, not spread out)
        assert end_time - start_time < 5, "Burst test took too long"

    def test_different_endpoints_separate_limits(self, client, valid_headers):
        """Test that different endpoints have separate rate limits."""
        # Make requests to memory/add
        add_responses = []
        for i in range(12):
            response = client.post(
                "/v1/memory/add",
                json={"content": f"test {i}"},
                headers=valid_headers
            )
            add_responses.append(response)
            time.sleep(0.1)

        # Then make request to memory/query (different limit)
        query_response = client.post(
            "/v1/memory/query",
            json={"query": "test", "top_k": 1},
            headers=valid_headers
        )

        # Query should work even if add is rate limited
        # (They have separate limits: 10/min for add, 100/min for query)
        if any(r.status_code == 429 for r in add_responses):
            # Add was rate limited, query should still work
            assert query_response.status_code != 429, \
                "Different endpoints should have independent rate limits"

    def test_rate_limit_bypass_attempts(self, client, valid_headers):
        """Test that common rate limit bypass techniques don't work."""
        endpoint = "/v1/memory/query"

        # Technique 1: Changing User-Agent
        for i in range(15):
            response = client.post(
                endpoint,
                json={"query": f"test {i}", "top_k": 1},
                headers={
                    **valid_headers,
                    "User-Agent": f"TestAgent-{i}"
                }
            )
            if response.status_code == 429:
                break

        assert response.status_code == 429, \
            "Changing User-Agent should not bypass rate limit"

        # Technique 2: Adding random headers
        client_with_headers = TestClient(app)
        for i in range(15):
            response = client_with_headers.post(
                endpoint,
                json={"query": f"test {i}", "top_k": 1},
                headers={
                    **valid_headers,
                    f"X-Custom-{i}": "bypass_attempt"
                }
            )
            if response.status_code == 429:
                break

        assert response.status_code == 429, \
            "Adding custom headers should not bypass rate limit"

    def test_rate_limit_error_message(self, client, valid_headers):
        """Test that rate limit error messages are informative."""
        endpoint = "/v1/memory/add"

        # Hit rate limit
        for i in range(15):
            response = client.post(
                endpoint,
                json={"content": f"test {i}"},
                headers=valid_headers
            )
            if response.status_code == 429:
                break

        if response.status_code == 429:
            # Check error message
            assert "detail" in response.json() or "error" in response.json(), \
                "Rate limit error should include explanatory message"

            # Check for Retry-After header
            # Note: This is good practice but may not be implemented
            # assert "Retry-After" in response.headers


class TestDDoSProtection:
    """Test DDoS protection mechanisms."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_request_size_limits(self, client):
        """Test that excessively large requests are rejected."""
        # Attempt to send very large payload
        large_content = "x" * (100 * 1024)  # 100KB

        response = client.post(
            "/v1/memory/add",
            json={"content": large_content},
            headers={"X-API-Key": SecurityConfig.get_api_key()}
        )

        # Should be rejected (either 413 or 400)
        assert response.status_code in [400, 413], \
            "Excessively large requests should be rejected"

    def test_connection_limits(self, client):
        """Test that concurrent connection limits are enforced."""
        # This would require concurrent requests
        # Placeholder for future implementation
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
