"""
Tests for Home Assistant Integration
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.services.home_assistant import HomeAssistantService, get_ha_service


@pytest.fixture
def ha_service():
    """Create a Home Assistant service instance for testing."""
    return HomeAssistantService(
        url="http://homeassistant.local:8123",
        token="test_token_123"
    )


@pytest.fixture
def ha_service_unconfigured():
    """Create an unconfigured Home Assistant service instance."""
    return HomeAssistantService(url=None, token=None)


class TestHomeAssistantService:
    """Test Home Assistant service initialization and configuration."""

    def test_service_initialization_with_config(self, ha_service):
        """Test that service initializes correctly with URL and token."""
        assert ha_service.url == "http://homeassistant.local:8123"
        assert ha_service.token == "test_token_123"
        assert ha_service.is_enabled() is True

    def test_service_initialization_without_config(self, ha_service_unconfigured):
        """Test that service initializes but is disabled without config."""
        assert ha_service_unconfigured.url == ""
        assert ha_service_unconfigured.token is None
        assert ha_service_unconfigured.is_enabled() is False

    def test_url_trailing_slash_removal(self):
        """Test that trailing slashes are removed from URL."""
        service = HomeAssistantService(
            url="http://homeassistant.local:8123/",
            token="test_token"
        )
        assert service.url == "http://homeassistant.local:8123"


class TestDeviceControl:
    """Test device control functionality."""

    @pytest.mark.asyncio
    async def test_control_device_turn_on_success(self, ha_service):
        """Test successful turn_on command."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            # Mock successful response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=[{"entity_id": "light.wohnzimmer", "state": "on"}])
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await ha_service.control_device("light.wohnzimmer", "turn_on")

            assert result["success"] is True
            assert result["entity_id"] == "light.wohnzimmer"
            assert result["action"] == "turn_on"
            assert result["service"] == "light.turn_on"

    @pytest.mark.asyncio
    async def test_control_device_set_brightness(self, ha_service):
        """Test setting brightness with value."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=[{}])
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await ha_service.control_device("light.wohnzimmer", "set_brightness", 128)

            assert result["success"] is True
            assert result["action"] == "set_brightness"

            # Verify brightness parameter was sent
            call_args = mock_post.call_args
            sent_data = call_args.kwargs['json']
            assert sent_data['brightness'] == 128

    @pytest.mark.asyncio
    async def test_control_device_set_temperature(self, ha_service):
        """Test setting temperature."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=[{}])
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await ha_service.control_device("climate.heizung", "set_temperature", 22.5)

            assert result["success"] is True

            # Verify temperature parameter
            call_args = mock_post.call_args
            sent_data = call_args.kwargs['json']
            assert sent_data['temperature'] == 22.5

    @pytest.mark.asyncio
    async def test_control_device_not_configured(self, ha_service_unconfigured):
        """Test that control fails when service is not configured."""
        result = await ha_service_unconfigured.control_device("light.test", "turn_on")

        assert result["success"] is False
        assert "nicht konfiguriert" in result["error"]

    @pytest.mark.asyncio
    async def test_control_device_http_error(self, ha_service):
        """Test handling of HTTP errors."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_response.text = AsyncMock(return_value="Entity not found")
            mock_post.return_value.__aenter__.return_value = mock_response

            result = await ha_service.control_device("light.nonexistent", "turn_on")

            assert result["success"] is False
            assert "HTTP 404" in result["error"]


class TestStateQuery:
    """Test state query functionality."""

    @pytest.mark.asyncio
    async def test_get_state_success(self, ha_service):
        """Test successful state query."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "entity_id": "light.wohnzimmer",
                "state": "on",
                "attributes": {"brightness": 255, "friendly_name": "Wohnzimmer Licht"},
                "last_changed": "2025-01-01T12:00:00",
                "last_updated": "2025-01-01T12:00:00"
            })
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await ha_service.get_state("light.wohnzimmer")

            assert result["success"] is True
            assert result["entity_id"] == "light.wohnzimmer"
            assert result["state"] == "on"
            assert result["attributes"]["brightness"] == 255

    @pytest.mark.asyncio
    async def test_get_state_not_found(self, ha_service):
        """Test state query for non-existent entity."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await ha_service.get_state("light.nonexistent")

            assert result["success"] is False
            assert "nicht gefunden" in result["error"]

    @pytest.mark.asyncio
    async def test_get_state_not_configured(self, ha_service_unconfigured):
        """Test that state query fails when not configured."""
        result = await ha_service_unconfigured.get_state("light.test")

        assert result["success"] is False
        assert "nicht konfiguriert" in result["error"]


class TestEntityListing:
    """Test entity listing functionality."""

    @pytest.mark.asyncio
    async def test_list_entities_all(self, ha_service):
        """Test listing all entities."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=[
                {
                    "entity_id": "light.wohnzimmer",
                    "state": "on",
                    "attributes": {"friendly_name": "Wohnzimmer"}
                },
                {
                    "entity_id": "switch.kaffeemaschine",
                    "state": "off",
                    "attributes": {"friendly_name": "Kaffeemaschine"}
                }
            ])
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await ha_service.list_entities()

            assert result["success"] is True
            assert result["count"] == 2
            assert len(result["entities"]) == 2

    @pytest.mark.asyncio
    async def test_list_entities_filtered_by_domain(self, ha_service):
        """Test listing entities filtered by domain."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=[
                {
                    "entity_id": "light.wohnzimmer",
                    "state": "on",
                    "attributes": {"friendly_name": "Wohnzimmer"}
                },
                {
                    "entity_id": "light.schlafzimmer",
                    "state": "off",
                    "attributes": {"friendly_name": "Schlafzimmer"}
                },
                {
                    "entity_id": "switch.kaffeemaschine",
                    "state": "off",
                    "attributes": {"friendly_name": "Kaffeemaschine"}
                }
            ])
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await ha_service.list_entities(domain="light")

            assert result["success"] is True
            assert result["count"] == 2
            assert result["domain_filter"] == "light"
            assert all(e["entity_id"].startswith("light.") for e in result["entities"])


class TestSingletonPattern:
    """Test singleton pattern for service instance."""

    def test_get_ha_service_singleton(self):
        """Test that get_ha_service returns same instance."""
        # Reset singleton for test
        import backend.services.home_assistant as ha_module
        ha_module._instance = None

        service1 = get_ha_service(url="http://test1:8123", token="token1")
        service2 = get_ha_service(url="http://test2:8123", token="token2")

        # Should be same instance (singleton pattern)
        assert service1 is service2
        # First initialization wins
        assert service1.url == "http://test1:8123"
