"""
Tests for Home Assistant LLM Tool Integration

Tests that verify LLM can call Home Assistant functions via tools.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from backend.tools.home_assistant_tools import (
    HomeAssistantControlTool,
    HomeAssistantStateTool,
    HomeAssistantListTool,
    get_home_assistant_tools,
    is_home_assistant_enabled
)


@pytest.fixture
def mock_ha_service():
    """Mock Home Assistant service."""
    service = MagicMock()
    service.is_enabled.return_value = True
    service.control_device = AsyncMock()
    service.get_state = AsyncMock()
    service.list_entities = AsyncMock()
    return service


class TestHomeAssistantControlTool:
    """Test Home Assistant control tool."""

    @pytest.mark.asyncio
    async def test_tool_turn_on_light(self, mock_ha_service):
        """Test LLM can turn on a light."""
        mock_ha_service.control_device.return_value = {
            "success": True,
            "entity_id": "light.wohnzimmer",
            "action": "turn_on"
        }

        with patch('backend.tools.home_assistant_tools.get_ha_service', return_value=mock_ha_service):
            tool = HomeAssistantControlTool()
            result = await tool._arun("light.wohnzimmer", "turn_on")

            assert "✅" in result
            assert "light.wohnzimmer" in result
            assert "eingeschaltet" in result
            mock_ha_service.control_device.assert_called_once_with("light.wohnzimmer", "turn_on", None)

    @pytest.mark.asyncio
    async def test_tool_set_brightness(self, mock_ha_service):
        """Test LLM can set brightness with value."""
        mock_ha_service.control_device.return_value = {
            "success": True,
            "entity_id": "light.wohnzimmer",
            "action": "set_brightness"
        }

        with patch('backend.tools.home_assistant_tools.get_ha_service', return_value=mock_ha_service):
            tool = HomeAssistantControlTool()
            result = await tool._arun("light.wohnzimmer", "set_brightness", 128)

            assert "✅" in result
            assert "128" in result
            mock_ha_service.control_device.assert_called_once_with("light.wohnzimmer", "set_brightness", 128)

    @pytest.mark.asyncio
    async def test_tool_set_temperature(self, mock_ha_service):
        """Test LLM can set temperature."""
        mock_ha_service.control_device.return_value = {
            "success": True,
            "entity_id": "climate.heizung",
            "action": "set_temperature"
        }

        with patch('backend.tools.home_assistant_tools.get_ha_service', return_value=mock_ha_service):
            tool = HomeAssistantControlTool()
            result = await tool._arun("climate.heizung", "set_temperature", 22.5)

            assert "✅" in result
            assert "22.5" in result
            mock_ha_service.control_device.assert_called_once_with("climate.heizung", "set_temperature", 22.5)

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, mock_ha_service):
        """Test tool handles errors gracefully."""
        mock_ha_service.control_device.return_value = {
            "success": False,
            "error": "Entity not found"
        }

        with patch('backend.tools.home_assistant_tools.get_ha_service', return_value=mock_ha_service):
            tool = HomeAssistantControlTool()
            result = await tool._arun("light.nonexistent", "turn_on")

            assert "❌" in result
            assert "Entity not found" in result

    @pytest.mark.asyncio
    async def test_tool_not_configured(self):
        """Test tool when HA is not configured."""
        mock_service = MagicMock()
        mock_service.is_enabled.return_value = False

        with patch('backend.tools.home_assistant_tools.get_ha_service', return_value=mock_service):
            tool = HomeAssistantControlTool()
            result = await tool._arun("light.test", "turn_on")

            assert "❌" in result
            assert "nicht konfiguriert" in result


class TestHomeAssistantStateTool:
    """Test Home Assistant state query tool."""

    @pytest.mark.asyncio
    async def test_tool_get_light_state(self, mock_ha_service):
        """Test LLM can query light state."""
        mock_ha_service.get_state.return_value = {
            "success": True,
            "entity_id": "light.wohnzimmer",
            "state": "on",
            "attributes": {
                "brightness": 255,
                "friendly_name": "Wohnzimmer Licht"
            }
        }

        with patch('backend.tools.home_assistant_tools.get_ha_service', return_value=mock_ha_service):
            tool = HomeAssistantStateTool()
            result = await tool._arun("light.wohnzimmer")

            assert "📊" in result
            assert "light.wohnzimmer" in result
            assert "on" in result
            assert "100%" in result  # Brightness at 255 = 100%
            mock_ha_service.get_state.assert_called_once_with("light.wohnzimmer")

    @pytest.mark.asyncio
    async def test_tool_get_climate_state(self, mock_ha_service):
        """Test LLM can query climate state."""
        mock_ha_service.get_state.return_value = {
            "success": True,
            "entity_id": "climate.heizung",
            "state": "heat",
            "attributes": {
                "temperature": 22.0,
                "current_temperature": 21.5,
                "friendly_name": "Heizung"
            }
        }

        with patch('backend.tools.home_assistant_tools.get_ha_service', return_value=mock_ha_service):
            tool = HomeAssistantStateTool()
            result = await tool._arun("climate.heizung")

            assert "📊" in result
            assert "22.0°C" in result  # Target temperature
            assert "21.5°C" in result  # Current temperature


class TestHomeAssistantListTool:
    """Test Home Assistant entity listing tool."""

    @pytest.mark.asyncio
    async def test_tool_list_all_entities(self, mock_ha_service):
        """Test LLM can list all entities."""
        mock_ha_service.list_entities.return_value = {
            "success": True,
            "count": 2,
            "entities": [
                {"entity_id": "light.wohnzimmer", "state": "on", "friendly_name": "Wohnzimmer"},
                {"entity_id": "switch.kaffeemaschine", "state": "off", "friendly_name": "Kaffeemaschine"}
            ]
        }

        with patch('backend.tools.home_assistant_tools.get_ha_service', return_value=mock_ha_service):
            tool = HomeAssistantListTool()
            result = await tool._arun(None)

            assert "📋" in result
            assert "2 gesamt" in result
            assert "light.wohnzimmer" in result
            assert "switch.kaffeemaschine" in result
            mock_ha_service.list_entities.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_tool_list_filtered_by_domain(self, mock_ha_service):
        """Test LLM can list entities filtered by domain."""
        mock_ha_service.list_entities.return_value = {
            "success": True,
            "count": 1,
            "domain_filter": "light",
            "entities": [
                {"entity_id": "light.wohnzimmer", "state": "on", "friendly_name": "Wohnzimmer"}
            ]
        }

        with patch('backend.tools.home_assistant_tools.get_ha_service', return_value=mock_ha_service):
            tool = HomeAssistantListTool()
            result = await tool._arun("light")

            assert "📋" in result
            assert "Typ: light" in result
            assert "light.wohnzimmer" in result
            mock_ha_service.list_entities.assert_called_once_with("light")


class TestToolIntegration:
    """Test full tool integration."""

    def test_get_all_tools(self):
        """Test that all tools are returned."""
        tools = get_home_assistant_tools()

        assert len(tools) == 3
        assert any(isinstance(t, HomeAssistantControlTool) for t in tools)
        assert any(isinstance(t, HomeAssistantStateTool) for t in tools)
        assert any(isinstance(t, HomeAssistantListTool) for t in tools)

    def test_tool_names_unique(self):
        """Test that tool names are unique."""
        tools = get_home_assistant_tools()
        names = [t.name for t in tools]

        assert len(names) == len(set(names))  # No duplicates

    def test_tool_descriptions_exist(self):
        """Test that all tools have descriptions."""
        tools = get_home_assistant_tools()

        for tool in tools:
            assert tool.description
            assert len(tool.description) > 20  # Meaningful description

    def test_is_home_assistant_enabled(self):
        """Test HA enabled check."""
        mock_service = MagicMock()
        mock_service.is_enabled.return_value = True

        with patch('backend.tools.home_assistant_tools.get_ha_service', return_value=mock_service):
            assert is_home_assistant_enabled() is True

    def test_is_home_assistant_disabled(self):
        """Test HA disabled check."""
        mock_service = MagicMock()
        mock_service.is_enabled.return_value = False

        with patch('backend.tools.home_assistant_tools.get_ha_service', return_value=mock_service):
            assert is_home_assistant_enabled() is False


class TestToolSchemas:
    """Test tool input schemas."""

    def test_control_tool_schema(self):
        """Test ControlDeviceInput schema."""
        tool = HomeAssistantControlTool()
        schema = tool.args_schema

        assert schema is not None
        assert "entity_id" in schema.model_fields
        assert "action" in schema.model_fields
        assert "value" in schema.model_fields

    def test_state_tool_schema(self):
        """Test GetStateInput schema."""
        tool = HomeAssistantStateTool()
        schema = tool.args_schema

        assert schema is not None
        assert "entity_id" in schema.model_fields

    def test_list_tool_schema(self):
        """Test ListEntitiesInput schema."""
        tool = HomeAssistantListTool()
        schema = tool.args_schema

        assert schema is not None
        assert "domain" in schema.model_fields
