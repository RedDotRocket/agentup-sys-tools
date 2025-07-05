"""Tests for System Tools plugin."""

import pytest
from agent.plugins.models import SkillContext, SkillInfo
from system_tools.plugin import Plugin


def test_plugin_registration():
    """Test that the plugin registers correctly."""
    plugin = Plugin()
    skill_info = plugin.register_skill()

    assert isinstance(skill_info, SkillInfo)
    assert skill_info.id == "system_tools"
    assert skill_info.name == "System Tools"


def test_plugin_execution():
    """Test basic plugin execution."""
    plugin = Plugin()

    # Create a mock context
    from unittest.mock import Mock
    task = Mock()
    context = SkillContext(task=task)

    result = plugin.execute_skill(context)

    assert result.success
    assert result.content
