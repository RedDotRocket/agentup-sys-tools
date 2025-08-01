import json
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest  # noqa: F401
from agent.plugins import CapabilityType, PluginDefinition, PluginValidationResult

from sys_tools.plugin import Plugin
from sys_tools.security import SecurityManager


class TestBasicFunctionality:
    """Test basic plugin functionality."""

    def test_plugin_registration(self):
        """Test that the plugin registers correctly."""
        plugin = Plugin()
        capabilities = plugin.register_capability()

        # Should return a list of capabilities
        assert isinstance(capabilities, list)
        assert len(capabilities) > 0  # Should have at least one capability

        # Check first capability (should be the main sys_tools capability)
        main_capability = capabilities[0]
        assert isinstance(main_capability, PluginDefinition)
        assert main_capability.id == "sys_tools"
        assert main_capability.name == "System Tools"
        assert CapabilityType.TEXT in main_capability.capabilities
        assert CapabilityType.AI_FUNCTION in main_capability.capabilities
        assert "system-tools" in main_capability.tags

    def test_config_validation(self):
        """Test configuration validation."""
        plugin = Plugin()

        # Valid config
        config = {
            "max_file_size": 5242880,  # 5MB
            "allow_command_execution": True,
        }

        result = plugin.validate_config(config)
        assert isinstance(result, PluginValidationResult)
        assert result.valid
        assert len(result.errors) == 0

    def test_ai_functions_registration(self):
        """Test AI function registration."""
        plugin = Plugin()
        functions = plugin.get_ai_functions()

        assert len(functions) == 11  # We registered 11 functions

        # Check function names
        function_names = [f.name for f in functions]
        expected_functions = [
            "file_read",
            "file_write",
            "file_exists",
            "file_info",
            "list_directory",
            "create_directory",
            "delete_file",
            "system_info",
            "working_directory",
            "execute_command",
        ]

        for func_name in expected_functions:
            assert func_name in function_names

    async def test_system_operations(self):
        """Test system operations that don't require file system access."""
        plugin = Plugin()

        # Test system info
        result = await plugin._system_info_internal()
        assert result["success"]
        assert "platform" in result["data"]
        assert "python_version" in result["data"]

        # Test working directory
        result = await plugin._working_directory_internal()
        assert result["success"]
        assert "path" in result["data"]

    async def test_command_execution(self):
        """Test safe command execution."""
        plugin = Plugin()

        # Test allowed command
        result = await plugin._execute_command_internal("echo 'Hello World'")
        assert result["success"]
        assert "Hello World" in result["data"]["stdout"]
        assert result["data"]["returncode"] == 0

        # Test disallowed command
        result = await plugin._execute_command_internal("rm -rf /")
        assert not result["success"]
        assert "not in allowed list" in result["error"]

    async def test_file_operations_with_temp_workspace(self):
        """Test file operations with a temporary workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)

            # Create plugin with custom security manager
            plugin = Plugin()
            plugin.security = SecurityManager(workspace_dir=str(temp_dir))

            # Test write file
            result = await plugin._file_write_internal("test.txt", "Hello, World!")
            assert result["success"]
            assert not result["data"]["overwritten"]

            # Verify file was created
            test_file = temp_dir / "test.txt"
            assert test_file.exists()
            assert test_file.read_text() == "Hello, World!"

            # Test read file
            result = await plugin._internal_file_read("test.txt")
            assert result["success"]
            assert result["data"]["content"] == "Hello, World!"

            # Test file exists
            result = await plugin._file_exists_internal("test.txt")
            assert result["success"]
            assert result["data"]["exists"]
            assert result["data"]["is_file"]

            # Test get file info
            result = await plugin._file_info_internal("test.txt")
            assert result["success"]
            assert result["data"]["name"] == "test.txt"
            assert result["data"]["size"] == len("Hello, World!")

            # Test directory operations
            result = await plugin._create_directory_internal("newdir")
            assert result["success"]

            new_dir = temp_dir / "newdir"
            assert new_dir.exists()
            assert new_dir.is_dir()

            # Test list directory
            result = await plugin._list_directory_internal(".")
            assert result["success"]
            assert result["data"]["count"] == 2  # test.txt and newdir

            names = [e["name"] for e in result["data"]["entries"]]
            assert "test.txt" in names
            assert "newdir" in names

            # Test delete file
            result = await plugin._delete_file_internal("test.txt")
            assert result["success"]
            assert not test_file.exists()

    async def test_security_features(self):
        """Test security features."""
        plugin = Plugin()

        # Test path traversal prevention
        result = await plugin._internal_file_read("../../../etc/passwd")
        assert not result["success"]
        assert "dangerous" in result["error"].lower()

        # Test disallowed command (use a command that's definitely not allowed)
        result = await plugin._execute_command_internal("sudo rm -rf /")
        assert not result["success"]
        assert "not in allowed list" in result["error"]

    async def test_natural_language_execution(self):
        """Test natural language execution."""
        plugin = Plugin()

        # Create a mock context without function call
        context = Mock()
        context.function_call = None
        context.task = Mock()
        context.task.history = [Mock(parts=[Mock(text="help me with files")])]

        result = await plugin.execute_capability(context)
        assert result.success
        assert "Available operations" in result.content

    async def test_ai_function_execution(self):
        """Test AI function execution via AI function wrapper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_dir = Path(tmpdir)
            test_file = temp_dir / "func_test.txt"
            test_file.write_text("Function test")

            # Create plugin with custom workspace
            plugin = Plugin()
            plugin.security = SecurityManager(workspace_dir=str(temp_dir))

            # Create proper context with task metadata (AgentUp's parameter passing)
            task = Mock()
            task.metadata = {"path": "func_test.txt"}

            context = Mock()
            context.task = task
            context.metadata = {"parameters": {}}

            # Test the AI function wrapper directly
            result = await plugin._ai_file_read(task, context)
            assert result.success

            data = json.loads(result.content)
            assert data["success"]
            assert data["data"]["content"] == "Function test"
