import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from agent.plugins import SkillCapability, SkillContext, SkillInfo, ValidationResult

from sys_tools.plugin import Plugin


class TestPluginRegistration:
    """Test plugin registration and configuration."""

    def test_plugin_registration(self):
        """Test that the plugin registers correctly."""
        plugin = Plugin()
        skill_info = plugin.register_skill()

        assert isinstance(skill_info, SkillInfo)
        assert skill_info.id == "sys_tools"
        assert skill_info.name == "System Tools"
        assert skill_info.version == "0.1.0"
        assert SkillCapability.TEXT in skill_info.capabilities
        assert SkillCapability.AI_FUNCTION in skill_info.capabilities
        assert "system-tools" in skill_info.tags

    def test_config_validation_valid(self):
        """Test configuration validation with valid config."""
        plugin = Plugin()

        # Valid config
        config = {
            "max_file_size": 5242880,  # 5MB
            "allow_command_execution": True,
        }

        result = plugin.validate_config(config)
        assert isinstance(result, ValidationResult)
        assert result.valid
        assert len(result.errors) == 0

    def test_config_validation_invalid(self):
        """Test configuration validation with invalid config."""
        plugin = Plugin()

        # Invalid config
        config = {"workspace_dir": "/nonexistent/path", "max_file_size": -1}

        result = plugin.validate_config(config)
        assert not result.valid
        assert len(result.errors) > 0
        assert any("does not exist" in error for error in result.errors)
        assert any("positive integer" in error for error in result.errors)

    def test_can_handle_task(self):
        """Test task routing logic."""
        plugin = Plugin()

        # Create contexts with different inputs
        def create_context(text):
            task = Mock()
            task.history = [Mock(parts=[Mock(text=text)])]
            return SkillContext(task=task)

        # High confidence tasks
        assert plugin.can_handle_task(create_context("read file test.txt")) == 1.0
        assert plugin.can_handle_task(create_context("list directory")) == 1.0
        assert plugin.can_handle_task(create_context("get system info")) == 1.0

        # Medium confidence tasks
        assert (
            plugin.can_handle_task(create_context("list files in this directory"))
            == 1.0
        )  # 'list files' matches exactly
        assert (
            plugin.can_handle_task(create_context("what's in this folder")) >= 0.8
        )  # 'folder' keyword

        # Low/no confidence tasks
        assert plugin.can_handle_task(create_context("hello world")) == 0.0
        assert plugin.can_handle_task(create_context("calculate 2+2")) == 0.0


class TestFileOperations:
    """Test file operations."""

    @pytest.fixture
    def plugin(self):
        """Create plugin instance."""
        return Plugin()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    async def test_read_file_success(self, plugin, temp_dir):
        """Test successful file reading."""
        # Create test file
        test_file = temp_dir / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)

        # Replace security manager with one that uses temp_dir as workspace
        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        result = await plugin._internal_read_file("test.txt")

        assert result["success"]
        assert result["data"]["content"] == test_content
        assert result["data"]["path"].endswith("test.txt")

    async def test_read_file_not_found(self, plugin, temp_dir):
        """Test reading non-existent file."""
        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        result = await plugin._internal_read_file("nonexistent.txt")

        assert not result["success"]
        assert "not found" in result["error"].lower()

    async def test_write_file_success(self, plugin, temp_dir):
        """Test successful file writing."""
        test_content = "Test content"

        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        result = await plugin._write_file_internal("output.txt", test_content)

        assert result["success"]
        test_file = temp_dir / "output.txt"
        assert test_file.exists()
        assert test_file.read_text() == test_content
        assert not result["data"]["overwritten"]

    async def test_write_file_overwrite(self, plugin, temp_dir):
        """Test file overwriting."""
        test_file = temp_dir / "existing.txt"
        test_file.write_text("Old content")
        new_content = "New content"

        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        result = await plugin._write_file_internal("existing.txt", new_content)

        assert result["success"]
        assert test_file.read_text() == new_content
        assert result["data"]["overwritten"]

    async def test_file_exists(self, plugin, temp_dir):
        """Test file existence check."""
        # Create test file
        test_file = temp_dir / "exists.txt"
        test_file.write_text("content")

        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        # Check existing file
        result = await plugin._file_exists_internal("exists.txt")
        assert result["success"]
        assert result["data"]["exists"]
        assert result["data"]["is_file"]
        assert not result["data"]["is_directory"]

        # Check non-existent file
        result = await plugin._file_exists_internal("nonexistent.txt")
        assert result["success"]
        assert not result["data"]["exists"]

    async def test_get_file_info(self, plugin, temp_dir):
        """Test getting file information."""
        test_file = temp_dir / "info.txt"
        test_content = "File info test"
        test_file.write_text(test_content)

        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        result = await plugin._get_file_info_internal("info.txt")

        assert result["success"]
        data = result["data"]
        assert data["name"] == "info.txt"
        assert data["size"] == len(test_content)
        assert data["is_file"]
        assert not data["is_directory"]
        assert "permissions" in data
        assert "modified" in data


class TestDirectoryOperations:
    """Test directory operations."""

    @pytest.fixture
    def plugin(self):
        """Create plugin instance."""
        return Plugin()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    async def test_list_directory(self, plugin, temp_dir):
        """Test directory listing."""
        # Create test structure
        (temp_dir / "file1.txt").write_text("content1")
        (temp_dir / "file2.py").write_text("content2")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "file3.txt").write_text("content3")

        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        # List root
        result = await plugin._list_directory_internal(".")
        assert result["success"]
        assert result["data"]["count"] == 3

        names = [e["name"] for e in result["data"]["entries"]]
        assert "file1.txt" in names
        assert "file2.py" in names
        assert "subdir" in names

        # List with pattern
        result = await plugin._list_directory_internal(".", pattern="*.txt")
        assert result["success"]
        assert result["data"]["count"] == 1
        assert result["data"]["entries"][0]["name"] == "file1.txt"

    async def test_list_directory_recursive(self, plugin, temp_dir):
        """Test recursive directory listing."""
        # Create nested structure
        (temp_dir / "a" / "b" / "c").mkdir(parents=True)
        (temp_dir / "file1.txt").write_text("1")
        (temp_dir / "a" / "file2.txt").write_text("2")
        (temp_dir / "a" / "b" / "file3.txt").write_text("3")

        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        result = await plugin._list_directory_internal(
            ".", pattern="*.txt", recursive=True
        )

        assert result["success"]
        assert result["data"]["count"] == 3
        paths = [e["path"] for e in result["data"]["entries"]]
        assert "file1.txt" in paths
        assert str(Path("a") / "file2.txt") in paths
        assert str(Path("a") / "b" / "file3.txt") in paths

    async def test_create_directory(self, plugin, temp_dir):
        """Test directory creation."""
        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        result = await plugin._create_directory_internal("newdir")

        assert result["success"]
        new_dir = temp_dir / "newdir"
        assert new_dir.exists()
        assert new_dir.is_dir()

    async def test_create_directory_nested(self, plugin, temp_dir):
        """Test nested directory creation."""
        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        result = await plugin._create_directory_internal("a/b/c", parents=True)

        assert result["success"]
        nested_dir = temp_dir / "a" / "b" / "c"
        assert nested_dir.exists()
        assert nested_dir.is_dir()

    async def test_delete_file(self, plugin, temp_dir):
        """Test file deletion."""
        test_file = temp_dir / "delete_me.txt"
        test_file.write_text("content")

        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        result = await plugin._delete_file_internal("delete_me.txt")

        assert result["success"]
        assert not test_file.exists()

    async def test_delete_directory(self, plugin, temp_dir):
        """Test directory deletion."""
        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        # Test empty directory
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        result = await plugin._delete_file_internal("empty")
        assert result["success"]
        assert not empty_dir.exists()

        # Test non-empty directory
        full_dir = temp_dir / "full"
        full_dir.mkdir()
        (full_dir / "file.txt").write_text("content")

        # Should fail without recursive
        result = await plugin._delete_file_internal("full")
        assert not result["success"]

        # Should succeed with recursive
        result = await plugin._delete_file_internal("full", recursive=True)
        assert result["success"]
        assert not full_dir.exists()


class TestSystemOperations:
    """Test system operations."""

    @pytest.fixture
    def plugin(self):
        """Create plugin instance."""
        return Plugin()

    async def test_get_system_info(self, plugin):
        """Test getting system information."""
        result = await plugin._get_system_info_internal()

        assert result["success"]
        data = result["data"]
        assert "platform" in data
        assert "python_version" in data
        assert "working_directory" in data
        assert "user" in data

    async def test_get_working_directory(self, plugin):
        """Test getting working directory."""
        result = await plugin._get_working_directory_internal()

        assert result["success"]
        data = result["data"]
        assert "path" in data
        assert "absolute" in data
        assert os.path.exists(data["path"])

    async def test_execute_command_success(self, plugin):
        """Test successful command execution."""
        result = await plugin._execute_command_internal("echo 'Hello World'")

        assert result["success"]
        data = result["data"]
        assert data["returncode"] == 0
        assert "Hello World" in data["stdout"]
        assert data["success"]

    async def test_execute_command_disallowed(self, plugin):
        """Test disallowed command execution."""
        # Try to execute a disallowed command
        result = await plugin._execute_command_internal("rm -rf /")

        assert not result["success"]
        assert "not in allowed list" in result["error"]


class TestSecurityFeatures:
    """Test security features."""

    @pytest.fixture
    def plugin(self):
        """Create plugin instance."""
        return Plugin()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    async def test_path_traversal_prevention(self, plugin):
        """Test that path traversal is prevented."""
        # Try to access parent directory
        result = await plugin._internal_read_file("../../../etc/passwd")

        assert not result["success"]
        assert "dangerous" in result["error"].lower()

    async def test_workspace_restriction(self, plugin, temp_dir):
        """Test workspace directory restriction."""
        # Create file outside workspace
        outside_file = temp_dir.parent / "outside.txt"
        outside_file.write_text("outside content")

        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        # Try to access file outside workspace using relative traversal
        result = await plugin._internal_read_file("../outside.txt")

        assert not result["success"]
        assert "dangerous" in result["error"].lower()

        # Clean up
        outside_file.unlink()

    async def test_file_size_limit(self, plugin, temp_dir):
        """Test file size limit enforcement."""
        # Create large file
        large_file = temp_dir / "large.txt"
        large_content = "x" * (11 * 1024 * 1024)  # 11MB
        large_file.write_text(large_content)

        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        result = await plugin._internal_read_file("large.txt")

        assert not result["success"]
        assert "exceeds maximum" in result["error"]


class TestAIFunctions:
    """Test AI function integration."""

    @pytest.fixture
    def plugin(self):
        """Create plugin instance."""
        return Plugin()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    async def test_get_ai_functions(self, plugin):
        """Test AI function registration."""
        functions = plugin.get_ai_functions()

        assert len(functions) == 10  # We registered 10 functions

        # Check function names
        function_names = [f.name for f in functions]
        assert "read_file" in function_names
        assert "write_file" in function_names
        assert "list_directory" in function_names
        assert "get_system_info" in function_names

        # Check a specific function
        read_func = next(f for f in functions if f.name == "read_file")
        assert read_func.description
        assert "path" in read_func.parameters["properties"]
        assert "path" in read_func.parameters["required"]

    async def test_function_call_execution(self, plugin, temp_dir):
        """Test execution via AI function wrapper."""
        # Create test file
        test_file = temp_dir / "func_test.txt"
        test_file.write_text("Function test")

        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        # Create proper context with task metadata (AgentUp's parameter passing)
        from sys_tools.plugin import SkillContext

        task = Mock()
        task.metadata = {"path": "func_test.txt"}

        context = SkillContext(task=task, metadata={"parameters": {}})

        # Test the AI function wrapper directly
        result = await plugin._ai_read_file(task, context)

        assert result.success
        data = json.loads(result.content)
        assert data["success"]
        assert data["data"]["content"] == "Function test"
