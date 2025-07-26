import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
from agent.plugins import CapabilityType, CapabilityInfo, PluginValidationResult

from sys_tools.plugin import Plugin


class TestPluginRegistration:
    """Test plugin registration and configuration."""

    def test_plugin_registration(self):
        """Test that the plugin registers correctly."""
        plugin = Plugin()
        capabilities = plugin.register_capability()

        # Should return a list of capabilities
        assert isinstance(capabilities, list)
        assert len(capabilities) > 0  # Should have at least one capability

        # Check first capability (should be the main sys_tools capability)
        main_capability = capabilities[0]
        assert isinstance(main_capability, CapabilityInfo)
        assert main_capability.id == "sys_tools"
        assert main_capability.name == "System Tools"
        assert CapabilityType.TEXT in main_capability.capabilities
        assert CapabilityType.AI_FUNCTION in main_capability.capabilities
        assert "system-tools" in main_capability.tags

    def test_config_validation_valid(self):
        """Test configuration validation with valid config."""
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
            context = Mock()
            context.task = Mock()
            context.task.history = [Mock()]
            context.task.history[0].parts = [Mock()]
            context.task.history[0].parts[0].root = Mock()
            context.task.history[0].parts[0].root.text = text
            return context

        # High confidence tasks
        assert plugin.can_handle_task(create_context("read file test.txt")) == 1.0
        assert plugin.can_handle_task(create_context("list directory")) == 1.0
        assert plugin.can_handle_task(create_context("get system info")) == 1.0
        assert plugin.can_handle_task(create_context("calculate hash of file")) == 1.0
        assert plugin.can_handle_task(create_context("get sha256 checksum")) == 1.0

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

    async def test_file_read_success(self, plugin, temp_dir):
        """Test successful file reading."""
        # Create test file
        test_file = temp_dir / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)

        # Replace security manager with one that uses temp_dir as workspace
        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        result = await plugin._internal_file_read("test.txt")

        assert result["success"]
        assert result["data"]["content"] == test_content
        assert result["data"]["path"].endswith("test.txt")

    async def test_file_read_not_found(self, plugin, temp_dir):
        """Test reading non-existent file."""
        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        result = await plugin._internal_file_read("nonexistent.txt")

        assert not result["success"]
        assert "not found" in result["error"].lower()

    async def test_file_write_success(self, plugin, temp_dir):
        """Test successful file writing."""
        test_content = "Test content"

        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        result = await plugin._file_write_internal("output.txt", test_content)

        assert result["success"]
        test_file = temp_dir / "output.txt"
        assert test_file.exists()
        assert test_file.read_text() == test_content
        assert not result["data"]["overwritten"]

    async def test_file_write_overwrite(self, plugin, temp_dir):
        """Test file overwriting."""
        test_file = temp_dir / "existing.txt"
        test_file.write_text("Old content")
        new_content = "New content"

        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        result = await plugin._file_write_internal("existing.txt", new_content)

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

    async def test_file_info(self, plugin, temp_dir):
        """Test getting file information."""
        test_file = temp_dir / "info.txt"
        test_content = "File info test"
        test_file.write_text(test_content)

        from sys_tools.security import SecurityManager

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))

        result = await plugin._file_info_internal("info.txt")

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

    async def test_system_info(self, plugin):
        """Test getting system information."""
        result = await plugin._system_info_internal()

        assert result["success"]
        data = result["data"]
        assert "platform" in data
        assert "python_version" in data
        assert "working_directory" in data
        assert "user" in data

    async def test_working_directory(self, plugin):
        """Test getting working directory."""
        result = await plugin._working_directory_internal()

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
        result = await plugin._internal_file_read("../../../etc/passwd")

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
        result = await plugin._internal_file_read("../outside.txt")

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

        result = await plugin._internal_file_read("large.txt")

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

        assert len(functions) == 11  # We registered 11 functions

        # Check function names
        function_names = [f.name for f in functions]
        assert "file_read" in function_names
        assert "file_write" in function_names
        assert "list_directory" in function_names
        assert "system_info" in function_names

        # Check a specific function
        read_func = next(f for f in functions if f.name == "file_read")
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


class TestFileHashing:
    """Test file hashing operations."""

    @pytest.fixture
    def plugin(self):
        """Create plugin instance."""
        return Plugin()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def _setup_plugin_security(self, plugin, temp_dir):
        """Helper to set up plugin security and hasher for test directory."""
        from sys_tools.security import SecurityManager
        from sys_tools.hashing import FileHasher

        plugin.security = SecurityManager(workspace_dir=str(temp_dir))
        plugin.hasher = FileHasher(plugin.security)

    async def test_hash_single_algorithm(self, plugin, temp_dir):
        """Test hashing with a single algorithm."""
        # Create test file with known content
        test_file = temp_dir / "hash_test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)

        self._setup_plugin_security(plugin, temp_dir)

        # Test default SHA256
        result = await plugin._file_hash_internal("hash_test.txt")

        assert result["success"]
        data = result["data"]
        assert "sha256" in data["hashes"]
        assert data["hashes"]["sha256"]["algorithm"] == "sha256"
        assert data["hashes"]["sha256"]["format"] == "hex"
        # Known SHA256 hash for "Hello, World!"
        assert (
            data["hashes"]["sha256"]["digest"]
            == "dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f"
        )

        # Check file info is included by default
        assert "file_info" in data
        assert data["file_info"]["name"] == "hash_test.txt"
        assert data["file_info"]["size"] == len(test_content)

    async def test_hash_multiple_algorithms(self, plugin, temp_dir):
        """Test hashing with multiple algorithms."""
        test_file = temp_dir / "multi_hash.txt"
        test_content = "Test content for multiple hashes"
        test_file.write_text(test_content)

        self._setup_plugin_security(plugin, temp_dir)

        result = await plugin._file_hash_internal(
            "multi_hash.txt", algorithms=["md5", "sha1", "sha256", "sha512"]
        )

        assert result["success"]
        data = result["data"]

        # Check all algorithms are present
        assert len(data["hashes"]) == 4
        assert "md5" in data["hashes"]
        assert "sha1" in data["hashes"]
        assert "sha256" in data["hashes"]
        assert "sha512" in data["hashes"]

        # Verify each hash has correct structure
        for algo in ["md5", "sha1", "sha256", "sha512"]:
            assert data["hashes"][algo]["algorithm"] == algo
            assert data["hashes"][algo]["format"] == "hex"
            assert len(data["hashes"][algo]["digest"]) > 0

    async def test_hash_base64_output(self, plugin, temp_dir):
        """Test hashing with base64 output format."""
        test_file = temp_dir / "base64_test.txt"
        test_file.write_text("Base64 test content")

        self._setup_plugin_security(plugin, temp_dir)

        result = await plugin._file_hash_internal(
            "base64_test.txt", algorithms=["sha256"], output_format="base64"
        )

        assert result["success"]
        data = result["data"]
        assert data["hashes"]["sha256"]["format"] == "base64"
        # Base64 strings should not contain only hex characters
        digest = data["hashes"]["sha256"]["digest"]
        assert any(c not in "0123456789abcdef" for c in digest.lower())
        assert digest.endswith("=") or len(digest) % 4 == 0  # Valid base64

    async def test_hash_without_file_info(self, plugin, temp_dir):
        """Test hashing without file information."""
        test_file = temp_dir / "no_info.txt"
        test_file.write_text("Content without info")

        self._setup_plugin_security(plugin, temp_dir)

        result = await plugin._file_hash_internal(
            "no_info.txt", include_file_info=False
        )

        assert result["success"]
        data = result["data"]
        assert "file_info" not in data
        assert "hashes" in data
        assert "sha256" in data["hashes"]

    async def test_hash_nonexistent_file(self, plugin, temp_dir):
        """Test hashing non-existent file."""
        self._setup_plugin_security(plugin, temp_dir)

        result = await plugin._file_hash_internal("nonexistent.txt")

        assert not result["success"]
        assert "not found" in result["error"].lower()

    async def test_hash_directory_error(self, plugin, temp_dir):
        """Test hashing a directory (should fail)."""
        test_dir = temp_dir / "subdir"
        test_dir.mkdir()

        self._setup_plugin_security(plugin, temp_dir)

        result = await plugin._file_hash_internal("subdir")

        assert not result["success"]
        assert "not a file" in result["error"].lower()

    async def test_hash_invalid_algorithm(self, plugin, temp_dir):
        """Test hashing with invalid algorithm."""
        test_file = temp_dir / "invalid_algo.txt"
        test_file.write_text("Test content")

        self._setup_plugin_security(plugin, temp_dir)

        result = await plugin._file_hash_internal(
            "invalid_algo.txt", algorithms=["invalid_algo"]
        )

        assert not result["success"]
        assert "unsupported algorithm" in result["error"].lower()

    async def test_hash_large_file(self, plugin, temp_dir):
        """Test hashing handles large files efficiently."""
        # Create a 1MB file
        test_file = temp_dir / "large.bin"
        large_content = b"x" * (1024 * 1024)  # 1MB
        test_file.write_bytes(large_content)

        self._setup_plugin_security(plugin, temp_dir)

        result = await plugin._file_hash_internal("large.bin")

        assert result["success"]
        data = result["data"]
        assert "sha256" in data["hashes"]
        assert data["file_info"]["size"] == 1024 * 1024

    async def test_ai_function_file_hash(self, plugin, temp_dir):
        """Test AI function wrapper for file_hash."""
        test_file = temp_dir / "ai_test.txt"
        test_file.write_text("AI function test")

        self._setup_plugin_security(plugin, temp_dir)

        # Check that file_hash is in AI functions
        functions = plugin.get_ai_functions()
        hash_func = next((f for f in functions if f.name == "file_hash"), None)
        assert hash_func is not None
        assert "path" in hash_func.parameters["properties"]
        assert "algorithms" in hash_func.parameters["properties"]
        assert "output_format" in hash_func.parameters["properties"]

        # Test AI function execution
        task = Mock()
        task.metadata = {"path": "ai_test.txt", "algorithms": ["sha256", "md5"]}

        context = Mock()
        context.task = task
        context.metadata = {"parameters": {}}

        result = await plugin._ai_file_hash(task, context)

        assert result.success
        data = json.loads(result.content)
        assert data["success"]
        assert len(data["data"]["hashes"]) == 2
        assert "sha256" in data["data"]["hashes"]
        assert "md5" in data["data"]["hashes"]
