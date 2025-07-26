import json
import os
import platform
import shutil
import subprocess
from typing import Any

import pluggy
from agent.plugins import (
    AIFunction,
    CapabilityType,
    CapabilityContext,
    CapabilityInfo,
    CapabilityResult,
    PluginValidationResult,
)

from .hashing import FileHasher
from .security import SecurityError, SecurityManager
from .utils import (
    create_error_response,
    create_success_response,
    format_file_size,
    format_timestamp,
    get_file_permissions,
    get_file_type,
    safe_read_text,
    safe_write_text,
)

hookimpl = pluggy.HookimplMarker("agentup")


# Capability configuration data with required scopes for each capability
CAPABILITIES_CONFIG = [
    {
        "id": "file_read",
        "name": "File Read",
        "description": "Read contents of files",
        "capabilities": [CapabilityType.AI_FUNCTION, CapabilityType.TEXT],
        "tags": ["files", "read", "io"],
        "required_scopes": ["files:read"],  # Plugin DECLARES scope requirement
        "ai_function": {
            "name": "file_read",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to read"},
                    "encoding": {"type": "string", "description": "Text encoding (default: utf-8)", "default": "utf-8"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "id": "file_write",
        "name": "File Write",
        "description": "Write content to files",
        "capabilities": [CapabilityType.AI_FUNCTION, CapabilityType.TEXT],
        "tags": ["files", "write", "io"],
        "required_scopes": ["files:write"],  # Plugin DECLARES scope requirement
        "ai_function": {
            "name": "file_write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file to write"},
                    "content": {"type": "string", "description": "Content to write to the file"},
                    "encoding": {"type": "string", "description": "Text encoding (default: utf-8)", "default": "utf-8"},
                    "create_parents": {"type": "boolean", "description": "Create parent directories if needed", "default": True},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "id": "file_exists",
        "name": "File Exists",
        "description": "Check if a file or directory exists",
        "capabilities": [CapabilityType.AI_FUNCTION],
        "tags": ["files", "check"],
        "required_scopes": ["files:read"],  # Plugin DECLARES scope requirement
        "ai_function": {
            "name": "file_exists",
            "description": "Check if a file or directory exists",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to check"}},
                "required": ["path"],
            },
        },
    },
    {
        "id": "file_info",
        "name": "File Info",
        "description": "Get detailed information about a file or directory",
        "capabilities": [CapabilityType.AI_FUNCTION],
        "tags": ["files", "info"],
        "required_scopes": ["files:read"],  # Plugin DECLARES scope requirement
        "ai_function": {
            "name": "file_info",
            "description": "Get detailed information about a file or directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path to the file or directory"}},
                "required": ["path"],
            },
        },
    },
    {
        "id": "list_directory",
        "name": "List Directory",
        "description": "List contents of a directory",
        "capabilities": [CapabilityType.AI_FUNCTION],
        "tags": ["directories", "list"],
        "required_scopes": ["files:read"],  # Plugin DECLARES scope requirement
        "ai_function": {
            "name": "list_directory",
            "description": "List contents of a directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: current directory)", "default": "."},
                    "pattern": {"type": "string", "description": "Glob pattern to filter results (e.g., '*.txt')"},
                    "recursive": {"type": "boolean", "description": "List recursively", "default": False},
                },
            },
        },
    },
    {
        "id": "create_directory",
        "name": "Create Directory",
        "description": "Create a new directory",
        "capabilities": [CapabilityType.AI_FUNCTION],
        "tags": ["directories", "create"],
        "required_scopes": ["files:write"],  # Plugin DECLARES scope requirement
        "ai_function": {
            "name": "create_directory",
            "description": "Create a new directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of directory to create"},
                    "parents": {"type": "boolean", "description": "Create parent directories if needed", "default": True},
                    "exist_ok": {"type": "boolean", "description": "Don't raise error if directory exists", "default": True},
                },
                "required": ["path"],
            },
        },
    },
    {
        "id": "delete_file",
        "name": "Delete File",
        "description": "Delete a file or directory",
        "capabilities": [CapabilityType.AI_FUNCTION],
        "tags": ["files", "delete"],
        "required_scopes": ["files:admin"],  # Plugin DECLARES scope requirement
        "ai_function": {
            "name": "delete_file",
            "description": "Delete a file or directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to delete"},
                    "recursive": {"type": "boolean", "description": "Delete directories recursively", "default": False},
                },
                "required": ["path"],
            },
        },
    },
    {
        "id": "system_info",
        "name": "System Info",
        "description": "Get system and platform information",
        "capabilities": [CapabilityType.AI_FUNCTION],
        "tags": ["system", "info"],
        "required_scopes": ["system:read"],  # Plugin DECLARES scope requirement
        "ai_function": {
            "name": "system_info",
            "description": "Get system and platform information",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "id": "working_directory",
        "name": "Working Directory",
        "description": "Get the current working directory",
        "capabilities": [CapabilityType.AI_FUNCTION],
        "tags": ["system", "directory"],
        "required_scopes": ["system:read"],  # Plugin DECLARES scope requirement
        "ai_function": {
            "name": "working_directory",
            "description": "Get the current working directory",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "id": "execute_command",
        "name": "Execute Command",
        "description": "Execute a safe shell command",
        "capabilities": [CapabilityType.AI_FUNCTION],
        "tags": ["system", "command", "execute"],
        "required_scopes": ["system:admin"],  # Plugin DECLARES scope requirement
        "ai_function": {
            "name": "execute_command",
            "description": "Execute a safe shell command (limited to whitelist)",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                },
                "required": ["command"],
            },
        },
    },
    {
        "id": "file_hash",
        "name": "File Hash",
        "description": "Compute cryptographic hash(es) for a file",
        "capabilities": [CapabilityType.AI_FUNCTION],
        "tags": ["files", "hash", "security"],
        "required_scopes": ["files:read"],  # Plugin DECLARES scope requirement
        "ai_function": {
            "name": "file_hash",
            "description": "Compute cryptographic hash(es) for a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "algorithms": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["md5", "sha1", "sha256", "sha512"]},
                        "description": "Hash algorithms to use",
                        "default": ["sha256"],
                    },
                },
                "required": ["path"],
            },
        },
    },
]


class Plugin:
    """Main plugin class for System Tools."""

    def __init__(self):
        """Initialize the plugin."""
        self.name = "system-tools"
        self.security = SecurityManager()
        self.hasher = FileHasher(self.security)

        # Build handler mapping from configuration using generic wrapper factory
        self._handlers = {
            config["id"]: self._create_ai_wrapper(config["id"], config["ai_function"]["name"])
            for config in CAPABILITIES_CONFIG
        }

    def _create_ai_wrapper(self, capability_id: str, function_name: str):
        """Create a generic AI wrapper function for a capability."""
        # Map function names to their internal implementation methods
        internal_methods = {
            "file_read": self._internal_file_read,
            "file_write": self._file_write_internal,
            "file_exists": self._file_exists_internal,
            "file_info": self._file_info_internal,
            "list_directory": self._list_directory_internal,
            "create_directory": self._create_directory_internal,
            "delete_file": self._delete_file_internal,
            "system_info": self._system_info_internal,
            "working_directory": self._working_directory_internal,
            "execute_command": self._execute_command_internal,
            "file_hash": self._file_hash_internal,
        }

        async def ai_wrapper(task, context: CapabilityContext) -> CapabilityResult:
            """Generic AI function wrapper."""
            # Extract parameters from context
            params = context.metadata.get("parameters", {})
            task_metadata = (
                task.metadata if hasattr(task, "metadata") and task.metadata else {}
            )
            if not params and task_metadata:
                params = task_metadata

            try:
                # Call the appropriate internal method
                internal_method = internal_methods[function_name]
                result = await internal_method(**params)
                return CapabilityResult(
                    content=json.dumps(result, indent=2),
                    success=result.get("success", True),
                    metadata={"capability": "sys_tools", "function": function_name},
                )
            except Exception as e:
                return CapabilityResult(
                    content=json.dumps(create_error_response(e, function_name)),
                    success=False,
                    error=str(e),
                )

        return ai_wrapper

    def _create_base_config_schema(self) -> dict:
        """Create the base configuration schema shared across capabilities."""
        return {
            "type": "object",
            "properties": {
                "workspace_dir": {
                    "type": "string",
                    "description": "Base directory for file operations (for security)"
                },
                "max_file_size": {
                    "type": "integer",
                    "description": "Maximum file size in bytes",
                    "default": 10485760
                },
                "allowed_extensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Allowed file extensions"
                }
            }
        }

    def _create_capability_info(self, config: dict) -> CapabilityInfo:
        """Create a CapabilityInfo object from configuration."""
        return CapabilityInfo(
            id=config["id"],
            name=config["name"],
            version="0.2.0",
            description=config["description"],
            plugin_name="sys_tools",
            capabilities=config["capabilities"],
            tags=config["tags"],
            config_schema=self._create_base_config_schema(),
            required_scopes=config.get("required_scopes", [])  # Plugin DECLARES scope requirement
        )

    @hookimpl
    def register_capability(self) -> list[CapabilityInfo]:
        """Register the system tools capabilities."""
        return [self._create_capability_info(config) for config in CAPABILITIES_CONFIG]

    @hookimpl
    def validate_config(self, config: dict) -> PluginValidationResult:
        """Validate capability configuration."""
        errors = []
        warnings = []

        # Validate workspace directory
        if "workspace_dir" in config:
            workspace = config["workspace_dir"]
            if not os.path.exists(workspace):
                errors.append(f"Workspace directory does not exist: {workspace}")
            elif not os.path.isdir(workspace):
                errors.append(f"Workspace path is not a directory: {workspace}")

        # Validate max file size
        if "max_file_size" in config:
            max_size = config["max_file_size"]
            if not isinstance(max_size, int) or max_size <= 0:
                errors.append("max_file_size must be a positive integer")
            elif max_size < 1024:
                warnings.append("max_file_size is very small (< 1KB)")

        return PluginValidationResult(
            valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    @hookimpl
    def can_handle_task(self, context: CapabilityContext) -> float:
        """Check if this capability can handle the task."""
        user_input = self._extract_user_input(context).lower()

        # Keywords and their confidence scores
        keywords = {
            # File operations
            "read file": 1.0,
            "read": 0.8,
            "open file": 1.0,
            "view file": 1.0,
            "write file": 1.0,
            "write": 0.8,
            "save file": 1.0,
            "create file": 1.0,
            "file exists": 1.0,
            "check file": 0.9,
            "file info": 1.0,
            "delete file": 1.0,
            "remove file": 1.0,
            # Directory operations
            "list directory": 1.0,
            "list files": 1.0,
            "ls": 0.9,
            "dir": 0.9,
            "create directory": 1.0,
            "mkdir": 1.0,
            "make directory": 1.0,
            "folder": 0.8,
            "directory": 0.8,
            # System operations
            "system info": 1.0,
            "system information": 1.0,
            "platform": 0.9,
            "working directory": 1.0,
            "pwd": 1.0,
            "current directory": 1.0,
            "execute": 0.9,
            "run command": 1.0,
            "shell": 0.8,
            # General file system
            "file system": 0.9,
            "filesystem": 0.9,
            "path": 0.7,
            # File hashing
            "hash": 1.0,
            "checksum": 1.0,
            "digest": 1.0,
            "sha256": 1.0,
            "sha512": 1.0,
            "sha1": 1.0,
            "md5": 1.0,
            "hash file": 1.0,
            "file hash": 1.0,
            "calculate hash": 1.0,
            "compute hash": 1.0,
            "verify hash": 1.0,
        }

        confidence = 0.0
        for keyword, score in keywords.items():
            if keyword in user_input:
                confidence = max(confidence, score)

        return confidence

    @hookimpl
    async def execute_capability(self, context: CapabilityContext) -> CapabilityResult:
        """Execute the capability logic."""
        try:
            # Get the specific capability being invoked
            capability_id = context.metadata.get("capability_id", "unknown")

            # Route to specific capability handler using the handlers mapping
            if capability_id in self._handlers:
                handler = self._handlers[capability_id]
                return await handler(context.task, context)
            else:
                # Fallback to natural language processing for unknown capabilities
                user_input = self._extract_user_input(context)
                return self._handle_natural_language(user_input)

        except SecurityError as e:
            return CapabilityResult(
                content=f"Security error: {str(e)}",
                success=False,
                error=str(e),
                metadata={"capability": context.metadata.get("capability_id", "unknown"), "error_type": "security"},
            )
        except Exception as e:
            return CapabilityResult(
                content=f"Error executing system tools: {str(e)}",
                success=False,
                error=str(e),
                metadata={"capability": context.metadata.get("capability_id", "unknown"), "error_type": type(e).__name__},
            )

    def _extract_user_input(self, context: CapabilityContext) -> str:
        """Extract user input from the task context."""
        if hasattr(context.task, "history") and context.task.history:
            last_msg = context.task.history[-1]
            if hasattr(last_msg, "parts") and last_msg.parts:
                return (
                    last_msg.parts[0].text if hasattr(last_msg.parts[0], "text") else ""
                )
        return ""

    def _handle_natural_language(self, user_input: str) -> CapabilityResult:
        """Handle natural language requests."""
        # Try to provide helpful guidance
        suggestions = [
            "Available operations:",
            "- Read file: 'read file <path>'",
            "- Write file: 'write file <path> with content <content>'",
            "- List directory: 'list files in <path>'",
            "- File info: 'get info for file <path>'",
            "- File hash: 'get hash for file <path>' or 'calculate sha256 of <path>'",
            "- System info: 'show system information'",
            "- Current directory: 'what is the current directory'",
            "",
            "Hash algorithms supported: MD5, SHA1, SHA256, SHA512",
            "For best results, use the AI function interface.",
        ]

        return CapabilityResult(
            content="\n".join(suggestions),
            success=True,
            metadata={"capability": "sys_tools", "type": "help"},
        )

    # File Operations
    async def _internal_file_read(
        self, path: str, encoding: str = "utf-8"
    ) -> dict[str, Any]:
        """Read contents of a file."""
        try:
            file_path = self.security.validate_path(path)
            self.security.validate_file_size(file_path)

            if not file_path.exists():
                return create_error_response(
                    FileNotFoundError(f"File not found: {path}"), "file_read"
                )

            if not file_path.is_file():
                return create_error_response(
                    ValueError(f"Path is not a file: {path}"), "file_read"
                )

            content = safe_read_text(file_path, encoding, self.security.max_file_size)

            return create_success_response(
                {
                    "path": str(file_path),
                    "content": content,
                    "encoding": encoding,
                    "size": len(content),
                },
                "file_read",
                f"Successfully read {format_file_size(len(content.encode()))}",
            )

        except Exception as e:
            return create_error_response(e, "file_read")

    async def _file_write(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
    ) -> dict[str, Any]:
        """Write content to a file."""
        try:
            file_path = self.security.validate_path(path)
            content = self.security.sanitize_content(content)

            # Check if we're overwriting
            exists = file_path.exists()

            safe_write_text(file_path, content, encoding, create_parents)

            return create_success_response(
                {
                    "path": str(file_path),
                    "size": len(content.encode()),
                    "encoding": encoding,
                    "overwritten": exists,
                },
                "file_write",
                f"Successfully {'updated' if exists else 'created'} file",
            )

        except Exception as e:
            return create_error_response(e, "file_write")

    async def _file_exists(self, path: str) -> dict[str, Any]:
        """Check if a file exists."""
        try:
            file_path = self.security.validate_path(path)
            exists = file_path.exists()

            return create_success_response(
                {
                    "path": str(file_path),
                    "exists": exists,
                    "is_file": file_path.is_file() if exists else None,
                    "is_directory": file_path.is_dir() if exists else None,
                },
                "file_exists",
            )

        except Exception as e:
            return create_error_response(e, "file_exists")

    async def _file_info(self, path: str) -> dict[str, Any]:
        """Get detailed information about a file."""
        try:
            file_path = self.security.validate_path(path)

            if not file_path.exists():
                return create_error_response(
                    FileNotFoundError(f"Path not found: {path}"), "file_info"
                )

            stat = file_path.stat()

            info = {
                "path": str(file_path),
                "name": file_path.name,
                "type": get_file_type(file_path),
                "size": stat.st_size,
                "size_human": format_file_size(stat.st_size),
                "permissions": get_file_permissions(file_path),
                "created": format_timestamp(stat.st_ctime),
                "modified": format_timestamp(stat.st_mtime),
                "accessed": format_timestamp(stat.st_atime),
                "is_file": file_path.is_file(),
                "is_directory": file_path.is_dir(),
                "is_symlink": file_path.is_symlink(),
            }

            if file_path.is_symlink():
                info["symlink_target"] = str(file_path.readlink())

            return create_success_response(info, "file_info")

        except Exception as e:
            return create_error_response(e, "file_info")

    # Directory Operations
    async def _list_directory(
        self, path: str = ".", pattern: str | None = None, recursive: bool = False
    ) -> dict[str, Any]:
        """List contents of a directory."""
        try:
            dir_path = self.security.validate_path(path)

            if not dir_path.exists():
                return create_error_response(
                    FileNotFoundError(f"Directory not found: {path}"), "list_directory"
                )

            if not dir_path.is_dir():
                return create_error_response(
                    ValueError(f"Path is not a directory: {path}"), "list_directory"
                )

            entries = []

            if recursive:
                # Use rglob for recursive listing
                paths = dir_path.rglob(pattern or "*")
            else:
                # Use glob for non-recursive listing
                paths = dir_path.glob(pattern or "*")

            for entry in sorted(paths):
                try:
                    stat = entry.stat()
                    entries.append(
                        {
                            "name": entry.name,
                            "path": str(entry.relative_to(dir_path)),
                            "type": "directory" if entry.is_dir() else "file",
                            "size": stat.st_size if entry.is_file() else None,
                            "modified": format_timestamp(stat.st_mtime),
                        }
                    )
                except Exception:
                    # Skip entries we can't stat
                    continue

            return create_success_response(
                {"path": str(dir_path), "count": len(entries), "entries": entries},
                "list_directory",
            )

        except Exception as e:
            return create_error_response(e, "list_directory")

    async def _create_directory(
        self, path: str, parents: bool = True, exist_ok: bool = True
    ) -> dict[str, Any]:
        """Create a directory."""
        try:
            dir_path = self.security.validate_path(path)

            if dir_path.exists() and not exist_ok:
                return create_error_response(
                    FileExistsError(f"Directory already exists: {path}"),
                    "create_directory",
                )

            dir_path.mkdir(parents=parents, exist_ok=exist_ok)

            return create_success_response(
                {"path": str(dir_path), "created": True},
                "create_directory",
                f"Directory created: {path}",
            )

        except Exception as e:
            return create_error_response(e, "create_directory")

    async def _delete_file(self, path: str, recursive: bool = False) -> dict[str, Any]:
        """Delete a file or directory."""
        try:
            file_path = self.security.validate_path(path)

            if not file_path.exists():
                return create_error_response(
                    FileNotFoundError(f"Path not found: {path}"), "delete_file"
                )

            if file_path.is_dir():
                if recursive:
                    shutil.rmtree(file_path)
                else:
                    file_path.rmdir()  # Only works for empty directories
            else:
                file_path.unlink()

            return create_success_response(
                {"path": str(file_path), "deleted": True},
                "delete_file",
                f"Successfully deleted: {path}",
            )

        except Exception as e:
            return create_error_response(e, "delete_file")

    # System Operations
    async def _system_info(self) -> dict[str, Any]:
        """Get system information."""
        try:
            info = {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "hostname": platform.node(),
                "python_version": platform.python_version(),
                "working_directory": os.getcwd(),
            }

            # Add OS-specific info
            if platform.system() != "Windows":
                info["user"] = os.environ.get("USER", "unknown")
            else:
                info["user"] = os.environ.get("USERNAME", "unknown")

            return create_success_response(info, "system_info")

        except Exception as e:
            return create_error_response(e, "system_info")

    async def _working_directory(self) -> dict[str, Any]:
        """Get current working directory."""
        try:
            cwd = os.getcwd()
            return create_success_response(
                {"path": cwd, "absolute": os.path.abspath(cwd)}, "working_directory"
            )
        except Exception as e:
            return create_error_response(e, "working_directory")

    async def _execute_command(self, command: str, timeout: int = 30) -> dict[str, Any]:
        """Execute a safe shell command."""
        try:
            # Validate command
            args = self.security.validate_command(command)

            # Execute command
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.security.workspace_dir),
            )

            return create_success_response(
                {
                    "command": command,
                    "args": args,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "success": result.returncode == 0,
                },
                "execute_command",
            )

        except subprocess.TimeoutExpired:
            return create_error_response(
                TimeoutError(f"Command timed out after {timeout} seconds"),
                "execute_command",
            )
        except Exception as e:
            return create_error_response(e, "execute_command")

    async def _file_hash_internal(
        self,
        path: str,
        algorithms: list[str] | None = None,
        output_format: str = "hex",
        include_file_info: bool = True,
    ) -> dict[str, Any]:
        """Internal file_hash implementation."""
        try:
            # Initialize hasher if not already done
            if not hasattr(self, "hasher"):
                self.hasher = FileHasher(self.security)

            # Use the hasher to compute file hash(es)
            result = self.hasher.hash_file_with_info(
                path, algorithms, output_format, include_file_info
            )
            return result
        except Exception as e:
            return create_error_response(e, "file_hash")

    # Direct method interfaces (called by AgentUp's function dispatcher)
    # These methods return JSON strings and handle direct function calls
    async def _file_read(self, path: str, encoding: str = "utf-8", **kwargs) -> str:
        """Direct method interface for file_read function calls."""
        try:
            result = await self._internal_file_read(path, encoding)
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "file_read")
            return json.dumps(error_result, indent=2)

    async def _file_write(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
        **kwargs,
    ) -> str:
        """Direct method interface for file_write function calls."""
        try:
            # Note: This conflicts with the internal method name, need to call differently
            from .utils import safe_write_text

            file_path = self.security.validate_path(path)
            content = self.security.sanitize_content(content)
            exists = file_path.exists()
            safe_write_text(file_path, content, encoding, create_parents)
            result = create_success_response(
                {
                    "path": str(file_path),
                    "size": len(content.encode()),
                    "encoding": encoding,
                    "overwritten": exists,
                },
                "file_write",
                f"Successfully {'updated' if exists else 'created'} file",
            )
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "file_write")
            return json.dumps(error_result, indent=2)

    # Internal implementations that return dictionaries (for AI function wrappers)
    async def _file_write_internal(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
    ) -> dict[str, Any]:
        """Internal file_write implementation."""
        try:
            file_path = self.security.validate_path(path)
            content = self.security.sanitize_content(content)
            exists = file_path.exists()
            safe_write_text(file_path, content, encoding, create_parents)
            return create_success_response(
                {
                    "path": str(file_path),
                    "size": len(content.encode()),
                    "encoding": encoding,
                    "overwritten": exists,
                },
                "file_write",
                f"Successfully {'updated' if exists else 'created'} file",
            )
        except Exception as e:
            return create_error_response(e, "file_write")

    async def _file_exists_internal(self, path: str) -> dict[str, Any]:
        """Internal file_exists implementation."""
        try:
            file_path = self.security.validate_path(path)
            exists = file_path.exists()
            return create_success_response(
                {
                    "path": str(file_path),
                    "exists": exists,
                    "is_file": file_path.is_file() if exists else None,
                    "is_directory": file_path.is_dir() if exists else None,
                },
                "file_exists",
            )
        except Exception as e:
            return create_error_response(e, "file_exists")

    async def _file_info_internal(self, path: str) -> dict[str, Any]:
        """Internal file_info implementation."""
        try:
            file_path = self.security.validate_path(path)
            if not file_path.exists():
                return create_error_response(
                    FileNotFoundError(f"Path not found: {path}"), "file_info"
                )
            stat = file_path.stat()
            info = {
                "path": str(file_path),
                "name": file_path.name,
                "type": get_file_type(file_path),
                "size": stat.st_size,
                "size_human": format_file_size(stat.st_size),
                "permissions": get_file_permissions(file_path),
                "created": format_timestamp(stat.st_ctime),
                "modified": format_timestamp(stat.st_mtime),
                "accessed": format_timestamp(stat.st_atime),
                "is_file": file_path.is_file(),
                "is_directory": file_path.is_dir(),
                "is_symlink": file_path.is_symlink(),
            }
            if file_path.is_symlink():
                info["symlink_target"] = str(file_path.readlink())
            return create_success_response(info, "file_info")
        except Exception as e:
            return create_error_response(e, "file_info")

    async def _list_directory_internal(
        self, path: str = ".", pattern: str | None = None, recursive: bool = False
    ) -> dict[str, Any]:
        """Internal list_directory implementation."""
        try:
            dir_path = self.security.validate_path(path)
            if not dir_path.exists():
                return create_error_response(
                    FileNotFoundError(f"Directory not found: {path}"), "list_directory"
                )
            if not dir_path.is_dir():
                return create_error_response(
                    ValueError(f"Path is not a directory: {path}"), "list_directory"
                )
            entries = []
            if recursive:
                paths = dir_path.rglob(pattern or "*")
            else:
                paths = dir_path.glob(pattern or "*")
            for entry in sorted(paths):
                try:
                    stat = entry.stat()
                    entries.append(
                        {
                            "name": entry.name,
                            "path": str(entry.relative_to(dir_path)),
                            "type": "directory" if entry.is_dir() else "file",
                            "size": stat.st_size if entry.is_file() else None,
                            "modified": format_timestamp(stat.st_mtime),
                        }
                    )
                except Exception:
                    continue
            return create_success_response(
                {"path": str(dir_path), "count": len(entries), "entries": entries},
                "list_directory",
            )
        except Exception as e:
            return create_error_response(e, "list_directory")

    async def _create_directory_internal(
        self, path: str, parents: bool = True, exist_ok: bool = True
    ) -> dict[str, Any]:
        """Internal create_directory implementation."""
        try:
            dir_path = self.security.validate_path(path)
            if dir_path.exists() and not exist_ok:
                return create_error_response(
                    FileExistsError(f"Directory already exists: {path}"),
                    "create_directory",
                )
            dir_path.mkdir(parents=parents, exist_ok=exist_ok)
            return create_success_response(
                {"path": str(dir_path), "created": True},
                "create_directory",
                f"Directory created: {path}",
            )
        except Exception as e:
            return create_error_response(e, "create_directory")

    async def _delete_file_internal(
        self, path: str, recursive: bool = False
    ) -> dict[str, Any]:
        """Internal delete_file implementation."""
        try:
            file_path = self.security.validate_path(path)
            if not file_path.exists():
                return create_error_response(
                    FileNotFoundError(f"Path not found: {path}"), "delete_file"
                )
            if file_path.is_dir():
                if recursive:
                    shutil.rmtree(file_path)
                else:
                    file_path.rmdir()
            else:
                file_path.unlink()
            return create_success_response(
                {"path": str(file_path), "deleted": True},
                "delete_file",
                f"Successfully deleted: {path}",
            )
        except Exception as e:
            return create_error_response(e, "delete_file")

    async def _system_info_internal(self) -> dict[str, Any]:
        """Internal system_info implementation."""
        try:
            info = {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "hostname": platform.node(),
                "python_version": platform.python_version(),
                "working_directory": os.getcwd(),
            }
            if platform.system() != "Windows":
                info["user"] = os.environ.get("USER", "unknown")
            else:
                info["user"] = os.environ.get("USERNAME", "unknown")
            return create_success_response(info, "system_info")
        except Exception as e:
            return create_error_response(e, "system_info")

    async def _working_directory_internal(self) -> dict[str, Any]:
        """Internal working_directory implementation."""
        try:
            cwd = os.getcwd()
            return create_success_response(
                {"path": cwd, "absolute": os.path.abspath(cwd)}, "working_directory"
            )
        except Exception as e:
            return create_error_response(e, "working_directory")

    async def _execute_command_internal(
        self, command: str, timeout: int = 30
    ) -> dict[str, Any]:
        """Internal execute_command implementation."""
        try:
            args = self.security.validate_command(command)
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.security.workspace_dir),
            )
            return create_success_response(
                {
                    "command": command,
                    "args": args,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "success": result.returncode == 0,
                },
                "execute_command",
            )
        except subprocess.TimeoutExpired:
            return create_error_response(
                TimeoutError(f"Command timed out after {timeout} seconds"),
                "execute_command",
            )
        except Exception as e:
            return create_error_response(e, "execute_command")

    async def _file_exists(self, path: str, **kwargs) -> str:
        """Direct method interface for file_exists function calls."""
        try:
            result = await self._file_exists_internal(path)
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "file_exists")
            return json.dumps(error_result, indent=2)

    async def _file_info(self, path: str, **kwargs) -> str:
        """Direct method interface for file_info function calls."""
        try:
            result = await self._file_info_internal(path)
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "file_info")
            return json.dumps(error_result, indent=2)

    async def _list_directory(
        self,
        path: str = ".",
        pattern: str | None = None,
        recursive: bool = False,
        **kwargs,
    ) -> str:
        """Direct method interface for list_directory function calls."""
        try:
            result = await self._list_directory_internal(path, pattern, recursive)
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "list_directory")
            return json.dumps(error_result, indent=2)

    async def _create_directory(
        self, path: str, parents: bool = True, exist_ok: bool = True, **kwargs
    ) -> str:
        """Direct method interface for create_directory function calls."""
        try:
            result = await self._create_directory_internal(path, parents, exist_ok)
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "create_directory")
            return json.dumps(error_result, indent=2)

    async def _delete_file(self, path: str, recursive: bool = False, **kwargs) -> str:
        """Direct method interface for delete_file function calls."""
        try:
            result = await self._delete_file_internal(path, recursive)
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "delete_file")
            return json.dumps(error_result, indent=2)

    async def _system_info(self, **kwargs) -> str:
        """Direct method interface for system_info function calls."""
        try:
            result = await self._system_info_internal()
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "system_info")
            return json.dumps(error_result, indent=2)

    async def _working_directory(self, **kwargs) -> str:
        """Direct method interface for working_directory function calls."""
        try:
            result = await self._working_directory_internal()
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "working_directory")
            return json.dumps(error_result, indent=2)

    async def _execute_command(self, command: str, timeout: int = 30, **kwargs) -> str:
        """Direct method interface for execute_command function calls."""
        try:
            result = await self._execute_command_internal(command, timeout)
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "execute_command")
            return json.dumps(error_result, indent=2)

    async def _file_hash(
        self,
        path: str,
        algorithms: list[str] | None = None,
        output_format: str = "hex",
        include_file_info: bool = True,
        **kwargs,
    ) -> str:
        """Direct method interface for file_hash function calls."""
        try:
            result = await self._file_hash_internal(
                path, algorithms, output_format, include_file_info
            )
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "file_hash")
            return json.dumps(error_result, indent=2)

    def _create_ai_function(self, config: dict) -> AIFunction:
        """Create an AIFunction from configuration."""
        ai_func_config = config["ai_function"]
        return AIFunction(
            name=ai_func_config["name"],
            description=ai_func_config["description"],
            parameters=ai_func_config["parameters"],
            handler=self._handlers[config["id"]],
        )

    @hookimpl
    def get_ai_functions(self, capability_id: str = None) -> list[AIFunction]:
        """Provide AI-callable functions for a specific capability."""
        # Return the specific function for this capability
        if capability_id:
            config = next((c for c in CAPABILITIES_CONFIG if c["id"] == capability_id), None)
            if config:
                return [self._create_ai_function(config)]
            return []

        # Return all functions (for backward compatibility)
        return [self._create_ai_function(config) for config in CAPABILITIES_CONFIG]
