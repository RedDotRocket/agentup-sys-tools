import json
import os
import platform
import shutil
import subprocess
from typing import Any

import pluggy
from agent.plugins import (
    AIFunction,
    SkillCapability,
    SkillContext,
    SkillInfo,
    SkillResult,
    ValidationResult,
)

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


class Plugin:
    """Main plugin class for System Tools."""

    def __init__(self):
        """Initialize the plugin."""
        self.name = "system-tools"
        self.security = SecurityManager()

    @hookimpl
    def register_skill(self) -> SkillInfo:
        """Register the skill with AgentUp."""
        return SkillInfo(
            id="sys_tools",
            name="System Tools",
            version="0.1.0",
            description="A plugin that provides System Tools functionality for reading, writing, executing files, working with folders",
            capabilities=[SkillCapability.TEXT, SkillCapability.AI_FUNCTION],
            tags=["system-tools", "file-io", "system"],
            config_schema={
                "type": "object",
                "properties": {
                    "workspace_dir": {
                        "type": "string",
                        "description": "Directory to restrict operations to",
                    },
                    "max_file_size": {
                        "type": "integer",
                        "description": "Maximum file size in bytes (default 10MB)",
                        "default": 10485760,
                    },
                    "allow_command_execution": {
                        "type": "boolean",
                        "description": "Allow safe command execution",
                        "default": True,
                    },
                },
            },
        )

    @hookimpl
    def validate_config(self, config: dict) -> ValidationResult:
        """Validate skill configuration."""
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

        return ValidationResult(
            valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    @hookimpl
    def can_handle_task(self, context: SkillContext) -> float:
        """Check if this skill can handle the task."""
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
        }

        confidence = 0.0
        for keyword, score in keywords.items():
            if keyword in user_input:
                confidence = max(confidence, score)

        return confidence

    @hookimpl
    async def execute_skill(self, context: SkillContext) -> SkillResult:
        """Execute the skill logic."""
        try:
            # For natural language requests (not AI function calls)
            user_input = self._extract_user_input(context)
            return self._handle_natural_language(user_input)

        except SecurityError as e:
            return SkillResult(
                content=f"Security error: {str(e)}",
                success=False,
                error=str(e),
                metadata={"skill": "sys_tools", "error_type": "security"},
            )
        except Exception as e:
            return SkillResult(
                content=f"Error executing system tools: {str(e)}",
                success=False,
                error=str(e),
                metadata={"skill": "sys_tools", "error_type": type(e).__name__},
            )

    def _extract_user_input(self, context: SkillContext) -> str:
        """Extract user input from the task context."""
        if hasattr(context.task, "history") and context.task.history:
            last_msg = context.task.history[-1]
            if hasattr(last_msg, "parts") and last_msg.parts:
                return (
                    last_msg.parts[0].text if hasattr(last_msg.parts[0], "text") else ""
                )
        return ""

    def _handle_natural_language(self, user_input: str) -> SkillResult:
        """Handle natural language requests."""
        # Try to provide helpful guidance
        suggestions = [
            "Available operations:",
            "- Read file: 'read file <path>'",
            "- Write file: 'write file <path> with content <content>'",
            "- List directory: 'list files in <path>'",
            "- File info: 'get info for file <path>'",
            "- System info: 'show system information'",
            "- Current directory: 'what is the current directory'",
            "",
            "For best results, use the AI function interface.",
        ]

        return SkillResult(
            content="\n".join(suggestions),
            success=True,
            metadata={"skill": "sys_tools", "type": "help"},
        )

    # File Operations
    async def _internal_read_file(
        self, path: str, encoding: str = "utf-8"
    ) -> dict[str, Any]:
        """Read contents of a file."""
        try:
            file_path = self.security.validate_path(path)
            self.security.validate_file_size(file_path)

            if not file_path.exists():
                return create_error_response(
                    FileNotFoundError(f"File not found: {path}"), "read_file"
                )

            if not file_path.is_file():
                return create_error_response(
                    ValueError(f"Path is not a file: {path}"), "read_file"
                )

            content = safe_read_text(file_path, encoding, self.security.max_file_size)

            return create_success_response(
                {
                    "path": str(file_path),
                    "content": content,
                    "encoding": encoding,
                    "size": len(content),
                },
                "read_file",
                f"Successfully read {format_file_size(len(content.encode()))}",
            )

        except Exception as e:
            return create_error_response(e, "read_file")

    async def _write_file(
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
                "write_file",
                f"Successfully {'updated' if exists else 'created'} file",
            )

        except Exception as e:
            return create_error_response(e, "write_file")

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

    async def _get_file_info(self, path: str) -> dict[str, Any]:
        """Get detailed information about a file."""
        try:
            file_path = self.security.validate_path(path)

            if not file_path.exists():
                return create_error_response(
                    FileNotFoundError(f"Path not found: {path}"), "get_file_info"
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

            return create_success_response(info, "get_file_info")

        except Exception as e:
            return create_error_response(e, "get_file_info")

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
    async def _get_system_info(self) -> dict[str, Any]:
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

            return create_success_response(info, "get_system_info")

        except Exception as e:
            return create_error_response(e, "get_system_info")

    async def _get_working_directory(self) -> dict[str, Any]:
        """Get current working directory."""
        try:
            cwd = os.getcwd()
            return create_success_response(
                {"path": cwd, "absolute": os.path.abspath(cwd)}, "get_working_directory"
            )
        except Exception as e:
            return create_error_response(e, "get_working_directory")

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

    # Direct method interfaces (called by AgentUp's function dispatcher)
    # These methods return JSON strings and handle direct function calls
    async def _read_file(self, path: str, encoding: str = "utf-8", **kwargs) -> str:
        """Direct method interface for read_file function calls."""
        try:
            result = await self._internal_read_file(path, encoding)
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "read_file")
            return json.dumps(error_result, indent=2)

    async def _write_file(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
        **kwargs,
    ) -> str:
        """Direct method interface for write_file function calls."""
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
                "write_file",
                f"Successfully {'updated' if exists else 'created'} file",
            )
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "write_file")
            return json.dumps(error_result, indent=2)

    # Internal implementations that return dictionaries (for AI function wrappers)
    async def _write_file_internal(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        create_parents: bool = True,
    ) -> dict[str, Any]:
        """Internal write_file implementation."""
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
                "write_file",
                f"Successfully {'updated' if exists else 'created'} file",
            )
        except Exception as e:
            return create_error_response(e, "write_file")

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

    async def _get_file_info_internal(self, path: str) -> dict[str, Any]:
        """Internal get_file_info implementation."""
        try:
            file_path = self.security.validate_path(path)
            if not file_path.exists():
                return create_error_response(
                    FileNotFoundError(f"Path not found: {path}"), "get_file_info"
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
            return create_success_response(info, "get_file_info")
        except Exception as e:
            return create_error_response(e, "get_file_info")

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

    async def _get_system_info_internal(self) -> dict[str, Any]:
        """Internal get_system_info implementation."""
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
            return create_success_response(info, "get_system_info")
        except Exception as e:
            return create_error_response(e, "get_system_info")

    async def _get_working_directory_internal(self) -> dict[str, Any]:
        """Internal get_working_directory implementation."""
        try:
            cwd = os.getcwd()
            return create_success_response(
                {"path": cwd, "absolute": os.path.abspath(cwd)}, "get_working_directory"
            )
        except Exception as e:
            return create_error_response(e, "get_working_directory")

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

    async def _get_file_info(self, path: str, **kwargs) -> str:
        """Direct method interface for get_file_info function calls."""
        try:
            result = await self._get_file_info_internal(path)
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "get_file_info")
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

    async def _get_system_info(self, **kwargs) -> str:
        """Direct method interface for get_system_info function calls."""
        try:
            result = await self._get_system_info_internal()
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "get_system_info")
            return json.dumps(error_result, indent=2)

    async def _get_working_directory(self, **kwargs) -> str:
        """Direct method interface for get_working_directory function calls."""
        try:
            result = await self._get_working_directory_internal()
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "get_working_directory")
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

    # AI Function Wrappers (AgentUp expects these to follow (task, context) signature)
    async def _ai_read_file(self, task, context: SkillContext) -> SkillResult:
        """AI function wrapper for read_file."""
        # Get parameters from task metadata (AgentUp's parameter passing mechanism)
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._internal_read_file(**params)
            return SkillResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"skill": "sys_tools", "function": "read_file"},
            )
        except Exception as e:
            return SkillResult(
                content=json.dumps(create_error_response(e, "read_file")),
                success=False,
                error=str(e),
            )

    async def _ai_write_file(self, task, context: SkillContext) -> SkillResult:
        """AI function wrapper for write_file."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._write_file_internal(**params)
            return SkillResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"skill": "sys_tools", "function": "write_file"},
            )
        except Exception as e:
            return SkillResult(
                content=json.dumps(create_error_response(e, "write_file")),
                success=False,
                error=str(e),
            )

    async def _ai_file_exists(self, task, context: SkillContext) -> SkillResult:
        """AI function wrapper for file_exists."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            # Call the internal method directly, not the string-returning direct method
            result = await self._file_exists_internal(**params)
            return SkillResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"skill": "sys_tools", "function": "file_exists"},
            )
        except Exception as e:
            return SkillResult(
                content=json.dumps(create_error_response(e, "file_exists")),
                success=False,
                error=str(e),
            )

    async def _ai_get_file_info(self, task, context: SkillContext) -> SkillResult:
        """AI function wrapper for get_file_info."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._get_file_info_internal(**params)
            return SkillResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"skill": "sys_tools", "function": "get_file_info"},
            )
        except Exception as e:
            return SkillResult(
                content=json.dumps(create_error_response(e, "get_file_info")),
                success=False,
                error=str(e),
            )

    async def _ai_list_directory(self, task, context: SkillContext) -> SkillResult:
        """AI function wrapper for list_directory."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._list_directory_internal(**params)
            return SkillResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"skill": "sys_tools", "function": "list_directory"},
            )
        except Exception as e:
            return SkillResult(
                content=json.dumps(create_error_response(e, "list_directory")),
                success=False,
                error=str(e),
            )

    async def _ai_create_directory(self, task, context: SkillContext) -> SkillResult:
        """AI function wrapper for create_directory."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._create_directory_internal(**params)
            return SkillResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"skill": "sys_tools", "function": "create_directory"},
            )
        except Exception as e:
            return SkillResult(
                content=json.dumps(create_error_response(e, "create_directory")),
                success=False,
                error=str(e),
            )

    async def _ai_delete_file(self, task, context: SkillContext) -> SkillResult:
        """AI function wrapper for delete_file."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._delete_file_internal(**params)
            return SkillResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"skill": "sys_tools", "function": "delete_file"},
            )
        except Exception as e:
            return SkillResult(
                content=json.dumps(create_error_response(e, "delete_file")),
                success=False,
                error=str(e),
            )

    async def _ai_get_system_info(self, task, context: SkillContext) -> SkillResult:
        """AI function wrapper for get_system_info."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._get_system_info_internal(**params)
            return SkillResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"skill": "sys_tools", "function": "get_system_info"},
            )
        except Exception as e:
            return SkillResult(
                content=json.dumps(create_error_response(e, "get_system_info")),
                success=False,
                error=str(e),
            )

    async def _ai_get_working_directory(
        self, task, context: SkillContext
    ) -> SkillResult:
        """AI function wrapper for get_working_directory."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._get_working_directory_internal(**params)
            return SkillResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"skill": "sys_tools", "function": "get_working_directory"},
            )
        except Exception as e:
            return SkillResult(
                content=json.dumps(create_error_response(e, "get_working_directory")),
                success=False,
                error=str(e),
            )

    async def _ai_execute_command(self, task, context: SkillContext) -> SkillResult:
        """AI function wrapper for execute_command."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._execute_command_internal(**params)
            return SkillResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"skill": "sys_tools", "function": "execute_command"},
            )
        except Exception as e:
            return SkillResult(
                content=json.dumps(create_error_response(e, "execute_command")),
                success=False,
                error=str(e),
            )

    @hookimpl
    def get_ai_functions(self) -> list[AIFunction]:
        """Provide AI-callable functions."""
        return [
            # File operations
            AIFunction(
                name="read_file",
                description="Read the contents of a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read",
                        },
                        "encoding": {
                            "type": "string",
                            "description": "Text encoding (default: utf-8)",
                            "default": "utf-8",
                        },
                    },
                    "required": ["path"],
                },
                handler=self._ai_read_file,
            ),
            AIFunction(
                name="write_file",
                description="Write content to a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file",
                        },
                        "encoding": {
                            "type": "string",
                            "description": "Text encoding (default: utf-8)",
                            "default": "utf-8",
                        },
                        "create_parents": {
                            "type": "boolean",
                            "description": "Create parent directories if needed",
                            "default": True,
                        },
                    },
                    "required": ["path", "content"],
                },
                handler=self._ai_write_file,
            ),
            AIFunction(
                name="file_exists",
                description="Check if a file or directory exists",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to check"}
                    },
                    "required": ["path"],
                },
                handler=self._ai_file_exists,
            ),
            AIFunction(
                name="get_file_info",
                description="Get detailed information about a file or directory",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file or directory",
                        }
                    },
                    "required": ["path"],
                },
                handler=self._ai_get_file_info,
            ),
            # Directory operations
            AIFunction(
                name="list_directory",
                description="List contents of a directory",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path (default: current directory)",
                            "default": ".",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern to filter results (e.g., '*.txt')",
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "List recursively",
                            "default": False,
                        },
                    },
                },
                handler=self._ai_list_directory,
            ),
            AIFunction(
                name="create_directory",
                description="Create a new directory",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path of directory to create",
                        },
                        "parents": {
                            "type": "boolean",
                            "description": "Create parent directories if needed",
                            "default": True,
                        },
                        "exist_ok": {
                            "type": "boolean",
                            "description": "Don't raise error if directory exists",
                            "default": True,
                        },
                    },
                    "required": ["path"],
                },
                handler=self._ai_create_directory,
            ),
            AIFunction(
                name="delete_file",
                description="Delete a file or directory",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to delete"},
                        "recursive": {
                            "type": "boolean",
                            "description": "Delete directories recursively",
                            "default": False,
                        },
                    },
                    "required": ["path"],
                },
                handler=self._ai_delete_file,
            ),
            # System operations
            AIFunction(
                name="get_system_info",
                description="Get system and platform information",
                parameters={"type": "object", "properties": {}},
                handler=self._ai_get_system_info,
            ),
            AIFunction(
                name="get_working_directory",
                description="Get the current working directory",
                parameters={"type": "object", "properties": {}},
                handler=self._ai_get_working_directory,
            ),
            AIFunction(
                name="execute_command",
                description="Execute a safe shell command (limited to whitelist)",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to execute",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds",
                            "default": 30,
                        },
                    },
                    "required": ["command"],
                },
                handler=self._ai_execute_command,
            ),
        ]
