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
    ValidationResult,
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


class Plugin:
    """Main plugin class for System Tools."""

    def __init__(self):
        """Initialize the plugin."""
        self.name = "system-tools"
        self.security = SecurityManager()
        self.hasher = FileHasher(self.security)

    @hookimpl
    def register_capability(self) -> list[CapabilityInfo]:
        """Register the system tools capabilities."""
        base_config_schema = {
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
        
        return [
            CapabilityInfo(
                id="file_read",
                name="File Read",
                version="0.2.0",
                description="Read contents of files",
                plugin_name="sys_tools",
                capabilities=[CapabilityType.AI_FUNCTION, CapabilityType.TEXT],
                tags=["files", "read", "io"],
                config_schema=base_config_schema
            ),
            CapabilityInfo(
                id="file_write",
                name="File Write",
                version="0.2.0",
                description="Write content to files",
                plugin_name="sys_tools",
                capabilities=[CapabilityType.AI_FUNCTION, CapabilityType.TEXT],
                tags=["files", "write", "io"],
                config_schema=base_config_schema
            ),
            CapabilityInfo(
                id="file_exists",
                name="File Exists",
                version="0.2.0",
                description="Check if a file or directory exists",
                plugin_name="sys_tools",
                capabilities=[CapabilityType.AI_FUNCTION],
                tags=["files", "check"],
                config_schema=base_config_schema
            ),
            CapabilityInfo(
                id="file_info",
                name="File Info",
                version="0.2.0",
                description="Get detailed information about a file or directory",
                plugin_name="sys_tools",
                capabilities=[CapabilityType.AI_FUNCTION],
                tags=["files", "info"],
                config_schema=base_config_schema
            ),
            CapabilityInfo(
                id="list_directory",
                name="List Directory",
                version="0.2.0",
                description="List contents of a directory",
                plugin_name="sys_tools",
                capabilities=[CapabilityType.AI_FUNCTION],
                tags=["directories", "list"],
                config_schema=base_config_schema
            ),
            CapabilityInfo(
                id="create_directory",
                name="Create Directory",
                version="0.2.0",
                description="Create a new directory",
                plugin_name="sys_tools",
                capabilities=[CapabilityType.AI_FUNCTION],
                tags=["directories", "create"],
                config_schema=base_config_schema
            ),
            CapabilityInfo(
                id="delete_file",
                name="Delete File",
                version="0.2.0",
                description="Delete a file or directory",
                plugin_name="sys_tools",
                capabilities=[CapabilityType.AI_FUNCTION],
                tags=["files", "delete"],
                config_schema=base_config_schema
            ),
            CapabilityInfo(
                id="system_info",
                name="System Info",
                version="0.2.0",
                description="Get system and platform information",
                plugin_name="sys_tools",
                capabilities=[CapabilityType.AI_FUNCTION],
                tags=["system", "info"],
                config_schema=base_config_schema
            ),
            CapabilityInfo(
                id="working_directory",
                name="Working Directory",
                version="0.2.0",
                description="Get the current working directory",
                plugin_name="sys_tools",
                capabilities=[CapabilityType.AI_FUNCTION],
                tags=["system", "directory"],
                config_schema=base_config_schema
            ),
            CapabilityInfo(
                id="execute_command",
                name="Execute Command",
                version="0.2.0",
                description="Execute a safe shell command",
                plugin_name="sys_tools",
                capabilities=[CapabilityType.AI_FUNCTION],
                tags=["system", "command", "execute"],
                config_schema=base_config_schema
            ),
            CapabilityInfo(
                id="file_hash",
                name="File Hash",
                version="0.2.0",
                description="Compute cryptographic hash(es) for a file",
                plugin_name="sys_tools",
                capabilities=[CapabilityType.AI_FUNCTION],
                tags=["files", "hash", "security"],
                config_schema=base_config_schema
            ),
        ]

    @hookimpl
    def validate_config(self, config: dict) -> ValidationResult:
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

        return ValidationResult(
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
            
            # Route to specific capability handler based on capability_id
            capability_map = {
                "file_read": self._ai_read_file,
                "file_write": self._ai_write_file,
                "file_exists": self._ai_file_exists,
                "file_info": self._ai_get_file_info,
                "list_directory": self._ai_list_directory,
                "create_directory": self._ai_create_directory,
                "delete_file": self._ai_delete_file,
                "system_info": self._ai_get_system_info,
                "working_directory": self._ai_get_working_directory,
                "execute_command": self._ai_execute_command,
                "file_hash": self._ai_get_file_hash,
            }
            
            if capability_id in capability_map:
                handler = capability_map[capability_id]
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

    async def _get_file_hash_internal(
        self,
        path: str,
        algorithms: list[str] | None = None,
        output_format: str = "hex",
        include_file_info: bool = True,
    ) -> dict[str, Any]:
        """Internal get_file_hash implementation."""
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
            return create_error_response(e, "get_file_hash")

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

    async def _get_file_hash(
        self,
        path: str,
        algorithms: list[str] | None = None,
        output_format: str = "hex",
        include_file_info: bool = True,
        **kwargs,
    ) -> str:
        """Direct method interface for get_file_hash function calls."""
        try:
            result = await self._get_file_hash_internal(
                path, algorithms, output_format, include_file_info
            )
            import json

            return json.dumps(result, indent=2)
        except Exception as e:
            import json

            error_result = create_error_response(e, "get_file_hash")
            return json.dumps(error_result, indent=2)

    # AI Function Wrappers (AgentUp expects these to follow (task, context) signature)
    async def _ai_read_file(self, task, context: CapabilityContext) -> CapabilityResult:
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
            return CapabilityResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"capability": "sys_tools", "function": "read_file"},
            )
        except Exception as e:
            return CapabilityResult(
                content=json.dumps(create_error_response(e, "read_file")),
                success=False,
                error=str(e),
            )

    async def _ai_write_file(self, task, context: CapabilityContext) -> CapabilityResult:
        """AI function wrapper for write_file."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._write_file_internal(**params)
            return CapabilityResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"capability": "sys_tools", "function": "write_file"},
            )
        except Exception as e:
            return CapabilityResult(
                content=json.dumps(create_error_response(e, "write_file")),
                success=False,
                error=str(e),
            )

    async def _ai_file_exists(self, task, context: CapabilityContext) -> CapabilityResult:
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
            return CapabilityResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"capability": "sys_tools", "function": "file_exists"},
            )
        except Exception as e:
            return CapabilityResult(
                content=json.dumps(create_error_response(e, "file_exists")),
                success=False,
                error=str(e),
            )

    async def _ai_get_file_info(self, task, context: CapabilityContext) -> CapabilityResult:
        """AI function wrapper for get_file_info."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._get_file_info_internal(**params)
            return CapabilityResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"capability": "sys_tools", "function": "get_file_info"},
            )
        except Exception as e:
            return CapabilityResult(
                content=json.dumps(create_error_response(e, "get_file_info")),
                success=False,
                error=str(e),
            )

    async def _ai_list_directory(self, task, context: CapabilityContext) -> CapabilityResult:
        """AI function wrapper for list_directory."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._list_directory_internal(**params)
            return CapabilityResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"capability": "sys_tools", "function": "list_directory"},
            )
        except Exception as e:
            return CapabilityResult(
                content=json.dumps(create_error_response(e, "list_directory")),
                success=False,
                error=str(e),
            )

    async def _ai_create_directory(self, task, context: CapabilityContext) -> CapabilityResult:
        """AI function wrapper for create_directory."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._create_directory_internal(**params)
            return CapabilityResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"capability": "sys_tools", "function": "create_directory"},
            )
        except Exception as e:
            return CapabilityResult(
                content=json.dumps(create_error_response(e, "create_directory")),
                success=False,
                error=str(e),
            )

    async def _ai_delete_file(self, task, context: CapabilityContext) -> CapabilityResult:
        """AI function wrapper for delete_file."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._delete_file_internal(**params)
            return CapabilityResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"capability": "sys_tools", "function": "delete_file"},
            )
        except Exception as e:
            return CapabilityResult(
                content=json.dumps(create_error_response(e, "delete_file")),
                success=False,
                error=str(e),
            )

    async def _ai_get_system_info(self, task, context: CapabilityContext) -> CapabilityResult:
        """AI function wrapper for get_system_info."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._get_system_info_internal(**params)
            return CapabilityResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"capability": "sys_tools", "function": "get_system_info"},
            )
        except Exception as e:
            return CapabilityResult(
                content=json.dumps(create_error_response(e, "get_system_info")),
                success=False,
                error=str(e),
            )

    async def _ai_get_working_directory(
        self, task, context: CapabilityContext
    ) -> CapabilityResult:
        """AI function wrapper for get_working_directory."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._get_working_directory_internal(**params)
            return CapabilityResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"capability": "sys_tools", "function": "get_working_directory"},
            )
        except Exception as e:
            return CapabilityResult(
                content=json.dumps(create_error_response(e, "get_working_directory")),
                success=False,
                error=str(e),
            )

    async def _ai_execute_command(self, task, context: CapabilityContext) -> CapabilityResult:
        """AI function wrapper for execute_command."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._execute_command_internal(**params)
            return CapabilityResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"capability": "sys_tools", "function": "execute_command"},
            )
        except Exception as e:
            return CapabilityResult(
                content=json.dumps(create_error_response(e, "execute_command")),
                success=False,
                error=str(e),
            )

    async def _ai_get_file_hash(self, task, context: CapabilityContext) -> CapabilityResult:
        """AI function wrapper for get_file_hash."""
        params = context.metadata.get("parameters", {})
        task_metadata = (
            task.metadata if hasattr(task, "metadata") and task.metadata else {}
        )
        if not params and task_metadata:
            params = task_metadata
        try:
            result = await self._get_file_hash_internal(**params)
            return CapabilityResult(
                content=json.dumps(result, indent=2),
                success=result.get("success", True),
                metadata={"capability": "sys_tools", "function": "get_file_hash"},
            )
        except Exception as e:
            return CapabilityResult(
                content=json.dumps(create_error_response(e, "get_file_hash")),
                success=False,
                error=str(e),
            )

    @hookimpl  
    def get_ai_functions(self, capability_id: str = None) -> list[AIFunction]:
        """Provide AI-callable functions for a specific capability."""
        # Map each capability to its specific AI function
        capability_functions = {
            "file_read": [
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
                )
            ],
            "file_write": [
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
                )
            ],
            "file_exists": [
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
                )
            ],
            "file_info": [
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
                )
            ],
            "list_directory": [
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
                )
            ],
            "create_directory": [
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
                )
            ],
            "delete_file": [
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
                )
            ],
            "system_info": [
                AIFunction(
                    name="get_system_info",
                    description="Get system and platform information",
                    parameters={"type": "object", "properties": {}},
                    handler=self._ai_get_system_info,
                )
            ],
            "working_directory": [
                AIFunction(
                    name="get_working_directory",
                    description="Get the current working directory",
                    parameters={"type": "object", "properties": {}},
                    handler=self._ai_get_working_directory,
                )
            ],
            "execute_command": [
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
                )
            ],
            "file_hash": [
                AIFunction(
                    name="get_file_hash",
                    description="Compute cryptographic hash(es) for a file (SHA256, SHA512, SHA1, MD5)",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to hash",
                            },
                            "algorithms": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of hash algorithms to use (default: ['sha256']). Options: md5, sha1, sha256, sha512",
                            },
                            "output_format": {
                                "type": "string",
                                "description": "Output format for hashes (default: 'hex')",
                                "enum": ["hex", "base64"],
                                "default": "hex",
                            },
                            "include_file_info": {
                                "type": "boolean",
                                "description": "Include file information in the response (default: true)",
                                "default": True,
                            },
                        },
                        "required": ["path"],
                    },
                    handler=self._ai_get_file_hash,
                )
            ],
        }
        
        # Return the specific function(s) for this capability
        if capability_id and capability_id in capability_functions:
            return capability_functions[capability_id]
        
        # Fallback: return all functions (for backward compatibility)
        all_functions = []
        for functions in capability_functions.values():
            all_functions.extend(functions)
        return all_functions
