# Sys Tools Plugin for AgentUp

<p align="center">
  <img src="static/sys_tools.png" alt="SYS Tools Plugin" width="400"/>
</p>

This plugin provides safe, controlled access to system operations including file I/O, directory management, and system information retrieval.

## Tool Capabilities

### File Operations
- **Read File** - Read text file contents with size limits and encoding support
- **Write File** - Write content to files with atomic operations and parent directory creation
- **Check File Exists** - Verify if a file or directory exists
- **Get File Info** - Retrieve detailed metadata about files and directories

### Directory Operations
- **List Directory** - List directory contents with pattern matching and recursive options
- **Create Directory** - Create directories with parent creation support
- **Delete File/Directory** - Safely delete files and directories with recursive option

### System Operations
- **Get System Info** - Retrieve platform, architecture, and environment information
- **Get Working Directory** - Get current working directory path
- **Execute Command** - Execute whitelisted shell commands with timeout support

### Security and Validation
- Path validation and basic sandboxing to prevent directory traversal
- Configurable workspace restriction
- File size limits (default 10MB)
- Command whitelist for safe execution
- Input sanitization and validation

## Installation

### For Development
```bash
cd system-tools
pip install -e .
```

### From PyPI (when published)
```bash
pip install agentup-system-tools
```

### Via AgentUp CLI
```bash
agentup plugin install system-tools
```

## Configuration

Add the sys_tools skill to your agent's `agent_config.yaml`:

```yaml
skills:
  - skill_id: sys_tools
    name: System Tools
    description: System tools for basic operations
    input_mode: text
    output_mode: text
    routing_mode: ai
    config:
      # Optional: Restrict operations to specific directory (defaults to cwd)
      workspace_dir: "./workspace"
      # Optional: Maximum file size in bytes (default 10MB)
      max_file_size: 10485760
      # Optional: Allow safe command execution (default true)
      allow_command_execution: true
```

## Usage Examples

### Natural Language Usage

The plugin responds to natural language requests:

```
"Read the contents of config.json"
"List all Python files in the src directory"
"Create a new folder called outputs"
"What operating system am I running on?"
```

### AI Function Calls

The plugin provides AI-callable functions that can be used by LLMs:

#### Read a File
```json
{
  "name": "read_file",
  "parameters": {
    "path": "data/config.json",
    "encoding": "utf-8"
  }
}
```

#### Write a File
```json
{
  "name": "write_file",
  "parameters": {
    "path": "output/results.txt",
    "content": "Processing complete!",
    "create_parents": true
  }
}
```

#### List Directory Contents
```json
{
  "name": "list_directory",
  "parameters": {
    "path": "src",
    "pattern": "*.py",
    "recursive": true
  }
}
```

#### Get File Information
```json
{
  "name": "get_file_info",
  "parameters": {
    "path": "document.pdf"
  }
}
```

#### Execute Safe Command
```json
{
  "name": "execute_command",
  "parameters": {
    "command": "ls -la",
    "timeout": 30
  }
}
```

## Security Considerations

### Path Security
- All paths are validated to prevent directory traversal attacks
- Operations are restricted to the configured workspace directory
- Symbolic links are detected and reported
- Absolute paths are only allowed when explicitly configured

### Command Execution
- Only whitelisted commands can be executed:
  - File viewing: `ls`, `cat`, `head`, `tail`, `wc`
  - System info: `pwd`, `whoami`, `date`, `uname`, `hostname`
  - Search: `grep`, `find`, `which`
  - Environment: `env`, `printenv`
  - System status: `df`, `du`, `free`, `uptime`
- Commands are parsed to prevent injection attacks
- Execution timeout prevents hanging processes

### File Size Limits
- Default 10MB limit for file operations
- Configurable via `max_file_size` setting
- Large files are truncated with notification

## Response Format

All operations return a standardized response format:

### Success Response
```json
{
  "success": true,
  "data": {
    // Operation-specific data
  },
  "operation": "operation_name",
  "message": "Optional success message"
}
```

### Error Response
```json
{
  "success": false,
  "error": "Error description",
  "error_type": "ErrorClassName",
  "operation": "operation_name",
  "message": "operation_name failed: Error description"
}
```

## Development

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=sys_tools --cov-report=html

# Run specific test categories
pytest tests/test_sys_tools.py::TestFileOperations -v
```

### Adding New Operations

1. Add the operation handler to `plugin.py`:
```python
def _my_operation(self, param1: str) -> dict[str, Any]:
    """Perform my operation."""
    try:
        # Validate inputs
        validated_path = self.security.validate_path(param1)

        # Perform operation
        result = do_something(validated_path)

        # Return standardized response
        return create_success_response(
            {"result": result},
            "my_operation"
        )
    except Exception as e:
        return create_error_response(e, "my_operation")
```

2. Register it as an AI function in `get_ai_functions()`:
```python
AIFunction(
    name="my_operation",
    description="Description for LLM",
    parameters={
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Parameter description"
            }
        },
        "required": ["param1"]
    },
    handler=self._my_operation
)
```

3. Add tests in `test_sys_tools.py`

## Error Handling

The plugin provides detailed error messages for common scenarios:

- **File Not Found**: Clear indication of missing files
- **Permission Denied**: When lacking file system permissions
- **Security Violations**: Path traversal attempts, workspace violations
- **Size Limits**: When files exceed configured limits
- **Invalid Operations**: Type mismatches, invalid parameters

## Performance Considerations

- File operations use buffered I/O for efficiency
- Large directories are listed iteratively
- Commands have configurable timeouts
- Atomic writes prevent partial file corruption

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This plugin is part of the AgentUp ecosystem and follows the same licensing, the Apache License 2.0. See the LICENSE file for details.

## Support

For issues, questions, or contributions:
- Create an issue in the repository
- Check the CLAUDE.md file for development guidelines
- Refer to AgentUp documentation for plugin development
