# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this AgentUp plugin.

## Plugin Overview

This is an AgentUp plugin that provides System Tools functionality. It follows the A2A-specification compliant plugin architecture using the pluggy hook system.

### Capabilities and Required Scopes

The plugin provides the following capabilities with their required scopes:

**File Operations:**
- `file_read` (scope: `files:read`) - Read text file contents
- `file_write` (scope: `files:write`) - Write content to files
- `file_exists` (scope: `files:read`) - Check if file/directory exists
- `file_info` (scope: `files:read`) - Get file/directory metadata
- `file_hash` (scope: `files:read`) - Compute cryptographic hashes

**Directory Operations:**
- `list_directory` (scope: `files:read`) - List directory contents
- `create_directory` (scope: `files:write`) - Create directories
- `delete_file` (scope: `files:admin`) - Delete files/directories

**System Operations:**
- `system_info` (scope: `system:read`) - Get system information
- `working_directory` (scope: `system:read`) - Get current working directory
- `execute_command` (scope: `system:admin`) - Execute whitelisted commands

## Plugin Structure

```
system-tools/
├── src/
│   └── sys_tools/
│       ├── __init__.py
│       └── plugin.py           # Main plugin implementation
├── tests/
│   └── test_sys_tools.py
├── pyproject.toml              # Package configuration with AgentUp entry point
├── README.md                   # Plugin documentation
└── CLAUDE.md                   # This file
```

## Core Plugin Architecture

### Hook System
The plugin uses pluggy hooks to integrate with AgentUp:

- `@hookimpl def register_capability()` - **Required** - Registers the plugin's capabilities
- `@hookimpl def can_handle_task()` - **Required** - Determines if plugin can handle a task
- `@hookimpl def execute_capability()` - **Required** - Main capability execution logic
- `@hookimpl def validate_config()` - Optional - Validates plugin configuration
- `@hookimpl def get_ai_functions()` - Optional - Provides AI-callable functions per capability

### Entry Point
The plugin is registered via entry point in `pyproject.toml`:
```toml
[project.entry-points."agentup.skills"]
sys_tools = "sys_tools.plugin:Plugin"
```

## Development Guidelines

### Code Style
- Follow PEP 8 and Python best practices
- Use type hints throughout the codebase
- Use async/await for I/O operations
- Handle errors gracefully with proper A2A error responses

### Plugin Implementation Patterns

#### 1. Capability Registration
```python
@hookimpl
def register_capability(self) -> list[PluginDefinition]:
    return [self._create_capability_info(config) for config in CAPABILITIES_CONFIG]
```

Each capability is defined with:
- `id` - Unique identifier (e.g., "file_read")
- `name` - Human-readable name
- `description` - What the capability does
- `capabilities` - Types (e.g., AI_FUNCTION, TEXT)
- `tags` - For categorization
- `required_scopes` - Security scopes needed (e.g., ["files:read"])

#### 2. Task Routing
```python
@hookimpl
def can_handle_task(self, context: SkillContext) -> float:
    user_input = self._extract_user_input(context).lower()

    # Return confidence score (0.0 to 1.0)
    # Higher scores = more likely to handle the task
    keywords = {'keyword1': 1.0, 'keyword2': 0.8}

    confidence = 0.0
    for keyword, score in keywords.items():
        if keyword in user_input:
            confidence = max(confidence, score)

    return confidence
```

#### 3. Capability Execution
```python
@hookimpl
async def execute_capability(self, context: CapabilityContext) -> CapabilityResult:
    try:
        # Get the specific capability being invoked
        capability_id = context.metadata.get("capability_id", "unknown")
        
        # Route to specific capability handler
        if capability_id in self._handlers:
            handler = self._handlers[capability_id]
            return await handler(context.task, context)
            
        return CapabilityResult(
            content=json.dumps(result, indent=2),
            success=True,
            metadata={"capability": "sys_tools", "function": capability_id}
        )
    except Exception as e:
        return CapabilityResult(
            content=json.dumps(create_error_response(e, capability_id)),
            success=False,
            error=str(e)
        )
```

#### 4. AI Function Support
```python
@hookimpl
def get_ai_functions(self, capability_id: str = None) -> list[AIFunction]:
    # Return AI functions for a specific capability
    if capability_id:
        config = next((c for c in CAPABILITIES_CONFIG if c["id"] == capability_id), None)
        if config:
            return [self._create_ai_function(config)]
    # Return all functions
    return [self._create_ai_function(config) for config in CAPABILITIES_CONFIG]
```

Example AI function configuration:
```python
{
    "name": "file_read",
    "description": "Read the contents of a file",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read"},
            "encoding": {"type": "string", "description": "Text encoding", "default": "utf-8"}
        },
        "required": ["path"]
    }
}
```

### Error Handling
- Always return CapabilityResult objects from execute_capability
- Use success=False for errors
- Include descriptive error messages
- Log errors appropriately for debugging
- Handle SecurityError separately for security violations

### Testing
- Write comprehensive tests for all plugin functionality
- Test both success and error cases
- Mock external dependencies
- Use pytest and async test patterns
- Test each capability independently

### Configuration
- Define configuration schema per capability
- Validate configuration in validate_config() hook
- Use environment variables for sensitive data
- Provide sensible defaults
- Support workspace_dir, max_file_size, allowed_extensions

## Development Workflow

### Local Development
1. Install in development mode: `pip install -e .`
2. Create test agent: `agentup agent create test-agent --template minimal`
3. Configure plugin in agent's `agent_config.yaml`
4. Test with: `agentup agent serve`

### Testing
```bash
# Run tests
pytest tests/ -v

# Check plugin loading
agentup plugin list

# Validate plugin
agentup plugin validate sys_tools
```

### External Dependencies
- Use AgentUp's service registry for HTTP clients, databases, etc.
- Declare all dependencies in pyproject.toml
- Use async libraries for better performance

## Plugin Capabilities

### Available Capability Types
- `CapabilityType.TEXT` - Text processing
- `CapabilityType.MULTIMODAL` - Images, documents, etc.
- `CapabilityType.AI_FUNCTION` - LLM-callable functions
- `CapabilityType.STREAMING` - Streaming responses
- `CapabilityType.STATEFUL` - State management

This plugin uses `CapabilityType.AI_FUNCTION` and `CapabilityType.TEXT` for its capabilities.

### Middleware Support
Request middleware for common functionality:
- Rate limiting
- Caching
- Retry logic
- Logging
- Validation

### Service Integration
Access external services via AgentUp's service registry:
- HTTP clients
- Database connections
- Cache backends
- Message queues

## Best Practices

### Performance
- Use async/await for I/O operations
- Implement caching for expensive operations
- Use connection pooling for external APIs
- Minimize blocking operations

### Security
- Validate all inputs with SecurityManager
- Sanitize outputs to prevent injection
- Use path validation to prevent directory traversal
- Never log sensitive data
- Implement command whitelisting for execute_command
- Enforce workspace_dir restrictions
- Check file sizes before operations

### Maintainability
- Follow single responsibility principle
- Keep functions small and focused
- Use descriptive variable names
- Add docstrings to all public methods

## Common Patterns

### External API Integration
```python
async def _call_external_api(self, data):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.example.com/endpoint",
            json=data,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        response.raise_for_status()
        return response.json()
```

### State Management
```python
@hookimpl
def get_state_schema(self) -> dict:
    return {
        "type": "object",
        "properties": {
            "user_preferences": {"type": "object"},
            "session_data": {"type": "object"}
        }
    }
```

### Configuration Validation
```python
@hookimpl
def validate_config(self, config: dict) -> PluginValidationResult:
    errors = []
    warnings = []

    if not config.get("api_key"):
        errors.append("api_key is required")

    return PluginValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )
```

## Debugging Tips

### Common Issues
- Plugin not loading: Check entry point in pyproject.toml
- Functions not available: Verify get_ai_functions() returns valid schemas
- Routing not working: Debug can_handle_task() logic
- Configuration errors: Implement validate_config() hook

### Logging
```python
import logging
logger = logging.getLogger(__name__)

async def execute_capability(self, context: CapabilityContext) -> CapabilityResult:
    capability_id = context.metadata.get("capability_id", "unknown")
    logger.info(f"Processing {capability_id} request", extra={"capability": capability_id})
    # ... implementation
```

## Distribution

### Package Structure
- Follow Python package conventions
- Include comprehensive README.md
- Add LICENSE file
- Include CHANGELOG.md for version history

### Publishing
1. Test thoroughly with various agents
2. Update version in pyproject.toml
3. Build package: `python -m build`
4. Upload to PyPI: `python -m twine upload dist/*`

## Important Notes

### A2A Compliance
- All responses must be A2A-compliant
- Use proper task lifecycle management
- Follow A2A error response formats
- Implement proper message handling

### Framework Integration
- Leverage AgentUp's built-in features
- Use provided utilities and helpers
- Follow established patterns from other plugins
- Maintain compatibility with different agent templates

### Community Guidelines
- Write clear documentation
- Provide usage examples
- Follow semantic versioning
- Respond to issues and pull requests

## Resources

- [AgentUp Documentation](https://docs.agentup.dev)
- [Plugin Development Guide](https://docs.agentup.dev/plugins/development)
---

Remember: This plugin is part of the AgentUp ecosystem. Always consider how it integrates with other plugins and follows A2A standards for maximum compatibility and usefulness.
