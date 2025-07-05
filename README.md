# System Tools

A plugin that provides System Tools functionality for reading, writing, executing files, working with folders

## Installation

### For development:
```bash
cd system-tools
pip install -e .
```

### From PyPI (when published):
```bash
pip install system-tools
```

## Usage

This plugin provides the `system_tools` skill to AgentUp agents.

## Development

1. Edit `src/system_tools/plugin.py` to implement your skill logic
2. Test locally with an AgentUp agent
3. Publish to PyPI when ready

## Configuration

The skill can be configured in `agent_config.yaml`:

```yaml
skills:
  - skill_id: system_tools
    config:
      # Add your configuration options here
```
