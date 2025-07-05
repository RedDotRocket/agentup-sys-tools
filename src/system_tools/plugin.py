"""
System Tools plugin for AgentUp.

A plugin that provides System Tools functionality for reading, writing, executing files, working with folders
"""

import pluggy
from agent.plugins import SkillInfo, SkillContext, SkillResult, ValidationResult, SkillCapability

hookimpl = pluggy.HookimplMarker("agentup")


class Plugin:
    """Main plugin class for System Tools."""

    def __init__(self):
        """Initialize the plugin."""
        self.name = "system-tools"

    @hookimpl
    def register_skill(self) -> SkillInfo:
        """Register the skill with AgentUp."""
        return SkillInfo(
            id="system_tools",
            name="System Tools",
            version="0.1.0",
            description="A plugin that provides System Tools functionality for reading, writing, executing files, working with folders",
            capabilities=[SkillCapability.TEXT],
            tags=["system-tools", "custom"],
        )

    @hookimpl
    def validate_config(self, config: dict) -> ValidationResult:
        """Validate skill configuration."""
        # Add your validation logic here
        return ValidationResult(valid=True)

    @hookimpl
    def can_handle_task(self, context: SkillContext) -> bool:
        """Check if this skill can handle the task."""
        # Add your routing logic here
        # For now, return True to handle all tasks
        return True

    @hookimpl
    def execute_skill(self, context: SkillContext) -> SkillResult:
        """Execute the skill logic."""
        # Extract user input from the task
        user_input = self._extract_user_input(context)

        # Your skill logic here
        response = f"Processed by System Tools: {user_input}"

        return SkillResult(
            content=response,
            success=True,
            metadata={"skill": "system_tools"},
        )

    def _extract_user_input(self, context: SkillContext) -> str:
        """Extract user input from the task context."""
        if hasattr(context.task, "history") and context.task.history:
            last_msg = context.task.history[-1]
            if hasattr(last_msg, "parts") and last_msg.parts:
                return last_msg.parts[0].text if hasattr(last_msg.parts[0], "text") else ""
        return ""
