from typing import List
from tool_definition import ToolDefinition

class ToolMapping:
    def __init__(self):
        # Each tool is a dict: {"name": str, "protected": bool, "login_url": Optional[str]}
        self.tools: List[ToolDefinition] = []

    def add_tool(self, tool: ToolDefinition):
        self.tools.append(tool)

    def get_all_tools(self):
        return [tool for tool in self.tools]

    def get_tool(self, name: str):
        for tool in self.tools:
            if tool["name"] == name:
                return tool
        return None

    def get_protected_tool_names(self) -> list[str]:
        """Return a list of tool names that are protected."""
        return [tool["name"] for tool in self.tools if tool["protected"]]

    def get_protected_tools(self) -> list[str]:
        """Return a list of tools that are protected."""
        return [tool for tool in self.tools if tool.is_protected()]
