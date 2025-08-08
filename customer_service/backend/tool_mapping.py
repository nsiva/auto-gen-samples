class ToolMapping:
    def __init__(self):
        # Each tool is a dict: {"name": str, "protected": bool, "login_url": Optional[str]}
        self.tools = []

    def add_tool(self, name: str, protected: bool = False, login_url: str = None):
        self.tools.append({
            "name": name,
            "protected": protected,
            "login_url": login_url if protected else None
        })

    def is_protected(self, name: str) -> bool:
        for tool in self.tools:
            if tool["name"] == name:
                return tool["protected"]
        return False

    def get_login_url(self, name: str) -> str:
        for tool in self.tools:
            if tool["name"] == name and tool["protected"]:
                return tool["login_url"]
        return None

    def get_all_tools(self):
        return [tool["name"] for tool in self.tools]

    def get_tool_info(self, name: str):
        for tool in self.tools:
            if tool["name"] == name:
                return tool
        return None

    def get_protected_tools(self) -> list[str]:
        """Return a list of tool names that are protected."""
        return [tool["name"] for tool in self.tools if tool["protected"]]