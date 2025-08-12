class ToolDefinition:
    def __init__(self, name: str, protected: bool = False, login_url: str = None):
        # Each tool is a dict: {"name": str, "protected": bool, "login_url": Optional[str]}
        self.name = name
        self.protected = protected
        self.login_url = login_url if protected else None

    def is_protected(self) -> bool:
        return self.protected
    
    def get_login_url(self, name: str) -> str:
        return self.login_url

    def get_tool_info(self, name: str) -> "ToolDefinition":
        if self.name == name:
            return self
        return None
