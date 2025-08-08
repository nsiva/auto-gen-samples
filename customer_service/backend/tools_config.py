
# Example usage:
from config import AUTH_LOGIN_URL
from tool_mapping import ToolMapping  # Make sure this import path is correct


tool_mapping = ToolMapping()
tool_mapping.add_tool("get_order_status", protected=True, login_url=AUTH_LOGIN_URL)
tool_mapping.add_tool("get_refund_status", protected=True, login_url=AUTH_LOGIN_URL)
tool_mapping.add_tool("get_inventory_status", protected=False)