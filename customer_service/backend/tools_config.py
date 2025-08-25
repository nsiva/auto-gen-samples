
# Example usage:
from config import AUTH_LOGIN_URL
from tool_definition import ToolDefinition
from tool_mapping import ToolMapping  # Make sure this import path is correct


# tool_definitions = ToolDefinition()
# tool_definitions.add_tool("get_order_status", protected=True, login_url=AUTH_LOGIN_URL)
# tool_definitions.add_tool("get_refund_status", protected=True, login_url=AUTH_LOGIN_URL)
# tool_definitions.add_tool("get_inventory_status", protected=False)


tool_mappings = ToolMapping()
# MCP tool names (used by MCP server)
tool_mappings.add_tool( ToolDefinition("check_order_status", protected=True, login_url=AUTH_LOGIN_URL) )
tool_mappings.add_tool( ToolDefinition("track_refund", protected=True, login_url=AUTH_LOGIN_URL) )
tool_mappings.add_tool( ToolDefinition("lookup_inventory", protected=False) )

# AutoGen function names (used by /ask endpoint)
tool_mappings.add_tool( ToolDefinition("get_order_status", protected=True, login_url=AUTH_LOGIN_URL) )
tool_mappings.add_tool( ToolDefinition("get_refund_status", protected=True, login_url=AUTH_LOGIN_URL) )
tool_mappings.add_tool( ToolDefinition("get_inventory_status", protected=False) )