#!/usr/bin/env python3
"""
MCP Server for Customer Service - stdio version for Claude Desktop
"""

import asyncio
import json
import sys
from typing import Any, Dict
import logging
from logic import get_order_status, get_inventory_lookup, get_refund_tracking

# Configure logging to stderr so it doesn't interfere with stdio communication
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

class MCPServer:
    def __init__(self):
        self.tools = [
            {
                "name": "lookup_inventory",
                "description": "Look up item availability and price in inventory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "item_id": {
                            "type": "string",
                            "description": "The ID of the item to look up"
                        }
                    },
                    "required": ["item_id"]
                }
            },
            {
                "name": "check_order_status",
                "description": "Check the status of an order",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "The ID of the order to check"
                        }
                    },
                    "required": ["order_id"]
                }
            },
            {
                "name": "track_refund",
                "description": "Track the status of a refund",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "refund_id": {
                            "type": "string",
                            "description": "The ID of the refund to track"
                        }
                    },
                    "required": ["refund_id"]
                }
            }
        ]

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests"""
        try:
            method = request.get("method")
            request_id = request.get("id")
            params = request.get("params", {})

            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else "",
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "customer-service-mcp",
                            "version": "1.0.0"
                        }
                    }
                }

            elif method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else "",
                    "result": {
                        "tools": self.tools
                    }
                }

            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                result = await self._execute_tool(tool_name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else "",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": result
                            }
                        ]
                    }
                }

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id if request_id is not None else "",
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            # Always set id to "" if not present or None
            request_id = request.get("id") if isinstance(request, dict) and "id" in request and request.get("id") is not None else ""
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                }
            }

    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return the result"""
        try:
            if tool_name == "lookup_inventory":
                item_id = arguments.get("item_id")
                if not item_id:
                    raise ValueError("item_id is required")
                return get_inventory_lookup(item_id)
            elif tool_name == "check_order_status":
                order_id = arguments.get("order_id")
                if not order_id:
                    raise ValueError("order_id is required")
                return get_order_status(order_id)
            elif tool_name == "track_refund":
                refund_id = arguments.get("refund_id")
                if not refund_id:
                    raise ValueError("refund_id is required")
                return get_refund_tracking(refund_id)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error: {str(e)}"

async def main():
    """Main function for stdio-based MCP server"""
    logger.info("Before instantiating MCP Server...")
    mcp_server = MCPServer()
    logger.info("Customer Service MCP Server starting...")

    try:
        while True:
            # Read from stdin
            line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            logger.info(f"Received request: {line}")
            try:
                request = json.loads(line)
                response = await mcp_server.handle_request(request)
                response_json = json.dumps(response)
                print(response_json, flush=True)
                logger.info(f"Sent response: {response_json}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": "",
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {str(e)}"
                    }
                }
                print(json.dumps(error_response), flush=True)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": "",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    }
                }
                print(json.dumps(error_response), flush=True)
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    logger.info("Starting Customer Service MCP Server...")
    asyncio.run(main())