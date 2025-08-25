#!/bin/zsh

# MCP Server Launcher Script for Customer Service using uv
# Save this as start_mcp_server.sh and make it executable: chmod +x start_mcp_server.sh

# Set the path to your project directory
PROJECT_DIR="/Users/siva/lab/autogen-first/auto-gen-samples/customer_service/backend"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Run the MCP server using uv
uv run python mcp_server.py