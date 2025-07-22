#!/bin/zsh

# MCP Server Launcher Script for Customer Service
# Save this as mcp_launcher.sh and make it executable: chmod +x mcp_launcher.sh

# Set the path to your project directory
PROJECT_DIR="/Users/siva/lab/autogen-first/auto-gen-samples/customer_service/backend"

# Set the path to your virtual environment
VENV_DIR="/Users/siva/lab/autogen-first/auto-gen-samples/customer_service/backend/venv"

# Change to project directory
cd "$PROJECT_DIR" || exit 1

# Activate virtual environment and run the MCP server
source "$VENV_DIR/bin/activate" && python mcp_server.py