# Customer Service Backend

FastAPI server with AutoGen agents, MCP support, and RAG functionality.

## Quick Setup

```bash
# Install dependencies
uv sync

# Start FastAPI server (port 8000)
./start_server

# Start MCP server (separate terminal)
./start_mcp_server.sh
```

## Environment Variables Required

```bash
export OPENAI_API_KEY="your-key-here"
export SUPABASE_URL="your-supabase-url"
export SUPABASE_KEY="your-supabase-key"
```

## Endpoints

- `/docs` - API documentation
- `/ask` - Traditional AutoGen endpoint  
- `/ask-stream` - Streaming MCP endpoint
- `/mcp` - MCP HTTP endpoint
- `/mcp-ws` - MCP WebSocket endpoint