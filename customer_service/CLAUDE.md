# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a customer service application with both frontend and backend components:
- **Backend**: FastAPI server with AutoGen agents, MCP (Model Context Protocol) support, and RAG (Retrieval Augmented Generation)
- **Frontend**: Angular 20 application for customer FAQ interface

## Architecture

### Backend Structure (`backend/`)
- **FastAPI Application** (`app.py`): Main server with REST endpoints and MCP WebSocket support
- **AutoGen Agents** (`agents.py`): Multi-agent system with router, inventory, order status, and refund tracking agents
- **MCP Server** (`mcp_server.py`): Model Context Protocol server for external integrations
- **Authentication** (`authentication_dependencies.py`): Supabase-based user authentication
- **Business Logic** (`logic.py`): Core functions for inventory, orders, and refund operations
- **RAG Components**: Vector embeddings, document search, and answer generation (`vector_helper.py`, `models.py`)
- **Tool System**: Protected tool mappings and query prediction (`tools_config.py`, `tool_mapping.py`, `query_predictor.py`)

### Frontend Structure (`frontend/customer-service-chatgpt-faq-app/`)
- **Angular Components**: Chat interface and FAQ section components
- **Models**: TypeScript interfaces for FAQ data
- **Services**: HTTP client services for backend communication

## Development Commands

### Backend Development
```bash
# Navigate to backend directory
cd backend

# Install dependencies using uv
uv sync

# Start FastAPI development server
./start_server

# Alternative: Direct uv command
uv run uvicorn app:app --reload

# Start MCP server (for external MCP clients)
./start_mcp_server.sh
```

### Frontend Development
```bash
# Navigate to frontend directory  
cd frontend/customer-service-chatgpt-faq-app

# Install dependencies
npm install

# Start development server (port 4200)
npm start

# Build for production
npm run build

# Run tests
npm run test

# Generate new components
ng generate component component-name
```

## Key Integration Points

### Authentication Flow
- Backend uses Supabase for user authentication
- Some tools require authentication (`get_order_status`, `get_refund_status`)
- Token-based authentication for both REST API and MCP endpoints

### Agent System
- Router agent decides which specific function to call based on user queries
- Executor agent handles function execution
- Functions are registered with both caller and executor for AutoGen integration

### MCP Protocol Support
- HTTP endpoint: `/mcp` 
- WebSocket endpoint: `/mcp-ws`
- Supports authentication via Bearer tokens
- Available tools: `lookup_inventory`, `check_order_status`, `track_refund`

### RAG Implementation  
- Document storage in Supabase with vector embeddings
- `/documents` endpoint for adding documents (admin only)
- `/search` endpoint for semantic search with OpenAI embeddings
- Integration with OpenAI for answer generation

## Environment Requirements

### Backend
- Python 3.11+ with uv for dependency management
- Required environment variables:
  - `OPENAI_API_KEY`
  - `SUPABASE_URL`
  - `SUPABASE_KEY`
- FastAPI runs on port 8000

### Frontend  
- Node.js with npm
- Angular CLI 20.0.5
- Development server runs on port 4200
- CORS configured for localhost:4200 and localhost:3000

## Testing
- Frontend: Uses Karma with Jasmine test runner
- Backend: No specific test framework configured (add as needed)

## Common Development Tasks

### Adding New Agents
1. Define agent in `agents.py`
2. Register functions with the agent using `register_function`
3. Update router agent's system message with new capabilities

### Adding Protected Tools
1. Add tool to `tools_config.py`
2. Update `tool_mapping.py` with protection rules
3. Ensure authentication checks in relevant endpoints

### RAG Document Management
- Use `/documents` endpoint to add new knowledge base content
- Documents are automatically embedded using OpenAI text-embedding-ada-002
- Search uses vector similarity in Supabase PostgreSQL with pgvector

### Streaming vs Traditional Endpoints

#### **Traditional `/ask` Endpoint**
- Uses AutoGen multi-agent system (router + executor)
- Returns complete response after full processing
- Good for simple queries and standard workflows

#### **Streaming `/ask-stream` Endpoint**  
- Uses MCP server as backend with streaming orchestration
- Provides real-time progress updates via Server-Sent Events
- Handles multiple entity extraction (e.g., "items 123 and 456" → separate tool calls)
- Shows tool prediction, authentication, and execution progress
- Better user experience for complex multi-tool queries

#### **Frontend Integration**
- "Send" button → `/ask` endpoint (existing functionality)
- "Send (MCP Stream)" button → `/ask-stream` endpoint (streaming experience)
- Real-time progress bars and detailed event tracking
- Streaming UI updates with tool execution status