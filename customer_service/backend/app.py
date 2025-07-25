import asyncio
import json
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import openai
from agents import inventory_lookup_agent, order_status_agent, refund_tracking_agent, router_agent, executor
from logic import get_order_status, get_inventory_lookup, get_refund_tracking
from dotenv import load_dotenv
from mcp_server import MCPServer

import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Service API with MCP Support",
    description="A FastAPI application that serves both REST API endpoints and MCP server functionality",
    version="1.0.0"
)

# CORS for frontend on localhost:4200
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://localhost:3000"],  # Added port 3000 for MCP clients
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# REST API Models
# ========================

class LookupRequest(BaseModel):
    item_name: str

class StatusRequest(BaseModel):
    order_id: str

class RefundRequest(BaseModel):
    order_id: str
    reason: str

class QueryRequest(BaseModel):
    query: str
    history: list[dict[str, str]] = Field(default_factory=list)

# ========================
# MCP Protocol Models
# ========================

class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int]
    method: str
    params: Optional[Dict[str, Any]] = None

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Union[str, int]
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None

class MCPTool(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, Any]

# ========================
# MCP Server Implementation
# ========================

# ========================
# REST API Endpoints
# ========================

@app.post("/lookup")
def lookup(req: LookupRequest):
    """REST API endpoint for inventory lookup"""
    try:
        response_chat = executor.initiate_chat(
            inventory_lookup_agent, 
            message=f"Please look up item: {req.item_name}"
        )
        return {"message": response_chat.summary or "No response available"}
    except Exception as e:
        logger.error(f"Error in lookup endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/status")
def status(req: StatusRequest):
    """REST API endpoint for order status"""
    try:
        response_chat = executor.initiate_chat(
            order_status_agent, 
            message=f"What is the status of order ID: {req.order_id}"
        )
        return {"message": response_chat.summary or "No response available"}
    except Exception as e:
        logger.error(f"Error in status endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refund")
def refund(req: RefundRequest):
    """REST API endpoint for refund tracking"""
    try:
        response_chat = executor.initiate_chat(
            refund_tracking_agent, 
            message=f"Track refund for order ID: {req.order_id}. Reason: {req.reason}"
        )
        return {"message": response_chat.summary or "No response available"}
    except Exception as e:
        logger.error(f"Error in refund endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_customer_query(request: QueryRequest):
    """REST API endpoint for general customer queries"""
    try:
        autogen_messages = request.history + [{"role": "user", "content": request.query}]

        logger.info(f"Received query: {request.query}")

        # Remove leading non-user messages
        while autogen_messages and autogen_messages[0]["role"] != "user":
            logger.info(f"Removing non-user message: {autogen_messages[0]}")
            autogen_messages = autogen_messages[1:]

        # Validate all messages
        for i, msg in enumerate(autogen_messages):
            assert isinstance(msg, dict), f"Message {i} is not a dict: {msg}"
            assert "role" in msg, f"Message {i} missing 'role': {msg}"
            assert "content" in msg, f"Message {i} missing 'content': {msg}"
            assert msg["content"], f"Message {i} has empty 'content': {msg}"
            assert msg["role"] in ["user", "assistant"], f"Message {i} has invalid role: {msg}"

        if not autogen_messages:
            return {"response": "No valid user message found in conversation history."}

        response_chat = executor.initiate_chat(
            recipient=router_agent,
            message=request.query,
        )
        return {"response": response_chat.summary or "No response available"}
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================
# MCP Server Endpoints
# ========================

mcp_server = MCPServer()

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """HTTP endpoint for MCP requests"""
    try:
        body_bytes = await request.body()
        if not body_bytes:
            raise ValueError("Empty request body")
        body = json.loads(body_bytes)
        mcp_request = MCPRequest(**body)
        response = await mcp_server.handle_request(mcp_request)
        return response.dict(exclude_none=True)
    except Exception as e:
        logger.error(f"Error in MCP endpoint: {e}")
        return MCPResponse(
            id=0,
            error={
                "code": -32700,
                "message": f"Parse error: {str(e)}"
            }
        ).dict(exclude_none=True)
    
@app.websocket("/mcp-ws")
async def mcp_websocket(websocket: WebSocket):
    """WebSocket endpoint for MCP requests"""
    await websocket.accept()
    logger.info("MCP WebSocket connection established")
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            logger.info(f"Received MCP WebSocket message: {data}")
            
            try:
                # Parse JSON-RPC request
                request_data = json.loads(data)
                mcp_request = MCPRequest(**request_data)
                
                # Handle request
                response = await mcp_server.handle_request(mcp_request)
                
                # Send response
                response_json = json.dumps(response.dict(exclude_none=True))
                await websocket.send_text(response_json)
                logger.info(f"Sent MCP WebSocket response: {response_json}")
                
            except json.JSONDecodeError as e:
                error_response = MCPResponse(
                    id=0,
                    error={
                        "code": -32700,
                        "message": f"Parse error: {str(e)}"
                    }
                )
                await websocket.send_text(json.dumps(error_response.dict(exclude_none=True)))
            except Exception as e:
                logger.error(f"Error processing MCP WebSocket message: {e}")
                error_response = MCPResponse(
                    id=0,
                    error={
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    }
                )
                await websocket.send_text(json.dumps(error_response.dict(exclude_none=True)))
                
    except WebSocketDisconnect:
        logger.info("MCP WebSocket connection closed")
    except Exception as e:
        logger.error(f"Unexpected error in MCP WebSocket: {e}")

# ========================
# Health Check and Info Endpoints
# ========================

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Customer Service API with MCP Support",
        "version": "1.0.0",
        "endpoints": {
            "rest_api": {
                "lookup": "/lookup",
                "status": "/status", 
                "refund": "/refund",
                "ask": "/ask"
            },
            "mcp": {
                "http": "/mcp",
                "websocket": "/mcp-ws"
            }
        },
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}

@app.get("/mcp/tools")
async def list_mcp_tools():
    """List available MCP tools"""
    # Instantiate the MCPServer for use in FastAPI
    return {
        "tools": mcp_server.tools
    }

# ========================
# Startup Event
# ========================

@app.on_event("startup")
async def startup_event():
    logger.info("Customer Service API with MCP Support starting up...")
    logger.info("Available endpoints:")
    logger.info("  REST API: /lookup, /status, /refund, /ask")
    logger.info("  MCP HTTP: /mcp")
    logger.info("  MCP WebSocket: /mcp-ws")
    logger.info("  Documentation: /docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)