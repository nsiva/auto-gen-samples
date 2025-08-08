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
from query_predictor import predict_tools_with_entities, predictor, simulate_router_decision
from authentication_dependencies import get_current_user, get_optional_current_user, UserProfile
from config import AUTH_LOGIN_URL


from fastapi import FastAPI, Depends, HTTPException, status

from typing import Annotated

import logging

from tools_config import tool_mapping 

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

# Protected endpoint: requires authentication
@app.get("/protected_data")
async def read_protected_data(current_user: Annotated[UserProfile, Depends(get_current_user)]):
    """
    This endpoint requires a valid authentication token.
    Returns user-specific data.
    """
    return {"message": f"Welcome, {current_user.email}! This is protected data."}

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
async def ask_customer_query(request: QueryRequest,
    current_user: Optional[UserProfile] = Depends(get_optional_current_user)
):
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

        # STEP 1: PREDICT which tools will be called BEFORE execution
        prediction_result = predictor.predict_tools_hybrid(request.query)
        predicted_tools = prediction_result["predicted_tools"]
        
        logger.info(f"PREDICTED tools before execution: {predicted_tools}")
        logger.info(f"Prediction details: {prediction_result}")
        
        # STEP 2: CHECK AUTHENTICATION based on predictions
        protected_tools = tool_mapping.get_protected_tool_names() #['get_order_status', 'get_refund_status']
        predicted_protected_tools = [tool for tool in predicted_tools if tool in protected_tools]
        
        if predicted_protected_tools and current_user is None:
            logger.warning(f"Authentication required for predicted tools: {predicted_protected_tools}")
            raise HTTPException(
                status_code=401,
                detail=f"Authentication required. Per our prediction, this query will likely access: {', '.join(predicted_protected_tools)}"
            )
        
        # STEP 3: Execute the actual conversation
        response_chat = executor.initiate_chat(
            recipient=router_agent,
            message=request.query,
        )
        
        # STEP 4: Track what was ACTUALLY called (for comparison)
        # Extract tool calls from chat history
        tools_called = []
        if hasattr(response_chat, 'chat_history'):
            for message in response_chat.chat_history:
                if isinstance(message, dict):
                    # Check for tool calls in message content
                    content = message.get('content', '')
                    if content is not None:
                        if 'get_order_status' in content:
                            tools_called.append('get_order_status')
                        elif 'get_inventory_status' in content:
                            tools_called.append('get_inventory_status') 
                        elif 'get_refund_status' in content:
                            tools_called.append('get_refund_status')
                    
                    # Check for function calls in message metadata
                    if 'tool_calls' in message:
                        for tool_call in message['tool_calls']:
                            tools_called.append(tool_call.get('function', {}).get('name'))

        logger.info(f"ACTUAL tools called: {tools_called}")
        logger.info(f"Prediction accuracy: {set(predicted_tools) == set(tools_called)}")
                
        # Check if authentication is required
        #protected_tools = tool_mapping.get_protected_tools() #['get_order_status', 'get_refund_status']
        if any(tool in tools_called for tool in protected_tools):
            if current_user is None:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required for order status and refund tracking."
                )
        
        return {
            "response": response_chat.summary or "No response available",
            "predicted_tools": predicted_tools,
            "actual_tools_used": tools_called,
            "prediction_accuracy": prediction_result
        }
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-tools")
async def predict_tools_endpoint(request: QueryRequest):
    """Endpoint to predict which tools will be called for a query"""
    try:
        # Method 1: Keyword-based
        keyword_prediction = predictor.predict_tools_keyword_based(request.query)
        
        # Method 2: LLM-based  
        llm_prediction = predictor.predict_tools_llm_based(request.query)
        
        # Method 3: Entity extraction
        entity_prediction = predict_tools_with_entities(request.query)
        
        # Method 4: Router simulation
        router_simulation = simulate_router_decision(request.query)
        
        return {
            "query": request.query,
            "predictions": {
                "keyword_based": keyword_prediction,
                "llm_based": llm_prediction, 
                "entity_based": entity_prediction,
                "router_simulation": router_simulation
            },
            "consensus": list(set(keyword_prediction + llm_prediction + entity_prediction["predicted_tools"] + router_simulation))
        }
        
    except Exception as e:
        logger.error(f"Error in predict-tools endpoint: {e}")
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