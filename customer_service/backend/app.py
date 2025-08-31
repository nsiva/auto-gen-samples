import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import openai
from agents import inventory_lookup_agent, order_status_agent, refund_tracking_agent, router_agent, executor
from logic import get_order_status, get_inventory_lookup, get_refund_tracking
from dotenv import load_dotenv
from mcp_server import MCPServer
from models import DocumentInput, DocumentResponse, SearchResult, RAGSearchResponse
from query_predictor import predict_tools_with_entities, predictor, simulate_router_decision
from authentication_dependencies import get_current_user, get_optional_current_user, UserProfile
from config import AUTH_LOGIN_URL, SUPABASE_KEY, SUPABASE_URL
from urllib.parse import urlencode

from supabase import create_client, Client

from fastapi import FastAPI, Depends, HTTPException
from fastapi import status

from typing import Annotated

import logging

from tools_config import tool_mappings
from vector_helper import count_tokens, generate_embedding, generate_rag_answer
from mcp_streaming_orchestrator import MCPStreamingOrchestrator
from streaming_models import StreamingQueryRequest 

load_dotenv()

def build_auth_url_with_callback(request: Request) -> str:
    """Build auth URL with callback parameter based on request origin"""
    callback_url = f"{request.headers.get('origin', 'http://localhost:4200')}/auth/callback"
    separator = '&' if '?' in AUTH_LOGIN_URL else '?'
    return f"{AUTH_LOGIN_URL}{separator}callback_url={callback_url}"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP WebSocket session management
class MCPSession:
    def __init__(self, user: UserProfile, session_id: str):
        self.user = user
        self.session_id = session_id
        self.authenticated_at = datetime.now()
        self.last_activity = datetime.now()

# Global session storage
mcp_sessions: Dict[str, MCPSession] = {}
mcp_temp_tokens: Dict[str, Dict] = {}

async def get_user_from_cookie(auth_token: Optional[str]) -> Optional[UserProfile]:
    """Extract user from auth token cookie"""
    if not auth_token:
        return None
    
    try:
        user = await get_current_user(auth_token)
        return user
    except Exception as e:
        logger.warning(f"Cookie authentication failed: {e}")
        return None

async def try_authenticate_mcp_websocket(temp_token: str) -> Optional[str]:
    """Try to authenticate WebSocket connection using token - doesn't close connection on failure"""
    logger.info("=== TRYING MCP WEBSOCKET AUTHENTICATION ===")
    
    try:
        logger.info(f"Validating token: {temp_token[:8] + '...' if temp_token else 'None'}")
        
        # Check if temp token exists and is valid
        if temp_token not in mcp_temp_tokens:
            logger.warning("Invalid or expired MCP token")
            return None
            
        token_data = mcp_temp_tokens[temp_token]
        
        # Check if token is expired (5 minutes)
        created_at = token_data['created_at']
        expires_in = token_data['expires_in']
        if (datetime.now() - created_at).total_seconds() > expires_in:
            logger.warning("MCP token expired")
            del mcp_temp_tokens[temp_token]  # Clean up expired token
            return None
        
        # Get user from token data
        user = token_data['user']
        
        # Create session
        session_id = str(uuid.uuid4())
        mcp_sessions[session_id] = MCPSession(user, session_id)
        
        # Clean up the temporary token (one-time use)
        del mcp_temp_tokens[temp_token]
        
        logger.info(f"MCP WebSocket authenticated for user: {user.email}")
        return session_id
        
    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        return None

async def authenticate_mcp_websocket(websocket: WebSocket) -> Optional[str]:
    """Authenticate WebSocket connection using query parameter token (DEPRECATED - closes on failure)"""
    logger.info("=== AUTHENTICATING MCP WEBSOCKET (STRICT) ===")
    
    try:
        # Try to get temporary token from query parameters
        query_params = dict(websocket.query_params)
        logger.info(f"Query parameters: {query_params}")
        temp_token = query_params.get("token")
        logger.info(f"Extracted token: {temp_token[:8] + '...' if temp_token else 'None'}")
        
        if not temp_token:
            logger.warning("No MCP token found in query parameters")
            await websocket.close(code=4001, reason="Authentication token required")
            return None
        
        session_id = await try_authenticate_mcp_websocket(temp_token)
        if not session_id:
            await websocket.close(code=4001, reason="Invalid authentication token")
            return None
            
        return session_id
        
    except Exception as e:
        logger.error(f"WebSocket authentication error: {e}")
        await websocket.close(code=4001, reason="Authentication failed")
        return None

app = FastAPI(
    title="Customer Service API with MCP Support",
    description="A FastAPI application that serves both REST API endpoints and MCP server functionality",
    version="1.0.0"
)

# --- Supabase Client ---
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- OpenAI Client for Embeddings ---
from openai import OpenAI

import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
embedding_model = "text-embedding-ada-002"

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
async def ask_customer_query(query_request: QueryRequest,
    request: Request,
    current_user: Optional[UserProfile] = Depends(get_optional_current_user)
):
    """REST API endpoint for general customer queries"""
    try:
        autogen_messages = query_request.history + [{"role": "user", "content": query_request.query}]

        logger.info(f"Received query: {query_request.query}")

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
        prediction_result = predictor.predict_tools_hybrid(query_request.query)
        predicted_tools = prediction_result["predicted_tools"]
        
        logger.info(f"PREDICTED tools before execution: {predicted_tools}")
        logger.info(f"Prediction details: {prediction_result}")
        
        # STEP 2: CHECK AUTHENTICATION based on predictions
        protected_tools = tool_mappings.get_protected_tools() 
        protected_tool_names = [tool.name for tool in protected_tools]
        logger.info(f"Protected tools: {protected_tool_names}")
        predicted_protected_tools = [tool for tool in predicted_tools if tool in protected_tool_names]
        
        if predicted_protected_tools and current_user is None:
            logger.warning(f"Authentication required for predicted tools: {predicted_protected_tools}")
            raise HTTPException(
                status_code=401,
                detail=f"Authentication required. Per our prediction, this query will likely access: {', '.join(predicted_protected_tools)}",
                headers={"X-Auth-URL": build_auth_url_with_callback(request)}
            )
        
        # STEP 3: Execute the actual conversation
        response_chat = executor.initiate_chat(
            recipient=router_agent,
            message=query_request.query,
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
        if any(tool in tools_called for tool in protected_tool_names):
            if current_user is None:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required for order status and refund tracking.",
                    headers={"X-Auth-URL": build_auth_url_with_callback(request)}
                )
        
        return {
            "response": response_chat.summary or "No response available",
            "predicted_tools": predicted_tools,
            "actual_tools_used": tools_called,
            "prediction_accuracy": prediction_result
        }
    except HTTPException as httpException:
        # Re-raise HTTPException so FastAPI returns the correct status code
        raise httpException
    except Exception as genericException:
        logger.error(f"Error in ask endpoint: {genericException}")
        raise HTTPException(status_code=500, detail=str(genericException))

@app.post("/predict-tools")
async def predict_tools_endpoint(request: QueryRequest):
    """Endpoint to predict which tools will be called for a query"""
    try:
        # Method 1: Keyword-based
        keyword_prediction = predictor.predict_tools_keyword_based(query_request.query)
        
        # Method 2: LLM-based  
        llm_prediction = predictor.predict_tools_llm_based(query_request.query)
        
        entity_prediction = predict_tools_with_entities(query_request.query)
        
        # Method 3: Entity extraction
        router_simulation = simulate_router_decision(query_request.query)
        
        return {
            "query": query_request.query,
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
    
        # Method 4: Router simulation

@app.post("/ask-stream")
async def ask_streaming(
    streaming_request: StreamingQueryRequest,
    request: Request,
    current_user: Optional[UserProfile] = Depends(get_optional_current_user)
):
    """Streaming endpoint that uses MCP server as backend for tool execution"""
    try:
        # Initialize MCP streaming orchestrator
        orchestrator = MCPStreamingOrchestrator()
        
        # Stream the query execution
        async def generate_stream():
            async for event_data in orchestrator.stream_query_execution(
                query=streaming_request.query,
                current_user=current_user,
                conversation_history=streaming_request.history
            ):
                yield event_data
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in streaming ask endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================
# MCP Server Endpoints
# ========================

mcp_server = MCPServer()

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    headers = dict(request.headers)
    body = await request.json()

    """HTTP endpoint for MCP requests"""
    try:
        body_bytes = await request.body()
        if not body_bytes:
            raise ValueError("Empty request body")
        # body = json.loads(body_bytes)
        # mcp_request = MCPRequest(**body)
        response = await mcp_server.handle_request(body, headers=headers)
        return response #.dict(exclude_none=True)
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
    """WebSocket endpoint for MCP requests with optional authentication"""
    logger.info("=== NEW MCP WEBSOCKET CONNECTION ===")
    logger.info(f"WebSocket headers: {dict(websocket.headers)}")
    logger.info(f"Query params: {dict(websocket.query_params)}")
    
    await websocket.accept()
    logger.info("MCP WebSocket connection accepted")
    
    # Try to authenticate, but don't require it for public tools
    session_id = None
    query_params = dict(websocket.query_params)
    temp_token = query_params.get("token")
    
    if temp_token:
        logger.info("Token provided, attempting authentication...")
        session_id = await try_authenticate_mcp_websocket(temp_token)
        if session_id:
            logger.info(f"Authentication successful, session_id: {session_id}")
        else:
            logger.info("Authentication failed, but continuing for public tools")
    else:
        logger.info("No token provided, continuing as unauthenticated user")
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                request_json = json.loads(data)
                logger.info(f"Received MCP request: {request_json.get('method', 'unknown')}")

                # Update session activity if authenticated
                if session_id and session_id in mcp_sessions:
                    mcp_sessions[session_id].last_activity = datetime.now()
                
                # Check if this request requires authentication
                method = request_json.get("method")
                tool_name = None
                if method == "tools/call":
                    tool_name = request_json.get("params", {}).get("name")
                    logger.info(f"Tool call requested: {tool_name}")
                
                # Check if tool requires authentication
                requires_auth = tool_name and tool_mappings.is_protected(tool_name)
                logger.info(f"Tool requires auth: {requires_auth}")
                
                if requires_auth and not session_id:
                    # Tool requires auth but user is not authenticated
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": request_json.get("id", ""),
                        "error": {
                            "code": 401,
                            "message": f"Authentication required for tool: {tool_name}"
                        }
                    }
                    await websocket.send_text(json.dumps(error_response))
                    continue
                
                # Build headers for MCP server
                headers = {}
                if session_id and session_id in mcp_sessions:
                    session = mcp_sessions[session_id]
                    headers = {
                        "user_id": session.user.user_id,
                        "user_email": session.user.email
                    }

                response = await mcp_server.handle_request(request_json, headers=headers)
                await websocket.send_text(json.dumps(response))
                logger.info(f"Sent MCP WebSocket response for session {session_id}")
                
            except json.JSONDecodeError as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": "",
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {str(e)}"
                    }
                }
                await websocket.send_text(json.dumps(error_response))
            except Exception as e:
                logger.error(f"Error processing MCP WebSocket message: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request_json.get("id", "") if 'request_json' in locals() else "",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    }
                }
                await websocket.send_text(json.dumps(error_response))
                
    except WebSocketDisconnect:
        logger.info(f"MCP WebSocket connection closed for session {session_id}")
    except Exception as e:
        logger.error(f"Unexpected error in MCP WebSocket: {e}")
    finally:
        # Clean up session
        if session_id and session_id in mcp_sessions:
            del mcp_sessions[session_id]
            logger.info(f"Cleaned up MCP session: {session_id}")

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

# --- API Endpoints ---

@app.post("/documents", response_model=DocumentResponse)
async def post_document(
    document: DocumentInput
):
    """
    Posts a new document to the PostgreSQL vector database.
    Open to anyone - no authentication required.
    """
    try:
        logger.info(f"Anonymous user attempting to post a document.")
        embedding = await generate_embedding(document.content)
        
        # Insert into Supabase
        response = supabase.from_('documents').insert({
            "content": document.content,
            "embedding": embedding,
            "metadata": document.metadata
        }).execute()

        if response.data:
            logger.info(f"Document with ID '{response.data[0]['id']}' posted successfully by anonymous user.")
            return DocumentResponse(**response.data[0])
        else:
            logger.error(f"Failed to post document to Supabase. Error: {response.error.message}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to post document: {response.error.message}"
            )
    except HTTPException as e:
        logger.warning(f"HTTPException raised during document post: {e.detail}")
        raise # Re-raise FastAPI HTTPExceptions
    except Exception as e:
        logger.exception(f"An unexpected error occurred while posting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while posting the document."
        )

@app.get("/search", response_model=RAGSearchResponse)
async def search_documents(
    query: str,
    top_k: int = 25
):
    """
    Searches the PostgreSQL vector database for documents similar to the query string.
    Open to anyone - no authentication required.
    """
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty."
        )

    try:
        logger.info(f"Anonymous user is searching for: '{query}' (top_k={top_k})")
        
        query_embedding = await generate_embedding(query)
        num_query_tokens = count_tokens(query)

        # Perform similarity search in Supabase
        # The 'vector_distance' operator is typically used for similarity, e.g., <-> for L2 distance, <=> for cosine
        # Supabase's 'match_documents' function (if created as RPC) or raw SQL via `rpc`
        # For direct `pgvector` queries, you'd use `order by embedding <-> :query_embedding limit :top_k`
        # Supabase Python client's `rpc` method is ideal for custom SQL functions like match_documents
        
        # Assuming you have an RPC function named 'match_documents' in Supabase like:
        # CREATE OR REPLACE FUNCTION match_documents(query_embedding vector(1536), match_threshold float, match_count int)
        # RETURNS TABLE (
        #     id UUID,
        #     content TEXT,
        #     metadata JSONB,
        #     created_at TIMESTAMPTZ,
        #     similarity float
        # )
        # LANGUAGE plpgsql AS $$
        # BEGIN
        #   RETURN QUERY
        #   SELECT
        #     d.id,
        #     d.content,
        #     d.metadata,
        #     d.created_at,
        #     1 - (d.embedding <=> query_embedding) AS similarity -- Cosine similarity between 0 and 1
        #   FROM documents d
        #   ORDER BY d.embedding <=> query_embedding
        #   LIMIT match_count;
        # END;
        # $$;

        # Call the RPC function (replace 'match_documents' with your actual function name if different)
        response = supabase.rpc(
            'match_documents',
            {
                'query_embedding': query_embedding,
                'match_count': top_k,
                'match_threshold': 0.7 # Example threshold, adjust as needed
            }
        ).execute()

        # --- Corrected Direct Supabase Query for Vector Similarity ---
        # The `<->` operator calculates cosine distance. Lower values mean higher similarity.
        # We use `order` to sort by this distance and `limit` the results.
        # The `select` method allows including the distance/similarity if you create a custom view/function
        # or calculate it client-side. For `supabase-py`'s `from_().select().order().limit()`,
        # the distance is not returned as a column directly.
        # So we'll fetch the embeddings and calculate similarity client-side for `SearchResult` display.
        
        # 1. Fetch documents ordered by their cosine distance to the query embedding
        # The .text(f'embedding <-> \'{query_embedding}\'') part is how you apply operators
        # in the order_by clause using Supabase's PostgREST syntax when the client doesn't have a direct method.
        # However, for the `supabase-py` client, the more direct way is to use `order('embedding', operator='<->', value=query_embedding)`
        # but that also needs specific `PostgrestClient` version support or RPC.
        # The simplest workaround without RPC is to just order by the vector column,
        # hoping the underlying index uses the operator, and then compute similarity.

        # Let's use the .order() method directly by the embedding column and rely on the
        # default `pgvector` behavior for ordering by closest vector if an appropriate index is present.
        # Then we calculate similarity in Python.
        # response = supabase.from_('documents').select(
        #     'id, content, metadata, created_at, embedding' # Select embedding to calculate similarity client-side
        # ).order(
        #     'embedding',
        #     desc=False # Ascending order for distance (lower distance = higher similarity)
        # ).limit(top_k).execute()

        retrieved_results: List[SearchResult] = []
        if response.data:
            retrieved_results = [SearchResult(
                id=str(d['id']),
                content=d['content'],
                metadata=d['metadata'],
                created_at=d['created_at'],
                similarity=d.get('similarity', 0.0) # Ensure similarity is captured
            ) for d in response.data]
            logger.info(f"Retrieved {len(retrieved_results)} documents for RAG processing.")
        else:
            logger.warning(f"No documents retrieved for query '{query}'") # or RPC error: {response.error.message}")
            # Even if no docs, we can still try to generate an answer saying so, or return empty
            # For this RAG example, we'll proceed with potentially empty context

        # Extract content from retrieved documents
        retrieved_docs_content = [doc.content for doc in retrieved_results]

        # Generate answer using LLM
        llm_answer = await generate_rag_answer(query, retrieved_docs_content)
        num_answer_tokens = count_tokens(llm_answer)

        return RAGSearchResponse(
            query=query,
            answer=llm_answer,
            retrieved_documents=retrieved_results,
            num_query_tokens=num_query_tokens,
            num_answer_tokens=num_answer_tokens
        )
    except HTTPException as e:
        logger.warning(f"HTTPException raised during document search: {e.detail}")
        raise # Re-raise FastAPI HTTPExceptions
    except Exception as e:
        logger.exception(f"An unexpected error occurred while searching documents: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while searching documents."
        )

@app.get("/search_all", response_model=RAGSearchResponse)
async def search_all_documents(
    query: str,
    limit: int = 100
):
    """
    Retrieves all documents from the database without vector search, then generates RAG answer.
    Open to anyone - no authentication required.
    Useful for getting comprehensive answers when vector search might miss relevant content.
    """
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty."
        )

    try:
        logger.info(f"Anonymous user is searching all documents for: '{query}' (limit={limit})")
        
        num_query_tokens = count_tokens(query)

        # Retrieve ALL documents (or up to limit) without vector search
        response = supabase.from_('documents').select(
            'id, content, metadata, created_at'
        ).order(
            'created_at', desc=True  # Order by newest first
        ).limit(limit).execute()

        retrieved_results: List[SearchResult] = []
        if response.data:
            # Since we're not doing vector search, set similarity to 0.0 for all documents
            retrieved_results = [SearchResult(
                id=str(d['id']),
                content=d['content'],
                metadata=d['metadata'],
                created_at=d['created_at'],
                similarity=0.0  # No similarity calculation since we're not doing vector search
            ) for d in response.data]
            logger.info(f"Retrieved {len(retrieved_results)} documents (all documents approach) for RAG processing.")
        else:
            logger.warning(f"No documents found in database for query '{query}'")

        # Extract content from retrieved documents
        retrieved_docs_content = [doc.content for doc in retrieved_results]

        # Generate answer using LLM with all document content as context
        llm_answer = await generate_rag_answer(query, retrieved_docs_content)
        num_answer_tokens = count_tokens(llm_answer)

        return RAGSearchResponse(
            query=query,
            answer=llm_answer,
            retrieved_documents=retrieved_results,
            num_query_tokens=num_query_tokens,
            num_answer_tokens=num_answer_tokens
        )
    except HTTPException as e:
        logger.warning(f"HTTPException raised during search_all: {e.detail}")
        raise # Re-raise FastAPI HTTPExceptions
    except Exception as e:
        logger.exception(f"An unexpected error occurred while searching all documents: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while searching all documents."
        )

# --- Health Check Endpoint ---
@app.get("/doc_health")
async def doc_health_check():
    """Simple health check endpoint."""
    try:
        # Try to ping Supabase
        _ = supabase.from_('documents').select('id').limit(1).execute()
        # Try to call OpenAI (e.g., list models)
        _ = openai_client.models.list()
        logger.info("Health check passed.")
        return {"status": "ok", "message": "API is healthy and connected to Supabase and OpenAI."}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {e}"
        )
    
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}

@app.post("/auth/validate-token")
async def validate_token(
    current_user: UserProfile = Depends(get_current_user)
):
    """Validate authentication token - used by callback page"""
    return {
        "valid": True,
        "user": {
            "id": current_user.id,
            "email": current_user.email
        }
    }

@app.post("/auth/get-mcp-token")
async def get_mcp_token(
    current_user: UserProfile = Depends(get_current_user)
):
    """Get temporary token for MCP WebSocket connection"""
    logger.info("=== CREATING MCP TOKEN ===")
    logger.info(f"User: {current_user.email} (ID: {current_user.id})")
    
    # Create a temporary token for WebSocket authentication
    temp_token = str(uuid.uuid4())
    logger.info(f"Generated token: {temp_token[:8]}...")
    
    # Store user info with temp token (expires in 5 minutes)
    token_data = {
        'user': current_user,
        'created_at': datetime.now(),
        'expires_in': 300  # 5 minutes
    }
    
    mcp_temp_tokens[temp_token] = token_data
    logger.info(f"Stored token data. Total tokens in memory: {len(mcp_temp_tokens)}")
    
    response = {
        "success": True,
        "mcp_token": temp_token,
        "expires_in": 300,
        "user": {
            "id": current_user.id,
            "email": current_user.email
        }
    }
    
    logger.info("MCP token created successfully")
    return response

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