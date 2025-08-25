import asyncio
import json
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        protected_tools = tool_mappings.get_protected_tools() 
        protected_tool_names = [tool.name for tool in protected_tools]
        logger.info(f"Protected tools: {protected_tool_names}")
        predicted_protected_tools = [tool for tool in predicted_tools if tool in protected_tool_names]
        
        if predicted_protected_tools and current_user is None:
            logger.warning(f"Authentication required for predicted tools: {predicted_protected_tools}")
            raise HTTPException(
                status_code=401,
                detail=f"Authentication required. Per our prediction, this query will likely access: {', '.join(predicted_protected_tools)}",
                headers={"X-Auth-URL": AUTH_LOGIN_URL}
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
        if any(tool in tools_called for tool in protected_tool_names):
            if current_user is None:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required for order status and refund tracking.",
                    headers={"X-Auth-URL": AUTH_LOGIN_URL}
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
        keyword_prediction = predictor.predict_tools_keyword_based(request.query)
        
        # Method 2: LLM-based  
        llm_prediction = predictor.predict_tools_llm_based(request.query)
        
        entity_prediction = predict_tools_with_entities(request.query)
        
        # Method 3: Entity extraction
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
    
        # Method 4: Router simulation

@app.post("/ask-stream")
async def ask_streaming(
    request: StreamingQueryRequest,
    current_user: Optional[UserProfile] = Depends(get_optional_current_user)
):
    """Streaming endpoint that uses MCP server as backend for tool execution"""
    try:
        # Initialize MCP streaming orchestrator
        orchestrator = MCPStreamingOrchestrator()
        
        # Stream the query execution
        async def generate_stream():
            async for event_data in orchestrator.stream_query_execution(
                query=request.query,
                current_user=current_user,
                conversation_history=request.history
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
    """WebSocket endpoint for MCP requests"""
    await websocket.accept()
    logger.info("MCP WebSocket connection established")
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                request_json = json.loads(data)

                # Extract token from params if present
                token = None
                params = request_json.get("params", {})
                if "token" in params:
                    token = params["token"]

                # Build headers dict for handle_request
                headers = {}
                if token:
                    headers["authorization"] = f"Bearer {token}"

                response = await mcp_server.handle_request(request_json, headers=headers)
                await websocket.send_text(json.dumps(response))
                logger.info(f"Sent MCP WebSocket response: {response}")
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

# --- API Endpoints ---

@app.post("/documents", response_model=DocumentResponse)
async def post_document(
    document: DocumentInput,
    admin_user: UserProfile = Depends(get_current_user) # Only admin can post
):
    """
    Posts a new document to the PostgreSQL vector database.
    Requires admin authentication.
    """
    try:
        logger.info(f"Admin user '{admin_user.id}' attempting to post a document.")
        embedding = await generate_embedding(document.content)
        
        # Insert into Supabase
        response = supabase.from_('documents').insert({
            "content": document.content,
            "embedding": embedding,
            "metadata": document.metadata
        }).execute()

        if response.data:
            logger.info(f"Document with ID '{response.data[0]['id']}' posted successfully by admin '{admin_user.id}'.")
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
    top_k: int = 25,
    current_user: UserProfile = Depends(get_current_user) # All authenticated users can search
):
    """
    Searches the PostgreSQL vector database for documents similar to the query string.
    Requires any authenticated user.
    """
    if not query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty."
        )

    try:
        logger.info(f"User '{current_user.id}' is searching for: '{query}' (top_k={top_k})")
        
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
        # response = supabase.rpc(
        #     'match_documents',
        #     {
        #         'query_embedding': query_embedding,
        #         'match_count': top_k,
        #         'match_threshold': 0.7 # Example threshold, adjust as needed
        #     }
        # ).execute()

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
        response = supabase.from_('documents').select(
            'id, content, metadata, created_at, embedding' # Select embedding to calculate similarity client-side
        ).order(
            'embedding',
            desc=False # Ascending order for distance (lower distance = higher similarity)
        ).limit(top_k).execute()

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