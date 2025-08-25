import asyncio
import json
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional, Any
import logging
from streaming_models import (
    StreamEvent, PredictionEvent, AuthCheckEvent, ToolExecutionStartEvent,
    ToolExecutionCompleteEvent, FinalAnswerEvent, ErrorEvent, ProgressEvent,
    ToolExecutionInfo, StreamingProgress, StreamingSummary
)
from mcp_server import MCPServer
from query_predictor import predictor
from tools_config import tool_mappings
from authentication_dependencies import UserProfile

logger = logging.getLogger(__name__)


class MCPStreamingOrchestrator:
    """Orchestrator that uses MCP server as backend for tool execution with streaming capabilities"""
    
    def __init__(self):
        self.mcp_server = MCPServer()
        self.tool_mappings = tool_mappings
        
    async def stream_query_execution(
        self, 
        query: str, 
        current_user: Optional[UserProfile] = None,
        conversation_history: List[Dict[str, str]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream the execution of a query using MCP server as the backend tool provider
        """
        start_time = datetime.now()
        conversation_history = conversation_history or []
        tools_executed: List[ToolExecutionInfo] = []
        error_count = 0
        
        try:
            # Step 1: Predict tools that will be needed
            logger.info(f"Starting streaming execution for query: {query}")
            prediction_result = predictor.predict_tools_hybrid(query)
            predicted_tools = prediction_result["predicted_tools"]
            
            prediction_event = PredictionEvent(data={
                "predicted_tools": predicted_tools,
                "confidence": prediction_result.get("confidence", 0.0),
                "prediction_method": prediction_result.get("method", "hybrid")
            })
            yield f"data: {prediction_event.json()}\n\n"
            
            # Step 2: Authentication check
            async for auth_event in self._handle_authentication(predicted_tools, current_user):
                yield auth_event
            
            # Step 3: Query analysis and entity extraction
            entities = self._extract_entities(query)
            yield f"data: {ProgressEvent(data={'step': 'entity_extraction', 'entities': entities}).json()}\n\n"
            
            # Step 4: Tool orchestration and execution
            tools_to_execute = self._determine_execution_plan(predicted_tools, entities)
            
            progress = StreamingProgress(
                total_steps=len(tools_to_execute),
                completed_steps=0,
                current_step="starting_execution",
                percentage=0
            )
            
            yield f"data: {ProgressEvent(data=progress.dict()).json()}\n\n"
            
            # Execute tools using MCP server
            tool_results = []
            for i, tool_call in enumerate(tools_to_execute):
                try:
                    # Stream tool execution start
                    tool_start_time = datetime.now()
                    start_event = ToolExecutionStartEvent(data={
                        "tool_name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                        "step": i + 1,
                        "total_steps": len(tools_to_execute)
                    })
                    yield f"data: {start_event.json()}\n\n"
                    
                    # Execute tool via MCP server
                    mcp_request = {
                        "jsonrpc": "2.0",
                        "id": f"stream_{i}",
                        "method": "tools/call",
                        "params": {
                            "name": tool_call["name"],
                            "arguments": tool_call["arguments"]
                        }
                    }
                    
                    # Build headers for authentication if needed
                    headers = {}
                    if current_user and self._is_protected_tool(tool_call["name"]):
                        headers["authorization"] = f"Bearer {current_user.token}"
                    
                    mcp_response = await self.mcp_server.handle_request(mcp_request, headers)
                    
                    tool_end_time = datetime.now()
                    duration_ms = int((tool_end_time - tool_start_time).total_seconds() * 1000)
                    
                    # Extract result from MCP response
                    if "result" in mcp_response and "content" in mcp_response["result"]:
                        result = mcp_response["result"]["content"][0]["text"]
                    elif "error" in mcp_response:
                        result = f"Error: {mcp_response['error']['message']}"
                        error_count += 1
                    else:
                        result = "Unknown response format"
                        error_count += 1
                    
                    tool_results.append(result)
                    
                    # Record tool execution info
                    tool_info = ToolExecutionInfo(
                        tool_name=tool_call["name"],
                        arguments=tool_call["arguments"],
                        start_time=tool_start_time,
                        end_time=tool_end_time,
                        result=result,
                        duration_ms=duration_ms,
                        error=mcp_response.get("error", {}).get("message") if "error" in mcp_response else None
                    )
                    tools_executed.append(tool_info)
                    
                    # Stream tool execution complete
                    complete_event = ToolExecutionCompleteEvent(data={
                        "tool_name": tool_call["name"],
                        "result": result,
                        "duration_ms": duration_ms,
                        "success": "error" not in mcp_response,
                        "step": i + 1
                    })
                    yield f"data: {complete_event.json()}\n\n"
                    
                    # Update progress
                    progress.completed_steps = i + 1
                    progress.percentage = int((progress.completed_steps / progress.total_steps) * 100)
                    progress.current_step = f"completed_{tool_call['name']}"
                    
                    yield f"data: {ProgressEvent(data=progress.dict()).json()}\n\n"
                    
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error executing tool {tool_call['name']}: {e}")
                    error_event = ErrorEvent(data={
                        "tool_name": tool_call["name"],
                        "error": str(e),
                        "step": i + 1
                    })
                    yield f"data: {error_event.json()}\n\n"
            
            # Step 5: Generate final response
            final_response = self._aggregate_tool_results(query, tool_results, predicted_tools)
            
            end_time = datetime.now()
            total_duration_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Create summary
            summary = StreamingSummary(
                query=query,
                predicted_tools=predicted_tools,
                actual_tools_used=[tool["name"] for tool in tools_to_execute],
                prediction_accuracy=set(predicted_tools) == set([tool["name"] for tool in tools_to_execute]),
                total_execution_time_ms=total_duration_ms,
                tools_executed=tools_executed,
                conversation_steps=[],  # MCP doesn't have conversation steps
                final_response=final_response,
                error_count=error_count
            )
            
            # Stream final answer
            final_event = FinalAnswerEvent(data={
                "response": final_response,
                "summary": summary.dict(),
                "execution_time_ms": total_duration_ms,
                "tools_used": [tool["name"] for tool in tools_to_execute],
                "prediction_accuracy": summary.prediction_accuracy,
                "error_count": error_count
            })
            yield f"data: {final_event.json()}\n\n"
            
        except Exception as e:
            logger.error(f"Critical error in streaming orchestration: {e}")
            error_event = ErrorEvent(data={
                "error": str(e),
                "step": "orchestration",
                "critical": True
            })
            yield f"data: {error_event.json()}\n\n"
    
    async def _handle_authentication(self, predicted_tools: List[str], current_user: Optional[UserProfile]) -> AsyncGenerator[str, None]:
        """Handle authentication check and stream the result"""
        protected_tools = self.tool_mappings.get_protected_tools()
        protected_tool_names = [tool.name for tool in protected_tools]
        predicted_protected_tools = [tool for tool in predicted_tools if tool in protected_tool_names]
        
        auth_required = len(predicted_protected_tools) > 0
        auth_valid = current_user is not None
        
        auth_event = AuthCheckEvent(data={
            "required": auth_required,
            "valid": auth_valid,
            "protected_tools": predicted_protected_tools,
            "user_id": current_user.id if current_user else None
        })
        yield f"data: {auth_event.json()}\n\n"
        
        if auth_required and not auth_valid:
            error_event = ErrorEvent(data={
                "error": f"Authentication required for tools: {', '.join(predicted_protected_tools)}",
                "step": "authentication",
                "critical": True
            })
            yield f"data: {error_event.json()}\n\n"
            raise ValueError("Authentication required")
    
    def _extract_entities(self, query: str) -> Dict[str, str]:
        """Extract entities from the query for tool arguments"""
        entities = {}
        
        # Order ID extraction
        import re
        order_patterns = [
            r'\border[:\s#]*(\w+)',
            r'\b([A-Z]{2}\d{6,})\b',
            r'\bORD(\d{6,})\b'
        ]
        
        for pattern in order_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities['order_id'] = match.group(1)
                break
        
        # Item ID/name extraction - handle multiple items
        item_patterns = [
            r'\bitem[:\s]*([a-zA-Z0-9\s,-]+)',
            r'\bproduct[:\s]*([a-zA-Z0-9\s,-]+)',
            r'\bavailability[:\s]*for[:\s]*([a-zA-Z0-9\s,-]+)'
        ]
        
        for pattern in item_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                items_text = match.group(1).strip()
                # Split by common delimiters and extract individual item IDs
                items = re.findall(r'\b(\d+)\b', items_text)
                if items:
                    entities['item_ids'] = items  # Multiple items
                    entities['item_id'] = items[0]  # First item for backward compatibility
                break
        
        # Refund ID extraction
        refund_patterns = [
            r'\brefund[:\s#]*(\w+)',
            r'\bREF(\d{6,})\b'
        ]
        
        for pattern in refund_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                entities['refund_id'] = match.group(1)
                break
        
        return entities
    
    def _determine_execution_plan(self, predicted_tools: List[str], entities: Dict[str, str]) -> List[Dict[str, Any]]:
        """Determine which tools to execute and in what order"""
        execution_plan = []
        
        # Map predicted tools to MCP tool names and build arguments
        tool_mapping = {
            "get_inventory_status": "lookup_inventory",
            "get_order_status": "check_order_status", 
            "get_refund_status": "track_refund"
        }
        
        for tool in predicted_tools:
            mcp_tool_name = tool_mapping.get(tool, tool)
            
            # Handle inventory lookups for multiple items
            if mcp_tool_name == "lookup_inventory" and "item_ids" in entities:
                # Create separate tool calls for each item
                for item_id in entities["item_ids"]:
                    execution_plan.append({
                        "name": mcp_tool_name,
                        "arguments": {"item_id": item_id}
                    })
            elif mcp_tool_name == "lookup_inventory" and "item_id" in entities:
                # Single item case (backward compatibility)
                execution_plan.append({
                    "name": mcp_tool_name,
                    "arguments": {"item_id": entities["item_id"]}
                })
            elif mcp_tool_name == "check_order_status" and "order_id" in entities:
                execution_plan.append({
                    "name": mcp_tool_name,
                    "arguments": {"order_id": entities["order_id"]}
                })
            elif mcp_tool_name == "track_refund" and "refund_id" in entities:
                execution_plan.append({
                    "name": mcp_tool_name,
                    "arguments": {"refund_id": entities["refund_id"]}
                })
        
        return execution_plan
    
    def _is_protected_tool(self, tool_name: str) -> bool:
        """Check if a tool requires authentication"""
        protected_tools = self.tool_mappings.get_protected_tools()
        protected_names = [tool.name for tool in protected_tools]
        return tool_name in protected_names
    
    def _aggregate_tool_results(self, query: str, tool_results: List[str], predicted_tools: List[str]) -> str:
        """Aggregate multiple tool results into a coherent response"""
        if not tool_results:
            return "I wasn't able to find the information you requested. Please check your query and try again."
        
        if len(tool_results) == 1:
            return tool_results[0]
        
        # Multiple results - provide direct answers without repeating the question
        response_parts = []
        
        for i, result in enumerate(tool_results):
            if result and not result.startswith("Error:"):
                # Clean up the result formatting
                clean_result = result.strip()
                if clean_result:
                    response_parts.append(clean_result)
            elif result and result.startswith("Error:"):
                response_parts.append(result)
        
        if not response_parts:  # No valid results
            return "I encountered issues retrieving the information. Please try again or contact support."
            
        return "\n".join(response_parts)