from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class StreamEvent(BaseModel):
    """Base class for all streaming events"""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class PredictionEvent(StreamEvent):
    """Event sent when tool prediction is complete"""
    type: Literal["prediction"] = "prediction"


class AuthCheckEvent(StreamEvent):
    """Event sent when authentication check is complete"""
    type: Literal["auth_check"] = "auth_check"


class ToolExecutionStartEvent(StreamEvent):
    """Event sent when a tool starts executing"""
    type: Literal["tool_execution_start"] = "tool_execution_start"


class ToolExecutionCompleteEvent(StreamEvent):
    """Event sent when a tool completes execution"""
    type: Literal["tool_execution_complete"] = "tool_execution_complete"


class ConversationStepEvent(StreamEvent):
    """Event sent for each step in the agent conversation"""
    type: Literal["conversation_step"] = "conversation_step"


class FinalAnswerEvent(StreamEvent):
    """Event sent with the final aggregated response"""
    type: Literal["final_answer"] = "final_answer"


class ErrorEvent(StreamEvent):
    """Event sent when an error occurs during streaming"""
    type: Literal["error"] = "error"


class ProgressEvent(StreamEvent):
    """Event sent to update overall progress"""
    type: Literal["progress"] = "progress"


# Union type for all possible stream events
StreamEventType = Union[
    PredictionEvent,
    AuthCheckEvent,
    ToolExecutionStartEvent,
    ToolExecutionCompleteEvent,
    ConversationStepEvent,
    FinalAnswerEvent,
    ErrorEvent,
    ProgressEvent
]


class StreamingQueryRequest(BaseModel):
    """Request model for streaming queries"""
    query: str
    history: List[Dict[str, str]] = Field(default_factory=list)
    include_intermediate: bool = True
    include_progress: bool = True


class ToolExecutionInfo(BaseModel):
    """Information about tool execution"""
    tool_name: str
    arguments: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None


class StreamingProgress(BaseModel):
    """Progress information for streaming execution"""
    total_steps: int
    completed_steps: int
    current_step: str
    percentage: int = Field(ge=0, le=100)


class ConversationStep(BaseModel):
    """Individual step in the agent conversation"""
    step_number: int
    agent_name: str
    message_type: str  # "function_call", "function_result", "assistant_response"
    content: str
    function_name: Optional[str] = None
    function_args: Optional[Dict[str, Any]] = None
    function_result: Optional[Any] = None


class StreamingSummary(BaseModel):
    """Summary of the streaming execution"""
    query: str
    predicted_tools: List[str]
    actual_tools_used: List[str]
    prediction_accuracy: bool
    total_execution_time_ms: int
    tools_executed: List[ToolExecutionInfo]
    conversation_steps: List[ConversationStep]
    final_response: str
    error_count: int = 0
    warnings: List[str] = Field(default_factory=list)