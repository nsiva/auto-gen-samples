export interface StreamEvent {
  type: string;
  data: any;
  timestamp: string;
}

export interface PredictionEvent extends StreamEvent {
  type: 'prediction';
  data: {
    predicted_tools: string[];
    confidence: number;
    prediction_method: string;
  };
}

export interface AuthCheckEvent extends StreamEvent {
  type: 'auth_check';
  data: {
    required: boolean;
    valid: boolean;
    protected_tools: string[];
    user_id?: string;
  };
}

export interface ToolExecutionStartEvent extends StreamEvent {
  type: 'tool_execution_start';
  data: {
    tool_name: string;
    arguments: any;
    step: number;
    total_steps: number;
  };
}

export interface ToolExecutionCompleteEvent extends StreamEvent {
  type: 'tool_execution_complete';
  data: {
    tool_name: string;
    result: string;
    duration_ms: number;
    success: boolean;
    step: number;
  };
}

export interface ProgressEvent extends StreamEvent {
  type: 'progress';
  data: {
    total_steps: number;
    completed_steps: number;
    current_step: string;
    percentage: number;
  };
}

export interface FinalAnswerEvent extends StreamEvent {
  type: 'final_answer';
  data: {
    response: string;
    summary: StreamingSummary;
    execution_time_ms: number;
    tools_used: string[];
    prediction_accuracy: boolean;
    error_count: number;
  };
}

export interface ErrorEvent extends StreamEvent {
  type: 'error';
  data: {
    error: string;
    step: string;
    critical?: boolean;
    tool_name?: string;
  };
}

export interface StreamingSummary {
  query: string;
  predicted_tools: string[];
  actual_tools_used: string[];
  prediction_accuracy: boolean;
  total_execution_time_ms: number;
  tools_executed: ToolExecutionInfo[];
  conversation_steps: any[];
  final_response: string;
  error_count: number;
  warnings?: string[];
}

export interface ToolExecutionInfo {
  tool_name: string;
  arguments: any;
  start_time: string;
  end_time?: string;
  result?: any;
  error?: string;
  duration_ms?: number;
}

export interface ToolProgress {
  step: number;
  total: number;
  current: string;
  percentage: number;
}

export interface StreamingResponse {
  type: 'event' | 'complete' | 'error';
  event?: StreamEvent;
  error?: string;
  timestamp: Date;
}

export interface StreamingQueryRequest {
  query: string;
  history?: any[];
  include_intermediate?: boolean;
  include_progress?: boolean;
}

export interface ChatMessage {
  id?: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  streaming?: boolean;
  streamEvents?: StreamEvent[];
  toolsUsed?: string[];
  executionTime?: number;
  error?: boolean;
}

export interface StreamingChatState {
  isStreaming: boolean;
  currentProgress: ToolProgress;
  currentEvents: StreamEvent[];
  error?: string;
}