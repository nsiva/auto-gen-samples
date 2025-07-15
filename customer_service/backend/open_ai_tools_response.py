from typing import Optional, List, Any


class Function:
    arguments: str
    name: str

    def __init__(self, arguments: str, name: str) -> None:
        self.arguments = arguments
        self.name = name


class ToolCall:
    id: str
    function: Function
    type: str

    def __init__(self, id: str, function: Function, type: str) -> None:
        self.id = id
        self.function = function
        self.type = type


class ToolResponse:
    tool_call_id: str
    role: str
    content: str

    def __init__(self, tool_call_id: str, role: str, content: str) -> None:
        self.tool_call_id = tool_call_id
        self.role = role
        self.content = content


class ChatHistory:
    content: Optional[str]
    role: str
    name: Optional[str]
    tool_calls: Optional[List[ToolCall]]
    tool_responses: Optional[List[ToolResponse]]

    def __init__(self, content: Optional[str], role: str, name: Optional[str], tool_calls: Optional[List[ToolCall]], tool_responses: Optional[List[ToolResponse]]) -> None:
        self.content = content
        self.role = role
        self.name = name
        self.tool_calls = tool_calls
        self.tool_responses = tool_responses


class GPT35Turbo0125:
    cost: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __init__(self, cost: float, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> None:
        self.cost = cost
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class UsageCludingCachedInference:
    total_cost: float
    gpt_35_turbo_0125: GPT35Turbo0125

    def __init__(self, total_cost: float, gpt_35_turbo_0125: GPT35Turbo0125) -> None:
        self.total_cost = total_cost
        self.gpt_35_turbo_0125 = gpt_35_turbo_0125


class Cost:
    usage_including_cached_inference: UsageCludingCachedInference
    usage_excluding_cached_inference: UsageCludingCachedInference

    def __init__(self, usage_including_cached_inference: UsageCludingCachedInference, usage_excluding_cached_inference: UsageCludingCachedInference) -> None:
        self.usage_including_cached_inference = usage_including_cached_inference
        self.usage_excluding_cached_inference = usage_excluding_cached_inference


class Message:
    chat_id: None
    chat_history: List[ChatHistory]
    summary: str
    cost: Cost
    human_input: List[Any]

    def __init__(self, chat_id: None, chat_history: List[ChatHistory], summary: str, cost: Cost, human_input: List[Any]) -> None:
        self.chat_id = chat_id
        self.chat_history = chat_history
        self.summary = summary
        self.cost = cost
        self.human_input = human_input


class ChatResult2:
    message: Message

    def __init__(self, message: Message) -> None:
        self.message = message
