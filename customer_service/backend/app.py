from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
# from logic import lookup_item, check_order_status, process_refund
# IMPORT THE NEW AGENTS
from agents import inventory_lookup_agent, order_status_agent, refund_tracking_agent, router_agent, executor

from dotenv import load_dotenv


load_dotenv()

app = FastAPI()

# CORS for frontend on localhost:4200
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LookupRequest(BaseModel):
    item_name: str

class StatusRequest(BaseModel):
    order_id: str

class RefundRequest(BaseModel):
    order_id: str
    reason: str

# NOTE: For these direct calls, if these agents are designed to use tools,
# you'll still need an executor for them.
# The original setup might work if they are purely conversational, but
# if they're meant to call mock_inventory_lookup, mock_order_status etc.,
# they will need an executor to run those functions.
# For simplicity, I'm assuming these directly respond conversationally now.
# If they need to call tools, you'd do:
# result = executor.initiate_chat(inventory_lookup_agent, message=req.item_name)
@app.post("/lookup")
def lookup(req: LookupRequest):
    # This will initiate a chat directly with the InventoryLookupAgent.
    # If InventoryLookupAgent is expected to call a tool, it needs to be registered
    # with an executor (like 'executor' defined above) for that tool.
    # The 'register_function' lines in agents.py are currently only for `router_agent`.
    # If you want these direct calls to use tools, you need `inventory_lookup_agent.register_for_llm(...)`
    # and `executor.register_for_execution(...)` for each specific tool for EACH agent.
    # For now, let's assume these are just conversational agents for these endpoints.
    # If they are meant to call tools, the chat initiation needs to be from the executor.
    response_chat = executor.initiate_chat(inventory_lookup_agent, message=f"Please look up item: {req.item_name}")
    # The 'message' field of the last message in the chat is usually the most relevant.
    return {"message": response_chat.last_message["content"]}


@app.post("/status")
def status(req: StatusRequest):
    response_chat = executor.initiate_chat(order_status_agent, message=f"What is the status of order ID: {req.order_id}")
    return {"message": response_chat.last_message["content"]}

@app.post("/refund")
def refund(req: RefundRequest):
    # Depending on how refund_tracking_agent is designed to use the 'reason',
    # you might want to include it in the message to the agent.
    response_chat = executor.initiate_chat(refund_tracking_agent, message=f"Track refund for order ID: {req.order_id}. Reason: {req.reason}")
    return {"message": response_chat.last_message["content"]}

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_customer_query(request: QueryRequest):
    # Initiate chat with the Router Agent, using the executor as the human proxy
    # The router_agent will then decide which tool (function) to call.
    response_chat = executor.initiate_chat(
        recipient=router_agent,
        message=request.query
    )
    # The final answer is typically in the content of the last message from the assistant
    # after the entire conversation (including tool calls and results) is done.
    return {"response": response_chat.summary}