from autogen import AssistantAgent, UserProxyAgent, register_function
from config import llm_config
from logic import get_order_status, get_inventory_lookup, get_refund_tracking

inventory_lookup_agent = AssistantAgent(
    name="InventoryLookupAgent",
    system_message="You lookup item availability and price from our inventory system. Use the inventory_lookup tool for factual data.",
    llm_config=llm_config,  # Add your tool here
)

# Order status checker
order_status_agent = AssistantAgent(
    name="OrderStatusAgent",
    system_message="You check the current order status given an order ID.",
    llm_config=llm_config,
)

# Processes refund requests
refund_tracking_agent = AssistantAgent(
    name="RefundTrackingAgent",
    system_message="You track a refund request if it's approved by RefundAgent.",
    llm_config=llm_config,
)
# --- NEW: The Router Agent ---
# This AssistantAgent will use its LLM to decide which function (representing the other agents' capabilities) to call.
router_agent = AssistantAgent(
    name="CustomerQueryRouter",
    system_message="""You are a customer query routing assistant.
    Your primary job is to understand the customer's query and decide which specific tool/function to call to resolve it.
    Available tools are:
    - `get_inventory_status(item_id: str)`: Use this if the user is asking about item availability, stock, or price.
    - `get_order_status(order_id: str)`: Use this if the user is asking about an order's current status.
    - `get_refund_status(refund_id: str)`: Use this if the user is asking about a refund's status.

    Extract the necessary ID (item ID, order ID, or refund ID) from the user's query when calling the tool.
    If a tool is called, state clearly what you are doing.
    Once you get the result from the tool, summarize it for the customer.
    If you have completely answered the user's query, reply with TERMINATE.
    If you cannot identify a clear intent or necessary ID for a tool, state that you need more information.
    """,
    llm_config=llm_config,
)

# --- The UserProxyAgent acts as the executor for ALL tools ---
# This UserProxyAgent will execute the functions suggested by the router_agent
executor = UserProxyAgent(
    name="Executor",
    human_input_mode="NEVER", # Set to NEVER for automated execution
    max_consecutive_auto_reply=1, # Allow more replies to handle tool execution feedback
    #is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    #code_execution_config={"last_n_messages": 1, "work_dir": "coding"}, # Optional: for code execution. Not strictly needed for function calls if `register_function` handles it.
)

# --- Register MOCK functions with the Executor, and make them callable by the Router Agent ---
# The Router Agent suggests the call, the Executor runs the actual mock function.

register_function(
    get_inventory_lookup,
    caller=router_agent,
    executor=executor,
    name="get_inventory_status",
    description="Gets the availability and price for a given item ID from inventory. Requires 'item_id: str'."
)

register_function(
    get_order_status,
    caller=router_agent,
    executor=executor,
    name="get_order_status",
    description="Gets the current status of an order given an order ID. Requires 'order_id: str'."
)

register_function(
    get_refund_tracking,
    caller=router_agent,
    executor=executor,
    name="get_refund_status",
    description="Gets the status of a refund request given a refund ID. Requires 'refund_id: str'."
)

