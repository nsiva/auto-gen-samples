from autogen import AssistantAgent, UserProxyAgent, register_function
from config import llm_config

def inventory_lookup(item_name: str) -> str:
    inventory = {
        "pizza": "Pizza is available at $10.99",
        "burger": "Burger is available at $8.99",
        "soda": "Soda is available at $1.99"
    }
    for key, value in inventory.items():
        if key in item_name.lower():
            return value
    return f"{item_name} is not found in our inventory."

lookup_agent = AssistantAgent(
    name="LookupAgent",
    system_message="You lookup item availability and price from our inventory system. Use the inventory_lookup tool for factual data.",
    llm_config=llm_config,  # Add your tool here
)

# Order status checker
status_agent = AssistantAgent(
    name="StatusAgent",
    system_message="You check the current order status given an order ID.",
    llm_config=llm_config,
)

# Refund agent initiates refund
refund_agent = AssistantAgent(
    name="RefundAgent",
    system_message="You determine if a user is eligible for a refund and pass to processing.",
    llm_config=llm_config,
)

# Processes refund
refund_processor = AssistantAgent(
    name="RefundProcessor",
    system_message="You process a refund request if it's approved by RefundAgent.",
    llm_config=llm_config,
)

# Simulated user
user_proxy = UserProxyAgent(
    name="UserProxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,
)


# Register the function as a tool with required arguments
register_function(
    inventory_lookup,
    caller=lookup_agent,
    executor=user_proxy,
    name="inventory_lookup",
    description="Returns item availability and price from inventory."
)
