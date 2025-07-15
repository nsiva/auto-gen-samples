from autogen import ChatResult
from agents import lookup_agent, status_agent, refund_agent, refund_processor, user_proxy

def lookup_item_old(item_name: str) -> str:
    msg = f"Can you check availability and price for: {item_name}?"
    response = user_proxy.initiate_chat(lookup_agent, message=msg)
    
    return response # if response.message else f"{item_name} is not found in our inventory."


def lookup_item(item_name: str) -> str:
    msg = f"Can you check availability and price for: {item_name}?"
    import json
    chat_results = user_proxy.initiate_chat(lookup_agent, message=msg, summary_method="last_msg")
    

        # Accessing the chat history
    print("Chat History:")
    if chat_results:
        for message in chat_results.chat_history:
            print(message)
    else:
        print("No chat history available.")

    # Accessing the summary
    print("\nChat Summary:")
    print(chat_results.summary)

    if chat_results.summary:
        return chat_results.summary
    return None

    #     "burger": "Burger is available at $8.99",
    #     "soda": "Soda is available at $1.99"
    # }
    # lower_item_name = item_name.lower()
    # for key, value in inventory.items():
    #     if key in lower_item_name:
    #         return value
    # return f"{item_name} is not found in our inventory."

def check_order_status(order_id: str) -> str:
    msg = f"What is the status of order ID: {order_id}?"
    user_proxy.initiate_chat(status_agent, message=msg)
    return f"Status checked for order {order_id}"

def process_refund(order_id: str, reason: str) -> str:
    user_proxy.initiate_chat(
        [refund_agent, refund_processor],
        message=f"I want a refund for order {order_id} because: {reason}"
    )
    return f"Refund request initiated for order {order_id}"

