#from autogen import ChatResult
#from agents import inventory_lookup, lookup_agent, status_agent, refund_agent, refund_processor, user_proxy



# def lookup_item_old(item_name: str) -> str:
#     msg = f"Can you check availability and price for: {item_name}?"
#     response = user_proxy.initiate_chat(lookup_agent, message=msg)
    
#     return response # if response.message else f"{item_name} is not found in our inventory."



# def lookup_item(item_name: str) -> str:
#     msg = f"asldfkjalskdfjlk for: {item_name}?"
#     #print(f"Looking up item: {item_name} without using the agent directly.")
#     #return inventory_lookup(item_name)  # Call the registered function to perform the lookup
#     # Initiate a chat with the lookup agent and get the results

#     print(f"Looking up item: {item_name} using the agent directly.")
#     import json
#     chat_results = user_proxy.initiate_chat(lookup_agent, message=msg, summary_method="last_msg")
    

#         # Accessing the chat history
#     print("Chat History:")
#     if chat_results:
#         for message in chat_results.chat_history:
#             print(message)
#     else:
#         print("No chat history available.")

#     # Accessing the summary
#     print("\nChat Summary:")
#     print(chat_results.summary)

#     if chat_results.summary:
#         return chat_results.summary
#     return None

#     #     "burger": "Burger is available at $8.99",
#     #     "soda": "Soda is available at $1.99"
#     # }
#     # lower_item_name = item_name.lower()
#     # for key, value in inventory.items():
#     #     if key in lower_item_name:
#     #         return value
#     # return f"{item_name} is not found in our inventory."

# def check_order_status(order_id: str) -> str:
#     msg = f"What is the status of order ID: {order_id}?"
#     user_proxy.initiate_chat(status_agent, message=msg)
#     return f"Status checked for order {order_id}"

# def process_refund(order_id: str, reason: str) -> str:
#     user_proxy.initiate_chat(
#         [refund_agent, refund_processor],
#         message=f"I want a refund for order {order_id} because: {reason}"
#     )
#     return f"Refund request initiated for order {order_id}"



# Mock database functions
def get_order_status(order_id: str):
    if(order_id == "123"):
        return f"Order {order_id} is currently being processed."
    elif(order_id == "456"):
        return f"Order {order_id} has been delivered."
    else:   
        return f"Order {order_id} not found in our system."

def get_inventory_lookup(item_id: str):
    if (item_id == "123"):
        return f"Item {item_id} is in stock with 15 units available."
    elif (item_id == "456"):
        return f"Item {item_id} is out of stock."
    else:
        return f"Item {item_id} not found in our inventory."

def get_refund_tracking(refund_id: str):
    if refund_id == "123":
        return f"Refund {refund_id} has been approved and is being processed."
    elif refund_id == "456":
        return f"Refund {refund_id} has been denied due to policy restrictions."
    else:
        return f"Refund {refund_id} not found in our system."
