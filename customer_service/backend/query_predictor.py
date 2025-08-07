# Add to app.py or create a new file: query_predictor.py

import openai
from typing import List, Dict
import re
import logging

logger = logging.getLogger(__name__)

class QueryToolPredictor:
    def __init__(self):
        self.tool_descriptions = {
            "get_inventory_status": {
                "description": "Gets availability and price for items",
                "keywords": ["item", "inventory", "stock", "availability", "price", "product", "in stock", "available"],
                "patterns": [r"item\s+\d+", r"product\s+\d+", r"check\s+stock"]
            },
            "get_order_status": {
                "description": "Gets current status of orders", 
                "keywords": ["order", "status", "delivery", "shipped", "processing", "track", "tracking"],
                "patterns": [r"order\s+\d+", r"order\s+id", r"my\s+order"]
            },
            "get_refund_status": {
                "description": "Gets status of refund requests",
                "keywords": ["refund", "return", "money back", "refund status", "refund request"],
                "patterns": [r"refund\s+\d+", r"refund\s+id", r"return\s+\d+"]
            }
        }
    
    def predict_tools_keyword_based(self, query: str) -> List[str]:
        """Predict tools based on keyword matching"""
        query_lower = query.lower()
        predicted_tools = []
        
        for tool_name, info in self.tool_descriptions.items():
            # Check keywords
            if any(keyword in query_lower for keyword in info["keywords"]):
                predicted_tools.append(tool_name)
                continue
                
            # Check regex patterns
            if any(re.search(pattern, query_lower) for pattern in info["patterns"]):
                predicted_tools.append(tool_name)
        
        return predicted_tools
    
    def predict_tools_llm_based(self, query: str) -> List[str]:
        """Use LLM to predict which tools will be needed"""
        
        prediction_prompt = f"""
        Analyze this customer service query and predict which tools will be needed:
        
        Query: "{query}"
        
        Available tools:
        1. get_inventory_status - for checking item availability, stock, prices
        2. get_order_status - for checking order status, delivery, tracking
        3. get_refund_status - for checking refund status, returns
        
        Respond with ONLY a JSON array of tool names that will likely be called.
        Examples:
        - "What's the status of order 123?" -> ["get_order_status"]
        - "Is item 456 in stock and what's the price?" -> ["get_inventory_status"]
        - "Check my refund 789 and order 123 status" -> ["get_refund_status", "get_order_status"]
        
        Response:"""
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prediction_prompt}],
                temperature=0.1,
                max_tokens=100
            )
            
            import json
            predicted_tools = json.loads(response.choices[0].message.content.strip())
            return predicted_tools if isinstance(predicted_tools, list) else []
            
        except Exception as e:
            logger.error(f"LLM prediction failed: {e}")
            return []
    
    def predict_tools_hybrid(self, query: str) -> Dict[str, List[str]]:
        """Combine both keyword and LLM predictions"""
        keyword_prediction = self.predict_tools_keyword_based(query)
        llm_prediction = self.predict_tools_llm_based(query)
        
        # Combine and deduplicate
        all_predictions = list(set(keyword_prediction + llm_prediction))
        
        return {
            "predicted_tools": all_predictions,
            "keyword_based": keyword_prediction,
            "llm_based": llm_prediction,
            "confidence": "high" if keyword_prediction and llm_prediction else "medium"
        }

# Initialize predictor
predictor = QueryToolPredictor()