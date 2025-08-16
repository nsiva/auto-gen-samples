from typing import List
import openai
from fastapi import HTTPException, status
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the OpenAI client
openai_client = openai

# Define the embedding model name
embedding_model = "text-embedding-ada-002"  # or your preferred model

# --- Helper Functions ---
async def generate_embedding(text: str) -> List[float]:
    """Generates an embedding for the given text using OpenAI."""
    try:
        response = openai_client.embeddings.create(
            input=[text],
            model=embedding_model
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to generate embedding for text: {text[:50]}... Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not generate embedding for the document."
        )

import tiktoken

def count_tokens(text: str) -> int:
    """Counts tokens in a given string using OpenAI's tiktoken."""
    encoding = tiktoken.encoding_for_model(embedding_model)
    return len(encoding.encode(text))
