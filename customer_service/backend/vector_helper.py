from typing import List
import openai
from fastapi import HTTPException, status
import logging
import tiktoken

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the OpenAI client
openai_client = openai

# Define the embedding model name
embedding_model = "text-embedding-ada-002"  # or your preferred model
chat_model = "gpt-3.5-turbo" # Or "gpt-4" for higher quality
encoding = tiktoken.encoding_for_model(embedding_model)

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
            detail="Could not generate embedding for the document/query."
        )

def count_tokens(text: str) -> int:
    """Counts tokens in a given string using OpenAI's tiktoken."""
    return len(encoding.encode(text))

async def generate_rag_answer(query: str, retrieved_docs_content: List[str]) -> str:
    """Generates an answer using an LLM based on query and retrieved document content."""
    context = "\n\n".join(retrieved_docs_content)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the user's question concisely based only on the provided context. If the answer is not in the context, state that you cannot provide an answer based on the given information."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    try:
        response = openai_client.chat.completions.create(
            model=chat_model,
            messages=messages,
            max_tokens=500, # Limit the answer length
            temperature=0.7,
        )
        answer = response.choices[0].message.content.strip()
        logger.info(f"Generated RAG answer for query: '{query}'")
        return answer
    except Exception as e:
        logger.error(f"Failed to generate RAG answer for query: {query}. Error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not generate an answer from the retrieved documents."
        )