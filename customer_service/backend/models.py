# --- Pydantic Models ---
from pydantic import BaseModel
from typing import List, Optional

class DocumentInput(BaseModel):
    content: str
    metadata: dict = {}

class DocumentResponse(BaseModel):
    id: str
    content: str
    metadata: dict
    created_at: str # Adjust to datetime if you prefer Pydantic's datetime type

class SearchResult(BaseModel):
    id: str
    content: str
    metadata: dict
    created_at: str
    similarity: float # Include similarity score

class RAGSearchResponse(BaseModel):
    query: str
    answer: str
    retrieved_documents: List[SearchResult]
    num_query_tokens: int
    num_answer_tokens: Optional[int] = None