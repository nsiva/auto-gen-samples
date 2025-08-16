# --- Pydantic Models ---
from pydantic import BaseModel
from typing import List

class DocumentInput(BaseModel):
    content: str
    metadata: dict = {}

class DocumentResponse(BaseModel):
    id: str
    content: str
    metadata: dict
    created_at: str # Adjust to datetime if you prefer Pydantic's datetime type

class SearchResponse(BaseModel):
    query: str
    results: List[DocumentResponse]
    num_tokens: int