from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    sources: Optional[List[str]] = None

class DocumentUpload(BaseModel):
    content: str
    metadata: Optional[dict] = None

class DocumentResponse(BaseModel):
    id: str
    message: str
