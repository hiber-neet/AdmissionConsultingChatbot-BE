from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.langchain_service import langchain_service
import uuid

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())

        answer, sources = langchain_service.query(request.message)

        return ChatResponse(
            response=answer,
            conversation_id=conversation_id,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/simple", response_model=ChatResponse)
async def simple_chat(request: ChatRequest):
    try:
        conversation_id = request.conversation_id or str(uuid.uuid4())

        answer = langchain_service.chat(request.message)

        return ChatResponse(
            response=answer,
            conversation_id=conversation_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
