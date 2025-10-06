from fastapi import FastAPI
from app.api.routes import chat, documents
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)

app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])

@app.get("/")
async def root():
    return {"message": "FastAPI + LangChain + Qdrant + OpenAI API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
