from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from app.core.config import settings
from app.api.routes import knowledge_base, chat
from app.models.database import init_db
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
async def startup_event():
    init_db()
app.add_event_handler("startup",startup_event)


app.include_router(knowledge_base.router , prefix="/knowledge", tags=["Knowledge Base"])
app.include_router(chat.router , prefix="/chat", tags=["Chat"])
@app.get("/")
async def root():
    return {"message": "FastAPI + LangChain + Qdrant + OpenAI API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


