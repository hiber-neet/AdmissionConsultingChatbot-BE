from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from app.core.config import settings
from app.api.routes import (
    knowledge_base_controller,
    chat_controller,
    auth_controller,
    profile_controller,
    major_controller,
    specialization_controller,
    article_controller,
    users_controller,
    riasec_controller,
    permissions_controller,
    academic_score_controller
)
from app.models.database import init_db
import os
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

app.include_router(auth_controller.router, prefix="/auth", tags=["Authentication"])
app.include_router(users_controller.router, prefix="/users", tags=["Users"])
app.include_router(profile_controller.router, prefix="/profile", tags=["Profile"])
app.include_router(major_controller.router, prefix="/majors", tags=["Majors"])
app.include_router(specialization_controller.router, prefix="/specializations", tags=["Specializations"])
app.include_router(article_controller.router, prefix="/articles", tags=["Articles"])
app.include_router(knowledge_base_controller.router, prefix="/knowledge", tags=["Knowledge Base"])
app.include_router(chat_controller.router, prefix="/chat", tags=["Chat"])
app.include_router(riasec_controller.router, prefix="/riasec", tags=["RIASEC"])
app.include_router(permissions_controller.router, prefix="/permissions", tags=["Permissions"])
app.include_router(academic_score_controller.router, prefix="/academic-score", tags=["Academic Score"])

@app.get("/")
async def root():
    return {"message": "FastAPI + LangChain + Qdrant + OpenAI API"}



