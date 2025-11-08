from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.models.schemas import (
    ArticleCreate, ArticleUpdate, ArticleResponse, 
    ArticleStatusUpdate
)
from app.models.entities import Article, Users, Major, Specialization
from typing import List, Optional
from app.core.security import (
    get_current_user, verify_content_manager,
    verify_content_manager_leader
)
from datetime import datetime
from sqlalchemy import or_

router = APIRouter()

@router.post("", response_model=ArticleResponse)
async def create_article(
    article: ArticleCreate,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
    Create a new article (Content Manager only).
    Article will be created with 'draft' status.
    """
    if not current_user or not verify_content_manager(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only content managers can create articles"
        )

    # Validate major and specialization if provided
    if article.major_id:
        major = db.query(Major).filter(Major.major_id == article.major_id).first()
        if not major:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Major with id {article.major_id} not found"
            )

    if article.specialization_id:
        spec = db.query(Specialization).filter(
            Specialization.specialization_id == article.specialization_id
        ).first()
        if not spec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Specialization with id {article.specialization_id} not found"
            )

    # Create new article with draft status
    new_article = Article(
        title=article.title,
        description=article.description,
        url=article.url,
        status="draft",
        create_at=datetime.now(),
        created_by=current_user.user_id,
        major_id=article.major_id,
        specialization_id=article.specialization_id
    )

    db.add(new_article)
    db.commit()
    db.refresh(new_article)

    return new_article

@router.put("/{article_id}/status", response_model=ArticleResponse)
async def update_article_status(
    article_id: int,
    status_update: ArticleStatusUpdate,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
    Update article status (Content Manager Leader only)
    """
    if not current_user or not verify_content_manager_leader(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only content manager leaders can update article status"
        )

    article = db.query(Article).filter(Article.article_id == article_id).first()
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article with id {article_id} not found"
        )

    # Update status
    article.status = status_update.status
    db.commit()
    db.refresh(article)

    return article

@router.get("", response_model=List[ArticleResponse])
async def get_articles(
    db: Session = Depends(get_db),
    current_user: Optional[Users] = Depends(get_current_user)
):
    """
    Get articles based on user role:
    - Content managers can see all articles
    - Others can only see published articles
    """
    # Base query with joins for additional information
    query = (
        db.query(Article)
        .outerjoin(Major)
        .outerjoin(Specialization)
        .outerjoin(Users, Users.user_id == Article.created_by)
    )

    # Add filters based on user role
    if not current_user or not verify_content_manager(current_user):
        # Public users can only see published articles
        query = query.filter(Article.status == "published")

    articles = query.all()

    # Format response with additional information
    response = []
    for article in articles:
        article_data = ArticleResponse(
            article_id=article.article_id,
            title=article.title,
            description=article.description,
            url=article.url,
            status=article.status,
            create_at=article.create_at,
            created_by=article.created_by,
            major_id=article.major_id,
            specialization_id=article.specialization_id,
            author_name=article.author_user.full_name if article.author_user else None,
            major_name=article.major.major_name if article.major else None,
            specialization_name=article.specialization.specialization_name if article.specialization else None
        )
        response.append(article_data)

    return response

@router.get("/{article_id}", response_model=ArticleResponse)
async def get_article(
    article_id: int,
    db: Session = Depends(get_db),
    current_user: Optional[Users] = Depends(get_current_user)
):
    """
    Get a specific article:
    - Content managers can see any article
    - Others can only see published articles
    """
    article = (
        db.query(Article)
        .outerjoin(Major)
        .outerjoin(Specialization)
        .outerjoin(Users, Users.user_id == Article.created_by)
        .filter(Article.article_id == article_id)
        .first()
    )

    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article with id {article_id} not found"
        )

    # Check access permission
    if (not current_user or not verify_content_manager(current_user)) and article.status != "published":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This article is not published"
        )

    return ArticleResponse(
        article_id=article.article_id,
        title=article.title,
        description=article.description,
        url=article.url,
        status=article.status,
        create_at=article.create_at,
        created_by=article.created_by,
        major_id=article.major_id,
        specialization_id=article.specialization_id,
        author_name=article.author_user.full_name if article.author_user else None,
        major_name=article.major.major_name if article.major else None,
        specialization_name=article.specialization.specialization_name if article.specialization else None
    )