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
    verify_content_manager_leader, is_admin, has_permission
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
        link_image=article.link_image,
        note=article.note,
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

@router.put("/{article_id}", response_model=ArticleResponse)
async def update_article(
    article_id: int,
    article_update: ArticleUpdate,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
    Update article content.
    - Admin/Content Manager Leader: can edit any article
    - Content Manager: can only edit their own articles
    """
    if not current_user or not verify_content_manager(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only content managers can update articles"
        )

    # Get the article
    article = db.query(Article).filter(Article.article_id == article_id).first()
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article with id {article_id} not found"
        )

    # Check permissions: Admin/Leader can edit any, regular CM can only edit their own
    is_admin_user = is_admin(current_user)
    is_leader = verify_content_manager_leader(current_user)
    
    if not (is_admin_user or is_leader) and article.created_by != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only edit your own articles"
        )

    # Validate major and specialization if provided
    if article_update.major_id is not None:
        major = db.query(Major).filter(Major.major_id == article_update.major_id).first()
        if not major:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Major with id {article_update.major_id} not found"
            )

    if article_update.specialization_id is not None:
        spec = db.query(Specialization).filter(
            Specialization.specialization_id == article_update.specialization_id
        ).first()
        if not spec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Specialization with id {article_update.specialization_id} not found"
            )

    # Update fields if provided
    if article_update.title is not None:
        article.title = article_update.title
    if article_update.description is not None:
        article.description = article_update.description
    if article_update.url is not None:
        article.url = article_update.url
    if article_update.link_image is not None:
        article.link_image = article_update.link_image
    if article_update.note is not None:
        article.note = article_update.note
    if article_update.major_id is not None:
        article.major_id = article_update.major_id
    if article_update.specialization_id is not None:
        article.specialization_id = article_update.specialization_id

    db.commit()
    db.refresh(article)

    return article

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
    if not current_user or not (verify_content_manager_leader(current_user) or is_admin(current_user)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only content manager leaders or admins can update article status"
        )

    article = db.query(Article).filter(Article.article_id == article_id).first()
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article with id {article_id} not found"
        )

    # Update status
    article.status = status_update.status
    article.note = status_update.note
    db.commit()
    db.refresh(article)

    return article

@router.get("", response_model=List[ArticleResponse])
async def get_articles(
    db: Session = Depends(get_db),
    current_user: Optional[Users] = Depends(get_current_user)
):
    """
    Get articles based on user permissions:
    - Admin: can see all articles (any status except deleted)
    - Content manager: can see their own articles (any status except deleted) + all published articles
    - Other users: can only see published articles
    """
    # Base query with joins for additional information
    query = (
        db.query(Article)
        .filter(Article.status != "deleted")  # Exclude deleted articles for all users
        .outerjoin(Major)
        .outerjoin(Specialization)
        .outerjoin(Users, Users.user_id == Article.created_by)
    )

    # Apply filters based on permissions
    if not current_user:
        # Not authenticated: only published articles
        query = query.filter(Article.status == "published")
    elif is_admin(current_user):
        # Admin: can see all articles (already filtered deleted above)
        pass
    elif (
        has_permission(current_user, "content_manager")
        or has_permission(current_user, "content manager")
        or (
            current_user
            and current_user.permissions
            and any(
                (p.permission_name and "content" in p.permission_name.lower())
                for p in current_user.permissions
            )
        )
    ):
        # Content manager: can see all articles (already filtered deleted above)
        pass
    else:
        # Other users: only published articles
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
            specialization_name=article.specialization.specialization_name if article.specialization else None,
            note=article.note if article.note else None
        )
        response.append(article_data)

    return response

@router.get("/review", response_model=List[ArticleResponse])
async def get_draft_articles_for_review(
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
    Get all articles with 'draft' status for review.
    Accessible only by Admins or Content Manager Leaders.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    # Check for admin or content manager leader permissions
    is_admin_user = is_admin(current_user)
    is_leader = verify_content_manager_leader(current_user)

    if not (is_admin_user or is_leader):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only Admins or Content Manager Leaders can review articles"
        )

    # Query for draft articles
    draft_articles = (
        db.query(Article)
        .filter(Article.status == "draft")
        .outerjoin(Users, Users.user_id == Article.created_by)
        .outerjoin(Major)
        .outerjoin(Specialization)
        .all()
    )

    # Format response
    response = []
    for article in draft_articles:
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
            specialization_name=article.specialization.specialization_name if article.specialization else None,
            note=article.note if article.note else None
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
    Get a specific article based on user permissions:
    - Admin: can see any article (any status)
    - Content manager: can see their own articles (any status) + published articles
    - Content manager author: can see their own articles (any status)
    - Other users: can only see published articles
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
    can_view = False
    
    if not current_user:
        # Not authenticated: only published articles
        can_view = article.status == "published"
    elif is_admin(current_user):
        # Admin: can view any article
        can_view = True
    elif (
        has_permission(current_user, "content_manager")
        or has_permission(current_user, "content manager")
        or (
            current_user
            and current_user.permissions
            and any(
                (p.permission_name and "content" in p.permission_name.lower())
                for p in current_user.permissions
            )
        )
    ):
        # Content manager: can view any article (any status)
        can_view = True
    else:
        # Other users: only published articles
        can_view = article.status == "published"
    
    if not can_view:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to view this article"
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
        specialization_name=article.specialization.specialization_name if article.specialization else None,
        note=article.note if article.note else None
    )

@router.delete("/{article_id}", response_model=ArticleResponse)
async def delete_article(
    article_id: int,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
    Soft delete an article by changing its status to 'deleted'.
    - Admin/Content Manager Leader: can delete any article
    - Content Manager: can only delete their own articles
    """
    if not current_user or not verify_content_manager(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only content managers can delete articles"
        )

    # Get the article
    article = db.query(Article).filter(Article.article_id == article_id).first()
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article with id {article_id} not found"
        )

    # Check if already deleted
    if article.status == "deleted":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Article is already deleted"
        )

    # Check permissions: Admin/Leader can delete any, regular CM can only delete their own
    is_admin_user = is_admin(current_user)
    is_leader = verify_content_manager_leader(current_user)
    
    if not (is_admin_user or is_leader) and article.created_by != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own articles"
        )

    # Soft delete: change status to 'deleted'
    article.status = "deleted"
    db.commit()
    db.refresh(article)

    return article

@router.get("/users/{user_id}", response_model=List[ArticleResponse])
async def get_articles_by_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """
    Get all articles created by a specific user.
    - Admins and Content Manager Leaders can view any user's articles
    - Content Managers can view their own articles
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    # Check for admin or content manager leader permissions
    is_admin_user = is_admin(current_user)
    is_leader = verify_content_manager_leader(current_user)
    is_content_manager = has_permission(current_user, "Content Manager")
    
    # Allow if:
    # 1. User is Admin or Content Manager Leader (can view any user's articles)
    # 2. User is Content Manager viewing their own articles
    is_viewing_own_articles = (current_user.user_id == user_id)
    
    if not (is_admin_user or is_leader or (is_content_manager and is_viewing_own_articles)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to view articles by this user"
        )
    
    # Check if the target user exists
    target_user = db.query(Users).filter(Users.user_id == user_id).first()
    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id {user_id} not found"
        )

    # Query for articles by the specified user (exclude deleted)
    user_articles = (
        db.query(Article)
        .filter(Article.created_by == user_id)
        .filter(Article.status != "deleted")
        .outerjoin(Users, Users.user_id == Article.created_by)
        .outerjoin(Major)
        .outerjoin(Specialization)
        .all()
    )

    # Format response
    response = []
    for article in user_articles:
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
            specialization_name=article.specialization.specialization_name if article.specialization else None,
            note=article.note if article.note else None
        )
        response.append(article_data)

    return response
