from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import (
    create_access_token,
    get_password_hash,
    verify_password,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    get_current_user,
    verify_user_access
)
from app.models.database import get_db
from app.models.schemas import Token, UserCreate, UserResponse, LoginRequest
from app.models.entities import Users

router = APIRouter()


@router.post("/register", response_model=UserResponse)
def register(*, db: Session = Depends(get_db), user_in: UserCreate) -> Any:
    """
    Register a new user.
    """
    user = db.query(Users).filter(Users.email == user_in.email).first()
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )
    
    user = Users(
        email=user_in.email,
        full_name=user_in.full_name,
        password=get_password_hash(user_in.password),
        role_id=user_in.role_id,
        status= True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(
    db: Session = Depends(get_db),
    form_data: LoginRequest = None
) -> Any:
    """
    Login to get an access token for future requests.
    """
    if not form_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Login credentials required",
        )
    
    email = form_data.email
    password = form_data.password

    user = db.query(Users).filter(Users.email == email).first()
    if not user or not verify_password(password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    if not user.status:
        raise HTTPException(
            status_code=400, detail="Inactive user"
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return {
        "access_token": create_access_token(
            {"sub": user.email, "user_id": user.user_id}
        ),
            "token_type": "bearer",
        }