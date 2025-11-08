from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Request, Depends
from fastapi.security import OAuth2PasswordBearer
import os
from dotenv import load_dotenv
from app.models.schemas import TokenData
from app.models.database import get_db
from app.models.entities import Users
from sqlalchemy.orm import Session

# Load environment variables
load_dotenv()

# JWT Configuration from .env
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

# Password hashing
pwd_context = CryptContext(schemes=["pbkdf2_sha256", "bcrypt_sha256", "bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> TokenData:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        if email is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token_data = TokenData(email=email, user_id=user_id)
        return token_data
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_user_access(requesting_user_id: int, target_user_id: int):
    """
    Verify if the requesting user has access to view the target user's profile
    """
    if requesting_user_id != target_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to access this profile"
        )

def verify_content_manager(user: Users) -> bool:
    """
    Verify if user is a content manager
    """
    return user.role.role_id == 3 if user and user.role else False

def verify_content_manager_leader(user: Users) -> bool:
    """
    Verify if user is a content manager leader
    """
    return (user.role.role_id == 3 and
            user.content_manager_profile and 
            user.content_manager_profile.is_leader) if user and user.role else False

async def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[Users]:
    """
    Get current user from token in Authorization header.
    Returns None if no token is provided or token is invalid.
    """    
    auth_header = request.headers.get("Authorization")
    if not auth_header or "Bearer" not in auth_header:
        return None
        
    try:
        token = auth_header.split(" ")[1]
        token_data = verify_token(token)
        if token_data.email is None:
            return None
            
        user = db.query(Users).filter(Users.email == token_data.email).first()
        if user is None or not user.status:
            return None
            
        return user
    except (JWTError, Exception):
        return None
