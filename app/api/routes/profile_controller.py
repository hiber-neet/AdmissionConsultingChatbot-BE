from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.security import get_current_user, verify_user_access
from app.models.database import get_db
from app.models.schemas import UserProfileResponse
from app.models.entities import Users, Role

router = APIRouter()

@router.get("/{user_id}", response_model=UserProfileResponse)
async def get_user_profile(
    user_id: int,
    current_user: Users = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user profile information. Users can only access their own profile.
    """
    # Check if user has permission to access this profile
    verify_user_access(current_user.user_id, user_id)

    # Get user with role information
    user = db.query(Users).join(Role).filter(Users.user_id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Build profile response based on user's role
    profile_data = {
        "user_id": user.user_id,
        "full_name": user.full_name,
        "email": user.email,
        "role_name": user.role.role_name if user.role else None,
        "student_profile": None,
        "consultant_profile": None,
        "content_manager_profile": None,
        "admission_official_profile": None
    }

    # Add role-specific profile data if exists
    if user.customer_profile:
        profile_data["student_profile"] = {
            "interest": {
                "desired_major": user.customer_profile.interest.desired_major if user.customer_profile.interest else None,
                "region": user.customer_profile.interest.region if user.customer_profile.interest else None
            } if user.customer_profile.interest else None
        }
    
    if user.consultant_profile:
        profile_data["consultant_profile"] = {
            "status": user.consultant_profile.status,
            "is_leader": user.consultant_profile.is_leader
        }

    if user.content_manager_profile:
        profile_data["content_manager_profile"] = {
            "is_leader": user.content_manager_profile.is_leader
        }

    if user.admission_official_profile:
        profile_data["admission_official_profile"] = {
            "rating": user.admission_official_profile.rating,
            "current_sessions": user.admission_official_profile.current_sessions,
            "max_sessions": user.admission_official_profile.max_sessions,
            "status": user.admission_official_profile.status
        }

    return profile_data