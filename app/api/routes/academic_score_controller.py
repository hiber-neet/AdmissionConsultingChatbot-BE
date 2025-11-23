from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models import schemas, database, entities
from app.core.security import get_current_user

router = APIRouter()

@router.post("/upload", response_model=schemas.AcademicScoreResponse)
def upload_academic_score(
    academic_score: schemas.AcademicScoreCreate,
    db: Session = Depends(database.get_db),
    current_user: entities.Users = Depends(get_current_user)
):
    if not current_user.customer_profile:
        raise HTTPException(status_code=403, detail="User is not a customer")
        
    db_academic_score = entities.AcademicScore(**academic_score.dict(), customer_id=current_user.user_id)
    db.add(db_academic_score)
    db.commit()
    db.refresh(db_academic_score)
    return db_academic_score