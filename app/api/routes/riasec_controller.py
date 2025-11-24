from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models import schemas, database, entities
from app.core.security import get_current_user

router = APIRouter()

@router.post("/submit", response_model=schemas.RiasecResult)
def submit_riasec(
    riasec_result: schemas.RiasecResultCreate,
    db: Session = Depends(database.get_db),
    current_user: entities.Users = Depends(get_current_user)
):
    db_riasec_result = entities.RiasecResult(**riasec_result.dict(), customer_id=current_user.user_id)
    db.add(db_riasec_result)
    db.commit()
    db.refresh(db_riasec_result)
    return db_riasec_result