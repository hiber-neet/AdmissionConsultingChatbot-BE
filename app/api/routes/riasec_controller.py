from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models import schemas, database, entities
from app.core.security import get_current_user
from typing import Optional

router = APIRouter()

@router.post("/submit", response_model=schemas.RiasecResult)
def submit_riasec(
    riasec_result: schemas.RiasecResultCreate,
    db: Session = Depends(database.get_db),
    current_user: Optional[entities.Users] = Depends(get_current_user)
):
    user_id = current_user.user_id if current_user else None

    # Find an existing result, prioritizing user_id if available, otherwise by session_id
    if user_id:
        # User is logged in, try to find their result
        db_riasec_result = db.query(entities.RiasecResult).filter(entities.RiasecResult.customer_id == user_id).first()
        
        # If no result is found for the user, check if there's a guest result with the same session_id
        if not db_riasec_result:
            guest_result = db.query(entities.RiasecResult).filter(
                entities.RiasecResult.session_id == riasec_result.session_id,
                entities.RiasecResult.customer_id == None
            ).first()
            # If a guest result exists, associate it with the logged-in user
            if guest_result:
                db_riasec_result = guest_result
                db_riasec_result.customer_id = user_id

    else:
        # User is a guest, find result by session_id
        db_riasec_result = db.query(entities.RiasecResult).filter(
            entities.RiasecResult.session_id == riasec_result.session_id,
            entities.RiasecResult.customer_id == None
        ).first()

    if db_riasec_result:
        # Update existing record
        for key, value in riasec_result.dict().items():
            setattr(db_riasec_result, key, value)
    else:
        # Create a new record
        db_riasec_result = entities.RiasecResult(
            **riasec_result.dict(),
            customer_id=user_id
        )
        db.add(db_riasec_result)
    
    db.commit()
    db.refresh(db_riasec_result)
    return db_riasec_result