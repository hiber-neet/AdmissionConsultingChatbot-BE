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

    # --- SỬA LẠI LOGIC TẠI ĐÂY ---

    # 1. Luôn tìm bản ghi theo Session ID hiện tại TRƯỚC (Bất kể đã login hay chưa)
    # Đây là bài test đang hiện trên màn hình của user
    current_session_result = db.query(entities.RiasecResult).filter(
        entities.RiasecResult.session_id == riasec_result.session_id
    ).first()

    target_result = None

    if current_session_result:
        # CASE A: Tìm thấy bài của Session này (Chính là bài Guest vừa làm)
        target_result = current_session_result
        
        # Nếu đang login, thực hiện "Claim" (Chiếm quyền) bài này
        if user_id:
            # Gán user id vào bài này
            target_result.customer_id = user_id
            
            # (Optional) Dọn dẹp: Nếu user này lỡ có bài cũ rích nào đó khác (gây thừa thãi)
            # thì xóa bài cũ đi.
            old_result = db.query(entities.RiasecResult).filter(
                entities.RiasecResult.customer_id == user_id,
                entities.RiasecResult.result_id != target_result.result_id
            ).first()
            if old_result:
                db.delete(old_result) # Xóa bài cũ để tránh duplicate logic sau này

    elif user_id:
        # CASE B: Session mới tinh, nhưng User đã login và có thể có bài cũ
        # Tìm bài cũ của user để update
        user_old_result = db.query(entities.RiasecResult).filter(
            entities.RiasecResult.customer_id == user_id
        ).first()
        
        if user_old_result:
            target_result = user_old_result
            # Cập nhật session id mới cho bài cũ
            target_result.session_id = riasec_result.session_id
    
    # --- THỰC HIỆN UPDATE HOẶC CREATE ---
    
    if target_result:
        # Update: Cập nhật điểm số
        # Exclude session_id để tránh việc tự gán lại chính nó (dù không lỗi nhưng thừa)
        update_data = riasec_result.dict(exclude={'session_id'}) 
        for key, value in update_data.items():
            setattr(target_result, key, value)
            
        # Đảm bảo session_id đúng (cho trường hợp Case B)
        if target_result.session_id != riasec_result.session_id:
             target_result.session_id = riasec_result.session_id
    else:
        # Create: Chưa có gì cả -> Tạo mới
        target_result = entities.RiasecResult(
            **riasec_result.dict(),
            customer_id=user_id
        )
        db.add(target_result)

    try:
        db.commit()
        db.refresh(target_result)
        return target_result
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Lỗi lưu kết quả: {str(e)}")