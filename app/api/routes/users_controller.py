from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.models.database import get_db
from app.models.schemas import (
    PermissionChangeRequest,
    PermissionRevokeRequest,
    BanUserRequest,
)
from app.models.entities import Users, UserPermission
from app.core.security import has_permission, get_current_user

router = APIRouter()


@router.post("/permissions/grant")
def grant_permission(
    payload: PermissionChangeRequest,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """Grant a permission to a user (admin only)."""
    if not current_user or not has_permission(current_user, "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required")

    # find target user
    target = db.query(Users).filter(Users.user_id == payload.user_id).first()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found")

    # validate permission exists
    from app.models.entities import Permission as PermissionModel
    from app.models.entities import ConsultantProfile as ConsultantProfileModel
    from app.models.entities import ContentManagerProfile as ContentManagerProfileModel
    from app.models.entities import AdmissionOfficialProfile as AdmissionOfficialProfileModel

    perm = db.query(PermissionModel).filter(PermissionModel.permission_id == payload.permission_id).first()
    if not perm:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Permission not found")

    # check existing
    existing = db.query(UserPermission).filter(
        UserPermission.user_id == target.user_id,
        UserPermission.permission_id == payload.permission_id
    ).first()
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User already has this permission")

    # add permission
    up = UserPermission(user_id=target.user_id, permission_id=payload.permission_id)
    db.add(up)

    # create related profile if necessary
    pname = (perm.permission_name or "").lower()
    if "consultant" in pname:
        cp = db.query(ConsultantProfileModel).filter(ConsultantProfileModel.consultant_id == target.user_id).first()
        if not cp:
            cp = ConsultantProfileModel(consultant_id=target.user_id, status=True, is_leader=bool(payload.consultant_is_leader))
            db.add(cp)

    if "content" in pname:
        cmp = db.query(ContentManagerProfileModel).filter(ContentManagerProfileModel.content_manager_id == target.user_id).first()
        if not cmp:
            cmp = ContentManagerProfileModel(content_manager_id=target.user_id, is_leader=bool(payload.content_manager_is_leader))
            db.add(cmp)

    if "admission" in pname or "official" in pname:
        ap = db.query(AdmissionOfficialProfileModel).filter(AdmissionOfficialProfileModel.admission_official_id == target.user_id).first()
        if not ap:
            ap = AdmissionOfficialProfileModel(admission_official_id=target.user_id, rating=0, current_sessions=0, max_sessions=10, status="available")
            db.add(ap)

    # Set user status to true when granting permission
    target.status = True

    db.commit()
    return {"message": "Permission granted"}


@router.delete("/permissions/revoke")
def revoke_permission(
    payload: PermissionRevokeRequest,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """Revoke a permission from a user (admin only). Admins cannot revoke permissions from other admins."""
    if not current_user or not has_permission(current_user, "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required")

    # find target user
    target = db.query(Users).filter(Users.user_id == payload.user_id).first()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found")

    # Determine if target has admin permission
    from app.models.entities import Permission as PermissionModel
    target_has_admin = any(p.permission_name and "admin" in p.permission_name.lower() for p in target.permissions or [])
    if target_has_admin and target.user_id != current_user.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot modify permissions of another admin")

    # find the UserPermission row
    up = db.query(UserPermission).filter(
        UserPermission.user_id == target.user_id,
        UserPermission.permission_id == payload.permission_id
    ).first()
    if not up:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Permission not assigned to user")

    # remove associated profile if applicable
    perm = db.query(PermissionModel).filter(PermissionModel.permission_id == payload.permission_id).first()
    if perm:
        pname = (perm.permission_name or "").lower()
        # import profile models
        from app.models.entities import ConsultantProfile as ConsultantProfileModel
        from app.models.entities import ContentManagerProfile as ContentManagerProfileModel
        from app.models.entities import AdmissionOfficialProfile as AdmissionOfficialProfileModel

        if "consultant" in pname:
            cp = db.query(ConsultantProfileModel).filter(ConsultantProfileModel.consultant_id == target.user_id).first()
            if cp:
                db.delete(cp)
        if "content" in pname:
            cmp = db.query(ContentManagerProfileModel).filter(ContentManagerProfileModel.content_manager_id == target.user_id).first()
            if cmp:
                db.delete(cmp)
        if "admission" in pname or "official" in pname:
            ap = db.query(AdmissionOfficialProfileModel).filter(AdmissionOfficialProfileModel.admission_official_id == target.user_id).first()
            if ap:
                db.delete(ap)

    # delete the permission link
    db.delete(up)
    db.flush()

    # If user has no remaining permissions, set status to False (ban)
    remaining = db.query(UserPermission).filter(UserPermission.user_id == target.user_id).all()
    if not remaining:
        target.status = False

    db.commit()
    return {"message": "Permission revoked"}


@router.post("/ban")
def ban_user(
    payload: BanUserRequest,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """Ban a user (admin only). Cannot ban another admin."""
    if not current_user or not has_permission(current_user, "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required")

    target = db.query(Users).filter(Users.user_id == payload.user_id).first()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found")

    # Prevent banning another admin
    target_is_admin = any(p.permission_name and "admin" in p.permission_name.lower() for p in (target.permissions or []))
    if target_is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot ban another admin")

    target.status = False
    db.commit()
    return {"message": "User has been banned"}


@router.post("/unban")
def unban_user(
    payload: BanUserRequest,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """Unban a user (admin only). Cannot unban another admin."""
    if not current_user or not has_permission(current_user, "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required")

    target = db.query(Users).filter(Users.user_id == payload.user_id).first()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found")

    # Prevent unbanning another admin
    target_is_admin = any(p.permission_name and "admin" in p.permission_name.lower() for p in (target.permissions or []))
    if target_is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot modify status of another admin")

    target.status = True
    db.commit()
    return {"message": "User has been unbanned"}
