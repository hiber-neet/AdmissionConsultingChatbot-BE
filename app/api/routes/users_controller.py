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
    """Grant one or more permissions to a user (admin only). Returns summary of changes."""
    if not current_user or not has_permission(current_user, "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin permission required")

    # find target user
    target = db.query(Users).filter(Users.user_id == payload.user_id).first()
    if not target:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Target user not found")

    # Normalize requested ids
    requested_ids = list(dict.fromkeys(payload.permission_ids or []))
    if not requested_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No permission ids provided")

    # validate permission ids exist
    from app.models.entities import Permission as PermissionModel
    from app.models.entities import ConsultantProfile as ConsultantProfileModel
    from app.models.entities import ContentManagerProfile as ContentManagerProfileModel
    from app.models.entities import AdmissionOfficialProfile as AdmissionOfficialProfileModel

    perms = db.query(PermissionModel).filter(PermissionModel.permission_id.in_(requested_ids)).all()
    found_ids = {p.permission_id for p in perms}
    missing = [pid for pid in requested_ids if pid not in found_ids]
    if missing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"missing_permission_ids": missing})

    added = []
    skipped = []

    # current permission ids
    current_ids = {p.permission_id for p in (target.permissions or [])}

    # add new permissions
    add_perms = [p for p in perms if p.permission_id not in current_ids]
    for perm in add_perms:
        db.add(UserPermission(user_id=target.user_id, permission_id=perm.permission_id))
        added.append(perm.permission_id)

    # collect skipped ids (already present)
    skipped = [pid for pid in requested_ids if pid in current_ids]

    # create related profiles if necessary (once per type)
    added_names = {(p.permission_name or "").lower() for p in add_perms}
    if any("consultant" in n for n in added_names):
        cp = db.query(ConsultantProfileModel).filter(ConsultantProfileModel.consultant_id == target.user_id).first()
        if not cp:
            cp = ConsultantProfileModel(consultant_id=target.user_id, status=True, is_leader=bool(payload.consultant_is_leader))
            db.add(cp)
    if any("content" in n for n in added_names):
        cmp = db.query(ContentManagerProfileModel).filter(ContentManagerProfileModel.content_manager_id == target.user_id).first()
        if not cmp:
            cmp = ContentManagerProfileModel(content_manager_id=target.user_id, is_leader=bool(payload.content_manager_is_leader))
            db.add(cmp)
    if any(("admission" in n) or ("official" in n) for n in added_names):
        ap = db.query(AdmissionOfficialProfileModel).filter(AdmissionOfficialProfileModel.admission_official_id == target.user_id).first()
        if not ap:
            ap = AdmissionOfficialProfileModel(admission_official_id=target.user_id, rating=0, current_sessions=0, max_sessions=10, status="available")
            db.add(ap)

    # If any permissions were added, set user status to True
    if added:
        target.status = True

    db.commit()
    return {"added": added, "skipped": skipped}


@router.delete("/permissions/revoke")
def revoke_permission(
    payload: PermissionRevokeRequest,
    db: Session = Depends(get_db),
    current_user: Users = Depends(get_current_user)
):
    """Revoke one or more permissions from a user (admin only). Admins cannot revoke permissions from other admins."""
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

    requested_ids = list(dict.fromkeys(payload.permission_ids or []))
    if not requested_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No permission ids provided")

    # Validate that the permission ids exist
    perms = db.query(PermissionModel).filter(PermissionModel.permission_id.in_(requested_ids)).all()
    found_ids = {p.permission_id for p in perms}
    missing = [pid for pid in requested_ids if pid not in found_ids]
    if missing:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"missing_permission_ids": missing})

    removed = []
    skipped = []

    # find existing UserPermission rows for requested ids
    ups = db.query(UserPermission).filter(
        UserPermission.user_id == target.user_id,
        UserPermission.permission_id.in_(requested_ids)
    ).all()
    existing_ids = {u.permission_id for u in ups}
    skipped = [pid for pid in requested_ids if pid not in existing_ids]

    # import profile models
    from app.models.entities import ConsultantProfile as ConsultantProfileModel
    from app.models.entities import ContentManagerProfile as ContentManagerProfileModel
    from app.models.entities import AdmissionOfficialProfile as AdmissionOfficialProfileModel

    # delete found links
    for u in ups:
        db.delete(u)
        removed.append(u.permission_id)

    db.flush()

    # After removals, recompute remaining permissions to decide profile cleanup
    remaining_perms = db.query(PermissionModel).join(UserPermission).filter(UserPermission.user_id == target.user_id).all()
    remaining_names = {(p.permission_name or "").lower() for p in remaining_perms}

    # Clean profiles if no remaining related permissions
    if not any("consultant" in name for name in remaining_names):
        cp = db.query(ConsultantProfileModel).filter(ConsultantProfileModel.consultant_id == target.user_id).first()
        if cp:
            db.delete(cp)
    if not any("content" in name for name in remaining_names):
        cmp = db.query(ContentManagerProfileModel).filter(ContentManagerProfileModel.content_manager_id == target.user_id).first()
        if cmp:
            db.delete(cmp)
    if not any(("admission" in name) or ("official" in name) for name in remaining_names):
        ap = db.query(AdmissionOfficialProfileModel).filter(AdmissionOfficialProfileModel.admission_official_id == target.user_id).first()
        if ap:
            db.delete(ap)

    # If user has no remaining permissions, set status to False (ban)
    remaining = db.query(UserPermission).filter(UserPermission.user_id == target.user_id).all()
    if not remaining:
        target.status = False

    db.commit()
    return {"removed": removed, "skipped": skipped}


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
