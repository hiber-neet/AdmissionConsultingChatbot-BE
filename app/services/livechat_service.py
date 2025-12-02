from datetime import datetime
from typing import Dict, List, Callable, Awaitable
from sqlalchemy.orm import Session

from app.models.database import SessionLocal
from app.models.entities import (
    ChatSession,
    ChatInteraction,
    LiveChatQueue,
    ParticipateChatSession,
    AdmissionOfficialProfile,
)


class LiveChatService:

    def __init__(self):
        # SSE subscribers
        self.sse_customers: Dict[int, List[Callable[[dict], Awaitable[None]]]] = {}
        self.sse_officials: Dict[int, List[Callable[[dict], Awaitable[None]]]] = {}

        # WebSocket chat connections
        self.active_sessions: Dict[int, List] = {}

    # ======================================================================
    # Helper: gửi SSE cho customer
    # ======================================================================
    async def send_customer_event(self, customer_id: int, data: dict):
        subs = self.sse_customers.get(customer_id, [])
        for send in subs:
            await send(data)

    # Helper: gửi SSE cho official
    async def send_official_event(self, official_id: int, data: dict):
        subs = self.sse_officials.get(official_id, [])
        for send in subs:
            await send(data)

    # Helper: đăng ký listener SSE
    def register_customer_sse(self, customer_id: int, callback):
        self.sse_customers.setdefault(customer_id, []).append(callback)

    def register_official_sse(self, official_id: int, callback):
        self.sse_officials.setdefault(official_id, []).append(callback)

    def unregister_customer_sse(self, customer_id: int, callback):
        if customer_id in self.sse_customers:
            self.sse_customers[customer_id].remove(callback)

    def unregister_official_sse(self, official_id: int, callback):
        if official_id in self.sse_officials:
            self.sse_officials[official_id].remove(callback)

    # ======================================================================
    # 1. CUSTOMER REQUEST QUEUE
    # ======================================================================
    async def customer_join_queue(self, customer_id: int, official_id: int):
        db = SessionLocal()

        queue_entry = LiveChatQueue(
            customer_id=customer_id,
            admission_official_id=official_id,
            status="waiting",
            created_at=datetime.now()
        )
        db.add(queue_entry)
        db.commit()
        db.refresh(queue_entry)

        # Gửi sự kiện cho chính student
        await self.send_customer_event(customer_id, {
            "event": "queued",
            "queue_id": queue_entry.id,
            "official_id": official_id
        })

        # Gửi update cho AO
        await self.send_official_event(official_id, {
            "event": "queue_updated"
        })

        return queue_entry

    # ======================================================================
    # 2. AO ACCEPT REQUEST
    # ======================================================================
    async def official_accept(self, official_id: int, queue_id: int):
        db = SessionLocal()

        queue_item = db.query(LiveChatQueue).filter_by(id=queue_id).first()
        if not queue_item:
            return {"error": "queue_not_found"}

        official = db.query(AdmissionOfficialProfile).filter_by(
            admission_official_id=official_id
        ).first()

        if official.current_sessions >= official.max_sessions:
            return {"error": "max_sessions_reached"}

        # Tạo live chat session
        session = ChatSession(
            session_type="live",
            start_time=datetime.now()
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        db.add_all([
            ParticipateChatSession(user_id=queue_item.customer_id, session_id=session.chat_session_id),
            ParticipateChatSession(user_id=official_id, session_id=session.chat_session_id),
        ])

        

        # SSE → notify student
        await self.send_customer_event(queue_item.customer_id, {
            "event": "accepted",
            "session_id": session.chat_session_id,
            "official_id": official_id
        })

        # SSE → update queue list cho AO
        await self.send_official_event(official_id, {
            "event": "queue_updated"
        })
        official.current_sessions += 1
        queue_item.status = "accepted"
        db.commit()
        db.close()
        return session

    # ======================================================================
    # 3. AO REJECT REQUEST
    # ======================================================================
    async def official_reject(self, official_id: int, queue_id: int, reason: str):
        db = SessionLocal()
        queue_item = db.query(LiveChatQueue).filter_by(id=queue_id).first()
        if not queue_item:
            return False

        queue_item.status = "rejected"
        db.commit()

        # notify student
        await self.send_customer_event(queue_item.customer_id, {
            "event": "rejected",
            "reason": reason
        })

        # notify AO update queue
        await self.send_official_event(official_id, {
            "event": "queue_updated"
        })

        return True

    # ======================================================================
    # 4. CHAT (WebSocket)
    # ======================================================================
    async def join_chat(self, websocket, session_id: int):
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = []
        self.active_sessions[session_id].append(websocket)

        await websocket.send_json({"event": "chat_connected"})

    async def broadcast_message(self, session_id: int, sender_id: int, message: str):
        db = SessionLocal()

        chat = ChatInteraction(
            session_id=session_id,
            sender_id=sender_id,
            message_text=message,
            timestamp=datetime.now(),
            is_from_bot=False
        )
        db.add(chat)
        db.commit()
        db.close()

        payload = {
            "event": "message",
            "session_id": session_id,
            "sender_id": sender_id,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }

        for conn in self.active_sessions.get(session_id, []):
            await conn.send_json(payload)


    # ===============================================================
    # 5. END SESSION
    # ===============================================================
    async def end_session(self, session_id: int, ended_by: int):
        db = SessionLocal()

        session = db.query(ChatSession).filter_by(chat_session_id=session_id).first()
        if not session:
            return {"error": "session_not_found"}

        session.end_time = datetime.now()
        db.commit()

        # tìm official để giảm session count
        official_part = db.query(ParticipateChatSession).filter(
            ParticipateChatSession.session_id == session_id
        ).all()

        # lấy official id
        official_id = None
        for p in official_part:
            if p.user_id != ended_by:
                official_id = p.user_id
                break

        if official_id:
            profile = db.query(AdmissionOfficialProfile).filter_by(
                admission_official_id=official_id
            ).first()
            if profile:
                profile.current_sessions -= 1
                db.commit()

        db.close()

        # push realtime cho cả student + official cùng session
        payload = {
            "event": "chat_ended",
            "session_id": session_id,
            "ended_by": ended_by
        }

        for conn in self.active_sessions.get(session_id, []):
            await conn.send_json(payload)

        # cleanup
        self.active_sessions.pop(session_id, None)

        return {"success": True}

    def get_my_status(self, customer_id: int):
        db = SessionLocal()
        item = db.query(LiveChatQueue) \
            .filter_by(customer_id=customer_id) \
            .order_by(LiveChatQueue.created_at.desc()) \
            .first()
        db.close()
        return item

    def delete_queue_item(self, queue_id: int):
        db = SessionLocal()
        item = db.query(LiveChatQueue).filter_by(id=queue_id).first()
        if item:
            db.delete(item)
            db.commit()
        db.close()
        return True

    def get_queue_list(self, official_id: int):
        db = SessionLocal()
        items = db.query(LiveChatQueue).filter_by(
            admission_official_id=official_id,
            status="waiting"
        ).all()
        db.close()
        return items
    
    def get_messages(self, session_id: int):
        db = SessionLocal()
        msgs = db.query(ChatInteraction) \
            .filter_by(session_id=session_id) \
            .order_by(ChatInteraction.timestamp.asc()) \
            .all()
        db.close()
        return msgs
