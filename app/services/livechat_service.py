from datetime import datetime
from typing import Dict, List, Callable, Awaitable
from sqlalchemy.orm import Session, joinedload

from app.models.database import SessionLocal
from app.models.entities import (
    ChatSession,
    ChatInteraction,
    LiveChatQueue,
    ParticipateChatSession,
    AdmissionOfficialProfile,
    Users,
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
        customer_id = None
        session_id = None

        try:
            queue_item = db.query(LiveChatQueue).filter_by(id=queue_id).first()
            if not queue_item:
                return {"error": "queue_not_found"}

            official = db.query(AdmissionOfficialProfile).filter_by(
                admission_official_id=official_id
            ).first()
            
            if not official:
                return {"error": "official_not_found"}

            if official.current_sessions >= official.max_sessions:
                return {"error": "max_sessions_reached"}

            # Store customer_id before any potential session issues
            customer_id = queue_item.customer_id

            # Tạo live chat session
            session = ChatSession(
                session_type="live",
                start_time=datetime.now()
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            session_id = session.chat_session_id

            # Create participants
            participant1 = ParticipateChatSession(user_id=customer_id, session_id=session_id)
            participant2 = ParticipateChatSession(user_id=official_id, session_id=session_id)
            
            db.add_all([participant1, participant2])

            # SSE → update queue list cho AO
            await self.send_official_event(official_id, {
                "event": "queue_updated"
            })
            official.current_sessions += 1
            queue_item.status = "accepted"
            db.commit()
            db.close()
            
            return session

        except Exception as e:
            db.rollback()
            db.close()
            print(f"ERROR in official_accept: {str(e)}")
            return {"error": f"internal_error: {str(e)}"}

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
        try:
            session = db.query(ChatSession).filter_by(chat_session_id=session_id).first()
            if not session:
                return {"error": "session_not_found"}

            # Check if session is already ended
            if session.end_time is not None:
                return {"error": "session_already_ended"}

            # Verify that the user ending the session is a participant
            participant = db.query(ParticipateChatSession).filter(
                ParticipateChatSession.session_id == session_id,
                ParticipateChatSession.user_id == ended_by
            ).first()
            
            if not participant:
                return {"error": "not_session_participant"}

            # Find all participants to identify the official
            all_participants = db.query(ParticipateChatSession).filter(
                ParticipateChatSession.session_id == session_id
            ).all()

            # Find the official (check if any participant is an admission official)
            official_id = None
            for p in all_participants:
                # Check if this user is an admission official
                profile = db.query(AdmissionOfficialProfile).filter_by(
                    admission_official_id=p.user_id
                ).first()
                if profile:
                    official_id = p.user_id
                    break

            # End the session
            session.end_time = datetime.now().date()
            
            # Decrease official's current session count if found
            if official_id:
                profile = db.query(AdmissionOfficialProfile).filter_by(
                    admission_official_id=official_id
                ).first()
                if profile and profile.current_sessions > 0:
                    profile.current_sessions -= 1

            db.commit()

            # Push realtime notification to all connected participants
            payload = {
                "event": "chat_ended",
                "session_id": session_id,
                "ended_by": ended_by
            }

            for conn in self.active_sessions.get(session_id, []):
                await conn.send_json(payload)

            # Cleanup WebSocket connections
            self.active_sessions.pop(session_id, None)

            return {"success": True}
            
        except Exception as e:
            db.rollback()
            return {"error": f"database_error: {str(e)}"}
        finally:
            db.close()

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
        ).options(
            # Eagerly load customer information
            joinedload(LiveChatQueue.customer)
        ).all()
        
        # Transform to include customer info
        result = []
        for item in items:
            queue_item_dict = {
                'id': item.id,
                'customer_id': item.customer_id,
                'admission_official_id': item.admission_official_id,
                'status': item.status,
                'created_at': item.created_at.isoformat() if item.created_at else None,
                'customer': {
                    'full_name': item.customer.full_name if item.customer else f'Customer {item.customer_id}',
                    'email': item.customer.email if item.customer else 'N/A',
                    'phone_number': item.customer.phone_number if item.customer else 'N/A'
                } if item.customer else {
                    'full_name': f'Customer {item.customer_id}',
                    'email': 'N/A', 
                    'phone_number': 'N/A'
                }
            }
            result.append(queue_item_dict)
        
        db.close()
        return result
    
    async def get_active_sessions(self, official_id: int):
        """Get all active chat sessions for an admission official"""
        db = SessionLocal()
        try:
            # Query for sessions where the official is a participant
            # Use start_time and end_time to determine active status (started but not ended)
            active_sessions_query = db.query(
                ChatSession.chat_session_id,
                ChatSession.start_time,
                ChatSession.session_type
            ).join(
                ParticipateChatSession, 
                ChatSession.chat_session_id == ParticipateChatSession.session_id
            ).filter(
                ParticipateChatSession.user_id == official_id,
                ChatSession.start_time.isnot(None),  # Session has started
                ChatSession.end_time.is_(None)  # Session hasn't ended (active)
            ).all()
            
            result = []
            for session in active_sessions_query:
                session_id, start_time, session_type = session
                
                # For each session, find the customer (the other participant)
                customer_participant = db.query(
                    ParticipateChatSession, Users.full_name
                ).join(
                    Users, ParticipateChatSession.user_id == Users.user_id
                ).filter(
                    ParticipateChatSession.session_id == session_id,
                    ParticipateChatSession.user_id != official_id
                ).first()
                
                if customer_participant:
                    participant, customer_name = customer_participant
                    
                    result.append({
                        'session_id': session_id,
                        'customer_id': participant.user_id,
                        'customer_name': customer_name,
                        'session_type': session_type or 'live',
                        'start_time': start_time.isoformat() + 'T00:00:00' if start_time else datetime.now().isoformat(),
                        'status': 'active'
                    })
            
            return result
            
        finally:
            db.close()
    
    def get_messages(self, session_id: int):
        db = SessionLocal()
        msgs = db.query(ChatInteraction) \
            .filter_by(session_id=session_id) \
            .order_by(ChatInteraction.timestamp.asc()) \
            .all()
        db.close()
        return msgs
