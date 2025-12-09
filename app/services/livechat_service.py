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
        dead_callbacks = []
        
        for send in subs:
            try:
                await send(data)
            except Exception as e:
                print(f"Dead SSE callback for customer {customer_id}: {e}")
                dead_callbacks.append(send)
        
        # Remove dead callbacks
        for dead in dead_callbacks:
            if customer_id in self.sse_customers:
                try:
                    self.sse_customers[customer_id].remove(dead)
                except ValueError:
                    pass

    # Helper: gửi SSE cho official
    async def send_official_event(self, official_id: int, data: dict):
        subs = self.sse_officials.get(official_id, [])
        dead_callbacks = []
        
        for send in subs:
            try:
                await send(data)
            except Exception as e:
                print(f"Dead SSE callback for official {official_id}: {e}")
                dead_callbacks.append(send)
        
        # Remove dead callbacks
        for dead in dead_callbacks:
            if official_id in self.sse_officials:
                try:
                    self.sse_officials[official_id].remove(dead)
                except ValueError:
                    pass

    # Helper: đăng ký listener SSE
    def register_customer_sse(self, customer_id: int, callback):
        # Log current connections for debugging
        current_count = len(self.sse_customers.get(customer_id, []))
        print(f"Registering SSE for customer {customer_id}. Current connections: {current_count}")
        
        self.sse_customers.setdefault(customer_id, []).append(callback)
        print(f"Customer {customer_id} now has {len(self.sse_customers[customer_id])} SSE connection(s)")

    def register_official_sse(self, official_id: int, callback):
        current_count = len(self.sse_officials.get(official_id, []))
        print(f"Registering SSE for official {official_id}. Current connections: {current_count}")
        
        self.sse_officials.setdefault(official_id, []).append(callback)
        print(f"Official {official_id} now has {len(self.sse_officials[official_id])} SSE connection(s)")

    def unregister_customer_sse(self, customer_id: int, callback):
        if customer_id in self.sse_customers:
            try:
                self.sse_customers[customer_id].remove(callback)
                remaining = len(self.sse_customers[customer_id])
                print(f"Unregistered SSE for customer {customer_id}. Remaining: {remaining}")
                
                # Clean up empty lists
                if remaining == 0:
                    del self.sse_customers[customer_id]
                    print(f"Removed empty SSE list for customer {customer_id}")
            except ValueError:
                print(f"Callback not found for customer {customer_id}")

    def unregister_official_sse(self, official_id: int, callback):
        if official_id in self.sse_officials:
            try:
                self.sse_officials[official_id].remove(callback)
                remaining = len(self.sse_officials[official_id])
                print(f"Unregistered SSE for official {official_id}. Remaining: {remaining}")
                
                # Clean up empty lists
                if remaining == 0:
                    del self.sse_officials[official_id]
                    print(f"Removed empty SSE list for official {official_id}")
            except ValueError:
                print(f"Callback not found for official {official_id}")
    
    def get_sse_connection_count(self, customer_id: int = None, official_id: int = None):
        """Get SSE connection count for debugging"""
        if customer_id:
            return len(self.sse_customers.get(customer_id, []))
        if official_id:
            return len(self.sse_officials.get(official_id, []))
        return {
            "customers": {cid: len(cbs) for cid, cbs in self.sse_customers.items()},
            "officials": {oid: len(cbs) for oid, cbs in self.sse_officials.items()}
        }

    # ======================================================================
    # 1. CUSTOMER REQUEST QUEUE
    # ======================================================================
    async def customer_join_queue(self, customer_id: int, official_id: int = None):
        db = SessionLocal()
        
        try:
            # Check if customer is banned
            customer = db.query(Users).filter(Users.user_id == customer_id).first()
            if not customer:
                return {"error": "customer_not_found"}
            if not customer.status:
                return {"error": "customer_banned"}
            
            # Auto-assign admission officer if not specified
            if official_id is None:
                # Find available admission officers (status=True, not banned)
                available_officials = db.query(AdmissionOfficialProfile).join(
                    Users, Users.user_id == AdmissionOfficialProfile.admission_official_id
                ).filter(
                    Users.status == True,  # Not banned
                    AdmissionOfficialProfile.status == "available",
                    AdmissionOfficialProfile.current_sessions < AdmissionOfficialProfile.max_sessions
                ).order_by(
                    AdmissionOfficialProfile.current_sessions.asc()  # Least loaded first
                ).all()
                
                if not available_officials:
                    return {"error": "no_officers_available", "message": "No admission officers are currently available. Please try again later."}
                
                # Assign to officer with least current sessions
                official_id = available_officials[0].admission_official_id
            else:
                # Verify specified officer exists and is available
                officer = db.query(AdmissionOfficialProfile).join(
                    Users, Users.user_id == AdmissionOfficialProfile.admission_official_id
                ).filter(
                    AdmissionOfficialProfile.admission_official_id == official_id,
                    Users.status == True
                ).first()
                
                if not officer:
                    return {"error": "official_not_found"}
                if not officer.status == "available":
                    return {"error": "official_not_available"}
                if officer.current_sessions >= officer.max_sessions:
                    return {"error": "official_at_capacity"}

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

            return {"success": True, "queue_id": queue_entry.id, "official_id": official_id, "status": "waiting"}
            
        finally:
            db.close()

    # ======================================================================
    # 1B. CUSTOMER CANCEL QUEUE REQUEST
    # ======================================================================
    async def customer_cancel_queue(self, customer_id: int):
        """
        Cancel a customer's pending queue request.
        Only works if status is 'waiting' (before acceptance).
        """
        db = SessionLocal()
        try:
            # Find customer's pending queue entry
            queue_entry = db.query(LiveChatQueue).filter(
                LiveChatQueue.customer_id == customer_id,
                LiveChatQueue.status == "waiting"
            ).first()
            
            if not queue_entry:
                return {"error": "no_pending_queue_request"}
            
            official_id = queue_entry.admission_official_id
            queue_id = queue_entry.id
            
            # Mark as canceled
            queue_entry.status = "canceled"
            db.commit()
            
            # Notify customer
            await self.send_customer_event(customer_id, {
                "event": "queue_canceled",
                "queue_id": queue_id,
                "message": "You have canceled your queue request"
            })
            
            # Notify admission official about queue update
            if official_id:
                await self.send_official_event(official_id, {
                    "event": "queue_updated",
                    "message": f"Customer {customer_id} canceled their request"
                })
            
            return {"success": True, "message": "Queue request canceled successfully"}
            
        except Exception as e:
            db.rollback()
            return {"error": f"database_error: {str(e)}"}
        finally:
            db.close()

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
            
            print(f"[Accept] ===== ACCEPTING REQUEST =====")
            print(f"[Accept] Queue ID: {queue_id}")
            print(f"[Accept] Customer ID: {customer_id}")
            print(f"[Accept] Official ID: {official_id}")

            # Tạo live chat session
            session = ChatSession(
                session_type="live",
                start_time=datetime.now()
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            session_id = session.chat_session_id
            
            print(f"[Accept] ✅ Created session_id: {session_id}")

            # Create participants
            participant1 = ParticipateChatSession(user_id=customer_id, session_id=session_id)
            participant2 = ParticipateChatSession(user_id=official_id, session_id=session_id)
            
            db.add_all([participant1, participant2])

            # Update queue status and official sessions
            official.current_sessions += 1
            queue_item.status = "accepted"
            db.commit()

            # CRITICAL: Send SSE event to CUSTOMER with session_id
            print(f"[Accept] 📤 Sending 'accepted' SSE to customer {customer_id} with session_id={session_id}")
            await self.send_customer_event(customer_id, {
                "event": "accepted",
                "session_id": session_id,
                "official_id": official_id,
                "queue_id": queue_id
            })

            # SSE → update queue list for admission official
            print(f"[Accept] 📤 Sending 'queue_updated' SSE to official {official_id}")
            await self.send_official_event(official_id, {
                "event": "queue_updated"
            })
            
            # Return session_id as dict BEFORE closing db session
            result = {
                "success": True,
                "chat_session_id": session_id,
                "session_id": session_id,  # Legacy compatibility
                "customer_id": customer_id,
                "official_id": official_id,
                "queue_id": queue_id
            }
            
            db.close()
            
            print(f"[Accept] 🎉 Returning result dict with session_id={session_id} to API")
            print(f"[Accept] ===== ACCEPTANCE COMPLETE =====\n")
            
            return result

        except Exception as e:
            db.rollback()
            db.close()
            print(f"ERROR in official_accept: {str(e)}")
            import traceback
            traceback.print_exc()
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
        print(f"[Join Chat] New WebSocket connection for session_id={session_id}")
        
        if session_id not in self.active_sessions:
            print(f"[Join Chat] Creating new session list for session_id={session_id}")
            self.active_sessions[session_id] = []
        
        self.active_sessions[session_id].append(websocket)
        connection_count = len(self.active_sessions[session_id])
        print(f"[Join Chat] Session {session_id} now has {connection_count} active connection(s)")

        await websocket.send_json({"event": "chat_connected"})
        print(f"[Join Chat] Sent chat_connected confirmation to new connection")

    async def broadcast_message(self, session_id: int, sender_id: int, message: str):
        db = SessionLocal()

        try:
            print(f"[Broadcast] Saving message to DB: session_id={session_id}, sender_id={sender_id}, message='{message}'")
            
            chat = ChatInteraction(
                session_id=session_id,
                sender_id=sender_id,
                message_text=message,
                timestamp=datetime.now().date(),  # Use .date() for Date column
                is_from_bot=False
            )
            db.add(chat)
            db.commit()
            db.refresh(chat)
            
            print(f"[Broadcast] Message saved with interaction_id={chat.interaction_id}")

            payload = {
                "event": "message",
                "session_id": session_id,
                "sender_id": sender_id,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "interaction_id": chat.interaction_id
            }

            # Send to all connections in this session
            active_connections = self.active_sessions.get(session_id, [])
            print(f"[Broadcast] Active connections for session {session_id}: {len(active_connections)}")
            
            if len(active_connections) == 0:
                print(f"[Broadcast] WARNING: No active WebSocket connections for session {session_id}!")
            
            for idx, conn in enumerate(active_connections):
                try:
                    print(f"[Broadcast] Sending to connection #{idx+1}...")
                    await conn.send_json(payload)
                    print(f"[Broadcast] Successfully sent to connection #{idx+1}")
                except Exception as e:
                    print(f"[Broadcast] Error sending to connection #{idx+1}: {e}")
                    # Remove broken connections
                    if conn in self.active_sessions[session_id]:
                        self.active_sessions[session_id].remove(conn)
                        print(f"[Broadcast] Removed broken connection #{idx+1}")

        except Exception as e:
            db.rollback()
            print(f"Error saving/broadcasting message: {e}")
            raise  # Re-raise to let WebSocket handler know
        finally:
            db.close()

    async def leave_chat(self, websocket, session_id: int):
        """Remove WebSocket connection from active session"""
        print(f"[Leave Chat] Removing connection from session_id={session_id}")
        
        if session_id in self.active_sessions:
            if websocket in self.active_sessions[session_id]:
                self.active_sessions[session_id].remove(websocket)
                remaining = len(self.active_sessions[session_id])
                print(f"[Leave Chat] Connection removed. Remaining connections: {remaining}")
            else:
                print(f"[Leave Chat] WARNING: WebSocket not found in session {session_id}")
            
            # Clean up empty session lists
            if not self.active_sessions[session_id]:
                del self.active_sessions[session_id]
                print(f"[Leave Chat] Session {session_id} has no more connections, removed from active_sessions")
        else:
            print(f"[Leave Chat] WARNING: Session {session_id} not found in active_sessions")


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
