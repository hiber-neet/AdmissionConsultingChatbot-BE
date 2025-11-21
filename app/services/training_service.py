from datetime import datetime
from typing import Dict, List
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_classic.memory import ConversationBufferMemory
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import uuid
import asyncio
from sqlalchemy.orm import Session
from app.models.entities import ChatInteraction, ChatSession, ParticipateChatSession, TrainingQuestionAnswer
from app.models.database import SessionLocal
from sqlalchemy.exc import SQLAlchemyError
from app.services.memory_service import MemoryManager

memory_service = MemoryManager()

class TrainingService:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.gemini_api_key,
            temperature=0.7
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=self.gemini_api_key
        )
        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        self.training_qa_collection = "training_qa"
        self.documents_collection = "knowledge_base_documents"
        self._init_collections()

    def _init_collections(self):
        try:
            self.qdrant_client.create_collection(
                collection_name=self.training_qa_collection,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
            )
        except:
            pass
            
        try:
            self.qdrant_client.create_collection(
                collection_name=self.documents_collection,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
            )
        except:
            pass

    def create_chat_session(self, user_id: int, session_type: str = "chatbot") -> int:
        """
        Tạo chat session mới
        
        Args:
            user_id: ID của user
            session_type: "chatbot" hoặc "live"
        
        Returns:
            session_id: ID của session vừa tạo
        """
        db = SessionLocal()
        try:
            session = ChatSession(
                session_type=session_type,
                start_time=datetime.datetime.now()
            )
            db.add(session)
            db.flush()
            
            # Add user vào participate table
            participate = ParticipateChatSession(
                user_id=user_id,
                session_id=session.chat_session_id
            )
            db.add(participate)
            db.commit()
            
            return session.chat_session_id
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Error creating session: {e}")
            raise
        finally:
            db.close()

    def get_session_history(self, session_id: int, limit: int = 50) -> List[Dict]:
        """
        Lấy lịch sử chat của session
        
        Returns:
            List of messages [{message_text, timestamp, is_from_bot}, ...]
        """
        db = SessionLocal()
        try:
            interactions = db.query(ChatInteraction).filter(
                ChatInteraction.session_id == session_id
            ).order_by(
                ChatInteraction.timestamp.asc()
            ).limit(limit).all()
            
            return [
                {
                    "message_text": i.message_text,
                    "timestamp": i.timestamp.isoformat() if i.timestamp else None,
                    "is_from_bot": i.is_from_bot,
                    "rating": i.rating
                }
                for i in interactions
            ]
        finally:
            db.close()
    
    def get_user_sessions(self, user_id: int) -> List[Dict]:
        """
        Lấy tất cả sessions của user (để hiển thị recent chats)
        
        Returns:
            List of sessions với preview message cuối cùng
        """
        db = SessionLocal()
        try:
            sessions = db.query(ChatSession).join(
                ParticipateChatSession
            ).filter(
                ParticipateChatSession.user_id == user_id
            ).order_by(
                ChatSession.start_time.desc()
            ).all()
            
            result = []
            for session in sessions:
                # Lấy message cuối cùng làm preview
                last_msg = db.query(ChatInteraction).filter(
                    ChatInteraction.session_id == session.chat_session_id
                ).order_by(
                    ChatInteraction.timestamp.desc()
                ).first()
                
                result.append({
                    "session_id": session.chat_session_id,
                    "session_type": session.session_type,
                    "start_time": session.start_time.isoformat() if session.start_time else None,
                    "last_message_preview": last_msg.message_text[:50] + "..." if last_msg else "",
                    "last_message_time": last_msg.timestamp.isoformat() if last_msg and last_msg.timestamp else None
                })
            
            return result
        finally:
            db.close()

    # ---------------------------
    # Query enrichment: dùng chat_history + last bot question để build a full query
    # ---------------------------
    async def enrich_query(self, session_id: str, user_message: str) -> str:
        memory = memory_service.get_memory(session_id)
        mem_vars = memory.load_memory_variables({})
        chat_history = mem_vars.get("chat_history", "")

        prompt = f"""
        Bạn là một trợ lý giúp chuyển các câu trả lời của người dùng thành các truy vấn tìm kiếm đầy đủ cho chatbot RAG tư vấn tuyển sinh.

        Cuộc hội thoại gần đây (theo thứ tự từ cũ đến mới):
        {chat_history}

        Phản hồi mới nhất của người dùng: "{user_message}"

        Nhiệm vụ: Dựa trên "cuộc hội thoại gần đây" và "phản hồi mới nhất của người dùng", bạn hãy đảm bảo tạo ra **một câu truy vấn tìm kiếm**, ngắn gọn, rõ ràng, cụ thể (bằng tiếng Việt), thể hiện đúng ý định của người dùng để gửi cho chatbot rag tư vấn để nó có thể hiểu yêu cầu của người dùng. "Chỉ tạo truy vấn nếu phản hồi của người dùng là phần tiếp nối hoặc làm rõ nội dung trong hội thoại trước đó.", nếu phản hồi của người dùng không trả lời hoặc không liên quan cho cuộc hội thoại gần đây thì hãy trả về chuỗi rỗng.

        Chỉ trả về **một dòng truy vấn duy nhất** (không thêm nội dung khác).  
        Ví dụ:
        - "Thông tin về ngành Công nghệ Thông tin tại trường XYZ"  
        - "Học phí ngành CNTT hệ chính quy năm 2025 tại trường XYZ"
        """
        # assume async predict exists
        enriched = await self.llm.ainvoke(prompt)
        # fallback: if empty use original
        enriched_txt = (enriched or "").strip().splitlines()[0] if enriched else user_message
        return enriched_txt   

    # ---------------------------
    # LLM relevance check: ensure enriched_query actually matches the training QA
    # ---------------------------
    async def llm_relevance_check(self, enriched_query: str, matched_question: str, answer: str) -> bool:
        prompt = f"""
        Bạn là chuyên gia đánh giá giữa câu hỏi tìm kiếm, câu hỏi trong cơ sở dữ liệu và câu trả lời cho 1 hệ thống chat RAG tuyển sinh, hãy suy luận. 

        Câu hỏi tìm kiếm (đã chuẩn hóa): "{enriched_query}"
        Câu hỏi DB: "{matched_question}"
        Câu trả lời chính thức: "{answer}"

        Hãy trả lời duy nhất chỉ một từ: "true" nếu câu hỏi DB phù hợp và trả lời đó hợp lý cho truy vấn tìm kiếm; "false" nếu chỉ trùng từ khóa hoặc không phù hợp.
        """
        res = await self.llm.ainvoke(prompt)
        if not res:
            return False
        r = res.strip().lower()
        return ("đúng" in r) or ("true" in r) or (r.startswith("đúng")) or (r.startswith("true"))

    async def load_session_history_to_memory(self, session_id: int, db: Session):
        memory = memory_service.get_memory(session_id)

        # Lấy lịch sử chat theo thứ tự thời gian
        interactions = (
            db.query(ChatInteraction)
            .filter(ChatInteraction.session_id == session_id)
            .order_by(ChatInteraction.timestamp.asc())
            .all()
        )

        last_user_msg = None
        for inter in interactions:
            if not inter.is_from_bot:
                # user message
                last_user_msg = inter.message_text
            else:
                # bot message -> kết hợp với user message trước đó (nếu có)
                memory.save_context(
                    {"input": last_user_msg or ""},
                    {"output": inter.message_text}
                )
                last_user_msg = None

        # Nếu cuối cùng là tin nhắn user chưa được phản hồi
        if last_user_msg:
            memory.save_context({"input": last_user_msg}, {"output": ""})

    async def stream_response_from_context(self, query: str, context: str, session_id: int = 1, user_id: int = 1):
        db = SessionLocal()
        
        try:
            # 🧩 1. Lưu tin nhắn người dùng
            user_msg = ChatInteraction(
                message_text=query,
                timestamp=datetime.now(),
                rating=None,
                is_from_bot=False,
                sender_id=user_id,
                session_id=session_id
            )
            db.add(user_msg)
            db.flush()  # flush để lấy ID nếu cần liên kết sau
        
            memory = memory_service.get_memory(session_id)
            mem_vars = memory.load_memory_variables({})
            chat_history = mem_vars.get("chat_history", "")
            """Stream phản hồi từ Gemini, từng chunk một."""
            prompt = f"""Bạn là một chatbot tư vấn tuyển sinh chuyên nghiệp của trường XYZ
            Đây là đoạn hội thoại trước: 
            {chat_history}
            === THÔNG TIN THAM KHẢO ===
            {context}
            === CÂU HỎI ===
            {query}
            === HƯỚNG DẪN ===
            - Trả lời bằng tiếng Việt
            - Thân thiện, chuyên nghiệp
            - Dựa vào thông tin tham khảo trên được cung cấp
            - Bạn là chatbot tư vấn tuyển sinh của trường xyz, nếu thông tin câu hỏi yêu câu tên 1 trường khác thì hãy nói rõ ra là không tìm thấy thông tin
            - Nếu không tìm thấy thông tin, hãy nói rõ và gợi ý liên hệ trực tiếp nhân viên tư vấn
            - Không bịa thêm thông tin ngoài context
            - Nếu câu hỏi chỉ là chào hỏi, hỏi thời tiết, hoặc các câu xã giao, hãy trả lời bằng lời chào thân thiện, giới thiệu về bản thân chatbot, KHÔNG kéo thêm thông tin chi tiết trong context.
            - Có thể **diễn đạt lại câu hỏi hoặc thông tin** một cách nhẹ nhàng, tự nhiên để người dùng dễ hiểu hơn, **nhưng tuyệt đối không thay đổi ý nghĩa hay thêm dữ kiện mới.**
            - Khi có thể, hãy **giải thích thêm bối cảnh hoặc gợi ý bước tiếp theo**, ví dụ:  
                “Bạn muốn mình gửi danh sách ngành đào tạo kèm chuyên ngành chi tiết không?”  
                hoặc  
                “Nếu bạn quan tâm học bổng, mình có thể nói rõ các loại học bổng hiện có nhé!”
            """
            full_response = ""
            async for chunk in self.llm.astream(prompt):
                yield chunk
                full_response += chunk
                await asyncio.sleep(0)  # Nhường event loop
            memory.save_context({"input": query}, {"output": full_response})  
            
            # === 🔥 Lưu bot response vào DB ===
            bot_msg = ChatInteraction(
                message_text=full_response,
                timestamp=datetime.now(),
                rating=None,
                is_from_bot=True,
                sender_id=None,
                session_id=session_id
            )
            db.add(bot_msg)

            # 🧩 5. Commit 1 lần duy nhất
            db.commit()
            print(f"💾 Saved both user+bot messages for session {session_id}")
        except SQLAlchemyError as e:
            db.rollback()
            print(f" Database error during chat transaction: {e}")
        finally:
            db.close()
    async def stream_response_from_qa(self, query: str, context: str, session_id: int = 1, user_id: int = 1):
      
        memory = memory_service.get_memory(session_id)
        mem_vars = memory.load_memory_variables({})
        chat_history = mem_vars.get("chat_history", "")
        prompt = f"""
        Bạn là chatbot tư vấn tuyển sinh của trường XYZ.
        Đây là đoạn hội thoại trước: 
        {chat_history}
        === CÂU TRẢ LỜI CHÍNH THỨC ===
        {context}

        === CÂU HỎI NGƯỜI DÙNG ===
        {query}

        === HƯỚNG DẪN TRẢ LỜI ===
        - Hãy đọc kỹ phần NGỮ CẢNH LIÊN QUAN, nhưng **chỉ sử dụng nó nếu thật sự có nội dung trùng khớp hoặc phù hợp với câu hỏi người dùng.**
        - Nếu phần CÂU TRẢ LỜI CHÍNH THỨC không liên quan rõ ràng đến câu hỏi, **đừng cố trả lời theo context** mà hãy nói:
        “Hiện chưa có thông tin chính xác cho câu hỏi này. Bạn có thể nói rõ chi tiết hơn được không?” 
        - Nếu phần trả lời chính thức không phù hợp với câu hỏi, hãy nói “Hiện chưa có thông tin cho câu hỏi này. Vui lòng liên hệ chuyên viên tư vấn.”
        - Bạn là chatbot tư vấn tuyển sinh của trường xyz, nhớ kiểm tra kĩ rõ ràng câu hỏi, nếu thông tin câu hỏi yêu câu tên 1 trường khác thì hãy nói rõ ra là không tìm thấy thông tin
        - Nếu câu hỏi chỉ là chào hỏi, hỏi thời tiết, hoặc các câu xã giao, hãy trả lời bằng lời chào thân thiện, giới thiệu về bản thân chatbot, KHÔNG kéo thêm thông tin chi tiết trong context.
        - Nếu câu hỏi quá mơ hồ, hãy hỏi lại để rõ hơn và chi tiết hơn về câu hỏi
        - Có thể **diễn đạt lại câu hỏi hoặc thông tin** một cách nhẹ nhàng, tự nhiên để người dùng dễ hiểu hơn, **nhưng tuyệt đối không thay đổi ý nghĩa hay thêm dữ kiện mới.**
        - Khi có thể, hãy **giải thích thêm bối cảnh hoặc gợi ý bước tiếp theo**, ví dụ:  
            “Bạn muốn mình gửi danh sách ngành đào tạo kèm chuyên ngành chi tiết không?”  
            hoặc  
            “Nếu bạn quan tâm học bổng, mình có thể nói rõ các loại học bổng hiện có nhé!”
        """
        full_response = ""
        async for chunk in self.llm.astream(prompt):
            yield chunk
            full_response += chunk 
            await asyncio.sleep(0)  # Nhường event loop

        memory.save_context({"input": query}, {"output": full_response})  
        print("Saved to memory. Current messages:", len(self.memory.chat_memory.messages)) 
    
    def add_document(self, document_id: int, content: str, metadata: dict = None):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Size optimal cho Vietnamese
            chunk_overlap=200     # Overlap to preserve context
        )
        chunks = text_splitter.split_text(content)
        
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            # Embed chunk
            embedding = self.embeddings.embed_query(chunk)
            point_id = str(uuid.uuid4())
            
            # Upsert to Qdrant
            self.qdrant_client.upsert(
                collection_name="knowledge_base_documents",
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "document_id": document_id,
                            "chunk_index": i,
                            "chunk_text": chunk,
                            "metadata": metadata or {},
                            "type": "document"
                        }
                    )
                ]
            )
            chunk_ids.append(point_id)
        
        return chunk_ids
    def add_training_qa(self, db: Session, intent_id: int, question_text: str, answer_text: str):
        """
        Add training Q&A pair vào Qdrant
        
        Chỉ embed question, không embed answer:
        - Answer stored ở DB, retrieve khi match found
        - Question dùng để search/match
        - Tiết kiệm storage, tăng search speed
        
        Args:
            question_id: Primary key của training Q&A
            intent_id: Intent này thuộc intent nào
            question_text: Question để embed
            answer_text: Answer (lưu ở DB, không embed)
        
        Returns:
            embedding_id: Qdrant point ID
        """
        new_qa = TrainingQuestionAnswer(
            question=question_text,
            answer=answer_text,
            intent_id=1,
            created_by=1
        )
        db.add(new_qa)
        db.commit()
        db.refresh(new_qa)
        # Embed question text
        embedding = self.embeddings.embed_query(question_text)
        point_id = str(uuid.uuid4())
        
        # Upsert vào training_qa collection
        # Metadata:
        # - question_id: Link về DB
        # - intent_id: Để track intent stats
        # - question_text: Lưu original text (optional, space saving)
        # - answer_text: Lưu answer (retrieve khi match)
        self.qdrant_client.upsert(
            collection_name=self.training_qa_collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "question_id": new_qa.question_id,
                        "intent_id": 1,
                        "question_text": question_text,
                        "answer_text": answer_text,
                        "type": "training_qa"
                    }
                )
            ]
        )
        
        return {
            "postgre_question_id": new_qa.question_id,
            "qdrant_question_id": point_id
        }
    def search_documents(self, query: str, top_k: int = 5):
        """
        Search documents (Fallback)
        
        Fallback path: Tìm document chunks khi training Q&A không match
        - Query → Embed → Search documents collection
        - Return top_k chunks
        - LLM sẽ synthesize answer từ chunks
        
        Args:
            query: User question
            top_k: Số chunks (lower score → fallback)
        
        Returns:
            List of document chunks
        """
        
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.qdrant_client.search(
            collection_name=self.documents_collection,
            query_vector=query_embedding,
            limit=top_k
        )
        
        return results
    
    def search_training_qa(self, query: str, top_k: int = 5):
        """
        Search training Q&A (Priority 1)
        
        Fast path: Tìm pre-approved answers
        - Query → Embed → Search training_qa collection
        - Return top_k matches
        - filter score > 0.8
        
        Args:
            query: User question
            top_k: Số results (default 5)
        
        Returns:
            List of search results with scores
        """
        
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.qdrant_client.search(
            collection_name=self.training_qa_collection,
            query_vector=query_embedding,
            limit=top_k
        )
        
        return results
    def hybrid_search(self, query: str):
        """
        Hybrid RAG Search Strategy
        
        PRIORITY SYSTEM (Cascade):
        1. TIER 1 - Training Q&A (score > 0.8)
           - Highest confidence, direct answer
           - No LLM needed, fast response
           
        2. TIER 2 - Training Q&A (0.7 < score <= 0.8)
           - Good match but not perfect
           - Use as primary answer + add document context
           
        3. TIER 3 - Document Search + LLM Generation
           - No training Q&A match
           - Search documents, LLM synthesize
           - Lower confidence, show sources
           
        4. TIER 4 - Fallback
           - Nothing found
           - Suggest live chat with officer
        
        Returns:
            {
                "response": str,
                "response_source": "training_qa" | "document" | "fallback",
                "confidence": float,
                "top_match": obj,
                "intent_id": int,
                "sources": list
            }
        """
        
        # STEP 1: Search training Q&A
        qa_results = self.search_training_qa(query, top_k=3)
        
        # TIER 1: Perfect match (score > 0.8)
        if qa_results and qa_results[0].score > 0.8:
            top_match = qa_results[0]
            return {
                "response_official_answer": top_match.payload.get("answer_text"),
                "response_source": "training_qa",
                "confidence": top_match.score,
                "top_match": top_match,
                "intent_id": top_match.payload.get("intent_id"),
                "question_id": top_match.payload.get("question_id"),
                "sources": []
            }
        
        
        # TIER 2: No training Q&A match, try documents
        else: doc_results = self.search_documents(query, top_k=5)
        return {
                "response": doc_results,
                "response_source": "document",
                "confidence": doc_results[0].score,
                "top_match": None,
                "intent_id": None,
                "sources": [r.payload.get("document_id") for r in doc_results]
            }
        
      

    

langchain_service = TrainingService()
