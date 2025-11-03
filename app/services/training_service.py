from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_classic.memory import ConversationBufferMemory
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
import uuid
import asyncio
from sqlalchemy.orm import Session
from app.models.entities import TrainingQuestionAnswer

class TrainingService:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=self.gemini_api_key,
            temperature=0.7
        )
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=self.gemini_api_key
        )
        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        self.previous_context = self.memory.load_memory_variables({}).get("chat_history", "")
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
    def llm_relevance_check(self, query: str, matched_question: str, answer: str) -> bool:
        prompt = f"""
        Câu hỏi người dùng: "{query}"
        Câu hỏi trong cơ sở dữ liệu: "{matched_question}"
        Câu trả lời: "{answer}"
        Hỏi: Hai câu hỏi này có cùng ý nghĩa không?
        Trả lời "true" hoặc "false".
        """
        result = self.llm(prompt)
        return "true" in result     

    async def stream_response_from_context(self, query: str, context: str):
        memory_vars = self.memory.load_memory_variables({})
        prev = memory_vars.get("chat_history", "")
        print("Content History (context):", self.previous_context)
        """Stream phản hồi từ Gemini, từng chunk một."""
        prompt = f"""Bạn là một chatbot tư vấn tuyển sinh chuyên nghiệp của trường XYZ
        Đây là đoạn hội thoại trước: 
        {self.previous_context}
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
        self.memory.save_context({"input": query}, {"output": full_response})  
        print("Saved to memory. Current messages:", len(self.memory.chat_memory.messages)) 
    async def stream_response_from_qa(self, query: str, context: str):
      
        memory_vars = self.memory.load_memory_variables({})
        prev = memory_vars.get("chat_history", "")
        print("Loaded memory (context):", bool(prev), "chars:", len(prev))
        prompt = f"""
        Bạn là chatbot tư vấn tuyển sinh của trường XYZ.
        Đây là đoạn hội thoại trước: 
        {self.previous_context}
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

        self.memory.save_context({"input": query}, {"output": full_response})  
        print("Saved to memory. Current messages:", len(self.memory.chat_memory.messages)) 
    # async def stream_response_from_hybrid(self, query: str, official_answer: str = "", additional_context: str = ""):
    #     """
    #     Stream phản hồi cho TIER 2 (hybrid giữa Training Q&A và Document).
    #     - official_answer: câu trả lời chính (từ training QA)
    #     - additional_context: thông tin mở rộng từ tài liệu (document)
    #     """

    #     prompt = f"""
    #     Bạn là chatbot tư vấn tuyển sinh chuyên nghiệp của trường Đại học XYZ.

    #     === CÂU TRẢ LỜI CHÍNH (từ cơ sở huấn luyện do chuyên viên cung cấp) ===
    #     {official_answer.strip() or "Không có dữ liệu huấn luyện cho câu hỏi này."}

    #     === THÔNG TIN THAM KHẢO BỔ SUNG (từ tài liệu chính thức) ===
    #     {additional_context.strip() or "Không có thông tin bổ sung."}

    #     === CÂU HỎI NGƯỜI DÙNG ===
    #     {query.strip()}

    #     === HƯỚNG DẪN TRẢ LỜI ===
    #     - Ưu tiên sử dụng **Câu trả lời chính** làm nội dung trung tâm khi bạn đã đánh giá kĩ càng, rõ ràng context có thật sự đúng với câu hỏi, yêu cầu của người dùng hay không, bạn có thể hỏi ngược lại người dùng nếu câu hỏi chưa khiến bạn chắc chắn, nếu phần CÂU TRẢ LỜI CHÍNH THỨC ở trên đã chứa thông tin trực tiếp liên quan, hãy sử dụng nguyên văn nội dung đó để trả lời.
    #     - Có thể **mở rộng hoặc làm rõ** bằng thông tin từ “Thông tin tham khảo bổ sung” 
    #     nếu thấy phù hợp, **nhưng không được thay đổi nội dung gốc**.
    #     - Trả lời bằng tiếng Việt, tự nhiên, thân thiện, dễ hiểu.
    #     - Không bịa thêm thông tin.
    #     - Bạn là chatbot tư vấn tuyển sinh của trường xyz, nhớ kiểm tra kĩ rõ ràng câu hỏi, nếu thông tin câu hỏi yêu câu tên 1 trường khác thì hãy nói rõ ra là không tìm thấy thông tin
    #     - Nếu câu hỏi chỉ là chào hỏi, hỏi thời tiết, hoặc các câu xã giao, hãy trả lời bằng lời chào thân thiện, giới thiệu về bản thân chatbot, KHÔNG kéo thêm thông tin chi tiết trong context.
    #     - Nếu câu hỏi quá mơ hồ, hãy hỏi lại để rõ hơn và chi tiết hơn về câu hỏi
    #     """

    #     async for chunk in self.llm.astream(prompt):
    #         yield chunk
    #         await asyncio.sleep(0) 
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
        
        # TIER 2: Good match (0.7 < score <= 0.8)
        # Dùng answer nhưng thêm document context để richer
        # elif qa_results and qa_results[0].score > 0.7:
        #     top_match = qa_results[0]
            
        #     # Thêm document context nếu available
        #     doc_results = self.search_documents(query, top_k=2)
        #     additional_context = "\n\n".join(
        #         [r.payload.get("chunk_text", "") for r in doc_results]
        #     )
            
            
            
        #     return {
        #         "response_official_answer": top_match.payload.get("answer_text"),
        #         "additional_context": additional_context,
        #         "response_source": "training_qa",
        #         "confidence": top_match.score,
        #         "top_match": top_match,
        #         "intent_id": top_match.payload.get("intent_id"),
        #         "question_id": top_match.payload.get("question_id"),
        #         "sources": [r.payload.get("document_id") for r in doc_results]
        #     }
        
        # TIER 3: No training Q&A match, try documents
        else: doc_results = self.search_documents(query, top_k=5)
        return {
                "response": doc_results,
                "response_source": "document",
                "confidence": doc_results[0].score,
                "top_match": None,
                "intent_id": None,
                "sources": [r.payload.get("document_id") for r in doc_results]
            }
        
        # TIER 4: Fallback - nothing found
        # fallback_response = """Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong cơ sở dữ liệu.

        # Để được hỗ trợ tốt hơn, bạn có thể:
        # 1. Đặt câu hỏi chi tiết hơn
        # 2. Liên hệ trực tiếp với phòng Tuyển sinh qua Live Chat
        # 3. Gọi hotline: [number]
        # 4. Email: [email]

        # Chúng tôi sẵn sàng giúp đỡ!"""

    

langchain_service = TrainingService()
