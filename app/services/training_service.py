from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters  import RecursiveCharacterTextSplitter
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


    async def stream_response_from_context(self, query: str, context: str):
        """Stream phản hồi từ Gemini, từng chunk một."""
        prompt = f"""Bạn là một chatbot tư vấn tuyển sinh chuyên nghiệp...
        === THÔNG TIN THAM KHẢO ===
        {context}
        === CÂU HỎI ===
        {query}
        === HƯỚNG DẪN ===
        - Trả lời bằng tiếng Việt
        - Dựa vào thông tin trên, không bịa đặt
        """

        async for chunk in self.llm.astream(prompt):
            yield chunk
            await asyncio.sleep(0)  # Nhường event loop

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
    def add_training_qa(self, db: Session, question_id: int, intent_id: int, question_text: str, answer_text: str):
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
            intent_id=intent_id,
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
                        "question_id": question_id,
                        "intent_id": intent_id,
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

    def generate_response_from_context(self, query: str, context: str):
        """
        Generate response từ document context (LLM-based)
        
        Dùng cho: Fallback path khi không có training Q&A match
        - Input: query + document chunks
        - Output: Natural language answer via Gemini
        - Risk: Có thể hallucinate nếu context không đủ
        - Mitigation: Confidence score thấp, suggest live chat
        
        Args:
            query: User question
            context: Document chunks concatenated
        
        Returns:
            Generated response
        """
        
        # Prompt engineering: Context + Query + Instructions
        prompt = f"""Bạn là một chatbot tư vấn tuyển sinh chuyên nghiệp của trường XYZ. 
Dựa trên thông tin dưới đây, hãy trả lời câu hỏi của sinh viên:

=== THÔNG TIN THAM KHẢO ===
{context}

=== CÂU HỎI CỦA SINH VIÊN ===
{query}

=== HƯỚNG DẪN ===
- Trả lời bằng tiếng Việt
- Thân thiện, chuyên nghiệp
- Dựa vào thông tin được cung cấp
- Nếu không tìm thấy thông tin, hãy nói rõ và gợi ý liên hệ trực tiếp
- Không bịa thêm thông tin ngoài context

=== TRẢ LỜI ===
"""
        
        response = self.llm.invoke(prompt)
        return response

langchain_service = TrainingService()
