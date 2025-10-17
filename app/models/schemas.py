from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import date


# ================= ROLE =================
class RoleBase(BaseModel):
    role_name: str


class RoleCreate(RoleBase):
    pass


class RoleResponse(RoleBase):
    role_id: int

    class Config:
        orm_mode = True


# ================= USER =================
class UserBase(BaseModel):
    full_name: str
    email: EmailStr
    status: Optional[str] = None


class UserCreate(UserBase):
    password: str
    role_id: Optional[int] = None


class UserUpdate(BaseModel):
    full_name: Optional[str]
    email: Optional[EmailStr]
    password: Optional[str]
    status: Optional[str]


class UserResponse(UserBase):
    user_id: int
    role_id: Optional[int]

    class Config:
        orm_mode = True


# ================= STUDENT PROFILE =================
class StudentProfileBase(BaseModel):
    interest_id: Optional[int]


class StudentProfileCreate(StudentProfileBase):
    pass


class StudentProfileResponse(StudentProfileBase):
    student_id: int

    class Config:
        orm_mode = True


# ================= INTEREST =================
class InterestBase(BaseModel):
    desired_major: Optional[str]
    region: Optional[str]


class InterestCreate(InterestBase):
    pass


class InterestResponse(InterestBase):
    interest_id: int

    class Config:
        orm_mode = True


# ================= CONSULTANT PROFILE =================
class ConsultantProfileBase(BaseModel):
    rating: Optional[int]
    status: Optional[str]
    is_leader: Optional[bool]


class ConsultantProfileResponse(ConsultantProfileBase):
    consultant_id: int

    class Config:
        orm_mode = True


# ================= CONTENT MANAGER PROFILE =================
class ContentManagerProfileBase(BaseModel):
    is_leader: bool


class ContentManagerProfileResponse(ContentManagerProfileBase):
    content_manager_id: int

    class Config:
        orm_mode = True


# ================= ADMISSION OFFICIAL PROFILE =================
class AdmissionOfficialProfileBase(BaseModel):
    rating: Optional[int]
    current_sessions: Optional[int]
    max_sessions: Optional[int]
    status: Optional[str]


class AdmissionOfficialProfileResponse(AdmissionOfficialProfileBase):
    admission_official_id: int

    class Config:
        orm_mode = True


# ================= CURRICULUM / MAJOR / COURSE =================
class CourseBase(BaseModel):
    name: str
    description: Optional[str]
    semester: Optional[str]
    major_id: Optional[int]


class CourseResponse(CourseBase):
    course_id: int

    class Config:
        orm_mode = True


class MajorBase(BaseModel):
    name: str
    description: Optional[str]
    curriculum_id: Optional[int]


class MajorResponse(MajorBase):
    major_id: int
    courses: Optional[List[CourseResponse]] = []

    class Config:
        orm_mode = True


class CurriculumBase(BaseModel):
    curriculum_name: str
    description: Optional[str]
    tuition_fee: Optional[float]
    image: Optional[str]


class CurriculumResponse(CurriculumBase):
    curriculum_id: int
    majors: Optional[List[MajorResponse]] = []

    class Config:
        orm_mode = True


# ================= INTENT / TRAINING QUESTION / FAQ =================
class IntentBase(BaseModel):
    intent_name: str
    description: Optional[str]


class IntentResponse(IntentBase):
    intent_id: int

    class Config:
        orm_mode = True


class TrainingQuestionBase(BaseModel):
    question: str
    answer: str
    intent_id: Optional[int]


class TrainingQuestionResponse(TrainingQuestionBase):
    question_id: int

    class Config:
        orm_mode = True


class FaqStatisticsBase(BaseModel):
    usage_count: int
    success_rate: float
    question_text: str
    last_used_at: Optional[date]
    intent_id: Optional[int]


class FaqStatisticsResponse(FaqStatisticsBase):
    faq_id: int

    class Config:
        orm_mode = True


# ================= KNOWLEDGE BASE =================
class KnowledgeBaseDocumentBase(BaseModel):
    title: str
    content: str
    category: Optional[str]
    created_at: Optional[date]
    update_at: Optional[date]
    created_by: Optional[int]


class KnowledgeBaseDocumentResponse(KnowledgeBaseDocumentBase):
    document_id: int

    class Config:
        orm_mode = True


class DocumentChunkBase(BaseModel):
    chunk_text: str
    created_at: Optional[date]
    document_id: int


class DocumentChunkResponse(DocumentChunkBase):
    chunk_id: int

    class Config:
        orm_mode = True


# ================= CHAT =================
class ChatInteractionBase(BaseModel):
    message_text: str
    timestamp: Optional[date]
    is_from_bot: bool
    sender_id: Optional[int]
    session_id: Optional[int]


class ChatInteractionResponse(ChatInteractionBase):
    interaction_id: int

    class Config:
        orm_mode = True


class ChatSessionBase(BaseModel):
    session_type: str
    start_time: Optional[date]
    end_time: Optional[date]
    feedback_rating: Optional[int]
    notes: Optional[str]
    student_id: Optional[int]
    admission_officer_id: Optional[int]


class ChatSessionResponse(ChatSessionBase):
    chat_session_id: int
    interactions: Optional[List[ChatInteractionResponse]] = []

    class Config:
        orm_mode = True


# ================= ARTICLE =================
class ArticleCategoryBase(BaseModel):
    category_name: str


class ArticleCategoryResponse(ArticleCategoryBase):
    category_id: int

    class Config:
        orm_mode = True


class ArticleBase(BaseModel):
    title: str
    content: str
    status: Optional[bool]
    author: str
    view_count: Optional[int]
    date_created: Optional[date]
    tag: Optional[str]
    category_id: Optional[int]


class ArticleResponse(ArticleBase):
    article_id: int

    class Config:
        orm_mode = True


# ================= RECOMMENDATION =================
class PersonalizedRecommendationBase(BaseModel):
    confidence_score: float
    user_id: int
    base_intent_id: int
    suggested_intent_id: int
    session_id: int


class PersonalizedRecommendationResponse(PersonalizedRecommendationBase):
    recommendation_id: int

    class Config:
        orm_mode = True
