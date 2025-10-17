from sqlalchemy import (
    Column, Integer, String, Boolean, Date, Float, ForeignKey, Text
)
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION, VECTOR
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy import create_engine

Base = declarative_base()

# ------------------------------------------------------------
# USERS & ROLES
# ------------------------------------------------------------

class Role(Base):
    __tablename__ = "Role"

    role_id = Column(Integer, primary_key=True)
    role_name = Column(String)

    users = relationship("Users", back_populates="role")


class Users(Base):
    __tablename__ = "Users"

    user_id = Column(Integer, primary_key=True)
    full_name = Column(String)
    email = Column(String)
    password = Column(String)
    status = Column(String)
    role_id = Column(Integer, ForeignKey("Role.role_id"))
    created_by = Column(Integer, ForeignKey("Users.user_id"), nullable=True)
    updated_by = Column(Integer, ForeignKey("Users.user_id"), nullable=True)

    role = relationship("Role", back_populates="users")


class UserRole(Base):
    __tablename__ = "UserRole"

    role_id = Column(Integer, ForeignKey("Role.role_id"))
    user_id = Column(Integer, ForeignKey("Users.user_id"))


# ------------------------------------------------------------
# STUDENT / CONSULTANT / ADMISSION / CONTENT MANAGER PROFILES
# ------------------------------------------------------------

class Interest(Base):
    __tablename__ = "Interest"

    interest_id = Column(Integer, primary_key=True)
    desired_major = Column(String)
    region = Column(String)

    student_profiles = relationship("StudentProfile", back_populates="interest")


class StudentProfile(Base):
    __tablename__ = "StudentProfile"

    student_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    interest_id = Column(Integer, ForeignKey("Interest.interest_id"))

    interest = relationship("Interest", back_populates="student_profiles")


class ConsultantProfile(Base):
    __tablename__ = "ConsultantProfile"

    consultant_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    rating = Column(Integer)
    status = Column(String)
    is_leader = Column(Boolean)


class AdmissionOfficialProfile(Base):
    __tablename__ = "AdmissionOfficalProfile"

    admission_official_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    rating = Column(Integer)
    current_sessions = Column(Integer)
    max_sessions = Column(Integer)
    status = Column(String)


class ContentManagerProfile(Base):
    __tablename__ = "ContentManagerProfile"

    content_manager_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    is_leader = Column(Boolean)


# ------------------------------------------------------------
# INTENT, TRAINING QUESTION, FAQ STATISTICS
# ------------------------------------------------------------

class Intent(Base):
    __tablename__ = "Intent"

    intent_id = Column(Integer, primary_key=True)
    intent_name = Column(String)
    description = Column(String)
    created_by = Column(Integer, ForeignKey("Users.user_id"))


class TrainingQuestion(Base):
    __tablename__ = "TrainingQuestion"

    question_id = Column(Integer, primary_key=True)
    question = Column(Text)
    answer = Column(Text)
    intent_id = Column(Integer, ForeignKey("Intent.intent_id"))
    created_by = Column(Integer, ForeignKey("Users.user_id"))
    approved_by = Column(Integer, ForeignKey("Users.user_id"))


class FaqStatistics(Base):
    __tablename__ = "FaqStatistics"

    faq_id = Column(Integer, primary_key=True)
    usage_count = Column(Integer)
    success_rate = Column(DOUBLE_PRECISION)
    question_text = Column(String)
    last_used_at = Column(Date)
    intent_id = Column(Integer, ForeignKey("Intent.intent_id"))


# ------------------------------------------------------------
# KNOWLEDGE BASE
# ------------------------------------------------------------

class KnowledgeBaseDocument(Base):
    __tablename__ = "KnowledgeBaseDocument"

    document_id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(Text)
    category = Column(String)
    created_at = Column(Date)
    update_at = Column(Date)
    created_by = Column(Integer, ForeignKey("Users.user_id"))

    chunks = relationship("DocumentChunk", back_populates="document")


class DocumentChunk(Base):
    __tablename__ = "DocumentChunk"

    chunk_id = Column(Integer, primary_key=True)
    chunk_text = Column(Text)
    embedding_vector = Column(VECTOR(1536))
    created_at = Column(Date)
    document_id = Column(Integer, ForeignKey("KnowledgeBaseDocument.document_id"))

    document = relationship("KnowledgeBaseDocument", back_populates="chunks")


# ------------------------------------------------------------
# CHAT SYSTEM
# ------------------------------------------------------------

class ChatSession(Base):
    __tablename__ = "ChatSession"

    chat_session_id = Column(Integer, primary_key=True)
    session_type = Column(String)
    start_time = Column(Date)
    end_time = Column(Date)
    feedback_rating = Column(Integer)
    notes = Column(Text)
    student_id = Column(Integer, ForeignKey("Users.user_id"))
    admission_officer_id = Column(Integer, ForeignKey("Users.user_id"))

    interactions = relationship("ChatInteraction", back_populates="session")


class ChatInteraction(Base):
    __tablename__ = "ChatInteraction"

    interaction_id = Column(Integer, primary_key=True)
    message_text = Column(Text)
    timestamp = Column(Date)
    is_from_bot = Column(Boolean)
    sender_id = Column(Integer, ForeignKey("Users.user_id"))
    session_id = Column(Integer, ForeignKey("ChatSession.chat_session_id"))

    session = relationship("ChatSession", back_populates="interactions")


class ParticipateChatSession(Base):
    __tablename__ = "ParticipateChatSession"

    user_id = Column(Integer, ForeignKey("Users.user_id"))
    session_id = Column(Integer, ForeignKey("ChatSession.chat_session_id"))


# ------------------------------------------------------------
# ARTICLE SYSTEM
# ------------------------------------------------------------

class ArticleCategory(Base):
    __tablename__ = "ArticleCategory"

    category_id = Column(Integer, primary_key=True)
    category_name = Column(String)

    articles = relationship("Article", back_populates="category")


class Article(Base):
    __tablename__ = "Article"

    article_id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(Text)
    status = Column(Boolean)
    author = Column(String)
    view_count = Column(Integer)
    date_created = Column(Date)
    tag = Column(String)
    category_id = Column(Integer, ForeignKey("ArticleCategory.category_id"))

    category = relationship("ArticleCategory", back_populates="articles")


# ------------------------------------------------------------
# CURRICULUM / MAJOR / COURSE
# ------------------------------------------------------------

class Curriculum(Base):
    __tablename__ = "Curriculum"

    curriculum_id = Column(Integer, primary_key=True)
    curriculum_name = Column(String)
    description = Column(Text)
    tuition_fee = Column(Float)
    image = Column(String)
    major_id = Column(Integer, ForeignKey("Major.major_id"))


class Major(Base):
    __tablename__ = "Major"

    major_id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(Text)
    curriculum_id = Column(Integer, ForeignKey("Curriculum.curriculum_id"))

    courses = relationship("Course", back_populates="major")


class Course(Base):
    __tablename__ = "Course"

    course_id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(Text)
    semester = Column(String)
    major_id = Column(Integer, ForeignKey("Major.major_id"))

    major = relationship("Major", back_populates="courses")


# ------------------------------------------------------------
# PERSONALIZED RECOMMENDATION
# ------------------------------------------------------------

class PersonalizedRecommendation(Base):
    __tablename__ = "PersionalizedRecommendation"

    recommendation_id = Column(Integer, primary_key=True)
    confidence_score = Column(DOUBLE_PRECISION)
    user_id = Column(Integer, ForeignKey("Users.user_id"))
    base_intent_id = Column(Integer, ForeignKey("Intent.intent_id"))
    suggested_intent_id = Column(Integer, ForeignKey("Intent.intent_id"))
    session_id = Column(Integer, ForeignKey("ChatSession.chat_session_id"))



