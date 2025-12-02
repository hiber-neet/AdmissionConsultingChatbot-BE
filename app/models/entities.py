import datetime
from sqlalchemy import (
    Column, Integer, String, Boolean, Date, Float, ForeignKey, Text
)
from datetime import datetime
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy import create_engine

Base = declarative_base()
# =====================
# USERS, ROLE, PERMISSION
# =====================
class Users(Base):
    __tablename__ = 'Users'
    
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    status = Column(Boolean, default=True)
    role_id = Column(Integer, ForeignKey('Role.role_id'), nullable=True)
    phone_number = Column(String, nullable=False)

    # Relationships
    role = relationship('Role', back_populates='users')

    # permissions (many-to-many)
    user_permissions = relationship('UserPermission', back_populates='user', cascade="all, delete-orphan")
    permissions = relationship('Permission', secondary='UserPermission', back_populates='users')

    # 1-1 profiles
    customer_profile = relationship('CustomerProfile', back_populates='user', uselist=False)
    consultant_profile = relationship('ConsultantProfile', back_populates='user', uselist=False)
    content_manager_profile = relationship('ContentManagerProfile', back_populates='user', uselist=False)
    admission_official_profile = relationship('AdmissionOfficialProfile', back_populates='user', uselist=False)

    # documents / knowledge base
    knowledge_documents = relationship('KnowledgeBaseDocument', back_populates='author', cascade="all, delete-orphan")
    document_chunks = relationship('DocumentChunk', back_populates='created_by_user', cascade="all, delete-orphan")

    # templates & articles & admissions
    templates = relationship('Template', back_populates='creator', cascade="all, delete-orphan")
    articles = relationship('Article', back_populates='author_user', cascade="all, delete-orphan")
    admission_informations = relationship('AdmissionInformation', back_populates='creator', cascade="all, delete-orphan")
    admission_forms = relationship('AdmissionForm', back_populates='user', cascade="all, delete-orphan")

    # chat, recommendations
    chat_interactions = relationship('ChatInteraction', back_populates='user', cascade="all, delete-orphan")
    personalized_recommendations = relationship('PersonalizedRecommendation', back_populates='user', cascade="all, delete-orphan")
    participate_sessions = relationship('ParticipateChatSession', back_populates='user')
    # training QA created/approved (two distinct relations)
    training_question_answers_created = relationship(
        "TrainingQuestionAnswer",
        foreign_keys="[TrainingQuestionAnswer.created_by]",
        back_populates="created_by_user",
        cascade="all, delete-orphan"
    )
    training_question_answers_approved = relationship(
        "TrainingQuestionAnswer",
        foreign_keys="[TrainingQuestionAnswer.approved_by]",
        back_populates="approved_by_user",
        cascade="all, delete-orphan"
    )


class UserPermission(Base):
    __tablename__ = 'UserPermission'
    
    permission_id = Column(Integer, ForeignKey("Permission.permission_id"), primary_key=True)
    user_id = Column(Integer, ForeignKey('Users.user_id'), primary_key=True)
    
    # Relationships
    user = relationship('Users', back_populates='user_permissions')
    permission = relationship('Permission', back_populates='user_permissions')


class Permission(Base):
    __tablename__ = 'Permission'
    
    permission_id = Column(Integer, primary_key=True, autoincrement=True)
    permission_name = Column(String)
    
    # Relationships
    user_permissions = relationship('UserPermission', back_populates='permission', cascade="all, delete-orphan")
    users = relationship('Users', secondary='UserPermission', back_populates='permissions')


class Role(Base):
    __tablename__ = 'Role'
    
    role_id = Column(Integer, primary_key=True, autoincrement=True)
    role_name = Column(String)
    
    # Relationships
    users = relationship('Users', back_populates='role')


# =====================
# CUSTOMER PROFILE (was StudentProfile)
# =====================
class CustomerProfile(Base):
    __tablename__ = 'CustomerProfile'
    
    customer_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    interest_id = Column(Integer, ForeignKey('Interest.interest_id'), nullable=True)
    # Relationships
    user = relationship('Users', back_populates='customer_profile')
    interest = relationship('Interest', back_populates='customer_profiles')
    academic_scores = relationship('AcademicScore', back_populates='customer', cascade="all, delete-orphan")
    riasec_results = relationship('RiasecResult', back_populates='customer', cascade="all, delete-orphan")
   
class LiveChatQueue(Base):
    __tablename__ = 'LiveChatQueue'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("Users.user_id"))
    admission_official_id = Column(Integer, ForeignKey("Users.user_id"), nullable=True)
    status = Column(String, default="waiting")  # waiting, accepted, canceled
    created_at = Column(Date, default=datetime.now)
    
    # relationships
    customer = relationship("Users", foreign_keys=[customer_id])
    admission_official = relationship("Users", foreign_keys=[admission_official_id])

class Interest(Base):
    __tablename__ = 'Interest'
    
    interest_id = Column(Integer, primary_key=True, autoincrement=True)
    desired_major = Column(String)
    region = Column(String)
    
    # Relationships
    customer_profiles = relationship('CustomerProfile', back_populates='interest', cascade="all, delete-orphan")


class AcademicScore(Base):
    __tablename__ = 'AcademicScore'
    
    score_id = Column(Integer, primary_key=True, autoincrement=True)
    subject_name = Column(String, nullable=False)
    score = Column(Float)
    customer_id = Column(Integer, ForeignKey('CustomerProfile.customer_id'))
    
    # Relationships
    customer = relationship('CustomerProfile', back_populates='academic_scores')


# =====================
# RIASEC
# =====================
class RiasecTrait(Base):
    __tablename__ = "RiasecTrait"
    
    trait_code_id = Column(Integer, primary_key=True, autoincrement=True)
    trait_code_name = Column(String, nullable=False)
    description = Column(String)
    
    # Relationships
    questions = relationship('RiasecQuestion', back_populates='trait', cascade="all, delete-orphan")


class RiasecQuestion(Base):
    __tablename__ = 'RiasecQuestion'
    
    question_id = Column(Integer, primary_key=True, autoincrement=True)
    question_name = Column(String, nullable=False)
    trait_code_id = Column(Integer, ForeignKey('RiasecTrait.trait_code_id'), nullable=False)
    
    # Relationships
    trait = relationship('RiasecTrait', back_populates='questions')






class RiasecResult(Base):
    __tablename__ = 'RiasecResult'
    
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    score_realistic = Column(Integer)
    score_investigative = Column(Integer)
    score_artistic = Column(Integer)
    score_social = Column(Integer)
    score_enterprising = Column(Integer)
    score_conventional = Column(Integer)
    result = Column(String)
    session_id = Column(String, unique=True)
    customer_id = Column(Integer, ForeignKey('CustomerProfile.customer_id'))
    
    # Relationships
    customer = relationship('CustomerProfile', back_populates='riasec_results')


# =====================
# CONSULTANT, MANAGER, ADMISSION OFFICIAL
# =====================
class ConsultantProfile(Base):
    __tablename__ = 'ConsultantProfile'
    
    consultant_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    status = Column(Boolean, default=True)
    is_leader = Column(Boolean, default=False)
   
    user = relationship('Users', back_populates='consultant_profile')


class ContentManagerProfile(Base):
    __tablename__ = 'ContentManagerProfile'
    
    content_manager_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    is_leader = Column(Boolean, default=False)
  
    user = relationship('Users', back_populates='content_manager_profile')


class AdmissionOfficialProfile(Base):
    __tablename__ = 'AdmissionOfficialProfile'
    
    admission_official_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    rating = Column(Integer)
    current_sessions = Column(Integer)
    max_sessions = Column(Integer)
    status = Column(String)
    
    user = relationship('Users', back_populates='admission_official_profile')


# =====================
# MAJOR, ADMISSION FORM
# =====================
class Major(Base):
    __tablename__ = 'Major'
    
    major_id = Column(Integer, primary_key=True, autoincrement=True)
    major_name = Column("name",String, nullable=False)
    created_by = Column(Integer, ForeignKey('Users.user_id'), nullable=True)
    
    admission_forms = relationship('AdmissionForm', back_populates='major', cascade="all, delete-orphan")
    articles = relationship('Article', back_populates='major', cascade="all, delete-orphan")


class AdmissionForm(Base):
    __tablename__ = 'AdmissionForm'
    
    form_id = Column(Integer, primary_key=True, autoincrement=True)
    fullname = Column(String)
    email = Column(String)
    phone_number = Column(String)
    major_id = Column(Integer, ForeignKey('Major.major_id'))
    campus = Column(String)
    submit_time = Column(Date)
    user_id = Column(Integer, ForeignKey('Users.user_id'))
    
    # Relationships
    user = relationship('Users', back_populates='admission_forms')
    major = relationship('Major', back_populates='admission_forms')


# =====================
# CHAT SYSTEM
# =====================
class ChatSession(Base):
    __tablename__ = 'ChatSession'
    
    chat_session_id = Column(Integer, primary_key=True, autoincrement=True)
    session_type = Column(String)
    start_time = Column(Date, default=datetime.now)
    end_time = Column(Date)
    feedback_rating = Column(Integer)
    notes = Column(String)
   
    interactions = relationship('ChatInteraction', back_populates='session', cascade="all, delete-orphan")
    participate_sessions = relationship('ParticipateChatSession', back_populates='session', cascade="all, delete-orphan")


class ParticipateChatSession(Base):
    __tablename__ = 'ParticipateChatSession'
    
    user_id = Column(Integer, ForeignKey('Users.user_id'), primary_key=True)
    session_id = Column(Integer, ForeignKey('ChatSession.chat_session_id'), primary_key=True)
    
    # Relationships
    session = relationship('ChatSession', back_populates='participate_sessions')
    user = relationship('Users', back_populates='participate_sessions')  


class ChatInteraction(Base):
    __tablename__ = 'ChatInteraction'
    
    interaction_id = Column(Integer, primary_key=True, autoincrement=True)
    message_text = Column(Text)
    timestamp = Column(Date, default=datetime.now)
    rating = Column(Integer)
    is_from_bot = Column(Boolean)
    sender_id = Column(Integer, ForeignKey('Users.user_id'))
    session_id = Column(Integer, ForeignKey('ChatSession.chat_session_id'))
    
    # Relationships
    user = relationship('Users', back_populates='chat_interactions')
    session = relationship('ChatSession', back_populates='interactions')


# =====================
# INTENT, FAQ, RECOMMENDATION, TRAINING QA
# =====================
class Intent(Base):
    __tablename__ = 'Intent'
    
    intent_id = Column(Integer, primary_key=True, autoincrement=True)
    intent_name = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(Date, default=datetime.now)
    created_by = Column(Integer, ForeignKey("Users.user_id"), nullable=True)
    
    faq_statistics = relationship('FaqStatistics', back_populates='intent', cascade="all, delete-orphan")
    training_questions = relationship('TrainingQuestionAnswer', back_populates='intent', cascade="all, delete-orphan")


class FaqStatistics(Base):
    __tablename__ = 'FaqStatistics'
    
    faq_id = Column(Integer, primary_key=True, autoincrement=True)
    usage_count = Column(Integer, default=0)
    success_rate = Column(Float)
    question_text = Column(String)
    answer_text = Column(String)
    rating = Column(Integer)
    last_used_at = Column(Date)
    intent_id = Column(Integer, ForeignKey('Intent.intent_id'))
    
    intent = relationship('Intent', back_populates='faq_statistics')


class PersonalizedRecommendation(Base):
    __tablename__ = 'PersonalizedRecommendation'
    
    recommendation_id = Column(Integer, primary_key=True, autoincrement=True)
    confidence_score = Column(Float)
    user_id = Column(Integer, ForeignKey('Users.user_id'))
    base_intent_id = Column(Integer, ForeignKey("Intent.intent_id"))
    suggested_intent_id = Column(Integer, ForeignKey("Intent.intent_id"))
    session_id = Column(Integer, ForeignKey("ChatSession.chat_session_id"))
    
    user = relationship('Users', back_populates='personalized_recommendations')


class TrainingQuestionAnswer(Base):
    __tablename__ = 'TrainingQuestionAnswer'
    
    question_id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(String)
    answer = Column(String)
    intent_id = Column(Integer, ForeignKey("Intent.intent_id"))
    created_by = Column(Integer, ForeignKey("Users.user_id"))
    approved_by = Column(Integer, ForeignKey("Users.user_id"))
    
    # Relationships
    intent = relationship("Intent", back_populates="training_questions")

    created_by_user = relationship(
        "Users", foreign_keys=[created_by], back_populates="training_question_answers_created"
    )
    approved_by_user = relationship(
        "Users", foreign_keys=[approved_by], back_populates="training_question_answers_approved"
    )


# -------------------- AdmissionInformation ---------------------------
class AdmissionInformation(Base):
    __tablename__ = "AdmissionInformation"

    id = Column(Integer, primary_key=True, autoincrement=True)
    academic_year = Column(String)
    target_applicant = Column(String)
    admission_method = Column(String)
    scholarship_infor = Column(String)
    create_at = Column(Date, default=datetime.now)
    update_at = Column(Date, onupdate=datetime.now)
    created_by = Column(Integer, ForeignKey("Users.user_id"))

    # Relationships
    creator = relationship("Users", back_populates="admission_informations")
# ---------------------------------------------------------------------


# --------------- KnowledgeBaseDocument & DocumentChunk ----------------
class KnowledgeBaseDocument(Base):
    __tablename__ = 'KnowledgeBaseDocument'
    
    document_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String)
    file_path = Column(String)
    category = Column(String)
    created_at = Column(Date, default=datetime.now)
    updated_at = Column(Date, onupdate=datetime.now)
    created_by = Column(Integer, ForeignKey('Users.user_id'))
    
    # Relationships
    chunks = relationship('DocumentChunk', back_populates='document', cascade="all, delete-orphan")
    author = relationship('Users', back_populates='knowledge_documents')
    

class DocumentChunk(Base):
    __tablename__ = 'DocumentChunk'
    
    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_text = Column(Text)
    embedding_vector = Column(String)  # Store as JSON or use vector extension
    created_at = Column(Date, default=datetime.now)
    document_id = Column(Integer, ForeignKey('KnowledgeBaseDocument.document_id'))
    created_by = Column(Integer, ForeignKey('Users.user_id'), nullable=True)

    # Relationships
    document = relationship('KnowledgeBaseDocument', back_populates='chunks')
    created_by_user = relationship('Users', back_populates='document_chunks')
# ---------------------------------------------------------------------


# -------------------- Template & Template_Field ----------------------
class Template(Base):
    __tablename__ = 'Template'
    
    template_id = Column(Integer, primary_key=True, autoincrement=True)
    template_name = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_by = Column(Integer, ForeignKey('Users.user_id'))
    
    # Relationships
    template_fields = relationship('Template_Field', back_populates='template', cascade="all, delete-orphan")
    creator = relationship('Users', back_populates='templates')


class Template_Field(Base):
    __tablename__ = 'Template_Field'
    
    template_field_id = Column(Integer, primary_key=True, autoincrement=True)
    field_name = Column(String)
    order_field = Column(Integer)
    field_type = Column(String)
    template_id = Column(Integer, ForeignKey("Template.template_id"))
    
    # Relationships
    template = relationship('Template', back_populates='template_fields')
# ---------------------------------------------------------------------


# ---------------- Specialization & Article ---------------------------
class Specialization(Base):
    __tablename__ = 'Specialization'
    
    specialization_id = Column(Integer, primary_key=True, autoincrement=True)
    specialization_name = Column(String, nullable=False)
    major_id = Column(Integer, ForeignKey('Major.major_id'), nullable=True)
    
    articles = relationship('Article', back_populates='specialization', cascade="all, delete-orphan")


class Article(Base):
    __tablename__ = "Article"

    article_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String)
    description = Column(String)
    url = Column(String)
    note = Column(String)
    # content = Column(Text)
    status = Column(String, default="draft")  # Values: draft, published, rejected, cancelled
    create_at = Column(Date, default=datetime.now)
    created_by = Column(Integer, ForeignKey("Users.user_id"))
    major_id = Column(Integer, ForeignKey('Major.major_id'), nullable=True)
    specialization_id = Column(Integer, ForeignKey('Specialization.specialization_id'), nullable=True)

    # Relationships
    author_user = relationship('Users', back_populates='articles')
    major = relationship('Major', back_populates='articles')
    specialization = relationship('Specialization', back_populates='articles')
# ---------------------------------------------------------------------import datetime
from sqlalchemy import (
    Column, Integer, String, Boolean, Date, Float, ForeignKey, Text
)
from datetime import datetime
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy import create_engine

Base = declarative_base()
# =====================
# USERS, ROLE, PERMISSION
# =====================
class Users(Base):
    __tablename__ = 'Users'
    
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    status = Column(Boolean, default=True)
    role_id = Column(Integer, ForeignKey('Role.role_id'), nullable=True)
    phone_number = Column(String, nullable=False)

    # Relationships
    role = relationship('Role', back_populates='users')

    # permissions (many-to-many)
    user_permissions = relationship('UserPermission', back_populates='user', cascade="all, delete-orphan")
    permissions = relationship('Permission', secondary='UserPermission', back_populates='users')

    # 1-1 profiles
    customer_profile = relationship('CustomerProfile', back_populates='user', uselist=False)
    consultant_profile = relationship('ConsultantProfile', back_populates='user', uselist=False)
    content_manager_profile = relationship('ContentManagerProfile', back_populates='user', uselist=False)
    admission_official_profile = relationship('AdmissionOfficialProfile', back_populates='user', uselist=False)

    # documents / knowledge base
    knowledge_documents = relationship('KnowledgeBaseDocument', back_populates='author', cascade="all, delete-orphan")
    document_chunks = relationship('DocumentChunk', back_populates='created_by_user', cascade="all, delete-orphan")

    # templates & articles & admissions
    templates = relationship('Template', back_populates='creator', cascade="all, delete-orphan")
    articles = relationship('Article', back_populates='author_user', cascade="all, delete-orphan")
    admission_informations = relationship('AdmissionInformation', back_populates='creator', cascade="all, delete-orphan")
    admission_forms = relationship('AdmissionForm', back_populates='user', cascade="all, delete-orphan")

    # chat, recommendations
    chat_interactions = relationship('ChatInteraction', back_populates='user', cascade="all, delete-orphan")
    personalized_recommendations = relationship('PersonalizedRecommendation', back_populates='user', cascade="all, delete-orphan")
    participate_sessions = relationship('ParticipateChatSession', back_populates='user')
    # training QA created/approved (two distinct relations)
    training_question_answers_created = relationship(
        "TrainingQuestionAnswer",
        foreign_keys="[TrainingQuestionAnswer.created_by]",
        back_populates="created_by_user",
        cascade="all, delete-orphan"
    )
    training_question_answers_approved = relationship(
        "TrainingQuestionAnswer",
        foreign_keys="[TrainingQuestionAnswer.approved_by]",
        back_populates="approved_by_user",
        cascade="all, delete-orphan"
    )


class UserPermission(Base):
    __tablename__ = 'UserPermission'
    
    permission_id = Column(Integer, ForeignKey("Permission.permission_id"), primary_key=True)
    user_id = Column(Integer, ForeignKey('Users.user_id'), primary_key=True)
    
    # Relationships
    user = relationship('Users', back_populates='user_permissions')
    permission = relationship('Permission', back_populates='user_permissions')


class Permission(Base):
    __tablename__ = 'Permission'
    
    permission_id = Column(Integer, primary_key=True, autoincrement=True)
    permission_name = Column(String)
    
    # Relationships
    user_permissions = relationship('UserPermission', back_populates='permission', cascade="all, delete-orphan")
    users = relationship('Users', secondary='UserPermission', back_populates='permissions')


class Role(Base):
    __tablename__ = 'Role'
    
    role_id = Column(Integer, primary_key=True, autoincrement=True)
    role_name = Column(String)
    
    # Relationships
    users = relationship('Users', back_populates='role')


# =====================
# CUSTOMER PROFILE (was StudentProfile)
# =====================
class CustomerProfile(Base):
    __tablename__ = 'CustomerProfile'
    
    customer_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    interest_id = Column(Integer, ForeignKey('Interest.interest_id'), nullable=True)
    # Relationships
    user = relationship('Users', back_populates='customer_profile')
    interest = relationship('Interest', back_populates='customer_profiles')
    academic_scores = relationship('AcademicScore', back_populates='customer', cascade="all, delete-orphan")
    riasec_results = relationship('RiasecResult', back_populates='customer', cascade="all, delete-orphan")
   
class LiveChatQueue(Base):
    __tablename__ = 'LiveChatQueue'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("Users.user_id"))
    admission_official_id = Column(Integer, ForeignKey("Users.user_id"), nullable=True)
    status = Column(String, default="waiting")  # waiting, accepted, canceled
    created_at = Column(Date, default=datetime.now)
    
    # relationships
    customer = relationship("Users", foreign_keys=[customer_id])
    admission_official = relationship("Users", foreign_keys=[admission_official_id])

class Interest(Base):
    __tablename__ = 'Interest'
    
    interest_id = Column(Integer, primary_key=True, autoincrement=True)
    desired_major = Column(String)
    region = Column(String)
    
    # Relationships
    customer_profiles = relationship('CustomerProfile', back_populates='interest', cascade="all, delete-orphan")


class AcademicScore(Base):
    __tablename__ = 'AcademicScore'
    
    score_id = Column(Integer, primary_key=True, autoincrement=True)
    subject_name = Column(String, nullable=False)
    score = Column(Float)
    customer_id = Column(Integer, ForeignKey('CustomerProfile.customer_id'))
    
    # Relationships
    customer = relationship('CustomerProfile', back_populates='academic_scores')


# =====================
# RIASEC
# =====================
class RiasecTrait(Base):
    __tablename__ = "RiasecTrait"
    
    trait_code_id = Column(Integer, primary_key=True, autoincrement=True)
    trait_code_name = Column(String, nullable=False)
    description = Column(String)
    
    # Relationships
    questions = relationship('RiasecQuestion', back_populates='trait', cascade="all, delete-orphan")


class RiasecQuestion(Base):
    __tablename__ = 'RiasecQuestion'
    
    question_id = Column(Integer, primary_key=True, autoincrement=True)
    question_name = Column(String, nullable=False)
    trait_code_id = Column(Integer, ForeignKey('RiasecTrait.trait_code_id'), nullable=False)
    
    # Relationships
    trait = relationship('RiasecTrait', back_populates='questions')






class RiasecResult(Base):
    __tablename__ = 'RiasecResult'
    
    result_id = Column(Integer, primary_key=True, autoincrement=True)
    score_realistic = Column(Integer)
    score_investigative = Column(Integer)
    score_artistic = Column(Integer)
    score_social = Column(Integer)
    score_enterprising = Column(Integer)
    score_conventional = Column(Integer)
    result = Column(String)
    session_id = Column(String, unique=True)
    customer_id = Column(Integer, ForeignKey('CustomerProfile.customer_id'))
    
    # Relationships
    customer = relationship('CustomerProfile', back_populates='riasec_results')


# =====================
# CONSULTANT, MANAGER, ADMISSION OFFICIAL
# =====================
class ConsultantProfile(Base):
    __tablename__ = 'ConsultantProfile'
    
    consultant_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    status = Column(Boolean, default=True)
    is_leader = Column(Boolean, default=False)
   
    user = relationship('Users', back_populates='consultant_profile')


class ContentManagerProfile(Base):
    __tablename__ = 'ContentManagerProfile'
    
    content_manager_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    is_leader = Column(Boolean, default=False)
  
    user = relationship('Users', back_populates='content_manager_profile')


class AdmissionOfficialProfile(Base):
    __tablename__ = 'AdmissionOfficialProfile'
    
    admission_official_id = Column(Integer, ForeignKey("Users.user_id"), primary_key=True)
    rating = Column(Integer)
    current_sessions = Column(Integer)
    max_sessions = Column(Integer)
    status = Column(String)
    
    user = relationship('Users', back_populates='admission_official_profile')


# =====================
# MAJOR, ADMISSION FORM
# =====================
class Major(Base):
    __tablename__ = 'Major'
    
    major_id = Column(Integer, primary_key=True, autoincrement=True)
    major_name = Column("name",String, nullable=False)
    created_by = Column(Integer, ForeignKey('Users.user_id'), nullable=True)
    
    admission_forms = relationship('AdmissionForm', back_populates='major', cascade="all, delete-orphan")
    articles = relationship('Article', back_populates='major', cascade="all, delete-orphan")


class AdmissionForm(Base):
    __tablename__ = 'AdmissionForm'
    
    form_id = Column(Integer, primary_key=True, autoincrement=True)
    fullname = Column(String)
    email = Column(String)
    phone_number = Column(String)
    major_id = Column(Integer, ForeignKey('Major.major_id'))
    campus = Column(String)
    submit_time = Column(Date)
    user_id = Column(Integer, ForeignKey('Users.user_id'))
    
    # Relationships
    user = relationship('Users', back_populates='admission_forms')
    major = relationship('Major', back_populates='admission_forms')


# =====================
# CHAT SYSTEM
# =====================
class ChatSession(Base):
    __tablename__ = 'ChatSession'
    
    chat_session_id = Column(Integer, primary_key=True, autoincrement=True)
    session_type = Column(String)
    start_time = Column(Date, default=datetime.now)
    end_time = Column(Date)
    feedback_rating = Column(Integer)
    notes = Column(String)
   
    interactions = relationship('ChatInteraction', back_populates='session', cascade="all, delete-orphan")
    participate_sessions = relationship('ParticipateChatSession', back_populates='session', cascade="all, delete-orphan")


class ParticipateChatSession(Base):
    __tablename__ = 'ParticipateChatSession'
    
    user_id = Column(Integer, ForeignKey('Users.user_id'), primary_key=True)
    session_id = Column(Integer, ForeignKey('ChatSession.chat_session_id'), primary_key=True)
    
    # Relationships
    session = relationship('ChatSession', back_populates='participate_sessions')
    user = relationship('Users', back_populates='participate_sessions')  


class ChatInteraction(Base):
    __tablename__ = 'ChatInteraction'
    
    interaction_id = Column(Integer, primary_key=True, autoincrement=True)
    message_text = Column(Text)
    timestamp = Column(Date, default=datetime.now)
    rating = Column(Integer)
    is_from_bot = Column(Boolean)
    sender_id = Column(Integer, ForeignKey('Users.user_id'))
    session_id = Column(Integer, ForeignKey('ChatSession.chat_session_id'))
    
    # Relationships
    user = relationship('Users', back_populates='chat_interactions')
    session = relationship('ChatSession', back_populates='interactions')


# =====================
# INTENT, FAQ, RECOMMENDATION, TRAINING QA
# =====================
class Intent(Base):
    __tablename__ = 'Intent'
    
    intent_id = Column(Integer, primary_key=True, autoincrement=True)
    intent_name = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(Date, default=datetime.now)
    created_by = Column(Integer, ForeignKey("Users.user_id"), nullable=True)
    
    faq_statistics = relationship('FaqStatistics', back_populates='intent', cascade="all, delete-orphan")
    training_questions = relationship('TrainingQuestionAnswer', back_populates='intent', cascade="all, delete-orphan")


class FaqStatistics(Base):
    __tablename__ = 'FaqStatistics'
    
    faq_id = Column(Integer, primary_key=True, autoincrement=True)
    usage_count = Column(Integer, default=0)
    success_rate = Column(Float)
    question_text = Column(String)
    answer_text = Column(String)
    rating = Column(Integer)
    last_used_at = Column(Date)
    intent_id = Column(Integer, ForeignKey('Intent.intent_id'))
    
    intent = relationship('Intent', back_populates='faq_statistics')


class PersonalizedRecommendation(Base):
    __tablename__ = 'PersonalizedRecommendation'
    
    recommendation_id = Column(Integer, primary_key=True, autoincrement=True)
    confidence_score = Column(Float)
    user_id = Column(Integer, ForeignKey('Users.user_id'))
    base_intent_id = Column(Integer, ForeignKey("Intent.intent_id"))
    suggested_intent_id = Column(Integer, ForeignKey("Intent.intent_id"))
    session_id = Column(Integer, ForeignKey("ChatSession.chat_session_id"))
    
    user = relationship('Users', back_populates='personalized_recommendations')


class TrainingQuestionAnswer(Base):
    __tablename__ = 'TrainingQuestionAnswer'
    
    question_id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(String)
    answer = Column(String)
    intent_id = Column(Integer, ForeignKey("Intent.intent_id"))
    created_by = Column(Integer, ForeignKey("Users.user_id"))
    approved_by = Column(Integer, ForeignKey("Users.user_id"))
    
    # Relationships
    intent = relationship("Intent", back_populates="training_questions")

    created_by_user = relationship(
        "Users", foreign_keys=[created_by], back_populates="training_question_answers_created"
    )
    approved_by_user = relationship(
        "Users", foreign_keys=[approved_by], back_populates="training_question_answers_approved"
    )


# -------------------- AdmissionInformation ---------------------------
class AdmissionInformation(Base):
    __tablename__ = "AdmissionInformation"

    id = Column(Integer, primary_key=True, autoincrement=True)
    academic_year = Column(String)
    target_applicant = Column(String)
    admission_method = Column(String)
    scholarship_infor = Column(String)
    create_at = Column(Date, default=datetime.now)
    update_at = Column(Date, onupdate=datetime.now)
    created_by = Column(Integer, ForeignKey("Users.user_id"))

    # Relationships
    creator = relationship("Users", back_populates="admission_informations")
# ---------------------------------------------------------------------


# --------------- KnowledgeBaseDocument & DocumentChunk ----------------
class KnowledgeBaseDocument(Base):
    __tablename__ = 'KnowledgeBaseDocument'
    
    document_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String)
    file_path = Column(String)
    category = Column(String)
    created_at = Column(Date, default=datetime.now)
    updated_at = Column(Date, onupdate=datetime.now)
    created_by = Column(Integer, ForeignKey('Users.user_id'))
    
    # Relationships
    chunks = relationship('DocumentChunk', back_populates='document', cascade="all, delete-orphan")
    author = relationship('Users', back_populates='knowledge_documents')
    

class DocumentChunk(Base):
    __tablename__ = 'DocumentChunk'
    
    chunk_id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_text = Column(Text)
    embedding_vector = Column(String)  # Store as JSON or use vector extension
    created_at = Column(Date, default=datetime.now)
    document_id = Column(Integer, ForeignKey('KnowledgeBaseDocument.document_id'))
    created_by = Column(Integer, ForeignKey('Users.user_id'), nullable=True)

    # Relationships
    document = relationship('KnowledgeBaseDocument', back_populates='chunks')
    created_by_user = relationship('Users', back_populates='document_chunks')
# ---------------------------------------------------------------------


# -------------------- Template & Template_Field ----------------------
class Template(Base):
    __tablename__ = 'Template'
    
    template_id = Column(Integer, primary_key=True, autoincrement=True)
    template_name = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_by = Column(Integer, ForeignKey('Users.user_id'))
    
    # Relationships
    template_fields = relationship('Template_Field', back_populates='template', cascade="all, delete-orphan")
    creator = relationship('Users', back_populates='templates')


class Template_Field(Base):
    __tablename__ = 'Template_Field'
    
    template_field_id = Column(Integer, primary_key=True, autoincrement=True)
    field_name = Column(String)
    order_field = Column(Integer)
    field_type = Column(String)
    template_id = Column(Integer, ForeignKey("Template.template_id"))
    
    # Relationships
    template = relationship('Template', back_populates='template_fields')
# ---------------------------------------------------------------------


# ---------------- Specialization & Article ---------------------------
class Specialization(Base):
    __tablename__ = 'Specialization'
    
    specialization_id = Column(Integer, primary_key=True, autoincrement=True)
    specialization_name = Column(String, nullable=False)
    major_id = Column(Integer, ForeignKey('Major.major_id'), nullable=True)
    
    articles = relationship('Article', back_populates='specialization', cascade="all, delete-orphan")


class Article(Base):
    __tablename__ = "Article"

    article_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String)
    description = Column(String)
    url = Column(String)
    note = Column(String)
    # content = Column(Text)
    status = Column(String, default="draft")  # Values: draft, published, rejected, cancelled
    create_at = Column(Date, default=datetime.now)
    created_by = Column(Integer, ForeignKey("Users.user_id"))
    major_id = Column(Integer, ForeignKey('Major.major_id'), nullable=True)
    specialization_id = Column(Integer, ForeignKey('Specialization.specialization_id'), nullable=True)

    # Relationships
    author_user = relationship('Users', back_populates='articles')
    major = relationship('Major', back_populates='articles')
    specialization = relationship('Specialization', back_populates='articles')
# ---------------------------------------------------------------------