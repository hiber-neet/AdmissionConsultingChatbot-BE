from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
from datetime import datetime, timedelta
from typing import List, Optional
import re
from app.models.database import get_db
from app.models import entities
from app.core.security import get_current_user, has_permission

router = APIRouter()

def check_analytics_permission(current_user: entities.Users = Depends(get_current_user)):
    """Check if user has permission to view analytics (Admin, Consultant, or Content Manager)"""
    if not current_user:
        raise HTTPException(status_code=403, detail="Not authenticated")

    # Use the standard has_permission function which handles admin bypassing
    # Permission names should match the database exactly (with spaces)
    is_admin = has_permission(current_user, "Admin")
    is_consultant = has_permission(current_user, "Consultant")  
    is_content_manager = has_permission(current_user, "Content Manager")

    print(f"DEBUG: User {current_user.user_id} - admin:{is_admin}, consultant:{is_consultant}, content_manager:{is_content_manager}")

    if not (is_admin or is_consultant or is_content_manager):
        # Debug: show actual permissions
        if current_user.permissions:
            actual_perms = [p.permission_name for p in current_user.permissions]
            print(f"DEBUG: Permission denied for user {current_user.user_id} with permissions {actual_perms}")
        else:
            print(f"DEBUG: Permission denied for user {current_user.user_id} - no permissions loaded")
            
        raise HTTPException(
            status_code=403,
            detail="Admin, Consultant, or Content Manager permission required"
        )
    
    return current_user

@router.get("/knowledge-gaps")
async def get_knowledge_gaps(
    days: int = Query(30, description="Number of days to look back"),
    min_frequency: int = Query(3, description="Minimum frequency to be considered a gap"),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_analytics_permission)
):
    """
    Get knowledge gaps - frequent user questions not covered in training data
    """
    try:
        # Calculate date threshold (convert to date for comparison)
        date_threshold = (datetime.now() - timedelta(days=days)).date()
        
        # Get all user questions from chat interactions with detailed temporal data
        user_questions = db.query(
            entities.ChatInteraction.message_text,
            func.count(entities.ChatInteraction.message_text).label('frequency'),
            func.max(entities.ChatInteraction.timestamp).label('last_asked'),
            func.min(entities.ChatInteraction.timestamp).label('first_asked')
        ).filter(
            and_(
                entities.ChatInteraction.is_from_bot == False,
                entities.ChatInteraction.timestamp >= date_threshold,
                entities.ChatInteraction.message_text.isnot(None),
                func.length(entities.ChatInteraction.message_text) > 10  # Filter out very short messages
            )
        ).group_by(
            entities.ChatInteraction.message_text
        ).having(
            func.count(entities.ChatInteraction.message_text) >= min_frequency
        ).order_by(
            desc('frequency')
        ).limit(20).all()
        
        # Get existing training questions with intent information (only approved)
        existing_training = db.query(
            entities.TrainingQuestionAnswer.question,
            entities.TrainingQuestionAnswer.intent_id,
            entities.Intent.intent_name
        ).outerjoin(
            entities.Intent, 
            entities.TrainingQuestionAnswer.intent_id == entities.Intent.intent_id
        ).filter(
            entities.TrainingQuestionAnswer.question.isnot(None),
            entities.TrainingQuestionAnswer.status == 'approved'
        ).all()
        
        existing_q_texts = [q.question.lower().strip() for q in existing_training if q.question]
        
        knowledge_gaps = []
        for idx, (question_text, frequency, last_asked, first_asked) in enumerate(user_questions):
            if not question_text:
                continue
                
            # Enhanced similarity check with multiple approaches
            question_lower = question_text.lower().strip()
            question_words = set(question_lower.split())
            
            is_covered = False
            best_match_score = 0
            best_match_intent_id = None
            best_match_intent_name = None
            
            # Check against each training question
            for training_q, intent_id, intent_name in existing_training:
                if not training_q:
                    continue
                training_q_lower = training_q.lower().strip()
                training_words = set(training_q_lower.split())
                
                # Method 1: Word overlap (existing)
                word_overlap = len(question_words & training_words)
                overlap_score = word_overlap / max(len(question_words), len(training_words), 1)
                
                # Method 2: Substring similarity
                substring_score = 0
                if training_q_lower in question_lower or question_lower in training_q_lower:
                    substring_score = 0.8
                
                # Method 3: Key phrase matching
                key_phrase_score = 0
                question_key_phrases = [phrase.strip() for phrase in question_lower.replace('?', '').split() if len(phrase) > 3]
                training_key_phrases = [phrase.strip() for phrase in training_q_lower.replace('?', '').split() if len(phrase) > 3]
                
                if question_key_phrases and training_key_phrases:
                    common_key_phrases = set(question_key_phrases) & set(training_key_phrases)
                    if common_key_phrases:
                        key_phrase_score = len(common_key_phrases) / len(question_key_phrases)
                
                # Combined similarity score
                combined_score = max(overlap_score, substring_score, key_phrase_score)
                
                # Consider it covered if similarity > 0.6 OR word overlap > 2
                if combined_score > 0.6 or word_overlap > 2:
                    is_covered = True
                    best_match_score = combined_score
                    best_match_intent_id = intent_id
                    best_match_intent_name = intent_name
                    break
                    
                # Track best match even if not covered
                if combined_score > best_match_score:
                    best_match_score = combined_score
                    best_match_intent_id = intent_id
                    best_match_intent_name = intent_name
            
            # Smart Grace Period Logic using temporal patterns
            grace_period_needed = False
            question_span_days = 0
            
            if first_asked and last_asked:
                question_span_days = (last_asked - first_asked).days
                recent_activity = (datetime.now().date() - last_asked).days
                
                # Grace period logic based on question patterns:
                # 1. If question was asked over multiple days, it shows persistence
                # 2. If recently asked (within 7 days), might need time for training to take effect
                # 3. If training data exists with partial match, give it time to prove effectiveness
                
                if (question_span_days >= 3 and recent_activity <= 7) or (best_match_score > 0.3 and recent_activity <= 3):
                    grace_period_needed = True
                    is_covered = False  # Keep in list during grace period
            
            if not is_covered:
                # Use intent name from best matching training question, or "Unclassified" if no match
                intent_name = best_match_intent_name if best_match_intent_name else "Unclassified"
                
                # Enhanced suggested action
                suggested_action = "Create comprehensive answer for this question"
                if grace_period_needed:
                    suggested_action = "Monitor effectiveness - recent activity detected"
                elif best_match_score > 0.3:
                    suggested_action = f"Improve existing answer - partial match found (Intent: {intent_name})"
                elif best_match_intent_name:
                    suggested_action = f"Consider adding to '{intent_name}' intent"
                
                knowledge_gaps.append({
                    "id": idx + 1,
                    "question": question_text,
                    "frequency": frequency,
                    "intent_id": best_match_intent_id,
                    "intent_name": intent_name,
                    "suggestedAction": suggested_action,
                    "last_asked": last_asked.strftime('%Y-%m-%d') if last_asked else None,
                    "first_asked": first_asked.strftime('%Y-%m-%d') if first_asked else None,
                    "question_span_days": question_span_days,
                    "match_score": best_match_score,
                    "in_grace_period": grace_period_needed
                })
        
        return knowledge_gaps
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing knowledge gaps: {str(e)}")

@router.get("/recent-questions")
async def get_recent_questions(
    limit: int = Query(5, description="Number of recent questions to return"),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_analytics_permission)
):
    """
    Get the most recent questions asked to the chatbot (from chatbot sessions only)
    """
    try:
        # Get the most recent user questions from chatbot sessions only
        # Join with ChatSession to filter by session_type = 'chatbot'
        recent_interactions = (
            db.query(entities.ChatInteraction)
            .join(entities.ChatSession, entities.ChatInteraction.session_id == entities.ChatSession.chat_session_id)
            .filter(entities.ChatSession.session_type == 'chatbot')
            .filter(entities.ChatInteraction.is_from_bot == False)
            .filter(entities.ChatInteraction.message_text.isnot(None))
            .order_by(desc(entities.ChatInteraction.timestamp))
            .limit(limit)
            .all()
        )
        
        questions = []
        for interaction in recent_interactions:
            # Get user info if available
            user_name = "Anonymous"
            if interaction.sender_id:
                user = db.query(entities.Users).filter(entities.Users.user_id == interaction.sender_id).first()
                if user:
                    user_name = user.full_name or user.username
            
            questions.append({
                "id": interaction.interaction_id,
                "question": interaction.message_text,
                "timestamp": interaction.timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(interaction.timestamp, datetime) else str(interaction.timestamp),
                "user_name": user_name,
                "rating": interaction.rating
            })
        
        return {
            "status": "success",
            "data": questions,
            "message": f"Retrieved {len(questions)} recent chatbot questions"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching recent questions: {str(e)}")

@router.get("/user-questions")
async def get_user_questions(
    days: int = Query(30, ge=1, le=90, description="Number of days to look back (max 90 days)"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(10, ge=1, le=100, description="Number of questions per page"),
    search: Optional[str] = Query(None, description="Search query to filter questions"),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_analytics_permission)
):
    """
    Get paginated user questions from chatbot sessions only (non-bot messages).
    Limited to maximum 90 days to prevent loading too much data.
    
    Status determination:
    - "answered": Question matches a training question in the database (exact or similar match)
    - "unanswered": Question does not have a corresponding training question
    """
    try:
        # Enforce maximum 90 days limit
        days = min(days, 90)
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Base query: get user questions from chatbot sessions
        base_query = (
            db.query(
                entities.ChatInteraction.interaction_id,
                entities.ChatInteraction.message_text,
                entities.ChatInteraction.timestamp
            )
            .join(entities.ChatSession, entities.ChatInteraction.session_id == entities.ChatSession.chat_session_id)
            .filter(
                entities.ChatSession.session_type == 'chatbot',
                entities.ChatInteraction.is_from_bot == False,
                entities.ChatInteraction.message_text.isnot(None),
                entities.ChatInteraction.timestamp >= cutoff_date
            )
        )
        
        # Apply search filter if provided
        if search:
            base_query = base_query.filter(
                entities.ChatInteraction.message_text.ilike(f'%{search}%')
            )
        
        # Get total count
        total_count = base_query.count()
        
        # Calculate pagination
        total_pages = (total_count + page_size - 1) // page_size
        offset = (page - 1) * page_size
        
        # Get paginated results
        questions = (
            base_query
            .order_by(desc(entities.ChatInteraction.timestamp))
            .limit(page_size)
            .offset(offset)
            .all()
        )
        
        # Format response and try to find matching intent from training data
        formatted_questions = []
        for q in questions:
            # Try to find matching training question to determine intent and answered status
            intent_name = "Uncategorized"
            has_answer = False
            
            # Look for matching training questions (exact match or similar, only approved)
            training_match = (
                db.query(entities.TrainingQuestionAnswer, entities.Intent.intent_name)
                .join(entities.Intent, entities.TrainingQuestionAnswer.intent_id == entities.Intent.intent_id)
                .filter(
                    or_(
                        entities.TrainingQuestionAnswer.question == q.message_text,
                        entities.TrainingQuestionAnswer.question.ilike(f'%{q.message_text}%')
                    ),
                    entities.TrainingQuestionAnswer.status == 'approved'
                )
                .first()
            )
            
            if training_match:
                intent_name = training_match[1]
                has_answer = True
            
            formatted_questions.append({
                "id": q.interaction_id,
                "question": q.message_text,
                "category": intent_name,
                "timestamp": q.timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(q.timestamp, datetime) else str(q.timestamp),
                "status": "answered" if has_answer else "unanswered"
            })
        
        return {
            "status": "success",
            "data": formatted_questions,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            },
            "message": f"Retrieved {len(formatted_questions)} user questions"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user questions: {str(e)}")

@router.get("/low-satisfaction-answers")
async def get_low_satisfaction_answers(
    threshold: float = Query(3.5, description="Satisfaction threshold below which answers are considered low"),
    min_usage: int = Query(5, description="Minimum usage count to be considered"),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_analytics_permission)
):
    """
    Get Q&A pairs with low user satisfaction ratings
    """
    try:
        # Get FAQ statistics with low ratings or success rates
        low_satisfaction_faqs = db.query(entities.FaqStatistics).filter(
            and_(
                or_(
                    entities.FaqStatistics.rating < threshold,
                    entities.FaqStatistics.success_rate < 0.7
                ),
                entities.FaqStatistics.usage_count >= min_usage,
                entities.FaqStatistics.question_text.isnot(None),
                entities.FaqStatistics.answer_text.isnot(None)
            )
        ).order_by(desc(entities.FaqStatistics.usage_count)).limit(10).all()
        
        # Also check chat interactions with low ratings
        low_rated_interactions = db.query(
            entities.ChatInteraction.message_text,
            func.avg(entities.ChatInteraction.rating).label('avg_rating'),
            func.count(entities.ChatInteraction.rating).label('rating_count')
        ).filter(
            and_(
                entities.ChatInteraction.is_from_bot == True,  # Bot responses
                entities.ChatInteraction.rating.isnot(None),
                entities.ChatInteraction.rating < threshold
            )
        ).group_by(
            entities.ChatInteraction.message_text
        ).having(
            func.count(entities.ChatInteraction.rating) >= 3  # At least 3 ratings
        ).order_by(desc('rating_count')).limit(10).all()
        
        confusing_answers = []
        
        # Process FAQ statistics
        for idx, faq in enumerate(low_satisfaction_faqs):
            feedback = "Users report answer needs improvement"
            if faq.rating and faq.rating < 2:
                feedback = "Users frequently rate this answer as unhelpful"
            elif faq.success_rate and faq.success_rate < 0.5:
                feedback = "Low success rate indicates users don't find this helpful"
            elif faq.rating and faq.rating < 3:
                feedback = "Users report answer could be clearer"
                
            suggestion = "Review and improve answer clarity, add more specific details"
            if "admission" in faq.question_text.lower():
                suggestion = "Provide specific admission requirements and timelines"
            elif "program" in faq.question_text.lower():
                suggestion = "Include detailed program information and requirements"
            
            confusing_answers.append({
                "id": idx + 1,
                "question": faq.question_text,
                "currentSatisfaction": round(faq.rating or 2.5, 1),
                "targetSatisfaction": 4.5,
                "feedback": feedback,
                "suggestion": suggestion,
                "usage_count": faq.usage_count,
                "success_rate": faq.success_rate
            })
        
        # Process low-rated chat interactions
        for idx, (message_text, avg_rating, rating_count) in enumerate(low_rated_interactions):
            if len(confusing_answers) >= 15:  # Limit total results
                break
                
            confusing_answers.append({
                "id": len(confusing_answers) + 1,
                "question": f"Question related to: {message_text[:100]}...",
                "currentSatisfaction": round(float(avg_rating), 1),
                "targetSatisfaction": 4.5,
                "feedback": f"Based on {rating_count} user ratings - users find this response unsatisfactory",
                "suggestion": "Review chat logs and improve response quality",
                "usage_count": rating_count,
                "success_rate": None
            })
        
        return confusing_answers
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing low satisfaction answers: {str(e)}")

@router.get("/trending-topics")
async def get_trending_topics(
    days: int = Query(14, description="Number of days to analyze for trends"),
    min_questions: int = Query(5, description="Minimum questions to be considered trending"),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_analytics_permission)
):
    """
    Get trending topics based on recent question patterns
    """
    try:
        # Calculate date thresholds (convert to date for comparison)
        recent_date = (datetime.now() - timedelta(days=days)).date()
        historical_date = (datetime.now() - timedelta(days=days*2)).date()
        
        # Define topic keywords for categorization
        topics_keywords = {
            "Study Abroad Programs": ["abroad", "exchange", "international", "overseas", "study abroad"],
            "AI and Computer Science": ["ai", "artificial intelligence", "computer science", "programming", "software", "tech"],
            "Admissions Process": ["admission", "apply", "application", "deadline", "requirement"],
            "Financial Aid": ["scholarship", "financial aid", "tuition", "fee", "cost", "money"],
            "Campus Life": ["campus", "dormitory", "housing", "student life", "facilities"],
            "Online Learning": ["online", "remote", "virtual", "distance learning", "e-learning"],
            "Career Services": ["career", "job", "internship", "employment", "placement"]
        }
        
        trending_topics = []
        
        for topic_name, keywords in topics_keywords.items():
            # Build keyword search condition
            keyword_conditions = []
            for keyword in keywords:
                keyword_conditions.append(
                    entities.ChatInteraction.message_text.ilike(f'%{keyword}%')
                )
            
            # Count recent questions
            recent_count = db.query(func.count(entities.ChatInteraction.interaction_id)).filter(
                and_(
                    entities.ChatInteraction.is_from_bot == False,
                    entities.ChatInteraction.timestamp >= recent_date,
                    or_(*keyword_conditions)
                )
            ).scalar() or 0
            
            # Count historical questions for comparison
            historical_count = db.query(func.count(entities.ChatInteraction.interaction_id)).filter(
                and_(
                    entities.ChatInteraction.is_from_bot == False,
                    entities.ChatInteraction.timestamp >= historical_date,
                    entities.ChatInteraction.timestamp < recent_date,
                    or_(*keyword_conditions)
                )
            ).scalar() or 0
            
            if recent_count >= min_questions:
                # Calculate growth rate
                if historical_count > 0:
                    growth_rate = round(((recent_count - historical_count) / historical_count) * 100)
                else:
                    growth_rate = 100 if recent_count > 0 else 0
                
                # Only include if showing growth or significant volume
                if growth_rate > 20 or recent_count >= min_questions * 2:
                    # Generate description and action based on topic
                    descriptions = {
                        "Study Abroad Programs": "Growing interest in international exchange opportunities and semester abroad programs",
                        "AI and Computer Science": "Increased inquiries about AI curriculum, computer science programs and tech career paths",
                        "Admissions Process": "Rising questions about application procedures, requirements and deadlines",
                        "Financial Aid": "More students seeking information about scholarships and financial support options",
                        "Campus Life": "Increased interest in campus facilities, housing options and student activities",
                        "Online Learning": "Growing demand for information about remote and hybrid learning options",
                        "Career Services": "Rising questions about career support, internships and job placement services"
                    }
                    
                    actions = {
                        "Study Abroad Programs": "Create dedicated section for international programs and partnerships",
                        "AI and Computer Science": "Highlight AI specializations and computer science curriculum details",
                        "Admissions Process": "Expand admission requirements and application process documentation", 
                        "Financial Aid": "Create comprehensive financial aid and scholarship information guide",
                        "Campus Life": "Document campus facilities, housing options and student life activities",
                        "Online Learning": "Add information about online and hybrid program offerings",
                        "Career Services": "Expand career services and internship opportunity information"
                    }
                    
                    trending_topics.append({
                        "id": len(trending_topics) + 1,
                        "topic": topic_name,
                        "growthRate": max(growth_rate, 0),  # Ensure non-negative
                        "questionsCount": recent_count,
                        "description": descriptions.get(topic_name, f"Increased activity in {topic_name.lower()} related questions"),
                        "action": actions.get(topic_name, f"Create more content about {topic_name.lower()}"),
                        "timeframe": f"last {days} days"
                    })
        
        # Sort by growth rate and question count
        trending_topics.sort(key=lambda x: (x["growthRate"], x["questionsCount"]), reverse=True)
        
        return trending_topics[:10]  # Return top 10
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing trending topics: {str(e)}")

@router.get("/content-statistics")
async def get_content_statistics(
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_analytics_permission)
):
    """
    Get content statistics for content managers
    """
    try:
        # Total articles count
        total_articles = db.query(func.count(entities.Article.article_id)).scalar() or 0
        
        # Published articles count
        published_articles = db.query(func.count(entities.Article.article_id)).filter(
            entities.Article.status == 'published'
        ).scalar() or 0
        
        # Draft articles count
        draft_articles = db.query(func.count(entities.Article.article_id)).filter(
            entities.Article.status == 'draft'
        ).scalar() or 0
        
        # Review articles count (assuming 'review' or 'pending' status)
        review_articles = db.query(func.count(entities.Article.article_id)).filter(
            or_(entities.Article.status == 'review', entities.Article.status == 'pending')
        ).scalar() or 0
        
        # Recent articles (last 10)
        recent_articles = db.query(entities.Article).order_by(
            desc(entities.Article.create_at)
        ).limit(10).all()
        
        # Articles by major
        articles_by_major = db.query(
            entities.Major.major_name,
            func.count(entities.Article.article_id).label('article_count')
        ).join(
            entities.Article, entities.Major.major_id == entities.Article.major_id
        ).group_by(
            entities.Major.major_name
        ).all()
        
        # Monthly trends - get articles created in last 6 months
        six_months_ago = datetime.now() - timedelta(days=180)
        monthly_articles = db.query(
            func.date_trunc('month', entities.Article.create_at).label('month'),
            func.count(entities.Article.article_id).label('total_articles')
        ).filter(
            entities.Article.create_at >= six_months_ago.date()
        ).group_by(
            func.date_trunc('month', entities.Article.create_at)
        ).order_by('month').all()
        
        # Get published articles count separately for each month
        published_monthly = db.query(
            func.date_trunc('month', entities.Article.create_at).label('month'),
            func.count(entities.Article.article_id).label('published_articles')
        ).filter(
            and_(
                entities.Article.create_at >= six_months_ago.date(),
                entities.Article.status == 'published'
            )
        ).group_by(
            func.date_trunc('month', entities.Article.create_at)
        ).all()
        
        # Combine the results
        published_dict = {month: count for month, count in published_monthly}
        monthly_trends = []
        for month, total_articles in monthly_articles:
            published_count = published_dict.get(month, 0)
            monthly_trends.append((month, total_articles, published_count))
        
        # Status distribution
        status_distribution = {}
        statuses = db.query(
            entities.Article.status,
            func.count(entities.Article.article_id)
        ).group_by(entities.Article.status).all()
        
        for status, count in statuses:
            status_distribution[status] = count
        
        return {
            "success": True,
            "data": {
                "overview": {
                    "total_articles": total_articles,
                    "published_articles": published_articles,
                    "draft_articles": draft_articles,
                    "review_articles": review_articles,
                    "my_articles": total_articles  # For now, assume all articles are "my articles"
                },
                "recent_articles": [
                    {
                        "article_id": article.article_id,
                        "title": article.title,
                        "author": "Admin",  # You might want to get actual author info
                        "status": article.status,
                        "created_at": article.create_at.isoformat() if article.create_at else None,
                        "major_id": article.major_id,
                        "specialization_id": article.specialization_id
                    }
                    for article in recent_articles
                ],
                "popular_articles": [
                    {
                        "article_id": article.article_id,
                        "title": article.title,
                        "author": "Admin",
                        "created_at": article.create_at.isoformat() if article.create_at else None,
                        "view_count": 0,  # You might want to add view tracking
                        "url": f"/articles/{article.article_id}"
                    }
                    for article in recent_articles[:5]  # Use recent articles as popular for now
                ],
                "articles_by_major": [
                    {
                        "major_name": major_name,
                        "article_count": article_count
                    }
                    for major_name, article_count in articles_by_major
                ],
                "monthly_trends": [
                    {
                        "month": month.strftime('%Y-%m') if month else None,
                        "total_articles": int(total_articles or 0),
                        "published_articles": int(published_articles or 0)
                    }
                    for month, total_articles, published_articles in monthly_trends
                ],
                "status_distribution": status_distribution,
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting content statistics: {str(e)}")

@router.get("/consultant-statistics")
async def get_consultant_statistics(
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_analytics_permission)
):
    """
    Get consultant dashboard statistics
    """
    try:
        # Total queries count (chatbot sessions only)
        total_queries = db.query(func.count(entities.ChatInteraction.interaction_id)).join(
            entities.ChatSession, entities.ChatInteraction.session_id == entities.ChatSession.chat_session_id
        ).filter(
            entities.ChatSession.session_type == 'chatbot'
        ).filter(
            entities.ChatInteraction.is_from_bot == False
        ).scalar() or 0
        
        # Queries in last 30 days for growth calculation
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_queries = db.query(func.count(entities.ChatInteraction.interaction_id)).join(
            entities.ChatSession, entities.ChatInteraction.session_id == entities.ChatSession.chat_session_id
        ).filter(
            entities.ChatSession.session_type == 'chatbot'
        ).filter(
            and_(
                entities.ChatInteraction.is_from_bot == False,
                entities.ChatInteraction.timestamp >= thirty_days_ago.date()
            )
        ).scalar() or 0
        
        # Previous 30 days for comparison
        sixty_days_ago = datetime.now() - timedelta(days=60)
        previous_queries = db.query(func.count(entities.ChatInteraction.interaction_id)).join(
            entities.ChatSession, entities.ChatInteraction.session_id == entities.ChatSession.chat_session_id
        ).filter(
            entities.ChatSession.session_type == 'chatbot'
        ).filter(
            and_(
                entities.ChatInteraction.is_from_bot == False,
                entities.ChatInteraction.timestamp >= sixty_days_ago.date(),
                entities.ChatInteraction.timestamp < thirty_days_ago.date()
            )
        ).scalar() or 0
        
        # Calculate growth rate
        queries_growth = 0
        if previous_queries > 0:
            queries_growth = round(((recent_queries - previous_queries) / previous_queries) * 100)
        elif recent_queries > 0:
            queries_growth = 100
        
        # Mock accuracy rate (you might want to calculate this based on ratings)
        accuracy_rate = 85  # Mock value
        accuracy_improvement = 5  # Mock value
        
        # Most active day (since we only have date, not time) - chatbot sessions only
        # Get the day of week with most questions in the last 30 days
        day_counts = {}
        day_names = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN']
        recent_interactions = db.query(entities.ChatInteraction.timestamp).join(
            entities.ChatSession, entities.ChatInteraction.session_id == entities.ChatSession.chat_session_id
        ).filter(
            entities.ChatSession.session_type == 'chatbot'
        ).filter(
            and_(
                entities.ChatInteraction.is_from_bot == False,
                entities.ChatInteraction.timestamp >= thirty_days_ago.date()
            )
        ).all()
        
        for interaction in recent_interactions:
            if interaction.timestamp:
                # Get day of week (0 = Monday, 6 = Sunday)
                day_of_week = interaction.timestamp.weekday() if hasattr(interaction.timestamp, 'weekday') else 0
                day_name = day_names[day_of_week]
                day_counts[day_name] = day_counts.get(day_name, 0) + 1
        
        # Find most active day
        most_active_time = max(day_counts.items(), key=lambda x: x[1])[0] if day_counts else "Monday"
        
        # Unanswered queries (knowledge gaps) - get actual count from knowledge gaps analysis
        # Get user questions from the last 30 days
        user_questions = (
            db.query(
                entities.ChatInteraction.message_text,
                func.count(entities.ChatInteraction.interaction_id).label('frequency')
            )
            .filter(entities.ChatInteraction.is_from_bot == False)
            .filter(entities.ChatInteraction.timestamp >= thirty_days_ago.date())
            .filter(entities.ChatInteraction.message_text.isnot(None))
            .filter(func.length(entities.ChatInteraction.message_text) > 10)  # Filter out very short messages
            .group_by(entities.ChatInteraction.message_text)
            .having(func.count(entities.ChatInteraction.interaction_id) >= 3)  # min_frequency = 3
            .all()
        )
        
        # Get all training questions for comparison (only approved)
        training_questions = db.query(entities.TrainingQuestionAnswer.question).filter(
            entities.TrainingQuestionAnswer.status == 'approved'
        ).all()
        training_set = {q.question.lower().strip() for q in training_questions if q.question}
        
        # Count knowledge gaps (user questions not in training data)
        unanswered_queries = 0
        for user_q, frequency in user_questions:
            if user_q:
                user_q_clean = user_q.lower().strip()
                # Check if question is in training data
                if user_q_clean not in training_set:
                    # Check for partial matches
                    best_match_score = 0
                    for train_q in training_set:
                        common_words = set(user_q_clean.split()) & set(train_q.split())
                        if common_words:
                            score = len(common_words) / max(len(user_q_clean.split()), len(train_q.split()))
                            best_match_score = max(best_match_score, score)
                    
                    # If no good match found (< 60% similarity), count as knowledge gap
                    if best_match_score < 0.6:
                        unanswered_queries += 1
        
        # Questions over time (last 7 days) - chatbot sessions only
        seven_days_ago = datetime.now() - timedelta(days=7)
        questions_over_time = []
        for i in range(7):
            day = seven_days_ago + timedelta(days=i)
            day_queries = db.query(func.count(entities.ChatInteraction.interaction_id)).join(
                entities.ChatSession, entities.ChatInteraction.session_id == entities.ChatSession.chat_session_id
            ).filter(
                entities.ChatSession.session_type == 'chatbot'
            ).filter(
                and_(
                    entities.ChatInteraction.is_from_bot == False,
                    func.date(entities.ChatInteraction.timestamp) == day.date()
                )
            ).scalar() or 0
            
            questions_over_time.append({
                "date": day.strftime('%Y-%m-%d'),
                "queries": day_queries
            })
        
        # Question categories - using actual data from database
        # Strategy: Use FaqStatistics which tracks question-answer pairs with intent_id and usage_count
        # This provides real data from actual chatbot interactions
        
        # Get question categories from FaqStatistics (real usage data from chatbot)
        # FaqStatistics.usage_count represents how many times each Q&A pair was used
        faq_stats_by_intent = (
            db.query(
                entities.Intent.intent_name,
                func.sum(entities.FaqStatistics.usage_count).label('total_count')
            )
            .join(entities.FaqStatistics, entities.Intent.intent_id == entities.FaqStatistics.intent_id)
            .filter(entities.FaqStatistics.usage_count > 0)
            .group_by(entities.Intent.intent_name)
            .all()
        )
        
        # Category colors mapping
        category_colors = {
            "Admissions": "#3B82F6",
            "Academic Programs": "#10B981",
            "Financial Aid": "#F59E0B",
            "Campus Life": "#8B5CF6",
            "Career Services": "#EF4444",
            "Other": "#6B7280"
        }
        
        # Map intent names to categories
        def map_intent_to_category(intent_name: str) -> str:
            """Map intent name to category based on keywords"""
            if not intent_name:
                return "Other"
            
            intent_lower = intent_name.lower()
            
            # Check for admission-related keywords
            if any(kw in intent_lower for kw in ['admission', 'tuyển sinh', 'hồ sơ', 'nộp đơn', 'application', 'apply', 'tuyển', 'điều kiện']):
                return "Admissions"
            # Check for academic/program-related keywords
            elif any(kw in intent_lower for kw in ['academic', 'program', 'major', 'ngành', 'chương trình', 'học', 'course', 'curriculum', 'degree', 'chuyên ngành']):
                return "Academic Programs"
            # Check for financial aid-related keywords
            elif any(kw in intent_lower for kw in ['financial', 'scholarship', 'tuition', 'học phí', 'học bổng', 'fee', 'cost', 'tài chính']):
                return "Financial Aid"
            # Check for campus life-related keywords
            elif any(kw in intent_lower for kw in ['campus', 'life', 'housing', 'dormitory', 'ký túc', 'cơ sở', 'facility', 'student life']):
                return "Campus Life"
            # Check for career-related keywords
            elif any(kw in intent_lower for kw in ['career', 'job', 'internship', 'nghề nghiệp', 'việc làm', 'employment', 'tốt nghiệp']):
                return "Career Services"
            else:
                return "Other"
        
        # Initialize category counts
        category_counts = {
            "Admissions": 0,
            "Academic Programs": 0,
            "Financial Aid": 0,
            "Campus Life": 0,
            "Career Services": 0,
            "Other": 0
        }
        
        # Count questions by category from FaqStatistics (real usage data)
        # usage_count represents how many times each Q&A pair was actually used
        for intent_name, count in faq_stats_by_intent:
            category = map_intent_to_category(intent_name)
            category_counts[category] += int(count) if count else 0
        
        # If FaqStatistics is empty or has very little data, fallback to direct user questions analysis
        total_from_faq = sum(count for _, count in faq_stats_by_intent)
        if total_from_faq == 0:
            # Fallback: Get user questions directly and match with training questions
            user_questions = (
                db.query(
                    entities.ChatInteraction.message_text,
                    func.count(entities.ChatInteraction.interaction_id).label('frequency')
                )
                .join(entities.ChatSession, entities.ChatInteraction.session_id == entities.ChatSession.chat_session_id)
                .filter(
                    and_(
                        entities.ChatSession.session_type == 'chatbot',
                        entities.ChatInteraction.is_from_bot == False,
                        entities.ChatInteraction.timestamp >= thirty_days_ago.date(),
                        entities.ChatInteraction.message_text.isnot(None),
                        func.length(entities.ChatInteraction.message_text) > 10
                    )
                )
                .group_by(entities.ChatInteraction.message_text)
                .limit(200)  # Limit to avoid performance issues
                .all()
            )
            
            # Get training questions with intents for matching
            training_with_intents = (
                db.query(
                    entities.TrainingQuestionAnswer.question,
                    entities.Intent.intent_name
                )
                .join(entities.Intent, entities.TrainingQuestionAnswer.intent_id == entities.Intent.intent_id)
                .filter(entities.TrainingQuestionAnswer.status == 'approved')
                .all()
            )
            
            # Create a dictionary of training questions for faster lookup
            training_questions_dict = {}
            for train_q, intent_name in training_with_intents:
                if train_q:
                    train_q_lower = train_q.lower().strip()
                    training_questions_dict[train_q_lower] = intent_name
            
            # Match user questions with training questions and categorize
            for user_q, frequency in user_questions:
                if not user_q:
                    continue
                
                user_q_lower = user_q.lower().strip()
                matched = False
                
                # Try exact match first
                if user_q_lower in training_questions_dict:
                    intent_name = training_questions_dict[user_q_lower]
                    category = map_intent_to_category(intent_name)
                    category_counts[category] += frequency
                    matched = True
                else:
                    # Try partial/substring match
                    for train_q_lower, intent_name in training_questions_dict.items():
                        if train_q_lower in user_q_lower or user_q_lower in train_q_lower:
                            category = map_intent_to_category(intent_name)
                            category_counts[category] += frequency
                            matched = True
                            break
                        
                        # Check word overlap for similarity
                        user_words = set(user_q_lower.split())
                        train_words = set(train_q_lower.split())
                        common_words = user_words & train_words
                        if len(common_words) >= 2:
                            overlap_ratio = len(common_words) / max(len(user_words), len(train_words), 1)
                            if overlap_ratio > 0.5:
                                category = map_intent_to_category(intent_name)
                                category_counts[category] += frequency
                                matched = True
                                break
                
                # If not matched, try keyword-based classification
                if not matched:
                    user_q_lower = user_q.lower()
                    keyword_matched = False
                    
                    # Define category keywords for fallback
                    category_keywords = {
                        'Admissions': ['admission', 'application', 'requirement', 'deadline', 'gpa', 'apply', 'tuyển sinh', 'hồ sơ', 'điểm', 'yêu cầu', 'nộp', 'hạn chót'],
                        'Academic Programs': ['program', 'major', 'course', 'curriculum', 'degree', 'ngành', 'chuyên ngành', 'khóa học', 'chương trình', 'học phần'],
                        'Financial Aid': ['scholarship', 'tuition', 'fee', 'cost', 'học phí', 'học bổng', 'tài chính'],
                        'Campus Life': ['campus', 'dormitory', 'housing', 'facility', 'cơ sở', 'ký túc xá', 'câu lạc bộ'],
                        'Career Services': ['career', 'internship', 'job', 'employment', 'nghề nghiệp', 'việc làm', 'tốt nghiệp']
                    }
                    
                    for category, keywords in category_keywords.items():
                        if any(keyword in user_q_lower for keyword in keywords):
                            category_counts[category] += frequency
                            keyword_matched = True
                            break
                    
                    if not keyword_matched:
                        category_counts["Other"] += frequency
        
        # Build question_categories list with format: name, value, color
        question_categories = []
        for category_name, count in category_counts.items():
            if count > 0:  # Only include categories with questions
                question_categories.append({
                    "name": category_name,
                    "value": count,
                    "color": category_colors.get(category_name, "#6B7280")
                })
        
        # Sort by value (descending)
        question_categories.sort(key=lambda x: x["value"], reverse=True)
        
        return {
            "status": "success",
            "data": {
                "overview_stats": {
                    "total_queries": total_queries,
                    "queries_growth": queries_growth,
                    "accuracy_rate": accuracy_rate,
                    "accuracy_improvement": accuracy_improvement,
                    "most_active_time": most_active_time,
                    "unanswered_queries": unanswered_queries
                },
                "questions_over_time": questions_over_time,
                "question_categories": question_categories,
                "last_updated": datetime.now().isoformat()
            },
            "message": "Consultant statistics retrieved successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting consultant statistics: {str(e)}")

@router.get("/category-statistics")
async def get_category_statistics(
    days: int = Query(30, description="Number of days to look back"),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_analytics_permission)
):
    """
    Get category statistics - question distribution across categories with metrics
    """
    try:
        # Calculate date threshold
        date_threshold = (datetime.now() - timedelta(days=days)).date()
        
        # Get all user questions from chat interactions
        user_questions = db.query(
            entities.ChatInteraction.message_text,
            func.count(entities.ChatInteraction.interaction_id).label('frequency')
        ).filter(
            and_(
                entities.ChatInteraction.is_from_bot == False,
                entities.ChatInteraction.timestamp >= date_threshold
            )
        ).group_by(entities.ChatInteraction.message_text).all()
        
        # Define category keywords
        category_keywords = {
            'Admission Requirements': ['admission', 'application', 'requirement', 'deadline', 'gpa', 'grade', 'apply', 'entrance', 'tuyển sinh', 'hồ sơ', 'điểm', 'yêu cầu', 'thời hạn', 'nộp', 'hạn chót', 'thủ tục', 'tuyển', 'điều kiện', 'học bạ', 'thi', 'hồ sơ', 'tài liệu'],
            'Financial Aid': ['financial aid', 'scholarship', 'tuition', 'fee', 'cost', 'funding', 'loan', 'grant', 'học phí', 'tiền', 'bao nhiêu', 'học bổng', 'tài chính', 'vay', 'trợ cấp'],
            'Academic Programs': ['program', 'major', 'course', 'curriculum', 'degree', 'bachelor', 'master', 'subject', 'ngành', 'chuyên ngành', 'khóa học', 'chương trình', 'học phần', 'bằng cấp', 'cử nhân', 'thạc sĩ', 'tiến sĩ', 'môn học', 'khoa', 'đào tạo'],
            'Campus Life': ['campus', 'dormitory', 'housing', 'student life', 'activities', 'club', 'facility', 'tour', 'cơ sở', 'hoạt động', 'ký túc xá', 'câu lạc bộ', 'clb', 'khuôn viên', 'cơ sở vật chất', 'thư viện', 'nhà xe', 'canteen', 'nhà ăn', 'chỗ ở', 'ngoại khóa'],
            'Career Services': ['career', 'internship', 'job placement', 'employment', 'graduation rate', 'alumni', 'nghề nghiệp', 'việc làm', 'thực tập', 'tốt nghiệp', 'cựu sinh viên', 'hỗ trợ việc làm', 'tư vấn nghề nghiệp', 'đầu ra', 'công ty', 'doanh nghiệp', 'lương', 'tuyển dụng', 'ra trường'],
        }
        
        # Initialize category stats
        category_stats = {}
        for category in category_keywords.keys():
            category_stats[category] = {
                'category': category,
                'total_questions': 0,
                'total_times_asked': 0,
                'unique_questions': []
            }
        
        # Categorize questions
        for question_row in user_questions:
            question_text = question_row.message_text.lower()
            frequency = question_row.frequency
            categorized = False
            
            # Try to match with category keywords
            for category, keywords in category_keywords.items():
                if any(keyword in question_text for keyword in keywords):
                    category_stats[category]['total_questions'] += 1
                    category_stats[category]['total_times_asked'] += frequency
                    category_stats[category]['unique_questions'].append({
                        'question': question_row.message_text,
                        'frequency': frequency
                    })
                    categorized = True
                    break
            
            # If not categorized, put in "Other"
            if not categorized:
                if 'Khác' not in category_stats:
                    category_stats['Khác'] = {
                        'category': 'Khác',
                        'total_questions': 0,
                        'total_times_asked': 0,
                        'unique_questions': []
                    }
                category_stats['Khác']['total_questions'] += 1
                category_stats['Khác']['total_times_asked'] += frequency
                category_stats['Khác']['unique_questions'].append({
                    'question': question_row.message_text,
                    'frequency': frequency
                })
        
        # Convert to list and remove empty categories
        category_list = []
        for category_name, stats in category_stats.items():
            if stats['total_questions'] > 0:  # Only include categories with questions
                # Remove the unique_questions list from the response to keep it clean
                category_list.append({
                    'category': stats['category'],
                    'total_questions': stats['total_questions'],
                    'total_times_asked': stats['total_times_asked']
                })
        
        # Sort by total times asked (descending)
        category_list.sort(key=lambda x: x['total_times_asked'], reverse=True)
        
        return category_list
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting category statistics: {str(e)}")


# ==================== DASHBOARD ANALYTICS ====================

@router.get("/dashboard/metrics")
async def get_dashboard_metrics(
    days: int = Query(7, description="Number of days to look back"),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_analytics_permission)
):
    """
    Get key dashboard metrics:
    - Active chatbot sessions (chatbot sessions with no end_time)
    - Total customers (users with Student or Parent role)
    - Active live chat sessions (live sessions with no end_time)
    """
    try:
        # Active Chatbot Sessions: sessions with session_type = "chatbot" and end_time IS NULL
        active_chatbot_sessions = db.query(entities.ChatSession).filter(
            and_(
                entities.ChatSession.session_type == "chatbot",
                entities.ChatSession.end_time.is_(None)
            )
        ).count()
        
        # Total Customers: count users with role_name = "Student" or "Parent"
        total_customers = db.query(entities.Users).join(
            entities.Role, entities.Users.role_id == entities.Role.role_id
        ).filter(
            or_(
                entities.Role.role_name == "Student",
                entities.Role.role_name == "Parent",
                entities.Role.role_name == "Customer"
            )
        ).count()
        
        # Active Live Chat Sessions: sessions with session_type = "live" and end_time IS NULL
        active_live_sessions = db.query(entities.ChatSession).filter(
            and_(
                entities.ChatSession.session_type == "chatlive",
                entities.ChatSession.end_time.is_(None)
            )
        ).count()
        
        return {
            "active_chatbot_sessions": active_chatbot_sessions,
            "total_customers": total_customers,
            "active_live_sessions": active_live_sessions,
            "period_days": days
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting dashboard metrics: {str(e)}")


@router.get("/dashboard/chatbot-requests")
async def get_chatbot_requests(
    days: int = Query(30, description="Number of days to get data for"),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_analytics_permission)
):
    """
    Get total chatbot requests over the last 30 days.
    Shows customer messages vs chatbot responses in chatbot sessions only.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get chatbot session IDs first (session_type = "chatbot")
        chatbot_session_ids = db.query(entities.ChatSession.chat_session_id).filter(
            entities.ChatSession.session_type == "chatbot"
        ).subquery()
        
        # Get customer messages (is_from_bot = False) grouped by date
        customer_messages_by_date = db.query(
            func.date(entities.ChatInteraction.timestamp).label('date'),
            func.count(entities.ChatInteraction.interaction_id).label('customer_count')
        ).filter(
            and_(
                entities.ChatInteraction.session_id.in_(chatbot_session_ids),
                entities.ChatInteraction.is_from_bot == False,
                entities.ChatInteraction.timestamp >= start_date
            )
        ).group_by(
            func.date(entities.ChatInteraction.timestamp)
        ).all()
        
        # Get chatbot responses (is_from_bot = True) grouped by date
        chatbot_messages_by_date = db.query(
            func.date(entities.ChatInteraction.timestamp).label('date'),
            func.count(entities.ChatInteraction.interaction_id).label('chatbot_count')
        ).filter(
            and_(
                entities.ChatInteraction.session_id.in_(chatbot_session_ids),
                entities.ChatInteraction.is_from_bot == True,
                entities.ChatInteraction.timestamp >= start_date
            )
        ).group_by(
            func.date(entities.ChatInteraction.timestamp)
        ).all()
        
        # Create date range
        date_data = {}
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%a')  # Mon, Tue, Wed format
            date_key = current_date.date()
            date_data[date_str] = {
                'name': date_str,
                'customer': 0,  # Customer messages (green - was "Resolved")
                'chatbot': 0,   # Chatbot responses (blue - was "Total")
                'date': date_key
            }
            current_date += timedelta(days=1)
        
        # Fill in customer messages data
        for msg in customer_messages_by_date:
            date_str = msg.date.strftime('%a')
            if date_str in date_data:
                date_data[date_str]['customer'] = msg.customer_count
        
        # Fill in chatbot messages data
        for msg in chatbot_messages_by_date:
            date_str = msg.date.strftime('%a')
            if date_str in date_data:
                date_data[date_str]['chatbot'] = msg.chatbot_count
        
        # Convert to list and sort by date
        result = list(date_data.values())
        result.sort(key=lambda x: x['date'])
        
        # Remove the date key from response
        for item in result:
            del item['date']
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting chatbot requests: {str(e)}")


@router.get("/dashboard/admission-stats")
async def get_admission_dashboard_stats(
    days: int = Query(30, description="Number of days for chatbot interactions"),
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(get_current_user)
):
    """
    Get statistics specifically for admission officer dashboard:
    - Ended chatbot sessions (last 30 days)
    - Total published articles
    - Current live chat queue count for the officer
    - Drafted articles count
    - Weekly article publication stats (last 7 days)
    - Intent distribution from training questions
    """
    try:
        # Check if user is an admission officer
        if not has_permission(current_user, "Admission Official"):
            raise HTTPException(status_code=403, detail="Admission Official permission required")
        
        # 1. Ended Chatbot Sessions (last 30 days)
        date_threshold = datetime.now() - timedelta(days=days)
        ended_chatbot_sessions = db.query(entities.ChatSession).filter(
            and_(
                entities.ChatSession.session_type == "chatbot",
                entities.ChatSession.end_time.isnot(None),
                entities.ChatSession.end_time >= date_threshold
            )
        ).count()
        
        # 2. Total Published Articles
        published_articles = db.query(entities.Article).filter(
            entities.Article.status == "published"
        ).count()
        
        # 3. Current Live Chat Queue Count for this officer
        queue_count = db.query(entities.LiveChatQueue).filter(
            and_(
                entities.LiveChatQueue.admission_official_id == current_user.user_id,
                entities.LiveChatQueue.status == "waiting"
            )
        ).count()
        
        # 4. Drafted Articles
        drafted_articles = db.query(entities.Article).filter(
            entities.Article.status == "draft"
        ).count()
        
        # 5. Weekly Article Publication Stats (last 7 days)
        week_threshold = datetime.now() - timedelta(days=7)
        weekly_articles = db.query(
            func.date(entities.Article.create_at).label('date'),
            func.count(entities.Article.article_id).label('count')
        ).filter(
            and_(
                entities.Article.status == "published",
                entities.Article.create_at >= week_threshold
            )
        ).group_by(
            func.date(entities.Article.create_at)
        ).order_by(
            func.date(entities.Article.create_at)
        ).all()
        
        # Create a dictionary from the query results
        articles_by_date = {date_obj: count for date_obj, count in weekly_articles}
        
        # Generate all 7 days with 0 for days without articles
        weekly_data = []
        for i in range(6, -1, -1):  # 6 days ago to today
            date_obj = (datetime.now() - timedelta(days=i)).date()
            count = articles_by_date.get(date_obj, 0)
            
            # Vietnamese day names
            day_names = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'CN']
            day_name = day_names[date_obj.weekday()]  # Mon=0 -> T2, Sun=6 -> CN
            
            weekly_data.append({
                "date": day_name,
                "articles": count
            })
        
        # 6. Intent Distribution from Training Questions (only approved)
        intent_stats = db.query(
            entities.Intent.intent_name,
            func.count(entities.TrainingQuestionAnswer.question_id).label('count')
        ).join(
            entities.TrainingQuestionAnswer,
            entities.Intent.intent_id == entities.TrainingQuestionAnswer.intent_id
        ).filter(
            entities.TrainingQuestionAnswer.status == 'approved'
        ).group_by(
            entities.Intent.intent_name
        ).order_by(
            desc('count')
        ).limit(10).all()
        
        intent_distribution = [
            {
                "topic": intent_name,
                "count": count
            }
            for intent_name, count in intent_stats
        ]
        
        return {
            "chatbot_interactions": ended_chatbot_sessions,
            "published_articles": published_articles,
            "queue_count": queue_count,
            "drafted_articles": drafted_articles,
            "weekly_articles": weekly_data,
            "intent_distribution": intent_distribution,
            "period_days": days
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting admission dashboard stats: {str(e)}")


@router.get("/dashboard/system-health")
async def get_system_health(
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_analytics_permission)
):
    """
    Get system health metrics
    """
    try:
        # Total knowledge base articles
        total_articles = db.query(entities.Article).filter(
            entities.Article.status == 'published'
        ).count()
        
        # Total knowledge documents (only approved)
        total_kb_docs = db.query(entities.KnowledgeBaseDocument).filter(
            entities.KnowledgeBaseDocument.status == 'approved'
        ).count()
        
        # Training QA pairs (only approved)
        total_qa_pairs = db.query(entities.TrainingQuestionAnswer).filter(
            entities.TrainingQuestionAnswer.status == 'approved'
        ).count()
        
        # Recent activity (errors, warnings, etc.)
        recent_failed_interactions = db.query(entities.ChatInteraction).filter(
            and_(
                entities.ChatInteraction.rating.isnot(None),
                entities.ChatInteraction.rating < 3,  # Low ratings
                entities.ChatInteraction.timestamp >= datetime.now() - timedelta(hours=24)
            )
        ).count()
        
        # Total users count (all users in the system)
        total_users = db.query(entities.Users).count()
        
        return {
            "total_users": total_users,
            "total_articles": total_articles,
            "total_qa_pairs": total_qa_pairs,
            "total_kb_docs": total_kb_docs,
            "recent_errors": recent_failed_interactions
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system health: {str(e)}")