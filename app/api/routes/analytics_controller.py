from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_
from datetime import datetime, timedelta
from typing import List, Optional
import re
from app.models.database import get_db
from app.models import entities
from app.core.security import get_current_user

router = APIRouter()

def check_analytics_permission(current_user: entities.Users = Depends(get_current_user)):
    """Check if user has permission to view analytics (Admin, Consultant, or Manager)"""
    if not current_user:
        raise HTTPException(status_code=403, detail="Not authenticated")

    try:
        user_perms_list = [p.permission_name.lower() for p in current_user.permissions] 
    except AttributeError:
        user_perms_list = [p.lower() for p in current_user.permissions]

    is_admin_or_consultant = "admin" in user_perms_list or "consultant" in user_perms_list
    is_manager = "manager" in user_perms_list

    if not (is_admin_or_consultant or is_manager):
        raise HTTPException(
            status_code=403,
            detail="Admin, Consultant, or Manager permission required"
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
        
        # Get existing training questions (no created_at field available)
        existing_training = db.query(entities.TrainingQuestionAnswer.question).filter(
            entities.TrainingQuestionAnswer.question.isnot(None)
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
            
            # Check against each training question
            for training_q in existing_q_texts:
                training_words = set(training_q.split())
                
                # Method 1: Word overlap (existing)
                word_overlap = len(question_words & training_words)
                overlap_score = word_overlap / max(len(question_words), len(training_words), 1)
                
                # Method 2: Substring similarity
                substring_score = 0
                if training_q in question_lower or question_lower in training_q:
                    substring_score = 0.8
                
                # Method 3: Key phrase matching
                key_phrase_score = 0
                question_key_phrases = [phrase.strip() for phrase in question_lower.replace('?', '').split() if len(phrase) > 3]
                training_key_phrases = [phrase.strip() for phrase in training_q.replace('?', '').split() if len(phrase) > 3]
                
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
                    break
                    
                # Track best match even if not covered
                if combined_score > best_match_score:
                    best_match_score = combined_score
            
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
                # Determine priority based on frequency and recency
                if frequency >= 15:
                    priority = "high"
                elif frequency >= 8:
                    priority = "medium"
                else:
                    priority = "low"
                
                # Enhanced category detection
                category = "General"
                question_lower = question_text.lower()
                if any(word in question_lower for word in ['admission', 'apply', 'application', 'admit', 'entrance', 'enroll']):
                    category = "Admissions"
                elif any(word in question_lower for word in ['course', 'program', 'study', 'academic', 'class', 'curriculum', 'major', 'degree']):
                    category = "Academic"
                elif any(word in question_lower for word in ['fee', 'cost', 'tuition', 'scholarship', 'financial', 'loan', 'grant', 'funding']):
                    category = "Financial"
                elif any(word in question_lower for word in ['campus', 'dormitory', 'housing', 'facility', 'student life', 'activities']):
                    category = "Campus Life"
                elif any(word in question_lower for word in ['career', 'job', 'employment', 'internship', 'graduation']):
                    category = "Career Services"
                
                # Enhanced suggested action
                suggested_action = f"Create comprehensive answer for this {category.lower()} question"
                if grace_period_needed:
                    suggested_action = f"Monitor effectiveness - recent activity detected for {category.lower()} question"
                elif best_match_score > 0.3:
                    suggested_action = f"Improve existing answer - partial match found for {category.lower()} question"
                
                knowledge_gaps.append({
                    "id": idx + 1,
                    "question": question_text,
                    "frequency": frequency,
                    "priority": priority,
                    "category": category,
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

@router.get("/analytics-summary")
async def get_analytics_summary(
    db: Session = Depends(get_db),
    current_user: entities.Users = Depends(check_analytics_permission)
):
    """
    Get summary statistics for the analytics dashboard
    """
    try:
        # Count knowledge gaps (recent questions without answers)
        date_threshold = (datetime.now() - timedelta(days=30)).date()
        
        user_questions_count = db.query(func.count(func.distinct(entities.ChatInteraction.message_text))).filter(
            and_(
                entities.ChatInteraction.is_from_bot == False,
                entities.ChatInteraction.timestamp >= date_threshold,
                func.length(entities.ChatInteraction.message_text) > 10
            )
        ).scalar() or 0
        
        existing_qa_count = db.query(func.count(entities.TrainingQuestionAnswer.question_id)).scalar() or 0
        
        # Estimate knowledge gaps (simplified)
        knowledge_gaps_count = max(0, int(user_questions_count * 0.3))  # Rough estimate
        
        # Count low satisfaction answers
        low_satisfaction_count = db.query(func.count(entities.FaqStatistics.faq_id)).filter(
            or_(
                entities.FaqStatistics.rating < 3.5,
                entities.FaqStatistics.success_rate < 0.7
            )
        ).scalar() or 0
        
        # Count trending topics (mock for now since it's complex to calculate dynamically)
        trending_topics_count = 4  # This would be calculated from the actual trending analysis
        
        # Total chat interactions for context
        total_interactions = db.query(func.count(entities.ChatInteraction.interaction_id)).scalar() or 0
        
        return {
            "knowledge_gaps_count": knowledge_gaps_count,
            "low_satisfaction_count": low_satisfaction_count,
            "trending_topics_count": trending_topics_count,
            "total_interactions": total_interactions,
            "existing_qa_count": existing_qa_count,
            "user_questions_count": user_questions_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting analytics summary: {str(e)}")

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
            'Admission Requirements': ['admission', 'application', 'requirement', 'deadline', 'gpa', 'grade', 'apply', 'entrance'],
            'Financial Aid': ['financial aid', 'scholarship', 'tuition', 'fee', 'cost', 'funding', 'loan', 'grant'],
            'Academic Programs': ['program', 'major', 'course', 'curriculum', 'degree', 'bachelor', 'master', 'subject'],
            'Campus Life': ['campus', 'dormitory', 'housing', 'student life', 'activities', 'club', 'facility', 'tour'],
            'Career Services': ['career', 'internship', 'job placement', 'employment', 'graduation rate', 'alumni']
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
                if 'Other' not in category_stats:
                    category_stats['Other'] = {
                        'category': 'Other',
                        'total_questions': 0,
                        'total_times_asked': 0,
                        'unique_questions': []
                    }
                category_stats['Other']['total_questions'] += 1
                category_stats['Other']['total_times_asked'] += frequency
                category_stats['Other']['unique_questions'].append({
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