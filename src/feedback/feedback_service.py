"""
Feedback Service

Collects and aggregates user feedback for quality improvement.

CRITICAL: Feedback NEVER directly retrains models or modifies rules.
All suggestions require human review before implementation.
"""

import uuid
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Optional, Dict, Any

from models.feedback import (
    UserFeedback,
    FeedbackAggregation,
    ThresholdSuggestion,
    FeedbackSummary,
)


class FeedbackStore:
    """
    Simple in-memory feedback store.
    
    In production, this would be backed by a database.
    """
    
    def __init__(self):
        self._feedback: List[UserFeedback] = []
    
    def save(self, feedback: UserFeedback):
        """Save a feedback entry."""
        self._feedback.append(feedback)
    
    def get_all(self) -> List[UserFeedback]:
        """Get all feedback."""
        return self._feedback.copy()
    
    def get_by_user(self, user_id: str) -> List[UserFeedback]:
        """Get feedback from a specific user."""
        return [f for f in self._feedback if f.user_id == user_id]
    
    def get_by_rule(self, rule_id: str) -> List[UserFeedback]:
        """Get feedback related to a specific rule."""
        return [f for f in self._feedback if f.related_rule_id == rule_id]
    
    def get_by_context_type(self, context_type: str) -> List[UserFeedback]:
        """Get feedback for a specific context type."""
        return [f for f in self._feedback if f.context_type == context_type]
    
    def clear(self):
        """Clear all feedback."""
        self._feedback = []


class FeedbackService:
    """
    Collects and aggregates user feedback.
    
    IMPORTANT:
    1. Feedback NEVER directly modifies rules or retrains models
    2. All suggestions are for HUMAN REVIEW only
    3. We track patterns to improve explanations
    """
    
    # Minimum feedback count before generating suggestions
    MIN_FEEDBACK_FOR_SUGGESTION = 10
    
    # Threshold for flagging rules that need review
    HELPFULNESS_THRESHOLD = 0.6  # 60%
    
    def __init__(self, feedback_store: Optional[FeedbackStore] = None):
        """
        Initialize Feedback Service.
        
        Args:
            feedback_store: Store for feedback data (creates new if None)
        """
        self.store = feedback_store or FeedbackStore()
    
    def submit_feedback(
        self,
        user_id: str,
        context_type: str,
        context_id: str,
        rating: str,
        comment: Optional[str] = None,
        related_rule_id: Optional[str] = None,
    ) -> UserFeedback:
        """
        Submit user feedback.
        
        Args:
            user_id: ID of the user submitting feedback
            context_type: Type of item being rated (coach_response, meal_fix, etc.)
            context_id: ID of the specific item
            rating: helpful, not_helpful, or incorrect
            comment: Optional detailed comment
            related_rule_id: Optional rule ID if feedback is about a rule
            
        Returns:
            Created UserFeedback object
        """
        feedback = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=datetime.now(),
            context_type=context_type,
            context_id=context_id,
            rating=rating,
            comment=comment,
            related_rule_id=related_rule_id,
        )
        
        self.store.save(feedback)
        return feedback
    
    def get_rule_feedback_summary(self, rule_id: str) -> FeedbackAggregation:
        """
        Aggregate feedback for a specific rule.
        
        Used for suggesting threshold adjustments (not automatic).
        """
        feedbacks = self.store.get_by_rule(rule_id)
        
        return FeedbackAggregation(
            rule_id=rule_id,
            total_feedback=len(feedbacks),
            helpful_count=sum(1 for f in feedbacks if f.rating == "helpful"),
            not_helpful_count=sum(1 for f in feedbacks if f.rating == "not_helpful"),
            incorrect_count=sum(1 for f in feedbacks if f.rating == "incorrect"),
            common_complaints=self._extract_common_complaints(feedbacks),
        )
    
    def suggest_threshold_adjustment(
        self, 
        rule_id: str,
        current_threshold: float = 0.0,
    ) -> Optional[ThresholdSuggestion]:
        """
        Suggest threshold adjustment based on feedback.
        
        CRITICAL: Returns suggestion for HUMAN REVIEW - NEVER auto-applies.
        
        Args:
            rule_id: ID of the rule to analyze
            current_threshold: Current threshold value
            
        Returns:
            ThresholdSuggestion if enough data and issues detected, else None
        """
        summary = self.get_rule_feedback_summary(rule_id)
        
        # Not enough data
        if summary.total_feedback < self.MIN_FEEDBACK_FOR_SUGGESTION:
            return None
        
        # Calculate unhelpful ratio
        unhelpful_ratio = (
            summary.not_helpful_count + summary.incorrect_count
        ) / summary.total_feedback
        
        # Rule needs review if too many find it unhelpful
        if unhelpful_ratio > (1 - self.HELPFULNESS_THRESHOLD):
            
            # Determine suggested action based on complaints
            complaints_lower = [c.lower() for c in summary.common_complaints]
            
            if any("too strict" in c or "too sensitive" in c for c in complaints_lower):
                suggested_action = "decrease"  # Make rule less strict
            elif any("not strict" in c or "missed" in c for c in complaints_lower):
                suggested_action = "increase"  # Make rule more strict
            else:
                suggested_action = "review"
            
            return ThresholdSuggestion(
                rule_id=rule_id,
                current_threshold=current_threshold,
                suggested_action=suggested_action,
                reason=f"{unhelpful_ratio*100:.0f}% of users found this rule unhelpful/incorrect",
                common_complaints=summary.common_complaints,
                confidence=min(0.9, summary.total_feedback / 50),  # More feedback = more confidence
                requires_human_review=True,  # ALWAYS True
            )
        
        return None
    
    def get_system_summary(self) -> FeedbackSummary:
        """
        Get system-wide feedback summary.
        
        Used for monitoring overall quality.
        """
        all_feedback = self.store.get_all()
        
        if not all_feedback:
            return FeedbackSummary(
                computed_at=datetime.now(),
                total_feedback_count=0,
                average_helpfulness=0.0,
                rules_needing_review=[],
                top_complaints=[],
            )
        
        # Calculate average helpfulness
        helpful_count = sum(1 for f in all_feedback if f.rating == "helpful")
        avg_helpfulness = helpful_count / len(all_feedback)
        
        # Find rules needing review
        rule_ids = set(f.related_rule_id for f in all_feedback if f.related_rule_id)
        rules_needing_review = []
        
        for rule_id in rule_ids:
            summary = self.get_rule_feedback_summary(rule_id)
            if summary.needs_review:
                rules_needing_review.append(rule_id)
        
        # Get top complaints
        all_complaints = self._extract_common_complaints(all_feedback, limit=5)
        
        return FeedbackSummary(
            computed_at=datetime.now(),
            total_feedback_count=len(all_feedback),
            average_helpfulness=avg_helpfulness,
            rules_needing_review=rules_needing_review,
            top_complaints=all_complaints,
        )
    
    def get_feedback_by_context(
        self, 
        context_type: str
    ) -> Dict[str, int]:
        """Get rating distribution for a context type."""
        feedbacks = self.store.get_by_context_type(context_type)
        
        distribution = Counter(f.rating for f in feedbacks)
        
        return {
            "helpful": distribution.get("helpful", 0),
            "not_helpful": distribution.get("not_helpful", 0),
            "incorrect": distribution.get("incorrect", 0),
            "total": len(feedbacks),
        }
    
    def _extract_common_complaints(
        self, 
        feedbacks: List[UserFeedback], 
        limit: int = 3
    ) -> List[str]:
        """
        Extract common themes from feedback comments.
        
        Simple keyword-based extraction (not NLP).
        """
        # Only consider negative feedback with comments
        negative_with_comments = [
            f.comment for f in feedbacks 
            if f.rating in ["not_helpful", "incorrect"] and f.comment
        ]
        
        if not negative_with_comments:
            return []
        
        # Count keywords
        keywords = defaultdict(int)
        complaint_keywords = [
            "too strict", "too sensitive", "not strict", "missed",
            "wrong", "incorrect", "annoying", "not helpful",
            "confusing", "unclear", "too frequent", "should not",
        ]
        
        for comment in negative_with_comments:
            comment_lower = comment.lower()
            for keyword in complaint_keywords:
                if keyword in comment_lower:
                    keywords[keyword] += 1
        
        # Return top complaints
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in sorted_keywords[:limit]]
