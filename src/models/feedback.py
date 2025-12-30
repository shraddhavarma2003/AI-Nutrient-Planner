"""
Feedback Data Models

Defines data structures for user feedback collection and aggregation.
Feedback is used for rule tuning suggestions, NOT automatic retraining.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Literal


@dataclass
class UserFeedback:
    """
    User feedback on a suggestion or response.
    
    Used for:
    1. Rule threshold tuning suggestions (requires human review)
    2. Explanation improvement
    3. Quality tracking
    
    IMPORTANT: Feedback NEVER directly modifies rules or retrains models.
    """
    feedback_id: str
    user_id: str
    timestamp: datetime
    
    # What is being rated
    context_type: Literal["coach_response", "meal_fix", "recipe", "insight", "analytics"]
    context_id: str  # ID of the item being rated
    
    # Rating
    rating: Literal["helpful", "not_helpful", "incorrect"]
    
    # Optional comment for detailed feedback
    comment: Optional[str] = None
    
    # Optional: which rule triggered the response
    related_rule_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "feedback_id": self.feedback_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "context_type": self.context_type,
            "context_id": self.context_id,
            "rating": self.rating,
            "comment": self.comment,
            "related_rule_id": self.related_rule_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "UserFeedback":
        """Create from dictionary."""
        return cls(
            feedback_id=data["feedback_id"],
            user_id=data["user_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            context_type=data["context_type"],
            context_id=data["context_id"],
            rating=data["rating"],
            comment=data.get("comment"),
            related_rule_id=data.get("related_rule_id"),
        )


@dataclass
class FeedbackAggregation:
    """
    Aggregated feedback for a specific rule or feature.
    
    Used for generating threshold adjustment suggestions
    that require human review.
    """
    rule_id: str
    total_feedback: int
    helpful_count: int
    not_helpful_count: int
    incorrect_count: int
    common_complaints: List[str]
    
    @property
    def helpfulness_ratio(self) -> float:
        """Calculate ratio of helpful feedback."""
        if self.total_feedback == 0:
            return 0.0
        return self.helpful_count / self.total_feedback
    
    @property
    def needs_review(self) -> bool:
        """Check if rule needs human review based on feedback."""
        if self.total_feedback < 10:
            return False  # Not enough data
        return self.helpfulness_ratio < 0.6  # Less than 60% helpful
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "total_feedback": self.total_feedback,
            "helpful_count": self.helpful_count,
            "not_helpful_count": self.not_helpful_count,
            "incorrect_count": self.incorrect_count,
            "helpfulness_ratio": round(self.helpfulness_ratio, 2),
            "common_complaints": self.common_complaints,
            "needs_review": self.needs_review,
        }


@dataclass
class ThresholdSuggestion:
    """
    Suggestion for rule threshold adjustment.
    
    IMPORTANT: This is a SUGGESTION for human review.
    It NEVER automatically modifies rules.
    """
    rule_id: str
    current_threshold: float
    suggested_action: Literal["review", "increase", "decrease", "no_change"]
    reason: str
    common_complaints: List[str]
    confidence: float  # How confident we are in the suggestion
    
    # ALWAYS True - we never auto-adjust
    requires_human_review: bool = True
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "current_threshold": self.current_threshold,
            "suggested_action": self.suggested_action,
            "reason": self.reason,
            "common_complaints": self.common_complaints,
            "confidence": round(self.confidence, 2),
            "requires_human_review": self.requires_human_review,
        }


@dataclass
class FeedbackSummary:
    """
    Summary of feedback across the system.
    
    Used for monitoring and quality tracking.
    """
    computed_at: datetime
    total_feedback_count: int
    average_helpfulness: float
    rules_needing_review: List[str]
    top_complaints: List[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "computed_at": self.computed_at.isoformat(),
            "total_feedback_count": self.total_feedback_count,
            "average_helpfulness": round(self.average_helpfulness, 2),
            "rules_needing_review": self.rules_needing_review,
            "top_complaints": self.top_complaints,
        }
