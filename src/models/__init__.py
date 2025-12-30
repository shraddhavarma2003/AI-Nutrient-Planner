# Models Package
from .food import Food, NutritionInfo
from .user import UserProfile, HealthCondition, DailyTargets, DailyIntake
from .conversation import ConversationMessage, ConversationContext, CoachResponse
from .analytics_models import (
    MealLogEntry, 
    DailyNutritionSummary, 
    NutrientTrend, 
    HealthScore,
    BehaviorPattern,
    AnalyticsSnapshot,
)
from .feedback import UserFeedback, FeedbackAggregation, ThresholdSuggestion

__all__ = [
    # Phase 1
    "Food", "NutritionInfo", "UserProfile", "HealthCondition", 
    "DailyTargets", "DailyIntake",
    # Phase 3 - Conversation
    "ConversationMessage", "ConversationContext", "CoachResponse",
    # Phase 3 - Analytics
    "MealLogEntry", "DailyNutritionSummary", "NutrientTrend", 
    "HealthScore", "BehaviorPattern", "AnalyticsSnapshot",
    # Phase 3 - Feedback
    "UserFeedback", "FeedbackAggregation", "ThresholdSuggestion",
]
