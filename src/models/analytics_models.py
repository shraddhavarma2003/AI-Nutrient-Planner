"""
Analytics Data Models

Defines data structures for meal logging, trend analysis,
health scoring, and behavior pattern detection.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import List, Optional, Dict, Any, Tuple, Literal


@dataclass
class MealLogEntry:
    """
    Single meal log entry.
    
    Stores a complete record of a logged meal including
    all foods, total nutrition, and any rule violations.
    """
    log_id: str
    user_id: str
    timestamp: datetime
    
    # Food data (stored as dictionaries for simplicity)
    foods: List[Dict[str, Any]]
    
    # Aggregated nutrition for the meal
    total_nutrition: Dict[str, float]
    
    # Rule violations detected
    violations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Derived fields for analytics
    day_of_week: int = 0  # 0=Monday, 6=Sunday
    hour_of_day: int = 0
    
    def __post_init__(self):
        """Compute derived fields."""
        self.day_of_week = self.timestamp.weekday()
        self.hour_of_day = self.timestamp.hour
    
    @property
    def is_weekend(self) -> bool:
        """Check if meal was on weekend."""
        return self.day_of_week >= 5
    
    @property
    def meal_type(self) -> str:
        """Infer meal type from hour."""
        if 5 <= self.hour_of_day < 11:
            return "breakfast"
        elif 11 <= self.hour_of_day < 15:
            return "lunch"
        elif 15 <= self.hour_of_day < 18:
            return "snack"
        elif 18 <= self.hour_of_day < 22:
            return "dinner"
        else:
            return "late_night"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "log_id": self.log_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "foods": self.foods,
            "total_nutrition": self.total_nutrition,
            "violations": self.violations,
            "day_of_week": self.day_of_week,
            "hour_of_day": self.hour_of_day,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MealLogEntry":
        """Create from dictionary."""
        return cls(
            log_id=data["log_id"],
            user_id=data["user_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            foods=data["foods"],
            total_nutrition=data["total_nutrition"],
            violations=data.get("violations", []),
        )


@dataclass
class DailyNutritionSummary:
    """
    Daily aggregated nutrition data.
    
    Used for trend computation and daily target comparison.
    """
    date: date
    user_id: str
    
    # Aggregated values
    total_calories: float = 0.0
    total_protein_g: float = 0.0
    total_carbs_g: float = 0.0
    total_fat_g: float = 0.0
    total_sugar_g: float = 0.0
    total_fiber_g: float = 0.0
    total_sodium_mg: float = 0.0
    
    # Meal counts
    meal_count: int = 0
    violation_count: int = 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat(),
            "user_id": self.user_id,
            "total_calories": self.total_calories,
            "total_protein_g": self.total_protein_g,
            "total_carbs_g": self.total_carbs_g,
            "total_fat_g": self.total_fat_g,
            "total_sugar_g": self.total_sugar_g,
            "total_fiber_g": self.total_fiber_g,
            "total_sodium_mg": self.total_sodium_mg,
            "meal_count": self.meal_count,
            "violation_count": self.violation_count,
        }


@dataclass
class NutrientTrend:
    """
    Trend data for a specific nutrient.
    
    Contains historical values, averages, and gap analysis.
    """
    nutrient: str
    daily_values: List[Tuple[str, float]]  # (date_str, value)
    weekly_average: float
    target: float
    gap_percentage: float  # positive = deficit, negative = excess
    
    @property
    def is_deficient(self) -> bool:
        """Check if nutrient is below target."""
        return self.gap_percentage > 10  # 10% below target
    
    @property
    def is_excessive(self) -> bool:
        """Check if nutrient is above target."""
        return self.gap_percentage < -10  # 10% above target
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "nutrient": self.nutrient,
            "daily_values": self.daily_values,
            "weekly_average": round(self.weekly_average, 1),
            "target": self.target,
            "gap_percentage": round(self.gap_percentage, 1),
            "is_deficient": self.is_deficient,
            "is_excessive": self.is_excessive,
        }


@dataclass
class HealthScore:
    """
    Deterministic health score (0-100).
    
    Computed from multiple components with transparent weights.
    NEVER uses ML predictions - all calculations are explainable.
    """
    overall_score: int
    components: Dict[str, int]
    explanation: str
    computed_at: datetime
    
    # Grade based on score
    @property
    def grade(self) -> str:
        """Get letter grade for score."""
        if self.overall_score >= 90:
            return "A"
        elif self.overall_score >= 80:
            return "B"
        elif self.overall_score >= 70:
            return "C"
        elif self.overall_score >= 60:
            return "D"
        else:
            return "F"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "grade": self.grade,
            "components": self.components,
            "explanation": self.explanation,
            "computed_at": self.computed_at.isoformat(),
        }


@dataclass
class BehaviorPattern:
    """
    Detected unhealthy behavior pattern.
    
    Rules-based pattern detection with evidence and suggestions.
    """
    pattern_id: str
    description: str
    evidence: List[str]
    severity: Literal["low", "medium", "high"]
    suggestion: str
    detected_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "evidence": self.evidence,
            "severity": self.severity,
            "suggestion": self.suggestion,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class AnalyticsSnapshot:
    """
    Complete analytics snapshot for a user.
    
    Aggregates all analytics data for dashboard display.
    """
    user_id: str
    computed_at: datetime
    
    # Core metrics
    health_score: HealthScore
    trends: Dict[str, NutrientTrend]
    patterns: List[BehaviorPattern]
    
    # Summary insight
    primary_insight: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "user_id": self.user_id,
            "computed_at": self.computed_at.isoformat(),
            "health_score": self.health_score.to_dict(),
            "trends": {k: v.to_dict() for k, v in self.trends.items()},
            "patterns": [p.to_dict() for p in self.patterns],
            "primary_insight": self.primary_insight,
        }
