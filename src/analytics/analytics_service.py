"""
Analytics Service

Provides deterministic analytics and health score computation.
Analyzes historical food logs to compute trends, detect patterns,
and generate actionable insights.

DESIGN: All calculations are transparent and explainable.
No ML predictions - only deterministic computations.
"""

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple

from models.user import UserProfile, DailyTargets
from models.analytics_models import (
    MealLogEntry,
    DailyNutritionSummary,
    NutrientTrend,
    HealthScore,
    BehaviorPattern,
    AnalyticsSnapshot,
)


class MealLogStore:
    """
    Simple in-memory meal log store.
    
    In production, this would be backed by a database.
    """
    
    def __init__(self):
        self._logs: Dict[str, List[MealLogEntry]] = defaultdict(list)
    
    def add(self, entry: MealLogEntry):
        """Add a meal log entry."""
        self._logs[entry.user_id].append(entry)
    
    def get_by_user(self, user_id: str) -> List[MealLogEntry]:
        """Get all logs for a user."""
        return self._logs.get(user_id, [])
    
    def get_range(
        self, 
        user_id: str, 
        start_date: date, 
        end_date: date
    ) -> List[MealLogEntry]:
        """Get logs within a date range."""
        logs = self._logs.get(user_id, [])
        return [
            log for log in logs
            if start_date <= log.timestamp.date() <= end_date
        ]
    
    def get_today(self, user_id: str) -> List[MealLogEntry]:
        """Get today's logs."""
        today = date.today()
        return self.get_range(user_id, today, today)
    
    def clear(self, user_id: str):
        """Clear logs for a user."""
        self._logs[user_id] = []


class AnalyticsService:
    """
    Deterministic analytics and health score computation.
    
    DESIGN PRINCIPLES:
    1. All calculations are explainable
    2. No ML predictions - only deterministic computations
    3. Scores have transparent weights
    4. Patterns are detected with clear rules
    """
    
    # Health score weights (must sum to 100)
    SCORE_WEIGHTS = {
        "calorie_adherence": 25,
        "macro_balance": 20,
        "sugar_control": 20,
        "sodium_control": 15,
        "fiber_intake": 10,
        "meal_regularity": 10,
    }
    
    # Nutrients to track
    TRACKED_NUTRIENTS = [
        "calories", "protein_g", "carbs_g", "fat_g", 
        "sugar_g", "fiber_g", "sodium_mg"
    ]
    
    def __init__(
        self, 
        meal_log_store: MealLogStore,
        default_targets: Optional[DailyTargets] = None,
    ):
        """
        Initialize Analytics Service.
        
        Args:
            meal_log_store: Store for meal logs
            default_targets: Default targets if user doesn't have custom ones
        """
        self.store = meal_log_store
        self.default_targets = default_targets or DailyTargets()
    
    def log_meal(
        self,
        user_id: str,
        foods: List[Dict[str, Any]],
        violations: List[Dict[str, Any]] = None,
    ) -> MealLogEntry:
        """
        Log a meal for analytics.
        
        Args:
            user_id: User ID
            foods: List of food dictionaries
            violations: Optional list of violations
            
        Returns:
            Created MealLogEntry
        """
        # Aggregate nutrition
        total = {
            "calories": 0.0,
            "protein_g": 0.0,
            "carbs_g": 0.0,
            "fat_g": 0.0,
            "sugar_g": 0.0,
            "fiber_g": 0.0,
            "sodium_mg": 0.0,
        }
        
        for food in foods:
            for key in total:
                total[key] += food.get(key, 0.0)
        
        entry = MealLogEntry(
            log_id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=datetime.now(),
            foods=foods,
            total_nutrition=total,
            violations=violations or [],
        )
        
        self.store.add(entry)
        return entry
    
    def compute_daily_trends(
        self, 
        user_id: str, 
        targets: DailyTargets,
        days: int = 7
    ) -> Dict[str, NutrientTrend]:
        """
        Compute daily nutrient trends.
        
        Returns deterministic calculations, never predictions.
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days - 1)
        
        logs = self.store.get_range(user_id, start_date, end_date)
        
        # Aggregate by day
        daily_totals: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {n: 0.0 for n in self.TRACKED_NUTRIENTS}
        )
        
        for log in logs:
            day_str = log.timestamp.date().isoformat()
            for nutrient in self.TRACKED_NUTRIENTS:
                daily_totals[day_str][nutrient] += log.total_nutrition.get(nutrient, 0.0)
        
        # Build trends
        trends = {}
        
        for nutrient in self.TRACKED_NUTRIENTS:
            # Get target value
            target_attr = nutrient.replace("_g", "").replace("_mg", "")
            if hasattr(targets, nutrient):
                target = getattr(targets, nutrient)
            elif hasattr(targets, target_attr):
                target = getattr(targets, target_attr)
            else:
                target = 0
            
            # Build daily values
            daily_values = []
            for d in range(days):
                current_date = start_date + timedelta(days=d)
                day_str = current_date.isoformat()
                value = daily_totals[day_str][nutrient] if day_str in daily_totals else 0.0
                daily_values.append((day_str, value))
            
            # Calculate average
            non_zero_values = [v for _, v in daily_values if v > 0]
            weekly_avg = sum(non_zero_values) / len(non_zero_values) if non_zero_values else 0.0
            
            # Calculate gap
            gap_pct = ((target - weekly_avg) / target * 100) if target > 0 else 0.0
            
            trends[nutrient] = NutrientTrend(
                nutrient=nutrient,
                daily_values=daily_values,
                weekly_average=weekly_avg,
                target=target,
                gap_percentage=gap_pct,
            )
        
        return trends
    
    def compute_health_score(
        self, 
        user_id: str, 
        targets: DailyTargets,
        days: int = 7
    ) -> HealthScore:
        """
        Compute deterministic health score (0-100).
        
        Formula is transparent and explainable:
        - Each component scored 0-100
        - Weighted average gives final score
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days - 1)
        
        logs = self.store.get_range(user_id, start_date, end_date)
        
        if not logs:
            return HealthScore(
                overall_score=0,
                components={k: 0 for k in self.SCORE_WEIGHTS},
                explanation="No meal data available for scoring.",
                computed_at=datetime.now(),
            )
        
        components = {}
        
        # 1. Calorie adherence (25 points)
        avg_calories = self._average_nutrient(logs, "calories", days)
        target_calories = targets.calories
        calorie_deviation = abs(avg_calories - target_calories) / target_calories if target_calories > 0 else 0
        components["calorie_adherence"] = max(0, int(100 - calorie_deviation * 100))
        
        # 2. Macro balance (20 points) - protein, carbs, fat ratio
        components["macro_balance"] = self._score_macro_balance(logs, targets)
        
        # 3. Sugar control (20 points)
        avg_sugar = self._average_nutrient(logs, "sugar_g", days)
        target_sugar = targets.sugar_g
        sugar_excess = max(0, (avg_sugar - target_sugar) / target_sugar) if target_sugar > 0 else 0
        components["sugar_control"] = max(0, int(100 - sugar_excess * 100))
        
        # 4. Sodium control (15 points)
        avg_sodium = self._average_nutrient(logs, "sodium_mg", days)
        target_sodium = targets.sodium_mg
        sodium_excess = max(0, (avg_sodium - target_sodium) / target_sodium) if target_sodium > 0 else 0
        components["sodium_control"] = max(0, int(100 - sodium_excess * 100))
        
        # 5. Fiber intake (10 points)
        avg_fiber = self._average_nutrient(logs, "fiber_g", days)
        target_fiber = targets.fiber_g
        fiber_ratio = min(1.0, avg_fiber / target_fiber) if target_fiber > 0 else 1.0
        components["fiber_intake"] = int(fiber_ratio * 100)
        
        # 6. Meal regularity (10 points)
        components["meal_regularity"] = self._score_meal_regularity(logs, days)
        
        # Calculate weighted score
        overall = sum(
            components[k] * (self.SCORE_WEIGHTS[k] / 100)
            for k in components
        )
        
        # Generate explanation
        explanation = self._generate_score_explanation(components, overall)
        
        return HealthScore(
            overall_score=int(overall),
            components=components,
            explanation=explanation,
            computed_at=datetime.now(),
        )
    
    def detect_patterns(
        self, 
        user_id: str, 
        targets: DailyTargets,
        days: int = 14
    ) -> List[BehaviorPattern]:
        """
        Detect unhealthy behavior patterns.
        
        Rules-based pattern detection (not ML):
        - Weekend sugar spikes
        - Frequent sodium excess
        - Low fiber
        - Late night eating
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days - 1)
        
        logs = self.store.get_range(user_id, start_date, end_date)
        
        if not logs:
            return []
        
        patterns = []
        
        # Pattern 1: Weekend sugar spikes
        weekend_sugar = self._average_by_day_type(logs, "sugar_g", weekend=True)
        weekday_sugar = self._average_by_day_type(logs, "sugar_g", weekend=False)
        
        if weekend_sugar > weekday_sugar * 1.3 and weekend_sugar > 0:  # 30% higher
            patterns.append(BehaviorPattern(
                pattern_id="weekend_sugar_spike",
                description="You tend to consume more sugar on weekends",
                evidence=[
                    f"Weekend average: {weekend_sugar:.0f}g sugar",
                    f"Weekday average: {weekday_sugar:.0f}g sugar",
                ],
                severity="medium" if weekend_sugar > weekday_sugar * 1.5 else "low",
                suggestion="Consider preparing healthy weekend snacks in advance.",
            ))
        
        # Pattern 2: Frequent high sodium
        sodium_excess_days = self._count_excess_days(logs, "sodium_mg", targets.sodium_mg)
        
        if sodium_excess_days >= 3:
            patterns.append(BehaviorPattern(
                pattern_id="frequent_high_sodium",
                description="Frequent high sodium days detected",
                evidence=[
                    f"{sodium_excess_days} days exceeded sodium limit ({targets.sodium_mg}mg) in {days} days",
                ],
                severity="high" if sodium_excess_days > days // 2 else "medium",
                suggestion="Check for hidden sodium in processed foods and condiments.",
            ))
        
        # Pattern 3: Chronic low fiber
        fiber_deficit_days = self._count_deficit_days(logs, "fiber_g", targets.fiber_g * 0.5)
        
        if fiber_deficit_days >= days // 2:
            patterns.append(BehaviorPattern(
                pattern_id="chronic_low_fiber",
                description="Consistently low fiber intake",
                evidence=[
                    f"{fiber_deficit_days} days with less than 50% of fiber target",
                ],
                severity="medium",
                suggestion="Add whole grains, vegetables, or legumes to each meal.",
            ))
        
        # Pattern 4: Late night eating
        late_night_meals = [log for log in logs if log.meal_type == "late_night"]
        
        if len(late_night_meals) >= 3:
            avg_late_calories = sum(
                m.total_nutrition.get("calories", 0) for m in late_night_meals
            ) / len(late_night_meals)
            
            patterns.append(BehaviorPattern(
                pattern_id="late_night_eating",
                description="Frequent late-night eating detected",
                evidence=[
                    f"{len(late_night_meals)} meals after 10 PM",
                    f"Average {avg_late_calories:.0f} calories per late meal",
                ],
                severity="low",
                suggestion="Try to finish eating at least 2-3 hours before bed.",
            ))
        
        return patterns
    
    def generate_insight(
        self, 
        user_id: str, 
        targets: DailyTargets
    ) -> str:
        """Generate a single, actionable insight for the user."""
        patterns = self.detect_patterns(user_id, targets)
        trends = self.compute_daily_trends(user_id, targets)
        
        # Priority 1: High severity patterns
        high_severity = [p for p in patterns if p.severity == "high"]
        if high_severity:
            pattern = high_severity[0]
            return f"⚠️ {pattern.description}. {pattern.suggestion}"
        
        # Priority 2: Medium severity patterns
        medium_severity = [p for p in patterns if p.severity == "medium"]
        if medium_severity:
            pattern = medium_severity[0]
            return f"💡 {pattern.description}. {pattern.suggestion}"
        
        # Priority 3: Nutrient gaps (deficits)
        deficient = [(n, t) for n, t in trends.items() if t.is_deficient]
        if deficient:
            nutrient, trend = deficient[0]
            nutrient_name = nutrient.replace("_g", "").replace("_mg", "").title()
            return f"📊 Your {nutrient_name} intake is {abs(trend.gap_percentage):.0f}% below target. Consider adding more {self._get_food_suggestions(nutrient)}."
        
        # Priority 4: Nutrient excess
        excessive = [(n, t) for n, t in trends.items() if t.is_excessive and n in ["sugar_g", "sodium_mg"]]
        if excessive:
            nutrient, trend = excessive[0]
            nutrient_name = nutrient.replace("_g", "").replace("_mg", "").title()
            return f"📉 Your {nutrient_name} intake is {abs(trend.gap_percentage):.0f}% above target."
        
        # Default: Everything looks good
        return "✅ Great job! Your nutrition is on track this week."
    
    def get_snapshot(
        self, 
        user_id: str, 
        targets: DailyTargets
    ) -> AnalyticsSnapshot:
        """Get complete analytics snapshot for dashboard."""
        return AnalyticsSnapshot(
            user_id=user_id,
            computed_at=datetime.now(),
            health_score=self.compute_health_score(user_id, targets),
            trends=self.compute_daily_trends(user_id, targets),
            patterns=self.detect_patterns(user_id, targets),
            primary_insight=self.generate_insight(user_id, targets),
        )
    
    # Helper methods
    
    def _average_nutrient(
        self, 
        logs: List[MealLogEntry], 
        nutrient: str, 
        days: int
    ) -> float:
        """Calculate average daily value for a nutrient."""
        total = sum(log.total_nutrition.get(nutrient, 0) for log in logs)
        days_with_data = len(set(log.timestamp.date() for log in logs))
        return total / days_with_data if days_with_data > 0 else 0.0
    
    def _score_macro_balance(
        self, 
        logs: List[MealLogEntry], 
        targets: DailyTargets
    ) -> int:
        """Score macro nutrient balance."""
        # Target ratios (example: 20% protein, 50% carbs, 30% fat)
        total_protein = sum(log.total_nutrition.get("protein_g", 0) for log in logs)
        total_carbs = sum(log.total_nutrition.get("carbs_g", 0) for log in logs)
        total_fat = sum(log.total_nutrition.get("fat_g", 0) for log in logs)
        
        total_cals = total_protein * 4 + total_carbs * 4 + total_fat * 9
        
        if total_cals == 0:
            return 50  # Neutral score
        
        actual_protein_pct = (total_protein * 4 / total_cals) * 100
        actual_carbs_pct = (total_carbs * 4 / total_cals) * 100
        actual_fat_pct = (total_fat * 9 / total_cals) * 100
        
        # Target ratios from targets
        target_cals = targets.calories
        target_protein_pct = (targets.protein_g * 4 / target_cals) * 100 if target_cals > 0 else 20
        target_carbs_pct = (targets.carbs_g * 4 / target_cals) * 100 if target_cals > 0 else 50
        target_fat_pct = (targets.fat_g * 9 / target_cals) * 100 if target_cals > 0 else 30
        
        # Score based on deviation from targets
        protein_dev = abs(actual_protein_pct - target_protein_pct)
        carbs_dev = abs(actual_carbs_pct - target_carbs_pct)
        fat_dev = abs(actual_fat_pct - target_fat_pct)
        
        avg_deviation = (protein_dev + carbs_dev + fat_dev) / 3
        
        return max(0, int(100 - avg_deviation * 2))
    
    def _score_meal_regularity(self, logs: List[MealLogEntry], days: int) -> int:
        """Score meal regularity (eating consistent times/days)."""
        days_with_meals = len(set(log.timestamp.date() for log in logs))
        
        if days_with_meals >= days - 1:
            return 100
        elif days_with_meals >= days * 0.7:
            return 80
        elif days_with_meals >= days * 0.5:
            return 60
        else:
            return 40
    
    def _average_by_day_type(
        self, 
        logs: List[MealLogEntry], 
        nutrient: str, 
        weekend: bool
    ) -> float:
        """Calculate average for weekends or weekdays."""
        filtered = [log for log in logs if log.is_weekend == weekend]
        if not filtered:
            return 0.0
        
        # Group by day
        days: Dict[str, float] = defaultdict(float)
        for log in filtered:
            days[log.timestamp.date().isoformat()] += log.total_nutrition.get(nutrient, 0)
        
        return sum(days.values()) / len(days) if days else 0.0
    
    def _count_excess_days(
        self, 
        logs: List[MealLogEntry], 
        nutrient: str, 
        threshold: float
    ) -> int:
        """Count days where nutrient exceeded threshold."""
        days: Dict[str, float] = defaultdict(float)
        for log in logs:
            days[log.timestamp.date().isoformat()] += log.total_nutrition.get(nutrient, 0)
        
        return sum(1 for v in days.values() if v > threshold)
    
    def _count_deficit_days(
        self, 
        logs: List[MealLogEntry], 
        nutrient: str, 
        threshold: float
    ) -> int:
        """Count days where nutrient was below threshold."""
        days: Dict[str, float] = defaultdict(float)
        for log in logs:
            days[log.timestamp.date().isoformat()] += log.total_nutrition.get(nutrient, 0)
        
        return sum(1 for v in days.values() if v < threshold)
    
    def _generate_score_explanation(
        self, 
        components: Dict[str, int], 
        overall: float
    ) -> str:
        """Generate human-readable explanation of score."""
        # Find strongest and weakest components
        sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)
        strongest = sorted_components[0]
        weakest = sorted_components[-1]
        
        lines = []
        
        if overall >= 80:
            lines.append("Excellent overall nutrition!")
        elif overall >= 60:
            lines.append("Good nutrition with room for improvement.")
        else:
            lines.append("Your nutrition needs attention.")
        
        lines.append(f"Strongest: {strongest[0].replace('_', ' ').title()} ({strongest[1]}%)")
        lines.append(f"Focus area: {weakest[0].replace('_', ' ').title()} ({weakest[1]}%)")
        
        return " ".join(lines)
    
    def _get_food_suggestions(self, nutrient: str) -> str:
        """Get food suggestions for a nutrient."""
        suggestions = {
            "protein_g": "lean meats, eggs, legumes, or Greek yogurt",
            "fiber_g": "whole grains, vegetables, fruits, or legumes",
            "carbs_g": "whole grains, fruits, or starchy vegetables",
            "fat_g": "healthy fats like avocado, nuts, or olive oil",
            "calories": "nutrient-dense foods",
            "sugar_g": "naturally sweet foods like fruits",
            "sodium_mg": "home-cooked meals with fresh ingredients",
        }
        return suggestions.get(nutrient, "nutrient-rich foods")
