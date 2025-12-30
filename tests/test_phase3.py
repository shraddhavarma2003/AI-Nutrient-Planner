"""
Tests for Phase 3: Interaction, Analytics & System Maturity

Tests cover:
- Virtual Coach intent classification and responses
- Analytics service trend computation and health scores
- Feedback service collection and aggregation
- Monitoring service event logging
"""

import sys
import os

# Add src to path for direct Python execution (pytest handles this via pytest.ini)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from datetime import datetime, timedelta, date
import uuid

from models.food import Food, NutritionInfo, FoodCategory
from models.user import UserProfile, HealthCondition, DailyIntake, DailyTargets
from models.conversation import ConversationContext, ConversationMessage, CoachResponse
from models.analytics_models import MealLogEntry, HealthScore, BehaviorPattern
from models.feedback import UserFeedback, FeedbackAggregation

from rules.engine import RuleEngine, Severity
from coach.virtual_coach import VirtualCoach, IntentType
from analytics.analytics_service import AnalyticsService, MealLogStore
from feedback.feedback_service import FeedbackService, FeedbackStore


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def rule_engine():
    return RuleEngine()


@pytest.fixture
def diabetic_user():
    return UserProfile(
        user_id="test-diabetic",
        name="Test User",
        conditions=[HealthCondition.DIABETES],
        allergens=["peanuts"],
        daily_targets=DailyTargets.for_diabetes(),
    )


@pytest.fixture
def healthy_user():
    return UserProfile(
        user_id="test-healthy",
        name="Healthy User",
        conditions=[],
        allergens=[],
        daily_targets=DailyTargets(),
    )


@pytest.fixture
def high_sugar_food():
    return Food(
        food_id="candy-001",
        name="Chocolate Candy Bar",
        serving_size=50,
        serving_unit="g",
        nutrition=NutritionInfo(
            calories=250,
            protein_g=3,
            carbs_g=35,
            sugar_g=28,
            fat_g=12,
            saturated_fat_g=7,
            sodium_mg=50,
        ),
        allergens=[],  # Explicitly set to prevent AL-003 warning
        category=FoodCategory.SNACK,
    )


@pytest.fixture
def healthy_food():
    """A moderate fruit - may trigger caution for diabetics due to sugar."""
    return Food(
        food_id="apple-001",
        name="Apple",
        serving_size=150,
        serving_unit="g",
        nutrition=NutritionInfo(
            calories=80,
            protein_g=0.5,
            carbs_g=21,
            sugar_g=16,
            fiber_g=4,
            fat_g=0.3,
            sodium_mg=2,
        ),
        allergens=[],  # Explicitly set to prevent AL-003 warning
        category=FoodCategory.FRUIT,
    )


@pytest.fixture
def diabetic_safe_food():
    """A low-sugar vegetable that's safe for all conditions."""
    return Food(
        food_id="broccoli-001",
        name="Steamed Broccoli",
        serving_size=100,
        serving_unit="g",
        nutrition=NutritionInfo(
            calories=35,
            protein_g=2.8,
            carbs_g=7,
            sugar_g=1.7,  # Very low sugar
            fiber_g=2.6,
            fat_g=0.4,
            sodium_mg=33,
        ),
        allergens=[],  # Explicitly set to prevent AL-003 warning
        category=FoodCategory.VEGETABLE,
    )


@pytest.fixture
def peanut_food():
    return Food(
        food_id="peanut-001",
        name="Peanut Butter",
        serving_size=30,
        serving_unit="g",
        nutrition=NutritionInfo(
            calories=180,
            protein_g=7,
            carbs_g=6,
            sugar_g=3,
            fat_g=16,
            sodium_mg=150,
        ),
        allergens=["peanuts"],
        category=FoodCategory.PROTEIN,
    )


@pytest.fixture
def virtual_coach(rule_engine, diabetic_user):
    return VirtualCoach(rule_engine, diabetic_user)


@pytest.fixture
def meal_log_store():
    return MealLogStore()


@pytest.fixture
def analytics_service(meal_log_store):
    return AnalyticsService(meal_log_store)


@pytest.fixture
def feedback_store():
    return FeedbackStore()


@pytest.fixture
def feedback_service(feedback_store):
    return FeedbackService(feedback_store)


# =============================================================================
# VIRTUAL COACH TESTS
# =============================================================================

class TestVirtualCoachIntentClassification:
    """Tests for intent classification."""
    
    def test_classify_safety_check(self, virtual_coach):
        """Should classify safety check intents."""
        messages = [
            "Is this safe for me?",
            "Can I eat this?",
            "Should I avoid this food?",
        ]
        for msg in messages:
            intent = virtual_coach._classify_intent(msg)
            assert intent == IntentType.SAFETY_CHECK
    
    def test_classify_exercise_burn(self, virtual_coach):
        """Should classify exercise burn intents."""
        messages = [
            "How much exercise to burn this?",
            "How long to burn this off?",
        ]
        for msg in messages:
            intent = virtual_coach._classify_intent(msg)
            assert intent == IntentType.EXERCISE_BURN
    
    def test_classify_explanation(self, virtual_coach):
        """Should classify explanation intents."""
        messages = [
            "Why is this not recommended?",
            "Explain why I can't eat this",
            "What's wrong with this food?",
        ]
        for msg in messages:
            intent = virtual_coach._classify_intent(msg)
            assert intent == IntentType.EXPLANATION
    
    def test_default_to_general(self, virtual_coach):
        """Should default to general for unknown intents."""
        intent = virtual_coach._classify_intent("Hello there")
        assert intent == IntentType.GENERAL


class TestVirtualCoachResponses:
    """Tests for coach response generation."""
    
    def test_safe_food_response(self, virtual_coach, diabetic_safe_food):
        """Should approve truly safe food for diabetic user."""
        response = virtual_coach.respond("Is this safe for me?", food=diabetic_safe_food)
        
        assert response.safety_level == "safe"
        assert response.confidence == 1.0
        assert "safe" in response.message.lower() or "✅" in response.message
    
    def test_unsafe_food_response(self, virtual_coach, high_sugar_food):
        """Should warn about high sugar for diabetic."""
        response = virtual_coach.respond("Is this safe for me?", food=high_sugar_food)
        
        assert response.safety_level in ["caution", "blocked"]
        assert len(response.violations) > 0
    
    def test_allergen_blocked(self, virtual_coach, peanut_food):
        """Should block allergen foods."""
        response = virtual_coach.respond("Can I eat this?", food=peanut_food)
        
        assert response.safety_level == "blocked"
        assert "🚫" in response.message or "NOT SAFE" in response.message.upper()
    
    def test_exercise_burn_calculation(self, virtual_coach, diabetic_safe_food):
        """Should calculate exercise burn times."""
        response = virtual_coach.respond(
            "How much exercise to burn this?", 
            food=diabetic_safe_food
        )
        
        # Exercise burn is informational, should provide exercise data
        assert "35" in response.message or "calories" in response.message.lower()  # Broccoli calories
        assert "minutes" in response.message.lower()
    
    def test_context_memory(self, virtual_coach, diabetic_safe_food):
        """Should maintain conversation context."""
        virtual_coach.respond("Hello", food=diabetic_safe_food)
        virtual_coach.respond("Is this safe?", food=diabetic_safe_food)
        
        assert len(virtual_coach.context.messages) == 4  # 2 user + 2 assistant
    
    def test_context_limit(self, virtual_coach, diabetic_safe_food):
        """Should limit context to max messages."""
        virtual_coach.context.max_messages = 4
        
        for i in range(10):
            virtual_coach.respond(f"Message {i}", food=diabetic_safe_food)
        
        assert len(virtual_coach.context.messages) == 4


class TestVirtualCoachSafety:
    """Tests for medical rule override."""
    
    def test_rules_override_response(self, rule_engine, diabetic_user, peanut_food):
        """Medical rules must override any response."""
        coach = VirtualCoach(rule_engine, diabetic_user)
        
        # Even if handler somehow says safe, validation should override
        response = coach.respond("Is this safe?", food=peanut_food)
        
        assert response.safety_level == "blocked"
    
    def test_no_diagnosis(self, virtual_coach, high_sugar_food):
        """Coach should not provide medical diagnosis."""
        response = virtual_coach.respond(
            "Do I have diabetes?",
            food=high_sugar_food
        )
        
        # Should not claim to diagnose
        assert "diagnosis" not in response.message.lower()
        assert "doctor" not in response.message.lower() or "consult" in response.message.lower()


# =============================================================================
# ANALYTICS SERVICE TESTS
# =============================================================================

class TestMealLogStore:
    """Tests for meal log storage."""
    
    def test_add_and_retrieve(self, meal_log_store):
        """Should store and retrieve meal logs."""
        entry = MealLogEntry(
            log_id="test-1",
            user_id="user-1",
            timestamp=datetime.now(),
            foods=[{"name": "Apple", "calories": 80}],
            total_nutrition={"calories": 80, "sugar_g": 16},
        )
        
        meal_log_store.add(entry)
        logs = meal_log_store.get_by_user("user-1")
        
        assert len(logs) == 1
        assert logs[0].log_id == "test-1"
    
    def test_get_range(self, meal_log_store):
        """Should filter by date range."""
        today = date.today()
        yesterday = today - timedelta(days=1)
        
        meal_log_store.add(MealLogEntry(
            log_id="today",
            user_id="user-1",
            timestamp=datetime.combine(today, datetime.min.time()),
            foods=[],
            total_nutrition={},
        ))
        
        meal_log_store.add(MealLogEntry(
            log_id="yesterday",
            user_id="user-1",
            timestamp=datetime.combine(yesterday, datetime.min.time()),
            foods=[],
            total_nutrition={},
        ))
        
        today_logs = meal_log_store.get_range("user-1", today, today)
        assert len(today_logs) == 1
        assert today_logs[0].log_id == "today"


class TestAnalyticsService:
    """Tests for analytics computations."""
    
    def test_log_meal(self, analytics_service):
        """Should log meals correctly."""
        entry = analytics_service.log_meal(
            user_id="user-1",
            foods=[
                {"name": "Apple", "calories": 80, "sugar_g": 16},
                {"name": "Banana", "calories": 105, "sugar_g": 14},
            ],
        )
        
        assert entry.total_nutrition["calories"] == 185
        assert entry.total_nutrition["sugar_g"] == 30
    
    def test_compute_trends(self, analytics_service, diabetic_user):
        """Should compute nutrient trends."""
        # Add some meal data
        for i in range(3):
            analytics_service.log_meal(
                user_id=diabetic_user.user_id,
                foods=[{"calories": 500, "sugar_g": 20, "fiber_g": 10}],
            )
        
        trends = analytics_service.compute_daily_trends(
            diabetic_user.user_id,
            diabetic_user.daily_targets,
            days=7,
        )
        
        assert "calories" in trends
        assert "sugar_g" in trends
        assert trends["calories"].weekly_average > 0
    
    def test_compute_health_score(self, analytics_service, diabetic_user):
        """Should compute health score."""
        # Add balanced meal data
        analytics_service.log_meal(
            user_id=diabetic_user.user_id,
            foods=[{
                "calories": 600, 
                "protein_g": 30, 
                "carbs_g": 60,
                "fat_g": 20,
                "sugar_g": 15, 
                "fiber_g": 8,
                "sodium_mg": 500,
            }],
        )
        
        score = analytics_service.compute_health_score(
            diabetic_user.user_id,
            diabetic_user.daily_targets,
            days=7,
        )
        
        assert 0 <= score.overall_score <= 100
        assert len(score.components) == 6
        assert score.explanation != ""
        assert score.grade in ["A", "B", "C", "D", "F"]
    
    def test_empty_score(self, analytics_service, diabetic_user):
        """Should handle empty data gracefully."""
        score = analytics_service.compute_health_score(
            diabetic_user.user_id,
            diabetic_user.daily_targets,
        )
        
        assert score.overall_score == 0
        assert "No meal data" in score.explanation
    
    def test_detect_patterns(self, analytics_service, meal_log_store, diabetic_user):
        """Should detect behavior patterns."""
        targets = diabetic_user.daily_targets
        
        # Add high sodium meals on DIFFERENT DAYS (pattern detection aggregates by day)
        for i in range(5):
            entry = MealLogEntry(
                log_id=f"sodium-test-{i}",
                user_id=diabetic_user.user_id,
                timestamp=datetime.now() - timedelta(days=i),  # Different days
                foods=[{"sodium_mg": targets.sodium_mg * 1.5}],
                total_nutrition={"sodium_mg": targets.sodium_mg * 1.5},  # 50% over limit
            )
            meal_log_store.add(entry)
        
        patterns = analytics_service.detect_patterns(
            diabetic_user.user_id,
            targets,
            days=7,
        )
        
        # Should detect high sodium pattern (5 days >= 3 threshold)
        sodium_patterns = [p for p in patterns if "sodium" in p.pattern_id]
        assert len(sodium_patterns) >= 1
    
    def test_generate_insight(self, analytics_service, diabetic_user):
        """Should generate actionable insights."""
        insight = analytics_service.generate_insight(
            diabetic_user.user_id,
            diabetic_user.daily_targets,
        )
        
        assert len(insight) > 0
        # Should have an emoji indicator
        assert any(c in insight for c in ["✅", "⚠️", "💡", "📊", "📉"])


# =============================================================================
# FEEDBACK SERVICE TESTS
# =============================================================================

class TestFeedbackService:
    """Tests for feedback collection."""
    
    def test_submit_feedback(self, feedback_service):
        """Should store feedback."""
        feedback = feedback_service.submit_feedback(
            user_id="user-1",
            context_type="coach_response",
            context_id="resp-123",
            rating="helpful",
            comment="Very useful!",
        )
        
        assert feedback.feedback_id is not None
        assert feedback.rating == "helpful"
    
    def test_get_rule_summary(self, feedback_service):
        """Should aggregate feedback by rule."""
        # Add multiple feedbacks
        for rating in ["helpful", "helpful", "not_helpful"]:
            feedback_service.submit_feedback(
                user_id="user-1",
                context_type="coach_response",
                context_id=f"resp-{rating}",
                rating=rating,
                related_rule_id="sugar_high",
            )
        
        summary = feedback_service.get_rule_feedback_summary("sugar_high")
        
        assert summary.total_feedback == 3
        assert summary.helpful_count == 2
        assert summary.not_helpful_count == 1
        assert summary.helpfulness_ratio == pytest.approx(0.67, rel=0.1)
    
    def test_threshold_suggestion(self, feedback_service):
        """Should suggest threshold adjustment when needed."""
        # Add many unhelpful feedbacks
        for i in range(15):
            feedback_service.submit_feedback(
                user_id=f"user-{i}",
                context_type="coach_response",
                context_id=f"resp-{i}",
                rating="not_helpful" if i < 10 else "helpful",  # 67% unhelpful
                comment="too strict" if i < 5 else None,
                related_rule_id="sodium_check",
            )
        
        suggestion = feedback_service.suggest_threshold_adjustment(
            rule_id="sodium_check",
            current_threshold=1500,
        )
        
        assert suggestion is not None
        assert suggestion.requires_human_review is True  # CRITICAL
        assert suggestion.suggested_action in ["review", "decrease", "increase"]
    
    def test_no_suggestion_insufficient_data(self, feedback_service):
        """Should not suggest with insufficient data."""
        feedback_service.submit_feedback(
            user_id="user-1",
            context_type="coach_response",
            context_id="resp-1",
            rating="not_helpful",
            related_rule_id="test_rule",
        )
        
        suggestion = feedback_service.suggest_threshold_adjustment("test_rule")
        
        assert suggestion is None  # Not enough data
    
    def test_system_summary(self, feedback_service):
        """Should generate system-wide summary."""
        for i in range(5):
            feedback_service.submit_feedback(
                user_id=f"user-{i}",
                context_type="coach_response",
                context_id=f"resp-{i}",
                rating="helpful",
            )
        
        summary = feedback_service.get_system_summary()
        
        assert summary.total_feedback_count == 5
        assert summary.average_helpfulness == 1.0


# =============================================================================
# CONVERSATION CONTEXT TESTS
# =============================================================================

class TestConversationContext:
    """Tests for conversation context."""
    
    def test_add_message(self):
        """Should add messages to context."""
        ctx = ConversationContext(user_id="user-1", session_id="sess-1")
        ctx.add_message("user", "Hello")
        ctx.add_message("assistant", "Hi there!")
        
        assert len(ctx.messages) == 2
        assert ctx.messages[0].role == "user"
        assert ctx.messages[1].role == "assistant"
    
    def test_window_limit(self):
        """Should maintain message window limit."""
        ctx = ConversationContext(
            user_id="user-1", 
            session_id="sess-1",
            max_messages=3,
        )
        
        for i in range(5):
            ctx.add_message("user", f"Message {i}")
        
        assert len(ctx.messages) == 3
        assert ctx.messages[0].content == "Message 2"  # Oldest kept
    
    def test_serialization(self):
        """Should serialize and deserialize correctly."""
        ctx = ConversationContext(user_id="user-1", session_id="sess-1")
        ctx.add_message("user", "Test message")
        
        data = ctx.to_dict()
        restored = ConversationContext.from_dict(data)
        
        assert restored.user_id == ctx.user_id
        assert len(restored.messages) == 1
        assert restored.messages[0].content == "Test message"


# =============================================================================
# DATA MODEL TESTS
# =============================================================================

class TestMealLogEntry:
    """Tests for meal log entry."""
    
    def test_meal_type_detection(self):
        """Should detect meal type from hour."""
        breakfast = MealLogEntry(
            log_id="1",
            user_id="u1",
            timestamp=datetime(2024, 1, 1, 8, 0),  # 8 AM
            foods=[],
            total_nutrition={},
        )
        assert breakfast.meal_type == "breakfast"
        
        dinner = MealLogEntry(
            log_id="2",
            user_id="u1",
            timestamp=datetime(2024, 1, 1, 19, 0),  # 7 PM
            foods=[],
            total_nutrition={},
        )
        assert dinner.meal_type == "dinner"
    
    def test_weekend_detection(self):
        """Should detect weekend meals."""
        saturday = MealLogEntry(
            log_id="1",
            user_id="u1",
            timestamp=datetime(2024, 12, 21, 12, 0),  # Saturday
            foods=[],
            total_nutrition={},
        )
        assert saturday.is_weekend is True
        
        monday = MealLogEntry(
            log_id="2",
            user_id="u1",
            timestamp=datetime(2024, 12, 23, 12, 0),  # Monday
            foods=[],
            total_nutrition={},
        )
        assert monday.is_weekend is False


class TestHealthScore:
    """Tests for health score model."""
    
    def test_grade_calculation(self):
        """Should calculate correct letter grade."""
        score_a = HealthScore(
            overall_score=92,
            components={},
            explanation="",
            computed_at=datetime.now(),
        )
        assert score_a.grade == "A"
        
        score_c = HealthScore(
            overall_score=72,
            components={},
            explanation="",
            computed_at=datetime.now(),
        )
        assert score_c.grade == "C"
        
        score_f = HealthScore(
            overall_score=50,
            components={},
            explanation="",
            computed_at=datetime.now(),
        )
        assert score_f.grade == "F"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPhase3Integration:
    """Integration tests for Phase 3 components."""
    
    def test_full_coach_flow(
        self, 
        rule_engine, 
        diabetic_user, 
        high_sugar_food
    ):
        """Test complete coach interaction flow."""
        coach = VirtualCoach(rule_engine, diabetic_user)
        
        # User asks about food safety
        response1 = coach.respond("Is this safe for me?", food=high_sugar_food)
        assert response1.safety_level in ["caution", "blocked"]
        
        # User asks for explanation
        response2 = coach.respond("Why is it not recommended?", food=high_sugar_food)
        assert "sugar" in response2.message.lower()
        
        # User asks about exercise
        response3 = coach.respond("How much to burn this?", food=high_sugar_food)
        assert "minutes" in response3.message.lower()
        
        # Context should have all messages
        assert len(coach.context.messages) == 6  # 3 user + 3 assistant
    
    def test_analytics_to_insight_flow(
        self, 
        analytics_service, 
        diabetic_user
    ):
        """Test analytics to insight generation flow."""
        # Log some meals
        for i in range(5):
            analytics_service.log_meal(
                user_id=diabetic_user.user_id,
                foods=[{
                    "calories": 600,
                    "protein_g": 25,
                    "carbs_g": 70,
                    "fat_g": 20,
                    "sugar_g": 30,  # High sugar
                    "fiber_g": 5,
                    "sodium_mg": 800,
                }],
            )
        
        # Get snapshot
        snapshot = analytics_service.get_snapshot(
            diabetic_user.user_id,
            diabetic_user.daily_targets,
        )
        
        assert snapshot.health_score.overall_score >= 0
        assert snapshot.primary_insight != ""
        assert len(snapshot.trends) > 0
    
    def test_feedback_loop_integration(
        self, 
        rule_engine, 
        diabetic_user, 
        high_sugar_food,
        feedback_service,
    ):
        """Test feedback collection after coach response."""
        coach = VirtualCoach(rule_engine, diabetic_user)
        
        # Get coach response
        response = coach.respond("Is this safe?", food=high_sugar_food)
        
        # Submit feedback
        feedback = feedback_service.submit_feedback(
            user_id=diabetic_user.user_id,
            context_type="coach_response",
            context_id=coach.context.session_id,
            rating="helpful",
            related_rule_id="sugar_high" if response.violations else None,
        )
        
        assert feedback is not None
        
        # Check feedback is stored
        all_feedback = feedback_service.store.get_all()
        assert len(all_feedback) == 1
