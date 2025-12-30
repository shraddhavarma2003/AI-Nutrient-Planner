"""
Virtual Coach Service

Provides context-aware conversational interface for nutrition guidance.
Uses medical rules from Phase 1 and food analysis from Phase 2.

CRITICAL: Medical rules ALWAYS override AI-generated content.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable
from enum import Enum

from models.food import Food, NutritionInfo
from models.user import UserProfile, HealthCondition, DailyIntake
from models.conversation import ConversationContext, ConversationMessage, CoachResponse
from rules.engine import RuleEngine, RuleViolation, Severity


class IntentType(Enum):
    """Types of user intents the coach can handle."""
    SAFETY_CHECK = "safety_check"      # "Is this safe for me?"
    EXERCISE_BURN = "exercise_burn"    # "How much exercise to burn this?"
    EXPLANATION = "explanation"        # "Why is this not recommended?"
    MEAL_HELP = "meal_help"            # "Fix my meal" / "Make it healthier"
    NUTRITION_INFO = "nutrition_info"  # "How many calories in this?"
    GENERAL = "general"                # General questions


@dataclass
class ContextData:
    """Gathered context for response generation."""
    user_profile: UserProfile
    recent_meals: List[Dict[str, Any]] = field(default_factory=list)
    current_food: Optional[Food] = None
    daily_intake: Optional[DailyIntake] = None
    active_violations: List[RuleViolation] = field(default_factory=list)


@dataclass 
class RawResponse:
    """Internal response before validation."""
    message: str
    safety_level: str = "unknown"
    confidence: float = 1.0
    violations: List[RuleViolation] = field(default_factory=list)
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    override_reason: Optional[str] = None


class VirtualCoach:
    """
    Context-aware virtual nutrition coach.
    
    DESIGN PRINCIPLES:
    1. Medical rules ALWAYS override AI suggestions
    2. Responses are explainable and transparent
    3. Conversation memory is limited to prevent bloat
    4. Never provides medical diagnosis
    """
    
    # Intent classification patterns (ordered by priority)
    INTENT_PATTERNS = {
        IntentType.SAFETY_CHECK: [
            r"is (?:this|it) safe",
            r"can i (?:eat|have|consume)",
            r"should i (?:eat|avoid|skip)",
            r"safe for (?:me|my)",
            r"okay for (?:me|my|diabetic|hypertension)",
        ],
        IntentType.EXERCISE_BURN: [
            r"how (?:much|long) (?:to |exercise |walk |run )?burn",
            r"burn (?:this|off|these|it)",
            r"exercise (?:for|to)",
            r"calories (?:burn|burned)",
        ],
        IntentType.EXPLANATION: [
            r"why (?:is|was|isn't|can't|shouldn't)",
            r"explain",
            r"what's wrong",
            r"reason",
            r"not recommended",
        ],
        IntentType.MEAL_HELP: [
            r"fix (?:my|this) meal",
            r"make (?:it|this) health",
            r"alternative",
            r"instead of",
            r"swap",
            r"replace",
        ],
        IntentType.NUTRITION_INFO: [
            r"how many (?:calories|carbs|protein)",
            r"nutrition (?:info|information|facts)",
            r"what's in",
        ],
    }
    
    # Calorie burn rates (kcal per minute, average adult)
    EXERCISE_BURN_RATES = {
        "walking": 5,
        "jogging": 10,
        "running": 12,
        "cycling": 8,
        "swimming": 11,
        "yoga": 4,
        "weight_training": 6,
    }
    
    def __init__(
        self,
        rule_engine: RuleEngine,
        user_profile: UserProfile,
        meal_fixer: Optional[Any] = None,
    ):
        """
        Initialize Virtual Coach.
        
        Args:
            rule_engine: Rule engine for safety validation
            user_profile: User's health profile
            meal_fixer: Optional MealFixer for fix suggestions
        """
        self.rule_engine = rule_engine
        self.user = user_profile
        self.meal_fixer = meal_fixer
        self.context = ConversationContext(
            user_id=user_profile.user_id,
            session_id=str(uuid.uuid4()),
        )
        
        # Handler mapping
        self._handlers: Dict[IntentType, Callable] = {
            IntentType.SAFETY_CHECK: self._handle_safety_check,
            IntentType.EXERCISE_BURN: self._handle_exercise_burn,
            IntentType.EXPLANATION: self._handle_explanation,
            IntentType.MEAL_HELP: self._handle_meal_help,
            IntentType.NUTRITION_INFO: self._handle_nutrition_info,
            IntentType.GENERAL: self._handle_general,
        }
    
    def respond(
        self, 
        user_message: str, 
        food: Optional[Food] = None,
        daily_intake: Optional[DailyIntake] = None,
    ) -> CoachResponse:
        """
        Generate a response to user message.
        
        CRITICAL: Medical rules ALWAYS override AI-generated content.
        
        Args:
            user_message: User's question or message
            food: Optional food being discussed
            daily_intake: Optional current daily intake
            
        Returns:
            CoachResponse with message and safety information
        """
        try:
            # Step 1: Add message to context
            self.context.add_message("user", user_message)
            
            # Step 2: Classify intent
            intent = self._classify_intent(user_message)
            
            # Step 3: Gather context data
            context_data = self._gather_context(intent, food, daily_intake)
            
            # Step 4: Route to appropriate handler
            handler = self._handlers.get(intent, self._handle_general)
            raw_response = handler(user_message, context_data)
            
            # Step 5: CRITICAL - Validate response against rules
            validated_response = self._validate_response(raw_response, context_data)
            
            # Step 6: Build final response
            final_response = self._build_response(validated_response)
            
            # Step 7: Add to context
            self.context.add_message("assistant", final_response.message)
            
            return final_response
            
        except Exception as e:
            # Graceful fallback on error
            return self._handle_error(str(e))
    
    def _classify_intent(self, message: str) -> IntentType:
        """Classify user intent using pattern matching."""
        message_lower = message.lower().strip()
        
        for intent_type, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent_type
        
        return IntentType.GENERAL
    
    def _gather_context(
        self, 
        intent: IntentType, 
        food: Optional[Food],
        daily_intake: Optional[DailyIntake],
    ) -> ContextData:
        """Gather relevant context for response generation."""
        # Evaluate food against rules if provided
        violations = []
        if food:
            violations = self.rule_engine.evaluate(food, self.user, daily_intake)
        
        return ContextData(
            user_profile=self.user,
            current_food=food,
            daily_intake=daily_intake,
            active_violations=violations,
        )
    
    def _handle_safety_check(
        self, message: str, context: ContextData
    ) -> RawResponse:
        """Handle 'Is this safe?' type questions."""
        if not context.current_food:
            # RAG RULE: Never ask for manual input - state what's missing
            return RawResponse(
                message="I don't have any food context currently. Please scan a food image first using the camera icon, and I'll automatically analyze it for safety.",
                safety_level="unknown",
                confidence=1.0,
            )
        
        violations = context.active_violations
        verdict = self.rule_engine.get_final_verdict(violations)
        
        if verdict == Severity.BLOCK:
            return RawResponse(
                message=self._format_block_response(context.current_food, violations),
                safety_level="blocked",
                violations=violations,
                confidence=1.0,
            )
        elif verdict in [Severity.ALERT, Severity.WARN]:
            return RawResponse(
                message=self._format_caution_response(context.current_food, violations),
                safety_level="caution",
                violations=violations,
                confidence=1.0,
            )
        else:
            return RawResponse(
                message=self._format_safe_response(context.current_food),
                safety_level="safe",
                confidence=1.0,
            )
    
    def _handle_exercise_burn(
        self, message: str, context: ContextData
    ) -> RawResponse:
        """Handle 'How much exercise to burn?' questions."""
        # Get calories from food or daily intake
        if context.current_food:
            calories = context.current_food.nutrition.calories
            food_name = context.current_food.name
        elif context.daily_intake:
            calories = context.daily_intake.calories
            food_name = "today's intake"
        else:
            return RawResponse(
                message="Please share the food or meal you'd like to know about.",
                safety_level="unknown",
                confidence=1.0,
            )
        
        # Calculate burn times
        burn_times = {
            name: round(calories / rate)
            for name, rate in self.EXERCISE_BURN_RATES.items()
        }
        
        # Format response
        lines = [f"To burn off **{food_name}** ({calories:.0f} calories):\n"]
        for exercise, minutes in sorted(burn_times.items(), key=lambda x: x[1]):
            lines.append(f"• {exercise.replace('_', ' ').title()}: ~{minutes} minutes")
        
        lines.append("\n_Note: These are estimates based on average adult metabolism._")
        
        return RawResponse(
            message="\n".join(lines),
            safety_level="safe",
            confidence=0.9,  # Estimates
            data={"calories": calories, "burn_times": burn_times},
        )
    
    def _handle_explanation(
        self, message: str, context: ContextData
    ) -> RawResponse:
        """Handle 'Why is this not recommended?' questions."""
        violations = context.active_violations
        
        if not violations:
            if context.current_food:
                return RawResponse(
                    message=f"**{context.current_food.name}** doesn't have any issues for your health profile. It's safe to consume!",
                    safety_level="safe",
                    confidence=1.0,
                )
            return RawResponse(
                message="Please share the food you'd like me to explain.",
                safety_level="unknown",
                confidence=1.0,
            )
        
        # Build detailed explanation
        lines = ["Here's why this food needs attention:\n"]
        
        for i, violation in enumerate(violations, 1):
            severity_emoji = {
                Severity.BLOCK: "🚫",
                Severity.ALERT: "⚠️",
                Severity.WARN: "💡",
                Severity.ALLOW: "✅",
            }.get(violation.severity, "•")
            
            lines.append(f"{severity_emoji} **{violation.category}**: {violation.message}")
            if violation.suggestion:
                lines.append(f"   → {violation.suggestion}")
        
        # Add condition-specific context
        conditions = [c.value for c in context.user_profile.conditions]
        if conditions:
            lines.append(f"\n_These checks are specific to your conditions: {', '.join(conditions)}_")
        
        return RawResponse(
            message="\n".join(lines),
            safety_level="caution" if violations else "safe",
            violations=violations,
            confidence=1.0,
        )
    
    def _handle_meal_help(
        self, message: str, context: ContextData
    ) -> RawResponse:
        """Handle 'Fix my meal' / 'Make it healthier' questions."""
        if not context.current_food:
            return RawResponse(
                message="Please share the meal you'd like me to help improve.",
                safety_level="unknown",
                confidence=1.0,
            )
        
        # Use MealFixer if available
        if self.meal_fixer:
            fixes = self.meal_fixer.analyze_and_fix(
                [context.current_food], 
                context.user_profile,
                context.daily_intake
            )
            
            if fixes:
                lines = [f"Here are some ways to improve **{context.current_food.name}**:\n"]
                for fix in fixes[:3]:  # Limit to top 3
                    lines.append(f"• **{fix.fix_type.value.title()}**: {fix.fix_description}")
                    if fix.replacement_food:
                        lines.append(f"   → Try: {fix.replacement_food}")
                
                return RawResponse(
                    message="\n".join(lines),
                    safety_level="caution",
                    suggestions=[f.to_dict() for f in fixes],
                    confidence=0.9,
                )
        
        # Fallback: basic suggestions based on violations
        violations = context.active_violations
        if violations:
            lines = [f"To make **{context.current_food.name}** healthier:\n"]
            
            for v in violations[:3]:
                if "sugar" in v.rule_id.lower():
                    lines.append("• Consider a smaller portion or sugar-free alternative")
                elif "sodium" in v.rule_id.lower():
                    lines.append("• Look for low-sodium versions or rinse canned items")
                elif "fat" in v.rule_id.lower():
                    lines.append("• Choose grilled or baked instead of fried")
                elif "calorie" in v.rule_id.lower():
                    lines.append("• Try half portion or add more vegetables")
            
            return RawResponse(
                message="\n".join(lines),
                safety_level="caution",
                violations=violations,
                confidence=0.8,
            )
        
        return RawResponse(
            message=f"**{context.current_food.name}** looks good for your profile! No changes needed.",
            safety_level="safe",
            confidence=1.0,
        )
    
    def _handle_nutrition_info(
        self, message: str, context: ContextData
    ) -> RawResponse:
        """Handle nutrition information questions."""
        if not context.current_food:
            return RawResponse(
                message="Please share the food you'd like nutrition information for.",
                safety_level="unknown",
                confidence=1.0,
            )
        
        food = context.current_food
        n = food.nutrition
        
        lines = [
            f"**{food.name}** nutrition (per {food.serving_size}{food.serving_unit}):\n",
            f"• Calories: {n.calories:.0f}",
            f"• Protein: {n.protein_g:.1f}g",
            f"• Carbs: {n.carbs_g:.1f}g (Sugar: {n.sugar_g:.1f}g)",
            f"• Fat: {n.fat_g:.1f}g",
            f"• Fiber: {n.fiber_g:.1f}g",
            f"• Sodium: {n.sodium_mg:.0f}mg",
        ]
        
        return RawResponse(
            message="\n".join(lines),
            safety_level="safe",
            confidence=1.0,
            data={"nutrition": food.nutrition.__dict__},
        )
    
    def _handle_general(
        self, message: str, context: ContextData
    ) -> RawResponse:
        """Handle general questions."""
        # RAG RULE: Use food context if available, never ask for manual input
        lines = []
        
        if context.current_food:
            # We have food context - reference it
            food = context.current_food
            lines.append(f"I see you've scanned **{food.name}** ({food.nutrition.calories:.0f} cal).")
            lines.append("")
            lines.append("I can help you with:")
            lines.append("• **Safety check**: \"Is this safe for me?\"")
            lines.append("• **Exercise**: \"How much to burn this?\"")
            lines.append("• **Nutrition**: \"What's the breakdown?\"")
            lines.append("• **Healthier options**: \"How to make this better?\"")
        else:
            # No food context - guide to scan
            lines.append("I'm ready to help with your nutrition questions!")
            lines.append("")
            lines.append("**To get started:**")
            lines.append("1. Scan a food item using the camera icon")
            lines.append("2. I'll automatically detect the food and its nutrition")
            lines.append("3. Then ask me anything about it!")
            lines.append("")
            lines.append("I can analyze safety based on your medical profile, calculate exercise needs, and suggest healthier alternatives.")
        
        return RawResponse(
            message="\n".join(lines),
            safety_level="unknown" if not context.current_food else "safe",
            confidence=1.0,
        )
    
    def _validate_response(
        self, response: RawResponse, context: ContextData
    ) -> RawResponse:
        """
        CRITICAL: Ensure response doesn't contradict medical rules.
        
        This is the safety net - even if handler makes a mistake,
        this validation catches it.
        """
        if context.active_violations:
            verdict = self.rule_engine.get_final_verdict(context.active_violations)
            
            # If there are BLOCK violations, response MUST reflect that
            if verdict == Severity.BLOCK and response.safety_level not in ["blocked"]:
                return RawResponse(
                    message=self._format_block_response(
                        context.current_food, 
                        context.active_violations
                    ),
                    safety_level="blocked",
                    violations=context.active_violations,
                    confidence=1.0,
                    override_reason="Medical safety rule applied",
                )
        
        return response
    
    def _build_response(self, raw: RawResponse) -> CoachResponse:
        """Convert internal response to external format."""
        return CoachResponse(
            message=raw.message,
            safety_level=raw.safety_level,
            confidence=raw.confidence,
            violations=[v.to_dict() for v in raw.violations] if raw.violations else [],
            suggestions=raw.suggestions,
            data=raw.data,
            override_reason=raw.override_reason,
        )
    
    def _format_block_response(
        self, food: Optional[Food], violations: List[RuleViolation]
    ) -> str:
        """Format response for blocked foods."""
        food_name = food.name if food else "This food"
        
        lines = [f"🚫 **{food_name}** is NOT SAFE for you.\n"]
        
        for v in violations:
            if v.severity == Severity.BLOCK:
                lines.append(f"• {v.message}")
        
        lines.append("\n**Please avoid this food or consult your healthcare provider.**")
        
        return "\n".join(lines)
    
    def _format_caution_response(
        self, food: Optional[Food], violations: List[RuleViolation]
    ) -> str:
        """Format response for foods requiring caution."""
        food_name = food.name if food else "This food"
        
        lines = [f"⚠️ **{food_name}** requires caution.\n"]
        
        for v in violations:
            lines.append(f"• {v.message}")
            if v.suggestion:
                lines.append(f"   → {v.suggestion}")
        
        lines.append("\n_You can consume this in moderation, but consider the suggestions above._")
        
        return "\n".join(lines)
    
    def _format_safe_response(self, food: Optional[Food]) -> str:
        """Format response for safe foods."""
        food_name = food.name if food else "This food"
        return f"✅ **{food_name}** is safe for your health profile. Enjoy!"
    
    def _handle_error(self, error_message: str) -> CoachResponse:
        """Handle errors gracefully."""
        return CoachResponse(
            message="I encountered an issue processing your request. Please try again or rephrase your question.",
            safety_level="unknown",
            confidence=0.0,
            requires_manual_check=True,
            data={"error": error_message},
        )
    
    def clear_context(self):
        """Clear conversation context."""
        self.context.clear()
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current context."""
        return {
            "session_id": self.context.session_id,
            "message_count": len(self.context.messages),
            "last_activity": self.context.last_activity.isoformat(),
        }
