"""
Ethics and Safety Guards

Responsible AI layer that prevents:
- Medical diagnosis attempts
- Hallucinated health claims
- Misleading nutrition advice

All guards are deterministic and explainable.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum


class GuardType(Enum):
    """Type of safety guard triggered."""
    DIAGNOSIS_ATTEMPT = "diagnosis_attempt"
    HALLUCINATION_RISK = "hallucination_risk"
    UNSAFE_CLAIM = "unsafe_claim"
    MISSING_DISCLAIMER = "missing_disclaimer"


@dataclass
class GuardResult:
    """Result of a safety guard check."""
    passed: bool
    guard_type: Optional[GuardType] = None
    reason: Optional[str] = None
    suggested_response: Optional[str] = None


class AntiDiagnosisGuard:
    """
    Prevents the system from providing medical diagnoses.
    
    CRITICAL: This guard must NEVER be bypassed.
    """
    
    # Patterns that indicate diagnosis requests
    DIAGNOSIS_PATTERNS = [
        (r"do i have (diabetes|cancer|heart disease|hypertension|anemia)", "disease_query"),
        (r"diagnose (me|my|this)", "diagnose_request"),
        (r"what (disease|condition|illness) do i have", "condition_query"),
        (r"am i (sick|ill|diseased)", "health_status"),
        (r"(interpret|explain|analyze) my (blood|lab|test|medical) (results?|report)", "test_interpretation"),
        (r"what does my (blood sugar|a1c|cholesterol|bp|blood pressure) (mean|indicate)", "metric_interpretation"),
        (r"is my \w+ (normal|abnormal|high|low)", "metric_assessment"),
        (r"should i (take|stop|change) my (medication|medicine|drugs?)", "medication_advice"),
    ]
    
    SAFE_RESPONSE = """
⚠️ **I cannot provide medical diagnoses or interpret medical tests.**

I'm a nutrition assistant, not a medical professional. I can help with:

✅ **Nutritional information** about foods
✅ **Safety checks** based on your health profile
✅ **Healthy eating guidance** and meal suggestions
✅ **Exercise estimates** for calorie burn

For medical concerns, lab results, or medication questions, please consult a **healthcare professional**.
"""
    
    def check(self, user_message: str) -> GuardResult:
        """Check if message is requesting a diagnosis."""
        message_lower = user_message.lower()
        
        for pattern, category in self.DIAGNOSIS_PATTERNS:
            if re.search(pattern, message_lower):
                return GuardResult(
                    passed=False,
                    guard_type=GuardType.DIAGNOSIS_ATTEMPT,
                    reason=f"Diagnosis request detected: {category}",
                    suggested_response=self.SAFE_RESPONSE,
                )
        
        return GuardResult(passed=True)


class AntiHallucinationGuard:
    """
    Prevents hallucinated health claims in responses.
    
    Blocks:
    - Cure/treatment claims
    - Guaranteed outcomes
    - Invented nutrition data
    """
    
    # Patterns that should NEVER appear in responses
    BLOCKED_CLAIM_PATTERNS = [
        (r"(cures?|will cure) (cancer|diabetes|disease|\w+)", "cure_claim"),
        (r"(treats?|will treat) (your )?(cancer|diabetes|disease|\w+)", "treatment_claim"),
        (r"(prevents?|will prevent) (cancer|diabetes|heart disease)", "prevention_claim"),
        (r"guaranteed (to |results?|outcome)", "guarantee"),
        (r"100% (safe|effective|guaranteed)", "absolute_claim"),
        (r"(eliminates?|removes?) (all |your )?(toxins?|diseases?)", "detox_claim"),
        (r"(miracle|magic|secret) (cure|food|diet|solution)", "miracle_claim"),
        (r"(doctors? (don't want|hate)|big pharma)", "conspiracy"),
    ]
    
    # Required qualifiers for health-related statements
    REQUIRED_QUALIFIERS = [
        "may help",
        "can support",
        "generally considered",
        "according to",
        "based on",
        "consult",
    ]
    
    def check_response(self, response: str) -> GuardResult:
        """Check if response contains hallucinated claims."""
        response_lower = response.lower()
        
        for pattern, category in self.BLOCKED_CLAIM_PATTERNS:
            if re.search(pattern, response_lower):
                return GuardResult(
                    passed=False,
                    guard_type=GuardType.HALLUCINATION_RISK,
                    reason=f"Blocked claim detected: {category}",
                )
        
        return GuardResult(passed=True)
    
    def sanitize_response(self, response: str) -> str:
        """Remove or flag problematic claims from response."""
        sanitized = response
        
        for pattern, category in self.BLOCKED_CLAIM_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                # Replace with safe alternative
                sanitized = re.sub(
                    pattern,
                    "[claim removed - consult healthcare provider]",
                    sanitized,
                    flags=re.IGNORECASE,
                )
        
        return sanitized


class DisclaimerInjector:
    """
    Ensures all health-related responses include appropriate disclaimers.
    """
    
    STANDARD_DISCLAIMER = """
─────────────────────────────────────────
ℹ️ *This is nutritional guidance, not medical advice.*
   *Always consult a healthcare provider for medical decisions.*
─────────────────────────────────────────
"""
    
    SHORT_DISCLAIMER = "\n\n_ℹ️ Not medical advice. Consult a healthcare provider for medical concerns._"
    
    # Topics that require disclaimers
    DISCLAIMER_TRIGGERS = [
        r"(safe|unsafe|dangerous) (for|to)",
        r"(diabetes|hypertension|heart|obesity|allergy)",
        r"(blood sugar|cholesterol|sodium|calories)",
        r"(health|healthy|unhealthy)",
        r"(recommend|suggest|advise)",
    ]
    
    def needs_disclaimer(self, response: str) -> bool:
        """Check if response needs a disclaimer."""
        for pattern in self.DISCLAIMER_TRIGGERS:
            if re.search(pattern, response, re.IGNORECASE):
                return True
        return False
    
    def inject(self, response: str, short: bool = True) -> str:
        """Add disclaimer to response if needed."""
        if not self.needs_disclaimer(response):
            return response
        
        # Don't double-add disclaimers
        if "not medical advice" in response.lower():
            return response
        
        disclaimer = self.SHORT_DISCLAIMER if short else self.STANDARD_DISCLAIMER
        return response + disclaimer


class SafetyGuardChain:
    """
    Chain of all safety guards.
    
    Processes input and output through all guards in sequence.
    """
    
    def __init__(self):
        self.anti_diagnosis = AntiDiagnosisGuard()
        self.anti_hallucination = AntiHallucinationGuard()
        self.disclaimer = DisclaimerInjector()
    
    def check_input(self, user_message: str) -> Tuple[bool, Optional[str]]:
        """
        Check user input for blocked patterns.
        
        Returns:
            (should_proceed, override_response)
        """
        # Check for diagnosis attempts
        result = self.anti_diagnosis.check(user_message)
        if not result.passed:
            return False, result.suggested_response
        
        return True, None
    
    def process_output(self, response: str) -> str:
        """
        Process AI response through safety guards.
        
        - Sanitizes hallucinated claims
        - Adds disclaimers where needed
        """
        # Sanitize any hallucinated claims
        result = self.anti_hallucination.check_response(response)
        if not result.passed:
            response = self.anti_hallucination.sanitize_response(response)
        
        # Add disclaimer if needed
        response = self.disclaimer.inject(response)
        
        return response


# Global instance for easy access
safety_guards = SafetyGuardChain()
