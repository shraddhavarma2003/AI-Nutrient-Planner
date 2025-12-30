"""
Conversation Data Models

Defines conversation context and message structures for the Virtual Coach.
Supports limited conversation memory and context tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal


@dataclass
class ConversationMessage:
    """
    Single message in a conversation.
    
    Stores both user and assistant messages with metadata
    for context-aware responses.
    """
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMessage":
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationContext:
    """
    Context for a conversation session.
    
    Maintains limited message history and references to
    relevant data (recent meals, active violations).
    
    DESIGN: Memory is limited to prevent context bloat
    and ensure responses remain relevant.
    """
    user_id: str
    session_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    
    # Limited window - last N messages only
    max_messages: int = 10
    
    # References to recent data (IDs only, not full objects)
    recent_meal_ids: List[str] = field(default_factory=list)
    
    # Active violations from last food check
    active_violation_ids: List[str] = field(default_factory=list)
    
    # Session metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """
        Add a message to the context.
        
        Automatically trims old messages to maintain window limit.
        """
        message = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )
        self.messages.append(message)
        self.last_activity = datetime.now()
        
        # Maintain window limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_recent_messages(self, count: int = 5) -> List[ConversationMessage]:
        """Get the most recent N messages."""
        return self.messages[-count:]
    
    def clear(self):
        """Clear conversation history while preserving session info."""
        self.messages = []
        self.recent_meal_ids = []
        self.active_violation_ids = []
    
    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "max_messages": self.max_messages,
            "recent_meal_ids": self.recent_meal_ids,
            "active_violation_ids": self.active_violation_ids,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationContext":
        """Create from dictionary."""
        ctx = cls(
            user_id=data["user_id"],
            session_id=data["session_id"],
            max_messages=data.get("max_messages", 10),
            recent_meal_ids=data.get("recent_meal_ids", []),
            active_violation_ids=data.get("active_violation_ids", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
        )
        ctx.messages = [
            ConversationMessage.from_dict(m) for m in data.get("messages", [])
        ]
        return ctx


@dataclass
class CoachResponse:
    """
    Response from the Virtual Coach.
    
    Contains the message, safety information, and supporting data
    for transparency and explainability.
    """
    message: str
    safety_level: Literal["safe", "caution", "blocked", "unknown"]
    
    # Confidence in the response (1.0 for rule-based, lower for estimates)
    confidence: float = 1.0
    
    # Supporting data for transparency
    violations: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)
    
    # Flags
    requires_manual_check: bool = False
    override_reason: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "message": self.message,
            "safety_level": self.safety_level,
            "confidence": self.confidence,
            "violations": self.violations,
            "suggestions": self.suggestions,
            "data": self.data,
            "requires_manual_check": self.requires_manual_check,
            "override_reason": self.override_reason,
        }
