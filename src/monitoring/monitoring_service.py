"""
Monitoring Service

Lightweight monitoring for free-tier friendly deployment.
Logs events to JSON file for debugging and quality tracking.
"""

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Optional, Literal


@dataclass
class MonitoringEvent:
    """Event for monitoring and alerting."""
    event_type: str
    severity: Literal["info", "warning", "error", "critical"]
    message: str
    metadata: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "message": self.message,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class MonitoringService:
    """
    Lightweight monitoring for free-tier deployment.
    
    Features:
    - JSON line logging (JSONL) for easy parsing
    - File-based storage (no external dependencies)
    - Log rotation support
    """
    
    # Event types
    EVENT_OCR_FAILURE = "ocr_failure"
    EVENT_RULE_VIOLATION = "rule_violation"
    EVENT_COACH_ERROR = "coach_error"
    EVENT_ANALYTICS_ANOMALY = "analytics_anomaly"
    EVENT_FEEDBACK_SUBMITTED = "feedback_submitted"
    EVENT_API_ERROR = "api_error"
    
    def __init__(
        self, 
        log_dir: str = "logs",
        log_file: str = "monitoring.jsonl",
        max_file_size_mb: float = 10.0,
    ):
        """
        Initialize Monitoring Service.
        
        Args:
            log_dir: Directory for log files
            log_file: Name of the log file
            max_file_size_mb: Max file size before rotation
        """
        self.log_dir = log_dir
        self.log_file = log_file
        self.log_path = os.path.join(log_dir, log_file)
        self.max_file_size = max_file_size_mb * 1024 * 1024
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
    
    def log_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        metadata: Dict[str, Any] = None,
    ):
        """
        Log a monitoring event.
        
        Args:
            event_type: Type of event
            severity: info, warning, error, or critical
            message: Human-readable message
            metadata: Additional structured data
        """
        event = MonitoringEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            metadata=metadata or {},
            timestamp=datetime.now().isoformat(),
        )
        
        self._emit(event)
    
    def log_ocr_failure(
        self, 
        reason: str, 
        input_preview: str = "",
        user_id: Optional[str] = None,
    ):
        """Log OCR extraction failure."""
        self.log_event(
            event_type=self.EVENT_OCR_FAILURE,
            severity="warning",
            message=f"OCR extraction failed: {reason}",
            metadata={
                "input_preview": input_preview[:100] if input_preview else "",
                "user_id": user_id,
            },
        )
    
    def log_rule_violation(
        self,
        user_id: str,
        rule_id: str,
        severity: str,
        food_name: Optional[str] = None,
    ):
        """Log rule violation for analytics."""
        self.log_event(
            event_type=self.EVENT_RULE_VIOLATION,
            severity="info",
            message=f"Rule {rule_id} triggered",
            metadata={
                "user_id": user_id,
                "rule_id": rule_id,
                "rule_severity": severity,
                "food_name": food_name,
            },
        )
    
    def log_coach_error(
        self,
        error_message: str,
        user_id: Optional[str] = None,
        intent: Optional[str] = None,
    ):
        """Log Virtual Coach error."""
        self.log_event(
            event_type=self.EVENT_COACH_ERROR,
            severity="error",
            message=f"Coach error: {error_message}",
            metadata={
                "user_id": user_id,
                "intent": intent,
            },
        )
    
    def log_analytics_anomaly(
        self,
        anomaly_type: str,
        user_id: str,
        details: Dict[str, Any],
    ):
        """Log analytics anomaly detection."""
        self.log_event(
            event_type=self.EVENT_ANALYTICS_ANOMALY,
            severity="info",
            message=f"Analytics anomaly: {anomaly_type}",
            metadata={
                "user_id": user_id,
                "anomaly_type": anomaly_type,
                **details,
            },
        )
    
    def log_feedback(
        self,
        user_id: str,
        feedback_type: str,
        rating: str,
    ):
        """Log feedback submission."""
        self.log_event(
            event_type=self.EVENT_FEEDBACK_SUBMITTED,
            severity="info",
            message=f"Feedback submitted: {rating}",
            metadata={
                "user_id": user_id,
                "feedback_type": feedback_type,
                "rating": rating,
            },
        )
    
    def log_api_error(
        self,
        endpoint: str,
        error_message: str,
        status_code: Optional[int] = None,
    ):
        """Log API error."""
        self.log_event(
            event_type=self.EVENT_API_ERROR,
            severity="error",
            message=f"API error on {endpoint}: {error_message}",
            metadata={
                "endpoint": endpoint,
                "status_code": status_code,
            },
        )
    
    def _emit(self, event: MonitoringEvent):
        """Emit event to log file."""
        try:
            # Check if rotation needed
            self._rotate_if_needed()
            
            # Append to log file
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), default=str) + "\n")
        except Exception as e:
            # Fallback to stderr if file logging fails
            import sys
            print(f"Monitoring error: {e}", file=sys.stderr)
            print(json.dumps(event.to_dict(), default=str), file=sys.stderr)
    
    def _rotate_if_needed(self):
        """Rotate log file if it exceeds max size."""
        if not os.path.exists(self.log_path):
            return
        
        if os.path.getsize(self.log_path) > self.max_file_size:
            # Simple rotation: rename current file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rotated_name = f"{self.log_file}.{timestamp}"
            rotated_path = os.path.join(self.log_dir, rotated_name)
            os.rename(self.log_path, rotated_path)
    
    def get_recent_events(
        self, 
        count: int = 100, 
        event_type: Optional[str] = None,
    ) -> list:
        """
        Get recent monitoring events.
        
        Args:
            count: Number of events to retrieve
            event_type: Filter by event type
            
        Returns:
            List of event dictionaries
        """
        if not os.path.exists(self.log_path):
            return []
        
        events = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            event = json.loads(line)
                            if event_type is None or event.get("event_type") == event_type:
                                events.append(event)
                        except json.JSONDecodeError:
                            continue
        except Exception:
            return []
        
        return events[-count:]
    
    def get_error_count(self, hours: int = 24) -> Dict[str, int]:
        """
        Get count of errors in the last N hours.
        
        Returns:
            Dictionary of error type to count
        """
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(hours=hours)
        events = self.get_recent_events(count=1000)
        
        counts = {}
        for event in events:
            try:
                event_time = datetime.fromisoformat(event["timestamp"])
                if event_time >= cutoff and event.get("severity") in ["error", "critical"]:
                    event_type = event.get("event_type", "unknown")
                    counts[event_type] = counts.get(event_type, 0) + 1
            except (ValueError, KeyError):
                continue
        
        return counts
