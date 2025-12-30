"""
Security Utilities

Data protection, access control, and upload sanitization.
Designed for privacy-by-design.
"""

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Set, List
from enum import Enum


class UserRole(Enum):
    """User roles for access control."""
    GUEST = "guest"
    USER = "user"
    ADMIN = "admin"


@dataclass
class DataRetentionPolicy:
    """Data retention configuration."""
    medical_profile: str = "until_user_deletes"
    food_logs_days: int = 90
    chat_history: str = "session_only"
    uploaded_images: str = "immediate_delete"


class AccessControl:
    """
    Role-based access control for sensitive data.
    
    Principle: Least privilege - users only access their own data.
    """
    
    # Permission definitions
    PERMISSIONS = {
        UserRole.GUEST: frozenset([
            "view_public_nutrition",
        ]),
        UserRole.USER: frozenset([
            "view_public_nutrition",
            "read_own_profile",
            "write_own_profile",
            "read_own_logs",
            "write_own_logs",
            "use_coach",
            "submit_feedback",
        ]),
        UserRole.ADMIN: frozenset([
            "view_public_nutrition",
            "read_aggregated_stats",  # Never individual user data
            "view_system_health",
            "manage_rules",  # With audit logging
        ]),
    }
    
    def __init__(self):
        self._audit_log: List[dict] = []
    
    def check_permission(
        self,
        user_role: UserRole,
        action: str,
        resource_owner_id: Optional[str] = None,
        requester_id: Optional[str] = None,
    ) -> bool:
        """
        Check if role has permission for action.
        
        For user data, also checks ownership.
        """
        allowed = self.PERMISSIONS.get(user_role, frozenset())
        
        if action not in allowed:
            self._log_access_denied(user_role, action, requester_id)
            return False
        
        # For "own_" permissions, verify ownership
        if "own_" in action and resource_owner_id:
            if requester_id != resource_owner_id:
                self._log_access_denied(user_role, action, requester_id)
                return False
        
        return True
    
    def _log_access_denied(
        self,
        role: UserRole,
        action: str,
        user_id: Optional[str],
    ):
        """Log access denial for security auditing."""
        self._audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": "access_denied",
            "role": role.value,
            "action": action,
            "user_id_hash": hashlib.sha256(
                (user_id or "anonymous").encode()
            ).hexdigest()[:16],
        })
    
    def get_audit_log(self) -> List[dict]:
        """Get recent access denial events."""
        return self._audit_log[-100:]  # Last 100 events


class UploadSanitizer:
    """
    Secure file upload handling.
    
    Validates file type, size, and content.
    """
    
    ALLOWED_MIME_TYPES = frozenset([
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/gif",
    ])
    
    ALLOWED_EXTENSIONS = frozenset([
        ".jpg", ".jpeg", ".png", ".webp", ".gif"
    ])
    
    MAX_FILE_SIZE_MB = 10
    
    # Magic bytes for image formats
    IMAGE_SIGNATURES = {
        b'\xff\xd8\xff': "image/jpeg",
        b'\x89PNG': "image/png",
        b'RIFF': "image/webp",  # WebP starts with RIFF
        b'GIF8': "image/gif",
    }
    
    @dataclass
    class ValidationResult:
        valid: bool
        error: Optional[str] = None
        detected_type: Optional[str] = None
    
    def validate(
        self,
        file_content: bytes,
        filename: str,
        declared_content_type: str,
    ) -> "UploadSanitizer.ValidationResult":
        """
        Validate uploaded file.
        
        Checks:
        1. File extension
        2. Declared MIME type
        3. Actual content (magic bytes)
        4. File size
        """
        # Check extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            return self.ValidationResult(
                valid=False,
                error=f"File extension '{ext}' not allowed"
            )
        
        # Check declared MIME type
        if declared_content_type not in self.ALLOWED_MIME_TYPES:
            return self.ValidationResult(
                valid=False,
                error=f"Content type '{declared_content_type}' not allowed"
            )
        
        # Check file size
        size_mb = len(file_content) / (1024 * 1024)
        if size_mb > self.MAX_FILE_SIZE_MB:
            return self.ValidationResult(
                valid=False,
                error=f"File too large ({size_mb:.1f}MB > {self.MAX_FILE_SIZE_MB}MB)"
            )
        
        # Check actual content (magic bytes)
        detected_type = self._detect_file_type(file_content)
        if detected_type is None:
            return self.ValidationResult(
                valid=False,
                error="Could not verify file type from content"
            )
        
        if detected_type not in self.ALLOWED_MIME_TYPES:
            return self.ValidationResult(
                valid=False,
                error=f"Detected file type '{detected_type}' not allowed"
            )
        
        # Check for embedded executable content
        if self._contains_executable_markers(file_content):
            return self.ValidationResult(
                valid=False,
                error="Suspicious content detected in file"
            )
        
        return self.ValidationResult(
            valid=True,
            detected_type=detected_type
        )
    
    def _detect_file_type(self, content: bytes) -> Optional[str]:
        """Detect file type from magic bytes."""
        for signature, mime_type in self.IMAGE_SIGNATURES.items():
            if content[:len(signature)] == signature:
                return mime_type
        return None
    
    def _contains_executable_markers(self, content: bytes) -> bool:
        """Check for embedded executable content."""
        suspicious_patterns = [
            b'<script',
            b'<?php',
            b'<%',
            b'\x00MZ',  # Windows executable
            b'\x7fELF',  # Linux executable
        ]
        
        for pattern in suspicious_patterns:
            if pattern in content.lower() if isinstance(content, str) else pattern in content:
                return True
        return False


class DataAnonymizer:
    """
    Anonymize sensitive data for logging and analytics.
    """
    
    @staticmethod
    def hash_user_id(user_id: str) -> str:
        """Create one-way hash of user ID."""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    @staticmethod
    def mask_email(email: str) -> str:
        """Mask email for logging: john@example.com -> j***@e***.com"""
        if "@" not in email:
            return "***"
        
        local, domain = email.split("@", 1)
        domain_parts = domain.split(".")
        
        masked_local = local[0] + "***" if local else "***"
        masked_domain = domain_parts[0][0] + "***" if domain_parts else "***"
        
        return f"{masked_local}@{masked_domain}.{domain_parts[-1] if len(domain_parts) > 1 else 'com'}"
    
    @staticmethod
    def redact_pii(text: str) -> str:
        """Redact common PII patterns from text."""
        # Email
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL]',
            text
        )
        
        # Phone numbers (various formats)
        text = re.sub(
            r'\b(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            '[PHONE]',
            text
        )
        
        # SSN (US format)
        text = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            '[SSN]',
            text
        )
        
        return text


# Global instances
access_control = AccessControl()
upload_sanitizer = UploadSanitizer()
data_anonymizer = DataAnonymizer()
retention_policy = DataRetentionPolicy()
