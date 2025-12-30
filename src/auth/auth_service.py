"""
Authentication Service

Handles user registration, login, and JWT token management.
"""

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Optional, Tuple
from dataclasses import dataclass
import json
import base64

from .database import UserRepository, MedicalProfileRepository


# JWT Secret (in production, use environment variable)
JWT_SECRET = "ai-nutrition-secret-key-change-in-production"
JWT_EXPIRY_HOURS = 24


@dataclass
class AuthResult:
    """Result of authentication operation."""
    success: bool
    user_id: Optional[str] = None
    token: Optional[str] = None
    error: Optional[str] = None


class AuthService:
    """Authentication service for user management."""
    
    def __init__(self):
        self.user_repo = UserRepository()
    
    def register(self, email: str, password: str, name: str) -> AuthResult:
        """Register a new user."""
        # Validate input
        if not email or '@' not in email:
            return AuthResult(success=False, error="Invalid email address")
        
        if not password or len(password) < 6:
            return AuthResult(success=False, error="Password must be at least 6 characters")
        
        if not name or len(name) < 2:
            return AuthResult(success=False, error="Name must be at least 2 characters")
        
        # Check if email exists
        existing = self.user_repo.get_by_email(email)
        if existing:
            return AuthResult(success=False, error="Email already registered")
        
        # Create user
        user_id = str(uuid.uuid4())
        password_hash = self._hash_password(password)
        
        if self.user_repo.create(user_id, email, password_hash, name):
            token = self._create_token(user_id, email)
            return AuthResult(success=True, user_id=user_id, token=token)
        
        return AuthResult(success=False, error="Failed to create user")
    
    def login(self, email: str, password: str) -> AuthResult:
        """Login an existing user."""
        if not email or not password:
            return AuthResult(success=False, error="Email and password required")
        
        # Get user
        user = self.user_repo.get_by_email(email)
        if not user:
            return AuthResult(success=False, error="Invalid email or password")
        
        # Verify password
        password_hash = self._hash_password(password)
        if password_hash != user['password_hash']:
            return AuthResult(success=False, error="Invalid email or password")
        
        # Create token
        token = self._create_token(user['id'], user['email'])
        return AuthResult(success=True, user_id=user['id'], token=token)
    
    def verify_token(self, token: str) -> Optional[dict]:
        """Verify JWT token and return payload if valid."""
        try:
            # Decode token
            parts = token.split(".")
            if len(parts) != 3:
                return None
            
            # Decode payload
            payload_b64 = parts[1]
            # Add padding if needed
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding
            
            payload_json = base64.urlsafe_b64decode(payload_b64).decode('utf-8')
            payload = json.loads(payload_json)
            
            # Check expiry
            if datetime.fromisoformat(payload['exp']) < datetime.now():
                return None
            
            # Verify signature
            expected_signature = self._sign(parts[0] + "." + parts[1])
            if parts[2] != expected_signature:
                return None
            
            return payload
        except Exception as e:
            print(f"Token verification error: {e}")
            return None
    
    def get_user(self, user_id: str) -> Optional[dict]:
        """Get user by ID."""
        user = self.user_repo.get_by_id(user_id)
        if user:
            # Don't return password hash
            del user['password_hash']
            return user
        return None
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt."""
        # Simple hash (in production, use bcrypt)
        salted = f"{password}:{JWT_SECRET}"
        return hashlib.sha256(salted.encode()).hexdigest()
    
    def _create_token(self, user_id: str, email: str) -> str:
        """Create a JWT token."""
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "sub": user_id,
            "email": email,
            "exp": (datetime.now() + timedelta(hours=JWT_EXPIRY_HOURS)).isoformat(),
            "iat": datetime.now().isoformat(),
        }
        
        # Encode
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        
        # Sign
        signature = self._sign(f"{header_b64}.{payload_b64}")
        
        return f"{header_b64}.{payload_b64}.{signature}"
    
    def _sign(self, data: str) -> str:
        """Create HMAC signature."""
        return hashlib.sha256(f"{data}:{JWT_SECRET}".encode()).hexdigest()[:43]


# Global instance
auth_service = AuthService()
