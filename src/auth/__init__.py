"""
Authentication package.
"""

from .auth_service import AuthService, AuthResult, auth_service
from .database import (
    UserRepository,
    MedicalProfileRepository,
    UploadRepository,
    init_database,
)

__all__ = [
    "AuthService",
    "AuthResult",
    "auth_service",
    "UserRepository",
    "MedicalProfileRepository",
    "UploadRepository",
    "init_database",
]
