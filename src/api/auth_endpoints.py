"""
User Management API Endpoints for Open Deep Research

This module provides FastAPI endpoints for user authentication, profile management,
and account operations that integrate with the existing Supabase authentication system.
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, Field, validator
from supabase import Client, create_client
import asyncio
import re
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.warning("Supabase credentials not found. Authentication endpoints may not work properly.")
    supabase_client = None
else:
    supabase_client = create_client(supabase_url, supabase_key)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Security scheme
security = HTTPBearer()

# Create router for authentication endpoints
auth_router = APIRouter(prefix="/auth", tags=["authentication"])

# Pydantic models for request/response validation

class SignUpRequest(BaseModel):
    """User signup request model."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password (minimum 8 characters)")
    name: Optional[str] = Field(None, max_length=100, description="User full name")
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Za-z]', v):
            raise ValueError('Password must contain at least one letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one number')
        return v

class SignInRequest(BaseModel):
    """User signin request model."""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")

class PasswordResetRequest(BaseModel):
    """Password reset request model."""
    email: EmailStr = Field(..., description="User email address")

class PasswordUpdateRequest(BaseModel):
    """Password update request model."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password (minimum 8 characters)")
    
    @validator('new_password')
    def validate_new_password(cls, v):
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Za-z]', v):
            raise ValueError('Password must contain at least one letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one number')
        return v

class ProfileUpdateRequest(BaseModel):
    """Profile update request model."""
    name: Optional[str] = Field(None, max_length=100, description="User full name")
    avatar_url: Optional[str] = Field(None, description="User avatar URL")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")

class AuthResponse(BaseModel):
    """Authentication response model."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: Dict[str, Any] = Field(..., description="User information")

class UserProfileResponse(BaseModel):
    """User profile response model."""
    user_id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    name: Optional[str] = Field(None, description="User full name")
    avatar_url: Optional[str] = Field(None, description="User avatar URL")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    created_at: datetime = Field(..., description="Account creation date")
    updated_at: datetime = Field(..., description="Last update date")
    subscription_status: Optional[Dict[str, Any]] = Field(None, description="Subscription information")

class MessageResponse(BaseModel):
    """Generic message response model."""
    message: str = Field(..., description="Response message")
    success: bool = Field(default=True, description="Operation success status")

# Dependency functions

def get_supabase_client() -> Client:
    """Dependency to get Supabase client."""
    if not supabase_client:
        raise HTTPException(
            status_code=500,
            detail="Supabase client not initialized. Check environment variables."
        )
    return supabase_client

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    supabase: Client = Depends(get_supabase_client)
) -> Dict[str, Any]:
    """Get current authenticated user from JWT token."""
    try:
        # Verify the JWT token with Supabase
        response = await asyncio.to_thread(supabase.auth.get_user, credentials.credentials)
        
        if not response.user:
            raise HTTPException(
                status_code=401,
                detail="Invalid token or user not found"
            )
        
        return {
            "id": response.user.id,
            "email": response.user.email,
            "user_metadata": response.user.user_metadata or {},
            "created_at": response.user.created_at
        }
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=401,
            detail="Authentication failed"
        )

async def get_user_profile(user_id: str, supabase: Client) -> Optional[Dict[str, Any]]:
    """Get user profile from database."""
    try:
        result = supabase.table("user_profiles").select("*").eq("user_id", user_id).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"Error fetching user profile: {e}")
        return None

async def create_or_update_user_profile(
    user_id: str,
    email: str,
    name: Optional[str] = None,
    avatar_url: Optional[str] = None,
    preferences: Optional[Dict[str, Any]] = None,
    supabase: Client = None
) -> Dict[str, Any]:
    """Create or update user profile in database."""
    try:
        profile_data = {
            "user_id": user_id,
            "email": email,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        if name is not None:
            profile_data["name"] = name
        if avatar_url is not None:
            profile_data["avatar_url"] = avatar_url
        if preferences is not None:
            profile_data["preferences"] = preferences
        
        result = supabase.table("user_profiles").upsert(profile_data).execute()
        return result.data[0] if result.data else profile_data
        
    except Exception as e:
        logger.error(f"Error creating/updating user profile: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update user profile"
        )

# API Endpoints

@auth_router.post("/signup", response_model=AuthResponse)
@limiter.limit("5/minute")
async def signup(
    request: Request,
    signup_data: SignUpRequest,
    supabase: Client = Depends(get_supabase_client)
) -> AuthResponse:
    """
    Register a new user account.
    
    Creates a new user account with email and password, and initializes
    their profile in the database.
    """
    try:
        # Create user with Supabase Auth
        auth_response = await asyncio.to_thread(
            supabase.auth.sign_up,
            {
                "email": signup_data.email,
                "password": signup_data.password,
                "options": {
                    "data": {
                        "name": signup_data.name
                    }
                }
            }
        )
        
        if not auth_response.user:
            raise HTTPException(
                status_code=400,
                detail="Failed to create user account"
            )
        
        # Create user profile
        await create_or_update_user_profile(
            user_id=auth_response.user.id,
            email=signup_data.email,
            name=signup_data.name,
            supabase=supabase
        )
        
        logger.info(f"New user registered: {signup_data.email}")
        
        return AuthResponse(
            access_token=auth_response.session.access_token,
            refresh_token=auth_response.session.refresh_token,
            expires_in=auth_response.session.expires_in,
            user={
                "id": auth_response.user.id,
                "email": auth_response.user.email,
                "name": signup_data.name,
                "created_at": auth_response.user.created_at
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during signup"
        )

@auth_router.post("/signin", response_model=AuthResponse)
@limiter.limit("10/minute")
async def signin(
    request: Request,
    signin_data: SignInRequest,
    supabase: Client = Depends(get_supabase_client)
) -> AuthResponse:
    """
    Sign in an existing user.
    
    Authenticates user with email and password and returns JWT tokens.
    """
    try:
        # Authenticate with Supabase
        auth_response = await asyncio.to_thread(
            supabase.auth.sign_in_with_password,
            {
                "email": signin_data.email,
                "password": signin_data.password
            }
        )
        
        if not auth_response.user or not auth_response.session:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        # Get or create user profile
        profile = await get_user_profile(auth_response.user.id, supabase)
        if not profile:
            profile = await create_or_update_user_profile(
                user_id=auth_response.user.id,
                email=auth_response.user.email,
                name=auth_response.user.user_metadata.get("name"),
                supabase=supabase
            )
        
        logger.info(f"User signed in: {signin_data.email}")
        
        return AuthResponse(
            access_token=auth_response.session.access_token,
            refresh_token=auth_response.session.refresh_token,
            expires_in=auth_response.session.expires_in,
            user={
                "id": auth_response.user.id,
                "email": auth_response.user.email,
                "name": profile.get("name"),
                "avatar_url": profile.get("avatar_url"),
                "created_at": auth_response.user.created_at
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signin error: {e}")
        raise HTTPException(
            status_code=401,
            detail="Authentication failed"
        )

@auth_router.post("/signout", response_model=MessageResponse)
async def signout(
    current_user: Dict[str, Any] = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_client)
) -> MessageResponse:
    """
    Sign out the current user.
    
    Invalidates the user's session and JWT tokens.
    """
    try:
        # Sign out with Supabase
        await asyncio.to_thread(supabase.auth.sign_out)
        
        logger.info(f"User signed out: {current_user['email']}")
        
        return MessageResponse(
            message="Successfully signed out",
            success=True
        )
        
    except Exception as e:
        logger.error(f"Signout error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to sign out"
        )

@auth_router.post("/password-reset", response_model=MessageResponse)
@limiter.limit("3/minute")
async def request_password_reset(
    request: Request,
    reset_data: PasswordResetRequest,
    supabase: Client = Depends(get_supabase_client)
) -> MessageResponse:
    """
    Request a password reset email.
    
    Sends a password reset email to the specified address.
    """
    try:
        # Request password reset with Supabase
        await asyncio.to_thread(
            supabase.auth.reset_password_email,
            reset_data.email
        )
        
        logger.info(f"Password reset requested for: {reset_data.email}")
        
        return MessageResponse(
            message="Password reset email sent if account exists",
            success=True
        )
        
    except Exception as e:
        logger.error(f"Password reset error: {e}")
        # Always return success to prevent email enumeration
        return MessageResponse(
            message="Password reset email sent if account exists",
            success=True
        )

@auth_router.post("/password-update", response_model=MessageResponse)
@limiter.limit("5/minute")
async def update_password(
    request: Request,
    password_data: PasswordUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_client)
) -> MessageResponse:
    """
    Update user password.
    
    Updates the current user's password after verifying the current password.
    """
    try:
        # Verify current password by attempting to sign in
        try:
            await asyncio.to_thread(
                supabase.auth.sign_in_with_password,
                {
                    "email": current_user["email"],
                    "password": password_data.current_password
                }
            )
        except:
            raise HTTPException(
                status_code=400,
                detail="Current password is incorrect"
            )
        
        # Update password
        await asyncio.to_thread(
            supabase.auth.update_user,
            {"password": password_data.new_password}
        )
        
        logger.info(f"Password updated for user: {current_user['email']}")
        
        return MessageResponse(
            message="Password updated successfully",
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password update error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update password"
        )

@auth_router.get("/profile", response_model=UserProfileResponse)
async def get_profile(
    current_user: Dict[str, Any] = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_client)
) -> UserProfileResponse:
    """
    Get current user profile.
    
    Returns the current user's profile information including subscription status.
    """
    try:
        # Get user profile
        profile = await get_user_profile(current_user["id"], supabase)
        
        if not profile:
            # Create profile if it doesn't exist
            profile = await create_or_update_user_profile(
                user_id=current_user["id"],
                email=current_user["email"],
                name=current_user["user_metadata"].get("name"),
                supabase=supabase
            )
        
        # Get subscription status
        subscription_status = None
        try:
            sub_result = supabase.table("subscriptions").select("*").eq("user_id", current_user["id"]).eq("status", "active").execute()
            if sub_result.data:
                subscription = sub_result.data[0]
                subscription_status = {
                    "tier": subscription["tier"],
                    "status": subscription["status"],
                    "current_period_end": subscription["current_period_end"]
                }
        except Exception as e:
            logger.warning(f"Failed to get subscription status: {e}")
        
        return UserProfileResponse(
            user_id=profile["user_id"],
            email=profile["email"],
            name=profile.get("name"),
            avatar_url=profile.get("avatar_url"),
            preferences=profile.get("preferences", {}),
            created_at=datetime.fromisoformat(profile["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(profile["updated_at"].replace("Z", "+00:00")),
            subscription_status=subscription_status
        )
        
    except Exception as e:
        logger.error(f"Get profile error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve user profile"
        )

@auth_router.put("/profile", response_model=UserProfileResponse)
@limiter.limit("10/minute")
async def update_profile(
    request: Request,
    profile_data: ProfileUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_client)
) -> UserProfileResponse:
    """
    Update user profile.
    
    Updates the current user's profile information.
    """
    try:
        # Update user profile
        profile = await create_or_update_user_profile(
            user_id=current_user["id"],
            email=current_user["email"],
            name=profile_data.name,
            avatar_url=profile_data.avatar_url,
            preferences=profile_data.preferences,
            supabase=supabase
        )
        
        logger.info(f"Profile updated for user: {current_user['email']}")
        
        return UserProfileResponse(
            user_id=profile["user_id"],
            email=profile["email"],
            name=profile.get("name"),
            avatar_url=profile.get("avatar_url"),
            preferences=profile.get("preferences", {}),
            created_at=datetime.fromisoformat(profile["created_at"].replace("Z", "+00:00")),
            updated_at=datetime.fromisoformat(profile["updated_at"].replace("Z", "+00:00"))
        )
        
    except Exception as e:
        logger.error(f"Update profile error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to update user profile"
        )

@auth_router.delete("/account", response_model=MessageResponse)
@limiter.limit("1/hour")
async def delete_account(
    request: Request,
    current_user: Dict[str, Any] = Depends(get_current_user),
    supabase: Client = Depends(get_supabase_client)
) -> MessageResponse:
    """
    Delete user account.
    
    Permanently deletes the user's account and all associated data.
    """
    try:
        # Delete user profile and related data
        # Note: Supabase Auth user deletion requires admin privileges
        # For now, we'll mark the profile as inactive
        supabase.table("user_profiles").update({
            "is_active": False,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("user_id", current_user["id"]).execute()
        
        # Cancel any active subscriptions
        try:
            active_subs = supabase.table("subscriptions").select("id").eq("user_id", current_user["id"]).eq("status", "active").execute()
            for sub in active_subs.data:
                # This would typically cancel the Stripe subscription
                # For now, we'll mark it as canceled in the database
                supabase.table("subscriptions").update({
                    "status": "canceled",
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", sub["id"]).execute()
        except Exception as e:
            logger.warning(f"Failed to cancel subscriptions during account deletion: {e}")
        
        logger.info(f"Account deletion requested for user: {current_user['email']}")
        
        return MessageResponse(
            message="Account deletion initiated. Your account has been deactivated.",
            success=True
        )
        
    except Exception as e:
        logger.error(f"Account deletion error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete account"
        )

@auth_router.get("/verify-token")
async def verify_token(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Verify JWT token validity.
    
    Returns user information if token is valid.
    """
    return {
        "valid": True,
        "user": {
            "id": current_user["id"],
            "email": current_user["email"],
            "created_at": current_user["created_at"]
        }
    }

# Error handlers for rate limiting
@auth_router.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    response = JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "Rate limit exceeded",
            "detail": f"Rate limit exceeded: {exc.detail}",
            "retry_after": exc.retry_after
        }
    )
    response.headers["Retry-After"] = str(exc.retry_after)
    return response
