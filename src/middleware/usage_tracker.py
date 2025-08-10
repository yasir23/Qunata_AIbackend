"""
Usage Tracking Middleware for Open Deep Research

This module provides comprehensive usage tracking and rate limiting functionality:
- Monitor API calls, token usage, and research requests per user
- Integrate with subscription system for tier-based limits
- Enforce usage limits based on subscription tiers
- Track usage metrics for billing and analytics
"""

import os
import logging
import time
import json
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from supabase import Client, create_client
import asyncio
from functools import wraps

# Import database models
from ..database.models import (
    UsageRecord, 
    UsageType, 
    SubscriptionTierEnum,
    SubscriptionStatus
)

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.warning("Supabase credentials not found. Usage tracking may not work properly.")
    supabase_client = None
else:
    supabase_client = create_client(supabase_url, supabase_key)

# Subscription tier limits configuration
SUBSCRIPTION_LIMITS = {
    SubscriptionTierEnum.FREE: {
        "api_calls_per_hour": 100,
        "api_calls_per_day": 1000,
        "research_requests_per_hour": 5,
        "research_requests_per_day": 20,
        "token_usage_per_day": 50000,
        "concurrent_research_units": 2,
        "mcp_servers": ["reddit", "youtube"],
        "rag_queries_per_day": 50,
    },
    SubscriptionTierEnum.PRO: {
        "api_calls_per_hour": 1000,
        "api_calls_per_day": 10000,
        "research_requests_per_hour": 50,
        "research_requests_per_day": 200,
        "token_usage_per_day": 500000,
        "concurrent_research_units": 5,
        "mcp_servers": ["reddit", "youtube", "github"],
        "rag_queries_per_day": 500,
    },
    SubscriptionTierEnum.ENTERPRISE: {
        "api_calls_per_hour": -1,  # Unlimited
        "api_calls_per_day": -1,   # Unlimited
        "research_requests_per_hour": -1,  # Unlimited
        "research_requests_per_day": -1,   # Unlimited
        "token_usage_per_day": -1,  # Unlimited
        "concurrent_research_units": 20,
        "mcp_servers": ["reddit", "youtube", "github"],
        "rag_queries_per_day": -1,  # Unlimited
    }
}

class UsageTracker:
    """Core usage tracking functionality."""
    
    def __init__(self, supabase_client: Optional[Client] = None):
        self.supabase = supabase_client or supabase_client
        
    async def get_user_subscription_status(self, user_id: str) -> Dict[str, Any]:
        """Get user's current subscription status and limits."""
        if not self.supabase:
            logger.warning("Supabase client not available for subscription check")
            return {
                "tier": SubscriptionTierEnum.FREE,
                "status": SubscriptionStatus.ACTIVE,
                "limits": SUBSCRIPTION_LIMITS[SubscriptionTierEnum.FREE]
            }
        
        try:
            # Get user subscription from database
            response = await asyncio.to_thread(
                self.supabase.rpc,
                "get_user_subscription_status",
                {"user_uuid": user_id}
            )
            
            if response.data:
                subscription_data = response.data[0] if isinstance(response.data, list) else response.data
                tier = SubscriptionTierEnum(subscription_data.get("tier", "free"))
                status_val = SubscriptionStatus(subscription_data.get("status", "active"))
                
                return {
                    "tier": tier,
                    "status": status_val,
                    "limits": SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS[SubscriptionTierEnum.FREE]),
                    "subscription_id": subscription_data.get("subscription_id"),
                    "customer_id": subscription_data.get("customer_id")
                }
            else:
                # Default to free tier if no subscription found
                return {
                    "tier": SubscriptionTierEnum.FREE,
                    "status": SubscriptionStatus.ACTIVE,
                    "limits": SUBSCRIPTION_LIMITS[SubscriptionTierEnum.FREE]
                }
                
        except Exception as e:
            logger.error(f"Error getting subscription status for user {user_id}: {e}")
            # Default to free tier on error
            return {
                "tier": SubscriptionTierEnum.FREE,
                "status": SubscriptionStatus.ACTIVE,
                "limits": SUBSCRIPTION_LIMITS[SubscriptionTierEnum.FREE]
            }
    
    async def get_usage_stats(self, user_id: str, time_window: str = "day") -> Dict[str, int]:
        """Get current usage statistics for a user within a time window."""
        if not self.supabase:
            logger.warning("Supabase client not available for usage stats")
            return {}
        
        try:
            # Calculate time window
            now = datetime.now(timezone.utc)
            if time_window == "hour":
                start_time = now - timedelta(hours=1)
            elif time_window == "day":
                start_time = now - timedelta(days=1)
            else:
                start_time = now - timedelta(days=1)  # Default to day
            
            # Query usage records
            response = await asyncio.to_thread(
                self.supabase.table("usage_records").select("usage_type, quantity, tokens_consumed").gte(
                    "timestamp", start_time.isoformat()
                ).eq("user_id", user_id).execute
            )
            
            # Aggregate usage statistics
            stats = {
                "api_calls": 0,
                "research_requests": 0,
                "token_usage": 0,
                "mcp_server_calls": 0,
                "rag_queries": 0
            }
            
            for record in response.data:
                usage_type = record.get("usage_type")
                quantity = record.get("quantity", 0)
                tokens = record.get("tokens_consumed", 0)
                
                if usage_type == UsageType.API_CALL:
                    stats["api_calls"] += quantity
                elif usage_type == UsageType.RESEARCH_REQUEST:
                    stats["research_requests"] += quantity
                elif usage_type == UsageType.TOKEN_USAGE:
                    stats["token_usage"] += tokens
                elif usage_type == UsageType.MCP_SERVER_CALL:
                    stats["mcp_server_calls"] += quantity
                # Add RAG queries when implemented
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting usage stats for user {user_id}: {e}")
            return {}
    
    async def check_usage_limits(self, user_id: str, usage_type: UsageType, quantity: int = 1) -> Tuple[bool, str]:
        """Check if user can perform the requested action within their limits."""
        try:
            # Get subscription status and limits
            subscription_info = await self.get_user_subscription_status(user_id)
            limits = subscription_info["limits"]
            
            # Get current usage stats
            hourly_stats = await self.get_usage_stats(user_id, "hour")
            daily_stats = await self.get_usage_stats(user_id, "day")
            
            # Check limits based on usage type
            if usage_type == UsageType.API_CALL:
                hourly_limit = limits.get("api_calls_per_hour", -1)
                daily_limit = limits.get("api_calls_per_day", -1)
                current_hourly = hourly_stats.get("api_calls", 0)
                current_daily = daily_stats.get("api_calls", 0)
                
                if hourly_limit > 0 and current_hourly + quantity > hourly_limit:
                    return False, f"Hourly API call limit exceeded ({current_hourly}/{hourly_limit})"
                if daily_limit > 0 and current_daily + quantity > daily_limit:
                    return False, f"Daily API call limit exceeded ({current_daily}/{daily_limit})"
                    
            elif usage_type == UsageType.RESEARCH_REQUEST:
                hourly_limit = limits.get("research_requests_per_hour", -1)
                daily_limit = limits.get("research_requests_per_day", -1)
                current_hourly = hourly_stats.get("research_requests", 0)
                current_daily = daily_stats.get("research_requests", 0)
                
                if hourly_limit > 0 and current_hourly + quantity > hourly_limit:
                    return False, f"Hourly research request limit exceeded ({current_hourly}/{hourly_limit})"
                if daily_limit > 0 and current_daily + quantity > daily_limit:
                    return False, f"Daily research request limit exceeded ({current_daily}/{daily_limit})"
                    
            elif usage_type == UsageType.TOKEN_USAGE:
                daily_limit = limits.get("token_usage_per_day", -1)
                current_daily = daily_stats.get("token_usage", 0)
                
                if daily_limit > 0 and current_daily + quantity > daily_limit:
                    return False, f"Daily token usage limit exceeded ({current_daily}/{daily_limit})"
            
            return True, "Usage within limits"
            
        except Exception as e:
            logger.error(f"Error checking usage limits for user {user_id}: {e}")
            # Allow usage on error to avoid blocking users
            return True, "Error checking limits, allowing usage"
    
    async def record_usage(
        self, 
        user_id: str, 
        usage_type: UsageType, 
        quantity: int = 1,
        endpoint: Optional[str] = None,
        model_used: Optional[str] = None,
        tokens_consumed: Optional[int] = None,
        cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record usage in the database."""
        if not self.supabase:
            logger.warning("Supabase client not available for usage recording")
            return False
        
        try:
            # Get subscription info for subscription_id
            subscription_info = await self.get_user_subscription_status(user_id)
            subscription_id = subscription_info.get("subscription_id")
            
            # Create usage record
            usage_record = UsageRecord(
                user_id=user_id,
                subscription_id=subscription_id,
                usage_type=usage_type,
                quantity=quantity,
                endpoint=endpoint,
                model_used=model_used,
                tokens_consumed=tokens_consumed,
                cost=cost,
                metadata=metadata or {}
            )
            
            # Insert into database
            await asyncio.to_thread(
                self.supabase.table("usage_records").insert(
                    usage_record.model_dump(exclude_none=True)
                ).execute
            )
            
            logger.info(f"Recorded usage: {usage_type} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording usage for user {user_id}: {e}")
            return False

class UsageTrackingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for usage tracking and rate limiting."""
    
    def __init__(self, app, supabase_client: Optional[Client] = None):
        super().__init__(app)
        self.usage_tracker = UsageTracker(supabase_client)
        
        # Endpoints that should be tracked
        self.tracked_endpoints = {
            "/research/start": UsageType.RESEARCH_REQUEST,
            "/research/config": UsageType.API_CALL,
            "/research/status": UsageType.API_CALL,
            "/auth/profile": UsageType.API_CALL,
            "/auth/signin": UsageType.API_CALL,
            "/auth/signup": UsageType.API_CALL,
        }
        
        # Endpoints that should be excluded from tracking
        self.excluded_endpoints = {
            "/health",
            "/health/detailed",
            "/docs",
            "/openapi.json",
            "/",
        }
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request with usage tracking and rate limiting."""
        start_time = time.time()
        
        # Skip tracking for excluded endpoints
        if request.url.path in self.excluded_endpoints:
            return await call_next(request)
        
        # Extract user ID from request (if authenticated)
        user_id = await self._extract_user_id(request)
        
        # Skip tracking for unauthenticated requests to public endpoints
        if not user_id and request.url.path not in ["/auth/signin", "/auth/signup"]:
            return await call_next(request)
        
        # Determine usage type
        usage_type = self.tracked_endpoints.get(request.url.path, UsageType.API_CALL)
        
        # Check usage limits (only for authenticated users)
        if user_id:
            can_proceed, limit_message = await self.usage_tracker.check_usage_limits(
                user_id, usage_type
            )
            
            if not can_proceed:
                logger.warning(f"Usage limit exceeded for user {user_id}: {limit_message}")
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": {
                            "code": 429,
                            "message": limit_message,
                            "type": "usage_limit_exceeded",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    }
                )
        
        # Process the request
        try:
            response = await call_next(request)
            
            # Record usage (only for successful requests and authenticated users)
            if user_id and 200 <= response.status_code < 400:
                processing_time = time.time() - start_time
                
                # Extract additional metadata
                metadata = {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "processing_time": processing_time,
                    "user_agent": request.headers.get("user-agent"),
                    "ip_address": request.client.host if request.client else None,
                }
                
                # Record the usage
                await self.usage_tracker.record_usage(
                    user_id=user_id,
                    usage_type=usage_type,
                    quantity=1,
                    endpoint=request.url.path,
                    metadata=metadata
                )
            
            # Add usage headers to response
            if user_id:
                subscription_info = await self.usage_tracker.get_user_subscription_status(user_id)
                daily_stats = await self.usage_tracker.get_usage_stats(user_id, "day")
                
                response.headers["X-Subscription-Tier"] = subscription_info["tier"]
                response.headers["X-Daily-API-Calls"] = str(daily_stats.get("api_calls", 0))
                response.headers["X-Daily-Research-Requests"] = str(daily_stats.get("research_requests", 0))
                response.headers["X-Daily-Token-Usage"] = str(daily_stats.get("token_usage", 0))
            
            return response
            
        except Exception as e:
            logger.error(f"Error in usage tracking middleware: {e}")
            # Return the response even if tracking fails
            return await call_next(request)
    
    async def _extract_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request authorization header."""
        try:
            authorization = request.headers.get("authorization")
            if not authorization:
                return None
            
            # Parse Bearer token
            try:
                scheme, token = authorization.split()
                if scheme.lower() != "bearer":
                    return None
            except ValueError:
                return None
            
            # Verify token with Supabase (simplified version)
            if not supabase_client:
                return None
            
            response = await asyncio.to_thread(supabase_client.auth.get_user, token)
            user = response.user
            
            return user.id if user else None
            
        except Exception as e:
            logger.debug(f"Could not extract user ID: {e}")
            return None

# Utility functions for external use

async def get_user_subscription_limits(user_id: str) -> Dict[str, Any]:
    """Get subscription limits for a user."""
    tracker = UsageTracker()
    return await tracker.get_user_subscription_status(user_id)

async def check_user_can_access_mcp_server(user_id: str, server_name: str) -> bool:
    """Check if user can access a specific MCP server based on their subscription."""
    tracker = UsageTracker()
    subscription_info = await tracker.get_user_subscription_status(user_id)
    allowed_servers = subscription_info["limits"].get("mcp_servers", [])
    return server_name.lower() in allowed_servers

async def record_token_usage(user_id: str, tokens: int, model: str, endpoint: str) -> bool:
    """Record token usage for a user."""
    tracker = UsageTracker()
    return await tracker.record_usage(
        user_id=user_id,
        usage_type=UsageType.TOKEN_USAGE,
        quantity=1,
        endpoint=endpoint,
        model_used=model,
        tokens_consumed=tokens
    )

def require_subscription_tier(required_tier: SubscriptionTierEnum):
    """Decorator to require a specific subscription tier for endpoint access."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract user_id from kwargs or request
            user_id = kwargs.get("user_id")
            if not user_id:
                # Try to extract from request if available
                request = kwargs.get("request")
                if request:
                    middleware = UsageTrackingMiddleware(None)
                    user_id = await middleware._extract_user_id(request)
            
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Check subscription tier
            tracker = UsageTracker()
            subscription_info = await tracker.get_user_subscription_status(user_id)
            user_tier = subscription_info["tier"]
            
            # Define tier hierarchy
            tier_hierarchy = {
                SubscriptionTierEnum.FREE: 0,
                SubscriptionTierEnum.PRO: 1,
                SubscriptionTierEnum.ENTERPRISE: 2
            }
            
            if tier_hierarchy.get(user_tier, 0) < tier_hierarchy.get(required_tier, 0):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"This feature requires {required_tier} subscription or higher"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
