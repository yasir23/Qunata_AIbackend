"""
Subscription Management System for Open Deep Research

This module provides comprehensive subscription management functionality:
- Managing subscription tiers and user access
- Checking user limits and feature access
- Controlling access to MCP servers based on subscription
- Integration with Configuration system for dynamic limits
- Tier-based feature access control
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
from enum import Enum
from pydantic import BaseModel, Field
from supabase import Client, create_client
import asyncio

# Import existing models and configurations
from ..database.models import (
    SubscriptionTierEnum,
    SubscriptionStatus,
    UsageType
)
from .stripe_integration import SUBSCRIPTION_TIERS, SubscriptionTier

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.warning("Supabase credentials not found. Subscription management may not work properly.")
    supabase_client = None
else:
    supabase_client = create_client(supabase_url, supabase_key)

class FeatureAccess(BaseModel):
    """Feature access configuration for subscription tiers."""
    research_requests: bool = True
    mcp_servers: List[str] = Field(default_factory=list)
    api_access: bool = False
    priority_support: bool = False
    advanced_rag: bool = False
    custom_integrations: bool = False
    dedicated_support: bool = False
    github_mcp_access: bool = False

class SubscriptionLimits(BaseModel):
    """Subscription limits configuration."""
    concurrent_research_units: int = 1
    api_calls_per_hour: int = 100
    api_calls_per_day: int = 1000
    research_requests_per_hour: int = 5
    research_requests_per_day: int = 20
    token_usage_per_day: int = 50000
    rag_queries_per_day: int = 50
    monthly_research_requests: int = 10
    monthly_api_calls: int = 1000
    monthly_tokens: int = 50000

class UserSubscription(BaseModel):
    """User subscription information."""
    user_id: str
    subscription_id: Optional[str] = None
    customer_id: Optional[str] = None
    tier: SubscriptionTierEnum = SubscriptionTierEnum.FREE
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    cancel_at_period_end: bool = False
    features: FeatureAccess = Field(default_factory=FeatureAccess)
    limits: SubscriptionLimits = Field(default_factory=SubscriptionLimits)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Enhanced subscription tier configurations with detailed limits
ENHANCED_SUBSCRIPTION_TIERS = {
    SubscriptionTierEnum.FREE: {
        "name": "Free",
        "monthly_price": 0.0,
        "features": FeatureAccess(
            research_requests=True,
            mcp_servers=["reddit", "youtube"],
            api_access=False,
            priority_support=False,
            advanced_rag=False,
            github_mcp_access=False
        ),
        "limits": SubscriptionLimits(
            concurrent_research_units=2,
            api_calls_per_hour=100,
            api_calls_per_day=1000,
            research_requests_per_hour=5,
            research_requests_per_day=20,
            token_usage_per_day=50000,
            rag_queries_per_day=50,
            monthly_research_requests=10,
            monthly_api_calls=1000,
            monthly_tokens=50000
        )
    },
    SubscriptionTierEnum.PRO: {
        "name": "Pro",
        "monthly_price": 29.99,
        "features": FeatureAccess(
            research_requests=True,
            mcp_servers=["reddit", "youtube", "github"],
            api_access=True,
            priority_support=True,
            advanced_rag=True,
            github_mcp_access=True
        ),
        "limits": SubscriptionLimits(
            concurrent_research_units=5,
            api_calls_per_hour=1000,
            api_calls_per_day=10000,
            research_requests_per_hour=50,
            research_requests_per_day=200,
            token_usage_per_day=500000,
            rag_queries_per_day=500,
            monthly_research_requests=500,
            monthly_api_calls=10000,
            monthly_tokens=1000000
        )
    },
    SubscriptionTierEnum.ENTERPRISE: {
        "name": "Enterprise",
        "monthly_price": 99.99,
        "features": FeatureAccess(
            research_requests=True,
            mcp_servers=["reddit", "youtube", "github"],
            api_access=True,
            priority_support=True,
            advanced_rag=True,
            custom_integrations=True,
            dedicated_support=True,
            github_mcp_access=True
        ),
        "limits": SubscriptionLimits(
            concurrent_research_units=20,
            api_calls_per_hour=-1,  # Unlimited
            api_calls_per_day=-1,   # Unlimited
            research_requests_per_hour=-1,  # Unlimited
            research_requests_per_day=-1,   # Unlimited
            token_usage_per_day=-1,  # Unlimited
            rag_queries_per_day=-1,  # Unlimited
            monthly_research_requests=-1,  # Unlimited
            monthly_api_calls=-1,  # Unlimited
            monthly_tokens=-1  # Unlimited
        )
    }
}

class SubscriptionManager:
    """Main subscription management class."""
    
    def __init__(self, supabase_client: Optional[Client] = None):
        """Initialize the subscription manager."""
        self.supabase = supabase_client or supabase_client
        
    async def get_user_subscription(self, user_id: str) -> UserSubscription:
        """Get comprehensive user subscription information."""
        if not self.supabase:
            logger.warning("Supabase client not available, returning default free subscription")
            return self._get_default_subscription(user_id)
        
        try:
            # Get subscription from database
            response = await asyncio.to_thread(
                self.supabase.rpc,
                "get_user_subscription_status",
                {"user_uuid": user_id}
            )
            
            if response.data:
                subscription_data = response.data[0] if isinstance(response.data, list) else response.data
                tier = SubscriptionTierEnum(subscription_data.get("tier", "free"))
                status = SubscriptionStatus(subscription_data.get("status", "active"))
                
                # Get tier configuration
                tier_config = ENHANCED_SUBSCRIPTION_TIERS.get(tier, ENHANCED_SUBSCRIPTION_TIERS[SubscriptionTierEnum.FREE])
                
                return UserSubscription(
                    user_id=user_id,
                    subscription_id=subscription_data.get("subscription_id"),
                    customer_id=subscription_data.get("customer_id"),
                    tier=tier,
                    status=status,
                    current_period_start=subscription_data.get("current_period_start"),
                    current_period_end=subscription_data.get("current_period_end"),
                    cancel_at_period_end=subscription_data.get("cancel_at_period_end", False),
                    features=tier_config["features"],
                    limits=tier_config["limits"]
                )
            else:
                # Return default free subscription
                return self._get_default_subscription(user_id)
                
        except Exception as e:
            logger.error(f"Error getting user subscription for {user_id}: {e}")
            return self._get_default_subscription(user_id)
    
    def _get_default_subscription(self, user_id: str) -> UserSubscription:
        """Get default free subscription for a user."""
        tier_config = ENHANCED_SUBSCRIPTION_TIERS[SubscriptionTierEnum.FREE]
        return UserSubscription(
            user_id=user_id,
            tier=SubscriptionTierEnum.FREE,
            status=SubscriptionStatus.ACTIVE,
            features=tier_config["features"],
            limits=tier_config["limits"]
        )
    
    async def check_feature_access(self, user_id: str, feature: str) -> bool:
        """Check if user has access to a specific feature."""
        subscription = await self.get_user_subscription(user_id)
        
        # Check if subscription is active
        if subscription.status != SubscriptionStatus.ACTIVE:
            return False
        
        # Check feature access
        return getattr(subscription.features, feature, False)
    
    async def check_mcp_server_access(self, user_id: str, server_name: str) -> bool:
        """Check if user can access a specific MCP server."""
        subscription = await self.get_user_subscription(user_id)
        
        # Check if subscription is active
        if subscription.status != SubscriptionStatus.ACTIVE:
            return False
        
        # Check MCP server access
        allowed_servers = subscription.features.mcp_servers
        return server_name.lower() in [s.lower() for s in allowed_servers]
    
    async def check_github_mcp_access(self, user_id: str) -> bool:
        """Check if user has access to GitHub MCP server (Pro/Enterprise only)."""
        subscription = await self.get_user_subscription(user_id)
        
        # GitHub MCP access is only for Pro and Enterprise tiers
        return (
            subscription.status == SubscriptionStatus.ACTIVE and
            subscription.features.github_mcp_access and
            subscription.tier in [SubscriptionTierEnum.PRO, SubscriptionTierEnum.ENTERPRISE]
        )
    
    async def get_concurrent_research_limit(self, user_id: str) -> int:
        """Get the concurrent research units limit for a user."""
        subscription = await self.get_user_subscription(user_id)
        return subscription.limits.concurrent_research_units
    
    async def get_usage_limits(self, user_id: str) -> Dict[str, int]:
        """Get all usage limits for a user."""
        subscription = await self.get_user_subscription(user_id)
        return {
            "concurrent_research_units": subscription.limits.concurrent_research_units,
            "api_calls_per_hour": subscription.limits.api_calls_per_hour,
            "api_calls_per_day": subscription.limits.api_calls_per_day,
            "research_requests_per_hour": subscription.limits.research_requests_per_hour,
            "research_requests_per_day": subscription.limits.research_requests_per_day,
            "token_usage_per_day": subscription.limits.token_usage_per_day,
            "rag_queries_per_day": subscription.limits.rag_queries_per_day,
            "monthly_research_requests": subscription.limits.monthly_research_requests,
            "monthly_api_calls": subscription.limits.monthly_api_calls,
            "monthly_tokens": subscription.limits.monthly_tokens
        }
    
    async def check_usage_limit(self, user_id: str, usage_type: str, current_usage: int) -> Tuple[bool, str]:
        """Check if user is within usage limits for a specific usage type."""
        subscription = await self.get_user_subscription(user_id)
        
        # Get the appropriate limit
        limit_mapping = {
            "concurrent_research_units": subscription.limits.concurrent_research_units,
            "api_calls_per_hour": subscription.limits.api_calls_per_hour,
            "api_calls_per_day": subscription.limits.api_calls_per_day,
            "research_requests_per_hour": subscription.limits.research_requests_per_hour,
            "research_requests_per_day": subscription.limits.research_requests_per_day,
            "token_usage_per_day": subscription.limits.token_usage_per_day,
            "rag_queries_per_day": subscription.limits.rag_queries_per_day
        }
        
        limit = limit_mapping.get(usage_type)
        if limit is None:
            return True, "Unknown usage type"
        
        # -1 means unlimited
        if limit == -1:
            return True, "Unlimited usage"
        
        # Check if within limit
        if current_usage >= limit:
            return False, f"{usage_type} limit exceeded ({current_usage}/{limit})"
        
        return True, f"Within {usage_type} limit ({current_usage}/{limit})"
    
    async def get_tier_comparison(self) -> Dict[str, Any]:
        """Get comparison of all subscription tiers."""
        return {
            tier.value: {
                "name": config["name"],
                "monthly_price": config["monthly_price"],
                "features": config["features"].model_dump(),
                "limits": config["limits"].model_dump()
            }
            for tier, config in ENHANCED_SUBSCRIPTION_TIERS.items()
        }
    
    async def upgrade_subscription(self, user_id: str, new_tier: SubscriptionTierEnum) -> bool:
        """Upgrade user subscription to a new tier."""
        if not self.supabase:
            logger.warning("Supabase client not available for subscription upgrade")
            return False
        
        try:
            # Update subscription in database
            response = await asyncio.to_thread(
                self.supabase.table("subscriptions").update({
                    "tier": new_tier.value,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }).eq("user_id", user_id).execute
            )
            
            logger.info(f"Upgraded user {user_id} to {new_tier.value} tier")
            return True
            
        except Exception as e:
            logger.error(f"Error upgrading subscription for user {user_id}: {e}")
            return False
    
    async def cancel_subscription(self, user_id: str, at_period_end: bool = True) -> bool:
        """Cancel user subscription."""
        if not self.supabase:
            logger.warning("Supabase client not available for subscription cancellation")
            return False
        
        try:
            # Update subscription in database
            update_data = {
                "cancel_at_period_end": at_period_end,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            if not at_period_end:
                update_data["status"] = SubscriptionStatus.CANCELLED.value
                update_data["tier"] = SubscriptionTierEnum.FREE.value
            
            response = await asyncio.to_thread(
                self.supabase.table("subscriptions").update(update_data).eq("user_id", user_id).execute
            )
            
            logger.info(f"Cancelled subscription for user {user_id} (at_period_end: {at_period_end})")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling subscription for user {user_id}: {e}")
            return False
    
    async def get_subscription_analytics(self, user_id: str) -> Dict[str, Any]:
        """Get subscription analytics for a user."""
        subscription = await self.get_user_subscription(user_id)
        
        # Calculate subscription value and usage efficiency
        monthly_price = ENHANCED_SUBSCRIPTION_TIERS[subscription.tier]["monthly_price"]
        
        analytics = {
            "subscription_info": {
                "tier": subscription.tier.value,
                "status": subscription.status.value,
                "monthly_price": monthly_price,
                "features_count": len([f for f, v in subscription.features.model_dump().items() if v]),
            },
            "limits_summary": subscription.limits.model_dump(),
            "feature_access": subscription.features.model_dump(),
            "tier_benefits": {
                "mcp_servers": subscription.features.mcp_servers,
                "github_access": subscription.features.github_mcp_access,
                "advanced_features": subscription.features.advanced_rag,
                "support_level": "dedicated" if subscription.features.dedicated_support else "priority" if subscription.features.priority_support else "standard"
            }
        }
        
        return analytics

# Utility functions for external use

async def get_user_subscription_info(user_id: str) -> UserSubscription:
    """Get user subscription information."""
    manager = SubscriptionManager()
    return await manager.get_user_subscription(user_id)

async def check_user_feature_access(user_id: str, feature: str) -> bool:
    """Check if user has access to a specific feature."""
    manager = SubscriptionManager()
    return await manager.check_feature_access(user_id, feature)

async def get_user_concurrent_research_limit(user_id: str) -> int:
    """Get concurrent research units limit for a user."""
    manager = SubscriptionManager()
    return await manager.get_concurrent_research_limit(user_id)

async def check_github_mcp_access(user_id: str) -> bool:
    """Check if user has GitHub MCP server access."""
    manager = SubscriptionManager()
    return await manager.check_github_mcp_access(user_id)

async def get_subscription_tier_config(tier: SubscriptionTierEnum) -> Dict[str, Any]:
    """Get configuration for a specific subscription tier."""
    config = ENHANCED_SUBSCRIPTION_TIERS.get(tier, ENHANCED_SUBSCRIPTION_TIERS[SubscriptionTierEnum.FREE])
    return {
        "name": config["name"],
        "monthly_price": config["monthly_price"],
        "features": config["features"].model_dump(),
        "limits": config["limits"].model_dump()
    }

# Configuration integration functions

def get_tier_based_concurrent_limit(tier: SubscriptionTierEnum) -> int:
    """Get concurrent research units limit based on subscription tier."""
    config = ENHANCED_SUBSCRIPTION_TIERS.get(tier, ENHANCED_SUBSCRIPTION_TIERS[SubscriptionTierEnum.FREE])
    return config["limits"].concurrent_research_units

def get_tier_based_mcp_servers(tier: SubscriptionTierEnum) -> List[str]:
    """Get allowed MCP servers based on subscription tier."""
    config = ENHANCED_SUBSCRIPTION_TIERS.get(tier, ENHANCED_SUBSCRIPTION_TIERS[SubscriptionTierEnum.FREE])
    return config["features"].mcp_servers

def validate_tier_access(tier: SubscriptionTierEnum, feature: str) -> bool:
    """Validate if a tier has access to a specific feature."""
    config = ENHANCED_SUBSCRIPTION_TIERS.get(tier, ENHANCED_SUBSCRIPTION_TIERS[SubscriptionTierEnum.FREE])
    return getattr(config["features"], feature, False)
