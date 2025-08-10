"""
Stripe Service for subscription management and billing integration.

This module provides comprehensive Stripe integration for the Open Deep Research platform,
including subscription management, usage tracking, and billing operations.
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import stripe
from decimal import Decimal

logger = logging.getLogger(__name__)

class SubscriptionTier(Enum):
    """Subscription tiers with different features and limits."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class PaymentStatus(Enum):
    """Payment and subscription status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    UNPAID = "unpaid"
    TRIALING = "trialing"

@dataclass
class TierLimits:
    """Limits and features for each subscription tier."""
    name: str
    monthly_price: Decimal
    yearly_price: Decimal
    research_requests_per_month: int
    max_concurrent_research: int
    api_calls_per_month: int
    tokens_per_month: int
    mcp_servers_access: List[str]
    rag_context_enabled: bool
    priority_support: bool
    custom_integrations: bool
    data_retention_days: int
    export_formats: List[str]

@dataclass
class UserSubscription:
    """User subscription information."""
    user_id: str
    customer_id: str
    subscription_id: Optional[str]
    tier: SubscriptionTier
    status: PaymentStatus
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool
    trial_end: Optional[datetime]
    created_at: datetime
    updated_at: datetime

@dataclass
class UsageMetrics:
    """User usage metrics for billing."""
    user_id: str
    period_start: datetime
    period_end: datetime
    research_requests: int
    api_calls: int
    tokens_used: int
    mcp_calls: int
    storage_used_mb: int
    last_updated: datetime

class StripeService:
    """Comprehensive Stripe service for subscription management."""
    
    def __init__(self, api_key: Optional[str] = None, webhook_secret: Optional[str] = None):
        """
        Initialize Stripe service.
        
        Args:
            api_key: Stripe API key (defaults to environment variable)
            webhook_secret: Stripe webhook secret (defaults to environment variable)
        """
        self.api_key = api_key or os.getenv("STRIPE_SECRET_KEY")
        self.webhook_secret = webhook_secret or os.getenv("STRIPE_WEBHOOK_SECRET")
        
        if not self.api_key:
            raise ValueError("Stripe API key is required")
        
        stripe.api_key = self.api_key
        
        # Define subscription tiers and their limits
        self.tier_limits = {
            SubscriptionTier.FREE: TierLimits(
                name="Free",
                monthly_price=Decimal("0.00"),
                yearly_price=Decimal("0.00"),
                research_requests_per_month=10,
                max_concurrent_research=1,
                api_calls_per_month=100,
                tokens_per_month=50000,
                mcp_servers_access=["reddit"],
                rag_context_enabled=False,
                priority_support=False,
                custom_integrations=False,
                data_retention_days=30,
                export_formats=["json"]
            ),
            SubscriptionTier.PRO: TierLimits(
                name="Pro",
                monthly_price=Decimal("29.99"),
                yearly_price=Decimal("299.99"),
                research_requests_per_month=500,
                max_concurrent_research=5,
                api_calls_per_month=10000,
                tokens_per_month=2000000,
                mcp_servers_access=["reddit", "youtube", "github"],
                rag_context_enabled=True,
                priority_support=True,
                custom_integrations=False,
                data_retention_days=365,
                export_formats=["json", "csv", "pdf"]
            ),
            SubscriptionTier.ENTERPRISE: TierLimits(
                name="Enterprise",
                monthly_price=Decimal("199.99"),
                yearly_price=Decimal("1999.99"),
                research_requests_per_month=-1,  # Unlimited
                max_concurrent_research=20,
                api_calls_per_month=-1,  # Unlimited
                tokens_per_month=-1,  # Unlimited
                mcp_servers_access=["reddit", "youtube", "github"],
                rag_context_enabled=True,
                priority_support=True,
                custom_integrations=True,
                data_retention_days=-1,  # Unlimited
                export_formats=["json", "csv", "pdf", "xlsx", "docx"]
            )
        }
        
        # Stripe product and price IDs (would be set up in Stripe dashboard)
        self.stripe_price_ids = {
            (SubscriptionTier.PRO, "monthly"): os.getenv("STRIPE_PRO_MONTHLY_PRICE_ID"),
            (SubscriptionTier.PRO, "yearly"): os.getenv("STRIPE_PRO_YEARLY_PRICE_ID"),
            (SubscriptionTier.ENTERPRISE, "monthly"): os.getenv("STRIPE_ENTERPRISE_MONTHLY_PRICE_ID"),
            (SubscriptionTier.ENTERPRISE, "yearly"): os.getenv("STRIPE_ENTERPRISE_YEARLY_PRICE_ID"),
        }
        
        logger.info("Stripe service initialized successfully")
    
    async def create_customer(
        self, 
        user_id: str, 
        email: str, 
        name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create a new Stripe customer.
        
        Args:
            user_id: Internal user ID
            email: Customer email
            name: Customer name
            metadata: Additional metadata
            
        Returns:
            Stripe customer ID
        """
        try:
            customer_metadata = {"user_id": user_id}
            if metadata:
                customer_metadata.update(metadata)
            
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata=customer_metadata
            )
            
            logger.info(f"Created Stripe customer {customer.id} for user {user_id}")
            return customer.id
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to create Stripe customer for user {user_id}: {e}")
            raise
    
    async def create_subscription(
        self,
        customer_id: str,
        tier: SubscriptionTier,
        billing_cycle: str = "monthly",
        trial_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a new subscription for a customer.
        
        Args:
            customer_id: Stripe customer ID
            tier: Subscription tier
            billing_cycle: "monthly" or "yearly"
            trial_days: Number of trial days
            
        Returns:
            Subscription information
        """
        try:
            if tier == SubscriptionTier.FREE:
                raise ValueError("Cannot create Stripe subscription for free tier")
            
            price_id = self.stripe_price_ids.get((tier, billing_cycle))
            if not price_id:
                raise ValueError(f"No price ID configured for {tier.value} {billing_cycle}")
            
            subscription_params = {
                "customer": customer_id,
                "items": [{"price": price_id}],
                "metadata": {
                    "tier": tier.value,
                    "billing_cycle": billing_cycle
                }
            }
            
            if trial_days:
                subscription_params["trial_period_days"] = trial_days
            
            subscription = stripe.Subscription.create(**subscription_params)
            
            logger.info(f"Created subscription {subscription.id} for customer {customer_id}")
            
            return {
                "subscription_id": subscription.id,
                "status": subscription.status,
                "current_period_start": datetime.fromtimestamp(subscription.current_period_start),
                "current_period_end": datetime.fromtimestamp(subscription.current_period_end),
                "trial_end": datetime.fromtimestamp(subscription.trial_end) if subscription.trial_end else None,
                "cancel_at_period_end": subscription.cancel_at_period_end
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to create subscription for customer {customer_id}: {e}")
            raise
    
    async def update_subscription(
        self,
        subscription_id: str,
        new_tier: SubscriptionTier,
        billing_cycle: str = "monthly",
        prorate: bool = True
    ) -> Dict[str, Any]:
        """
        Update an existing subscription.
        
        Args:
            subscription_id: Stripe subscription ID
            new_tier: New subscription tier
            billing_cycle: "monthly" or "yearly"
            prorate: Whether to prorate the change
            
        Returns:
            Updated subscription information
        """
        try:
            if new_tier == SubscriptionTier.FREE:
                return await self.cancel_subscription(subscription_id)
            
            price_id = self.stripe_price_ids.get((new_tier, billing_cycle))
            if not price_id:
                raise ValueError(f"No price ID configured for {new_tier.value} {billing_cycle}")
            
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            stripe.Subscription.modify(
                subscription_id,
                items=[{
                    "id": subscription["items"]["data"][0]["id"],
                    "price": price_id,
                }],
                proration_behavior="create_prorations" if prorate else "none",
                metadata={
                    "tier": new_tier.value,
                    "billing_cycle": billing_cycle
                }
            )
            
            updated_subscription = stripe.Subscription.retrieve(subscription_id)
            
            logger.info(f"Updated subscription {subscription_id} to {new_tier.value}")
            
            return {
                "subscription_id": updated_subscription.id,
                "status": updated_subscription.status,
                "current_period_start": datetime.fromtimestamp(updated_subscription.current_period_start),
                "current_period_end": datetime.fromtimestamp(updated_subscription.current_period_end),
                "cancel_at_period_end": updated_subscription.cancel_at_period_end
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to update subscription {subscription_id}: {e}")
            raise
    
    async def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True
    ) -> Dict[str, Any]:
        """
        Cancel a subscription.
        
        Args:
            subscription_id: Stripe subscription ID
            at_period_end: Whether to cancel at period end or immediately
            
        Returns:
            Cancellation information
        """
        try:
            if at_period_end:
                subscription = stripe.Subscription.modify(
                    subscription_id,
                    cancel_at_period_end=True
                )
            else:
                subscription = stripe.Subscription.delete(subscription_id)
            
            logger.info(f"Canceled subscription {subscription_id} (at_period_end: {at_period_end})")
            
            return {
                "subscription_id": subscription.id,
                "status": subscription.status,
                "canceled_at": datetime.fromtimestamp(subscription.canceled_at) if subscription.canceled_at else None,
                "cancel_at_period_end": subscription.cancel_at_period_end
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to cancel subscription {subscription_id}: {e}")
            raise
    
    async def get_subscription_info(self, subscription_id: str) -> Dict[str, Any]:
        """
        Get subscription information.
        
        Args:
            subscription_id: Stripe subscription ID
            
        Returns:
            Subscription information
        """
        try:
            subscription = stripe.Subscription.retrieve(subscription_id)
            
            return {
                "subscription_id": subscription.id,
                "customer_id": subscription.customer,
                "status": subscription.status,
                "tier": subscription.metadata.get("tier", "unknown"),
                "billing_cycle": subscription.metadata.get("billing_cycle", "monthly"),
                "current_period_start": datetime.fromtimestamp(subscription.current_period_start),
                "current_period_end": datetime.fromtimestamp(subscription.current_period_end),
                "trial_end": datetime.fromtimestamp(subscription.trial_end) if subscription.trial_end else None,
                "cancel_at_period_end": subscription.cancel_at_period_end,
                "canceled_at": datetime.fromtimestamp(subscription.canceled_at) if subscription.canceled_at else None
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to get subscription info for {subscription_id}: {e}")
            raise
    
    async def create_checkout_session(
        self,
        customer_id: str,
        tier: SubscriptionTier,
        billing_cycle: str = "monthly",
        success_url: str = "",
        cancel_url: str = "",
        trial_days: Optional[int] = None
    ) -> str:
        """
        Create a Stripe Checkout session for subscription.
        
        Args:
            customer_id: Stripe customer ID
            tier: Subscription tier
            billing_cycle: "monthly" or "yearly"
            success_url: URL to redirect on success
            cancel_url: URL to redirect on cancel
            trial_days: Number of trial days
            
        Returns:
            Checkout session URL
        """
        try:
            if tier == SubscriptionTier.FREE:
                raise ValueError("Cannot create checkout session for free tier")
            
            price_id = self.stripe_price_ids.get((tier, billing_cycle))
            if not price_id:
                raise ValueError(f"No price ID configured for {tier.value} {billing_cycle}")
            
            session_params = {
                "customer": customer_id,
                "payment_method_types": ["card"],
                "line_items": [{
                    "price": price_id,
                    "quantity": 1,
                }],
                "mode": "subscription",
                "success_url": success_url,
                "cancel_url": cancel_url,
                "metadata": {
                    "tier": tier.value,
                    "billing_cycle": billing_cycle
                }
            }
            
            if trial_days:
                session_params["subscription_data"] = {
                    "trial_period_days": trial_days
                }
            
            session = stripe.checkout.Session.create(**session_params)
            
            logger.info(f"Created checkout session {session.id} for customer {customer_id}")
            return session.url
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to create checkout session for customer {customer_id}: {e}")
            raise
    
    async def create_billing_portal_session(
        self,
        customer_id: str,
        return_url: str = ""
    ) -> str:
        """
        Create a billing portal session for customer self-service.
        
        Args:
            customer_id: Stripe customer ID
            return_url: URL to return to after portal session
            
        Returns:
            Billing portal URL
        """
        try:
            session = stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=return_url,
            )
            
            logger.info(f"Created billing portal session for customer {customer_id}")
            return session.url
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to create billing portal session for customer {customer_id}: {e}")
            raise
    
    def get_tier_limits(self, tier: SubscriptionTier) -> TierLimits:
        """Get limits and features for a subscription tier."""
        return self.tier_limits[tier]
    
    def check_usage_limits(
        self, 
        tier: SubscriptionTier, 
        usage: UsageMetrics
    ) -> Dict[str, Any]:
        """
        Check if usage is within tier limits.
        
        Args:
            tier: Subscription tier
            usage: Current usage metrics
            
        Returns:
            Usage limit check results
        """
        limits = self.tier_limits[tier]
        
        checks = {
            "research_requests": {
                "used": usage.research_requests,
                "limit": limits.research_requests_per_month,
                "within_limit": limits.research_requests_per_month == -1 or usage.research_requests <= limits.research_requests_per_month,
                "percentage": (usage.research_requests / limits.research_requests_per_month * 100) if limits.research_requests_per_month > 0 else 0
            },
            "api_calls": {
                "used": usage.api_calls,
                "limit": limits.api_calls_per_month,
                "within_limit": limits.api_calls_per_month == -1 or usage.api_calls <= limits.api_calls_per_month,
                "percentage": (usage.api_calls / limits.api_calls_per_month * 100) if limits.api_calls_per_month > 0 else 0
            },
            "tokens": {
                "used": usage.tokens_used,
                "limit": limits.tokens_per_month,
                "within_limit": limits.tokens_per_month == -1 or usage.tokens_used <= limits.tokens_per_month,
                "percentage": (usage.tokens_used / limits.tokens_per_month * 100) if limits.tokens_per_month > 0 else 0
            }
        }
        
        overall_within_limits = all(check["within_limit"] for check in checks.values())
        
        return {
            "tier": tier.value,
            "overall_within_limits": overall_within_limits,
            "checks": checks,
            "period_start": usage.period_start.isoformat(),
            "period_end": usage.period_end.isoformat(),
            "last_updated": usage.last_updated.isoformat()
        }
    
    async def record_usage(
        self,
        user_id: str,
        usage_type: str,
        amount: int = 1,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record usage for billing purposes.
        
        Args:
            user_id: User ID
            usage_type: Type of usage (research_request, api_call, tokens, etc.)
            amount: Amount to record
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            # This would typically integrate with a database or analytics service
            # For now, we'll log the usage
            logger.info(f"Recording usage for user {user_id}: {usage_type} = {amount}")
            
            # In a real implementation, this would:
            # 1. Store usage in database
            # 2. Update real-time counters
            # 3. Check against limits
            # 4. Trigger alerts if approaching limits
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record usage for user {user_id}: {e}")
            return False
    
    def construct_webhook_event(self, payload: bytes, sig_header: str) -> stripe.Event:
        """
        Construct and verify a Stripe webhook event.
        
        Args:
            payload: Raw webhook payload
            sig_header: Stripe signature header
            
        Returns:
            Verified Stripe event
        """
        if not self.webhook_secret:
            raise ValueError("Webhook secret not configured")
        
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, self.webhook_secret
            )
            return event
        except ValueError as e:
            logger.error(f"Invalid payload in webhook: {e}")
            raise
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Invalid signature in webhook: {e}")
            raise
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get Stripe service statistics and configuration."""
        return {
            "api_key_configured": bool(self.api_key),
            "webhook_secret_configured": bool(self.webhook_secret),
            "supported_tiers": [tier.value for tier in SubscriptionTier],
            "tier_limits": {
                tier.value: asdict(limits) for tier, limits in self.tier_limits.items()
            },
            "price_ids_configured": {
                f"{tier.value}_{cycle}": bool(price_id) 
                for (tier, cycle), price_id in self.stripe_price_ids.items()
            }
        }
