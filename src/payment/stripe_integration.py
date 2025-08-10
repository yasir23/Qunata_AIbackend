"""
Stripe Payment Integration for Open Deep Research

This module provides comprehensive Stripe payment processing functionality including:
- Checkout session creation
- Webhook handling
- Subscription management
- Payment history tracking
- Usage tracking integration
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import stripe
from pydantic import BaseModel, Field
from supabase import Client
import json
import hmac
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Stripe
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

class SubscriptionTier(BaseModel):
    """Subscription tier configuration."""
    name: str
    price_id: str
    monthly_price: float
    features: Dict[str, Any]
    limits: Dict[str, int]

class PaymentIntent(BaseModel):
    """Payment intent data model."""
    id: str
    amount: int
    currency: str
    status: str
    customer_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

class Subscription(BaseModel):
    """Subscription data model."""
    id: str
    customer_id: str
    status: str
    current_period_start: datetime
    current_period_end: datetime
    tier: str
    price_id: str
    cancel_at_period_end: bool = False
    metadata: Dict[str, Any] = {}

class PaymentHistory(BaseModel):
    """Payment history record."""
    id: str
    customer_id: str
    amount: int
    currency: str
    status: str
    description: str
    created_at: datetime
    invoice_id: Optional[str] = None
    subscription_id: Optional[str] = None

class UsageRecord(BaseModel):
    """Usage tracking record."""
    user_id: str
    subscription_id: Optional[str] = None
    usage_type: str  # 'api_call', 'research_request', 'token_usage'
    quantity: int
    timestamp: datetime
    metadata: Dict[str, Any] = {}

# Subscription tier configurations
SUBSCRIPTION_TIERS = {
    "free": SubscriptionTier(
        name="Free",
        price_id="",  # No price ID for free tier
        monthly_price=0.0,
        features={
            "research_requests": True,
            "mcp_servers": ["reddit", "youtube"],
            "api_access": False,
            "priority_support": False,
            "advanced_rag": False
        },
        limits={
            "concurrent_research_units": 1,
            "monthly_research_requests": 10,
            "monthly_api_calls": 0,
            "monthly_tokens": 50000
        }
    ),
    "pro": SubscriptionTier(
        name="Pro",
        price_id=os.getenv("STRIPE_PRO_PRICE_ID", ""),
        monthly_price=29.99,
        features={
            "research_requests": True,
            "mcp_servers": ["reddit", "youtube", "github"],
            "api_access": True,
            "priority_support": True,
            "advanced_rag": True
        },
        limits={
            "concurrent_research_units": 5,
            "monthly_research_requests": 500,
            "monthly_api_calls": 10000,
            "monthly_tokens": 1000000
        }
    ),
    "enterprise": SubscriptionTier(
        name="Enterprise",
        price_id=os.getenv("STRIPE_ENTERPRISE_PRICE_ID", ""),
        monthly_price=99.99,
        features={
            "research_requests": True,
            "mcp_servers": ["reddit", "youtube", "github"],
            "api_access": True,
            "priority_support": True,
            "advanced_rag": True,
            "custom_integrations": True,
            "dedicated_support": True
        },
        limits={
            "concurrent_research_units": 20,
            "monthly_research_requests": -1,  # Unlimited
            "monthly_api_calls": -1,  # Unlimited
            "monthly_tokens": -1  # Unlimited
        }
    )
}

class StripePaymentProcessor:
    """Main Stripe payment processing class."""
    
    def __init__(self, supabase_client: Client):
        """Initialize the payment processor with Supabase client."""
        self.supabase = supabase_client
        self.webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        
        if not stripe.api_key:
            raise ValueError("STRIPE_SECRET_KEY environment variable is required")
    
    def create_customer(self, user_id: str, email: str, name: Optional[str] = None) -> str:
        """Create a new Stripe customer."""
        try:
            customer = stripe.Customer.create(
                email=email,
                name=name,
                metadata={"user_id": user_id}
            )
            
            # Store customer ID in Supabase
            self.supabase.table("user_profiles").upsert({
                "user_id": user_id,
                "stripe_customer_id": customer.id,
                "email": email,
                "name": name,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).execute()
            
            logger.info(f"Created Stripe customer {customer.id} for user {user_id}")
            return customer.id
            
        except stripe.error.StripeError as e:
            logger.error(f"Failed to create Stripe customer: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to store customer data: {e}")
            raise
    
    def get_or_create_customer(self, user_id: str, email: str, name: Optional[str] = None) -> str:
        """Get existing customer or create new one."""
        try:
            # Check if customer exists in Supabase
            result = self.supabase.table("user_profiles").select("stripe_customer_id").eq("user_id", user_id).execute()
            
            if result.data and result.data[0].get("stripe_customer_id"):
                return result.data[0]["stripe_customer_id"]
            
            # Create new customer
            return self.create_customer(user_id, email, name)
            
        except Exception as e:
            logger.error(f"Failed to get or create customer: {e}")
            raise

def create_checkout_session(
    user_id: str,
    price_id: str,
    success_url: str,
    cancel_url: str,
    customer_email: Optional[str] = None,
    trial_days: Optional[int] = None
) -> Dict[str, Any]:
    """Create a Stripe checkout session for subscription."""
    try:
        processor = StripePaymentProcessor(None)  # Will need Supabase client
        
        session_params = {
            "payment_method_types": ["card"],
            "line_items": [{
                "price": price_id,
                "quantity": 1,
            }],
            "mode": "subscription",
            "success_url": success_url,
            "cancel_url": cancel_url,
            "metadata": {"user_id": user_id},
            "allow_promotion_codes": True,
            "billing_address_collection": "required",
        }
        
        if customer_email:
            session_params["customer_email"] = customer_email
        
        if trial_days:
            session_params["subscription_data"] = {
                "trial_period_days": trial_days,
                "metadata": {"user_id": user_id}
            }
        
        session = stripe.checkout.Session.create(**session_params)
        
        logger.info(f"Created checkout session {session.id} for user {user_id}")
        
        return {
            "session_id": session.id,
            "url": session.url,
            "status": "created"
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Failed to create checkout session: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating checkout session: {e}")
        raise

def create_payment_intent(
    amount: int,
    currency: str = "usd",
    customer_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a payment intent for one-time payments."""
    try:
        intent_params = {
            "amount": amount,
            "currency": currency,
            "automatic_payment_methods": {"enabled": True},
            "metadata": metadata or {}
        }
        
        if customer_id:
            intent_params["customer"] = customer_id
        
        intent = stripe.PaymentIntent.create(**intent_params)
        
        logger.info(f"Created payment intent {intent.id}")
        
        return {
            "client_secret": intent.client_secret,
            "id": intent.id,
            "status": intent.status
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Failed to create payment intent: {e}")
        raise

def handle_webhook(payload: bytes, signature: str, supabase_client: Client) -> Dict[str, Any]:
    """Handle Stripe webhook events."""
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    
    if not webhook_secret:
        raise ValueError("STRIPE_WEBHOOK_SECRET environment variable is required")
    
    try:
        # Verify webhook signature
        event = stripe.Webhook.construct_event(payload, signature, webhook_secret)
        
        logger.info(f"Received webhook event: {event['type']}")
        
        # Handle different event types
        if event["type"] == "checkout.session.completed":
            return _handle_checkout_completed(event["data"]["object"], supabase_client)
        
        elif event["type"] == "customer.subscription.created":
            return _handle_subscription_created(event["data"]["object"], supabase_client)
        
        elif event["type"] == "customer.subscription.updated":
            return _handle_subscription_updated(event["data"]["object"], supabase_client)
        
        elif event["type"] == "customer.subscription.deleted":
            return _handle_subscription_deleted(event["data"]["object"], supabase_client)
        
        elif event["type"] == "invoice.payment_succeeded":
            return _handle_payment_succeeded(event["data"]["object"], supabase_client)
        
        elif event["type"] == "invoice.payment_failed":
            return _handle_payment_failed(event["data"]["object"], supabase_client)
        
        else:
            logger.info(f"Unhandled webhook event type: {event['type']}")
            return {"status": "ignored", "event_type": event["type"]}
    
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Webhook signature verification failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Webhook handling failed: {e}")
        raise

def _handle_checkout_completed(session: Dict[str, Any], supabase_client: Client) -> Dict[str, Any]:
    """Handle completed checkout session."""
    try:
        user_id = session.get("metadata", {}).get("user_id")
        customer_id = session.get("customer")
        subscription_id = session.get("subscription")
        
        if not user_id:
            logger.error("No user_id in checkout session metadata")
            return {"status": "error", "message": "Missing user_id"}
        
        # Update user profile with customer ID
        if customer_id:
            supabase_client.table("user_profiles").upsert({
                "user_id": user_id,
                "stripe_customer_id": customer_id,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).execute()
        
        logger.info(f"Checkout completed for user {user_id}, subscription {subscription_id}")
        
        return {"status": "success", "user_id": user_id, "subscription_id": subscription_id}
        
    except Exception as e:
        logger.error(f"Failed to handle checkout completion: {e}")
        raise

def _handle_subscription_created(subscription: Dict[str, Any], supabase_client: Client) -> Dict[str, Any]:
    """Handle subscription creation."""
    try:
        customer_id = subscription.get("customer")
        subscription_id = subscription.get("id")
        price_id = subscription["items"]["data"][0]["price"]["id"]
        
        # Get user ID from customer
        user_result = supabase_client.table("user_profiles").select("user_id").eq("stripe_customer_id", customer_id).execute()
        
        if not user_result.data:
            logger.error(f"No user found for customer {customer_id}")
            return {"status": "error", "message": "User not found"}
        
        user_id = user_result.data[0]["user_id"]
        
        # Determine subscription tier
        tier = "free"
        for tier_name, tier_config in SUBSCRIPTION_TIERS.items():
            if tier_config.price_id == price_id:
                tier = tier_name
                break
        
        # Store subscription in database
        supabase_client.table("subscriptions").insert({
            "id": subscription_id,
            "user_id": user_id,
            "customer_id": customer_id,
            "status": subscription.get("status"),
            "tier": tier,
            "price_id": price_id,
            "current_period_start": datetime.fromtimestamp(subscription.get("current_period_start"), timezone.utc).isoformat(),
            "current_period_end": datetime.fromtimestamp(subscription.get("current_period_end"), timezone.utc).isoformat(),
            "cancel_at_period_end": subscription.get("cancel_at_period_end", False),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": subscription.get("metadata", {})
        }).execute()
        
        logger.info(f"Created subscription {subscription_id} for user {user_id} with tier {tier}")
        
        return {"status": "success", "user_id": user_id, "tier": tier}
        
    except Exception as e:
        logger.error(f"Failed to handle subscription creation: {e}")
        raise

def _handle_subscription_updated(subscription: Dict[str, Any], supabase_client: Client) -> Dict[str, Any]:
    """Handle subscription updates."""
    try:
        subscription_id = subscription.get("id")
        price_id = subscription["items"]["data"][0]["price"]["id"]
        
        # Determine new tier
        tier = "free"
        for tier_name, tier_config in SUBSCRIPTION_TIERS.items():
            if tier_config.price_id == price_id:
                tier = tier_name
                break
        
        # Update subscription in database
        supabase_client.table("subscriptions").update({
            "status": subscription.get("status"),
            "tier": tier,
            "price_id": price_id,
            "current_period_start": datetime.fromtimestamp(subscription.get("current_period_start"), timezone.utc).isoformat(),
            "current_period_end": datetime.fromtimestamp(subscription.get("current_period_end"), timezone.utc).isoformat(),
            "cancel_at_period_end": subscription.get("cancel_at_period_end", False),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "metadata": subscription.get("metadata", {})
        }).eq("id", subscription_id).execute()
        
        logger.info(f"Updated subscription {subscription_id} to tier {tier}")
        
        return {"status": "success", "subscription_id": subscription_id, "tier": tier}
        
    except Exception as e:
        logger.error(f"Failed to handle subscription update: {e}")
        raise

def _handle_subscription_deleted(subscription: Dict[str, Any], supabase_client: Client) -> Dict[str, Any]:
    """Handle subscription cancellation."""
    try:
        subscription_id = subscription.get("id")
        
        # Update subscription status
        supabase_client.table("subscriptions").update({
            "status": "canceled",
            "tier": "free",  # Downgrade to free tier
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", subscription_id).execute()
        
        logger.info(f"Canceled subscription {subscription_id}")
        
        return {"status": "success", "subscription_id": subscription_id}
        
    except Exception as e:
        logger.error(f"Failed to handle subscription deletion: {e}")
        raise

def _handle_payment_succeeded(invoice: Dict[str, Any], supabase_client: Client) -> Dict[str, Any]:
    """Handle successful payment."""
    try:
        customer_id = invoice.get("customer")
        subscription_id = invoice.get("subscription")
        amount = invoice.get("amount_paid")
        currency = invoice.get("currency")
        
        # Get user ID
        user_result = supabase_client.table("user_profiles").select("user_id").eq("stripe_customer_id", customer_id).execute()
        
        if not user_result.data:
            logger.error(f"No user found for customer {customer_id}")
            return {"status": "error", "message": "User not found"}
        
        user_id = user_result.data[0]["user_id"]
        
        # Record payment history
        supabase_client.table("payment_history").insert({
            "id": invoice.get("id"),
            "user_id": user_id,
            "customer_id": customer_id,
            "subscription_id": subscription_id,
            "amount": amount,
            "currency": currency,
            "status": "succeeded",
            "description": invoice.get("description", "Subscription payment"),
            "created_at": datetime.fromtimestamp(invoice.get("created"), timezone.utc).isoformat(),
            "metadata": invoice.get("metadata", {})
        }).execute()
        
        logger.info(f"Recorded successful payment for user {user_id}")
        
        return {"status": "success", "user_id": user_id, "amount": amount}
        
    except Exception as e:
        logger.error(f"Failed to handle payment success: {e}")
        raise

def _handle_payment_failed(invoice: Dict[str, Any], supabase_client: Client) -> Dict[str, Any]:
    """Handle failed payment."""
    try:
        customer_id = invoice.get("customer")
        subscription_id = invoice.get("subscription")
        
        # Get user ID
        user_result = supabase_client.table("user_profiles").select("user_id").eq("stripe_customer_id", customer_id).execute()
        
        if not user_result.data:
            logger.error(f"No user found for customer {customer_id}")
            return {"status": "error", "message": "User not found"}
        
        user_id = user_result.data[0]["user_id"]
        
        # Record failed payment
        supabase_client.table("payment_history").insert({
            "id": invoice.get("id"),
            "user_id": user_id,
            "customer_id": customer_id,
            "subscription_id": subscription_id,
            "amount": invoice.get("amount_due"),
            "currency": invoice.get("currency"),
            "status": "failed",
            "description": invoice.get("description", "Subscription payment failed"),
            "created_at": datetime.fromtimestamp(invoice.get("created"), timezone.utc).isoformat(),
            "metadata": invoice.get("metadata", {})
        }).execute()
        
        logger.warning(f"Payment failed for user {user_id}")
        
        return {"status": "success", "user_id": user_id, "payment_status": "failed"}
        
    except Exception as e:
        logger.error(f"Failed to handle payment failure: {e}")
        raise

def get_subscription_status(user_id: str, supabase_client: Client) -> Optional[Dict[str, Any]]:
    """Get current subscription status for a user."""
    try:
        result = supabase_client.table("subscriptions").select("*").eq("user_id", user_id).eq("status", "active").execute()
        
        if not result.data:
            return None
        
        subscription = result.data[0]
        tier_config = SUBSCRIPTION_TIERS.get(subscription["tier"], SUBSCRIPTION_TIERS["free"])
        
        return {
            "subscription_id": subscription["id"],
            "tier": subscription["tier"],
            "status": subscription["status"],
            "current_period_end": subscription["current_period_end"],
            "cancel_at_period_end": subscription["cancel_at_period_end"],
            "features": tier_config.features,
            "limits": tier_config.limits
        }
        
    except Exception as e:
        logger.error(f"Failed to get subscription status: {e}")
        return None

def cancel_subscription(subscription_id: str) -> Dict[str, Any]:
    """Cancel a subscription at period end."""
    try:
        subscription = stripe.Subscription.modify(
            subscription_id,
            cancel_at_period_end=True
        )
        
        logger.info(f"Scheduled cancellation for subscription {subscription_id}")
        
        return {
            "status": "success",
            "subscription_id": subscription_id,
            "cancel_at_period_end": True,
            "current_period_end": subscription.current_period_end
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Failed to cancel subscription: {e}")
        raise

def update_subscription(subscription_id: str, new_price_id: str) -> Dict[str, Any]:
    """Update subscription to a new price/tier."""
    try:
        subscription = stripe.Subscription.retrieve(subscription_id)
        
        stripe.Subscription.modify(
            subscription_id,
            items=[{
                "id": subscription["items"]["data"][0]["id"],
                "price": new_price_id,
            }],
            proration_behavior="create_prorations"
        )
        
        logger.info(f"Updated subscription {subscription_id} to price {new_price_id}")
        
        return {"status": "success", "subscription_id": subscription_id}
        
    except stripe.error.StripeError as e:
        logger.error(f"Failed to update subscription: {e}")
        raise

def get_payment_history(user_id: str, supabase_client: Client, limit: int = 50) -> List[Dict[str, Any]]:
    """Get payment history for a user."""
    try:
        result = supabase_client.table("payment_history").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        
        return result.data or []
        
    except Exception as e:
        logger.error(f"Failed to get payment history: {e}")
        return []

def track_usage(
    user_id: str,
    usage_type: str,
    quantity: int,
    supabase_client: Client,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """Track usage for billing and rate limiting."""
    try:
        # Get current subscription
        subscription_result = supabase_client.table("subscriptions").select("*").eq("user_id", user_id).eq("status", "active").execute()
        
        subscription_id = None
        if subscription_result.data:
            subscription_id = subscription_result.data[0]["id"]
        
        # Record usage
        supabase_client.table("usage_records").insert({
            "user_id": user_id,
            "subscription_id": subscription_id,
            "usage_type": usage_type,
            "quantity": quantity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {}
        }).execute()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to track usage: {e}")
        return False

def get_usage_summary(user_id: str, supabase_client: Client, period_start: datetime, period_end: datetime) -> Dict[str, int]:
    """Get usage summary for a user within a time period."""
    try:
        result = supabase_client.table("usage_records").select("usage_type, quantity").eq("user_id", user_id).gte("timestamp", period_start.isoformat()).lte("timestamp", period_end.isoformat()).execute()
        
        usage_summary = {}
        for record in result.data or []:
            usage_type = record["usage_type"]
            quantity = record["quantity"]
            usage_summary[usage_type] = usage_summary.get(usage_type, 0) + quantity
        
        return usage_summary
        
    except Exception as e:
        logger.error(f"Failed to get usage summary: {e}")
        return {}

def check_usage_limits(user_id: str, usage_type: str, supabase_client: Client) -> Dict[str, Any]:
    """Check if user has exceeded usage limits."""
    try:
        # Get subscription status
        subscription_status = get_subscription_status(user_id, supabase_client)
        
        if not subscription_status:
            # Use free tier limits
            limits = SUBSCRIPTION_TIERS["free"].limits
        else:
            limits = subscription_status["limits"]
        
        # Get current month usage
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        usage_summary = get_usage_summary(user_id, supabase_client, month_start, now)
        
        # Check specific usage type limit
        limit_key = f"monthly_{usage_type}"
        limit = limits.get(limit_key, 0)
        current_usage = usage_summary.get(usage_type, 0)
        
        # -1 means unlimited
        if limit == -1:
            return {"allowed": True, "limit": -1, "current_usage": current_usage}
        
        return {
            "allowed": current_usage < limit,
            "limit": limit,
            "current_usage": current_usage,
            "remaining": max(0, limit - current_usage)
        }
        
    except Exception as e:
        logger.error(f"Failed to check usage limits: {e}")
        return {"allowed": False, "error": str(e)}
