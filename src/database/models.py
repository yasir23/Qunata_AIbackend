"""
Database models for Open Deep Research payment and user management system.

This module defines Pydantic models that correspond to Supabase database tables
for users, subscriptions, payments, and usage tracking.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr
from enum import Enum

class SubscriptionStatus(str, Enum):
    """Subscription status enumeration."""
    ACTIVE = "active"
    CANCELED = "canceled"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    PAST_DUE = "past_due"
    TRIALING = "trialing"
    UNPAID = "unpaid"

class SubscriptionTierEnum(str, Enum):
    """Subscription tier enumeration."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

class PaymentStatus(str, Enum):
    """Payment status enumeration."""
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PENDING = "pending"
    CANCELED = "canceled"
    REFUNDED = "refunded"

class UsageType(str, Enum):
    """Usage type enumeration."""
    API_CALL = "api_call"
    RESEARCH_REQUEST = "research_request"
    TOKEN_USAGE = "token_usage"
    MCP_SERVER_CALL = "mcp_server_call"

class UserProfile(BaseModel):
    """User profile model for storing user information and preferences."""
    
    user_id: str = Field(..., description="Supabase user ID")
    email: EmailStr = Field(..., description="User email address")
    name: Optional[str] = Field(None, description="User full name")
    stripe_customer_id: Optional[str] = Field(None, description="Stripe customer ID")
    avatar_url: Optional[str] = Field(None, description="User avatar URL")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences and settings")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Account creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    is_active: bool = Field(default=True, description="Whether the user account is active")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class Subscription(BaseModel):
    """Subscription model for managing user subscriptions."""
    
    id: str = Field(..., description="Stripe subscription ID")
    user_id: str = Field(..., description="Supabase user ID")
    customer_id: str = Field(..., description="Stripe customer ID")
    status: SubscriptionStatus = Field(..., description="Subscription status")
    tier: SubscriptionTierEnum = Field(..., description="Subscription tier")
    price_id: str = Field(..., description="Stripe price ID")
    current_period_start: datetime = Field(..., description="Current billing period start")
    current_period_end: datetime = Field(..., description="Current billing period end")
    cancel_at_period_end: bool = Field(default=False, description="Whether to cancel at period end")
    canceled_at: Optional[datetime] = Field(None, description="Cancellation timestamp")
    trial_start: Optional[datetime] = Field(None, description="Trial period start")
    trial_end: Optional[datetime] = Field(None, description="Trial period end")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Subscription creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional subscription metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PaymentHistory(BaseModel):
    """Payment history model for tracking all payment transactions."""
    
    id: str = Field(..., description="Stripe invoice/payment intent ID")
    user_id: str = Field(..., description="Supabase user ID")
    customer_id: str = Field(..., description="Stripe customer ID")
    subscription_id: Optional[str] = Field(None, description="Related subscription ID")
    amount: int = Field(..., description="Payment amount in cents")
    currency: str = Field(default="usd", description="Payment currency")
    status: PaymentStatus = Field(..., description="Payment status")
    description: str = Field(..., description="Payment description")
    invoice_url: Optional[str] = Field(None, description="Stripe invoice URL")
    receipt_url: Optional[str] = Field(None, description="Payment receipt URL")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Payment timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional payment metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class UsageRecord(BaseModel):
    """Usage record model for tracking API usage and billing."""
    
    id: Optional[str] = Field(None, description="Record ID (auto-generated)")
    user_id: str = Field(..., description="Supabase user ID")
    subscription_id: Optional[str] = Field(None, description="Related subscription ID")
    usage_type: UsageType = Field(..., description="Type of usage")
    quantity: int = Field(..., description="Usage quantity")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Usage timestamp")
    endpoint: Optional[str] = Field(None, description="API endpoint used")
    model_used: Optional[str] = Field(None, description="AI model used")
    tokens_consumed: Optional[int] = Field(None, description="Number of tokens consumed")
    cost: Optional[float] = Field(None, description="Cost of the usage")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional usage metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SubscriptionTierConfig(BaseModel):
    """Subscription tier configuration model."""
    
    tier: SubscriptionTierEnum = Field(..., description="Tier name")
    name: str = Field(..., description="Display name")
    description: str = Field(..., description="Tier description")
    monthly_price: float = Field(..., description="Monthly price in dollars")
    stripe_price_id: Optional[str] = Field(None, description="Stripe price ID")
    features: Dict[str, bool] = Field(..., description="Available features")
    limits: Dict[str, int] = Field(..., description="Usage limits (-1 for unlimited)")
    is_active: bool = Field(default=True, description="Whether the tier is available")
    sort_order: int = Field(default=0, description="Display order")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ApiKey(BaseModel):
    """API key model for user API access."""
    
    id: Optional[str] = Field(None, description="API key ID")
    user_id: str = Field(..., description="Supabase user ID")
    key_hash: str = Field(..., description="Hashed API key")
    name: str = Field(..., description="API key name/description")
    prefix: str = Field(..., description="API key prefix for identification")
    permissions: List[str] = Field(default_factory=list, description="API permissions")
    last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    is_active: bool = Field(default=True, description="Whether the key is active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class UsageSummary(BaseModel):
    """Usage summary model for aggregated usage data."""
    
    user_id: str = Field(..., description="Supabase user ID")
    period_start: datetime = Field(..., description="Summary period start")
    period_end: datetime = Field(..., description="Summary period end")
    api_calls: int = Field(default=0, description="Total API calls")
    research_requests: int = Field(default=0, description="Total research requests")
    tokens_used: int = Field(default=0, description="Total tokens consumed")
    mcp_server_calls: int = Field(default=0, description="Total MCP server calls")
    total_cost: float = Field(default=0.0, description="Total cost for the period")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Summary creation timestamp")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class WebhookEvent(BaseModel):
    """Webhook event model for tracking processed webhooks."""
    
    id: str = Field(..., description="Stripe event ID")
    event_type: str = Field(..., description="Webhook event type")
    processed: bool = Field(default=False, description="Whether the event was processed")
    processed_at: Optional[datetime] = Field(None, description="Processing timestamp")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Event creation timestamp")
    raw_data: Dict[str, Any] = Field(default_factory=dict, description="Raw webhook data")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# Request/Response models for API endpoints

class CreateCheckoutSessionRequest(BaseModel):
    """Request model for creating checkout sessions."""
    
    price_id: str = Field(..., description="Stripe price ID")
    success_url: str = Field(..., description="Success redirect URL")
    cancel_url: str = Field(..., description="Cancel redirect URL")
    trial_days: Optional[int] = Field(None, description="Trial period in days")
    
class CreateCheckoutSessionResponse(BaseModel):
    """Response model for checkout session creation."""
    
    session_id: str = Field(..., description="Stripe session ID")
    url: str = Field(..., description="Checkout URL")
    status: str = Field(..., description="Session status")

class SubscriptionStatusResponse(BaseModel):
    """Response model for subscription status."""
    
    subscription_id: Optional[str] = Field(None, description="Subscription ID")
    tier: SubscriptionTierEnum = Field(..., description="Current tier")
    status: Optional[SubscriptionStatus] = Field(None, description="Subscription status")
    current_period_end: Optional[datetime] = Field(None, description="Current period end")
    cancel_at_period_end: bool = Field(default=False, description="Scheduled for cancellation")
    features: Dict[str, bool] = Field(..., description="Available features")
    limits: Dict[str, int] = Field(..., description="Usage limits")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class UsageLimitsResponse(BaseModel):
    """Response model for usage limits check."""
    
    allowed: bool = Field(..., description="Whether usage is allowed")
    limit: int = Field(..., description="Usage limit (-1 for unlimited)")
    current_usage: int = Field(..., description="Current usage count")
    remaining: Optional[int] = Field(None, description="Remaining usage")
    reset_date: Optional[datetime] = Field(None, description="When limits reset")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class PaymentHistoryResponse(BaseModel):
    """Response model for payment history."""
    
    payments: List[PaymentHistory] = Field(..., description="Payment history records")
    total_count: int = Field(..., description="Total number of payments")
    has_more: bool = Field(..., description="Whether there are more records")

class UsageAnalyticsResponse(BaseModel):
    """Response model for usage analytics."""
    
    period_start: datetime = Field(..., description="Analytics period start")
    period_end: datetime = Field(..., description="Analytics period end")
    total_api_calls: int = Field(..., description="Total API calls")
    total_research_requests: int = Field(..., description="Total research requests")
    total_tokens: int = Field(..., description="Total tokens used")
    daily_usage: List[Dict[str, Any]] = Field(..., description="Daily usage breakdown")
    top_endpoints: List[Dict[str, Any]] = Field(..., description="Most used endpoints")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
