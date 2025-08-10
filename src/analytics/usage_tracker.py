"""
Comprehensive Usage Tracker for research operations and billing integration.

This module provides detailed usage tracking, quota enforcement, and billing integration
with Stripe for the Open Deep Research platform.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import redis
from collections import defaultdict, deque

# Import Stripe and Firebase integrations
try:
    from payments.stripe_service import StripeService, SubscriptionTier
    from payments.firebase_integration import FirebasePaymentIntegration
    BILLING_AVAILABLE = True
except ImportError:
    BILLING_AVAILABLE = False
    logging.warning("Billing integration not available")

logger = logging.getLogger(__name__)

class UsageType(Enum):
    """Types of usage to track."""
    RESEARCH_REQUEST = "research_request"
    API_CALL = "api_call"
    TOKEN_USAGE = "token_usage"
    MCP_CALL = "mcp_call"
    RAG_QUERY = "rag_query"
    SEARCH_QUERY = "search_query"
    DATA_STORAGE = "data_storage"
    EXPORT_REQUEST = "export_request"
    CONCURRENT_RESEARCH = "concurrent_research"

class QuotaStatus(Enum):
    """Quota status enumeration."""
    WITHIN_LIMITS = "within_limits"
    APPROACHING_LIMIT = "approaching_limit"
    LIMIT_EXCEEDED = "limit_exceeded"
    UNLIMITED = "unlimited"

@dataclass
class UsageMetric:
    """Individual usage metric data structure."""
    user_id: str
    usage_type: UsageType
    amount: int
    timestamp: datetime
    metadata: Dict[str, Any] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    cost_cents: Optional[int] = None

@dataclass
class QuotaLimit:
    """Quota limit configuration."""
    usage_type: UsageType
    limit: int  # -1 for unlimited
    period_hours: int = 24 * 30  # Default to monthly
    warning_threshold: float = 0.8  # 80%
    critical_threshold: float = 0.95  # 95%

@dataclass
class UsageSummary:
    """Usage summary for a user and period."""
    user_id: str
    period_start: datetime
    period_end: datetime
    usage_by_type: Dict[UsageType, int]
    total_cost_cents: int
    quota_status: Dict[UsageType, QuotaStatus]
    warnings: List[str]

class QuotaEnforcer:
    """Quota enforcement with real-time checking."""
    
    def __init__(self, usage_tracker: 'UsageTracker'):
        """
        Initialize quota enforcer.
        
        Args:
            usage_tracker: UsageTracker instance
        """
        self.usage_tracker = usage_tracker
        self.tier_limits = self._initialize_tier_limits()
        
    def _initialize_tier_limits(self) -> Dict[SubscriptionTier, List[QuotaLimit]]:
        """Initialize quota limits for each subscription tier."""
        return {
            SubscriptionTier.FREE: [
                QuotaLimit(UsageType.RESEARCH_REQUEST, 10, period_hours=24*30),
                QuotaLimit(UsageType.API_CALL, 100, period_hours=24*30),
                QuotaLimit(UsageType.TOKEN_USAGE, 50000, period_hours=24*30),
                QuotaLimit(UsageType.MCP_CALL, 50, period_hours=24*30),
                QuotaLimit(UsageType.RAG_QUERY, 0, period_hours=24*30),  # Not available
                QuotaLimit(UsageType.CONCURRENT_RESEARCH, 1, period_hours=1),
                QuotaLimit(UsageType.DATA_STORAGE, 100, period_hours=24*30),  # MB
                QuotaLimit(UsageType.EXPORT_REQUEST, 5, period_hours=24*30),
            ],
            SubscriptionTier.PRO: [
                QuotaLimit(UsageType.RESEARCH_REQUEST, 500, period_hours=24*30),
                QuotaLimit(UsageType.API_CALL, 10000, period_hours=24*30),
                QuotaLimit(UsageType.TOKEN_USAGE, 2000000, period_hours=24*30),
                QuotaLimit(UsageType.MCP_CALL, 5000, period_hours=24*30),
                QuotaLimit(UsageType.RAG_QUERY, 1000, period_hours=24*30),
                QuotaLimit(UsageType.CONCURRENT_RESEARCH, 5, period_hours=1),
                QuotaLimit(UsageType.DATA_STORAGE, 10000, period_hours=24*30),  # MB
                QuotaLimit(UsageType.EXPORT_REQUEST, 100, period_hours=24*30),
            ],
            SubscriptionTier.ENTERPRISE: [
                QuotaLimit(UsageType.RESEARCH_REQUEST, -1, period_hours=24*30),  # Unlimited
                QuotaLimit(UsageType.API_CALL, -1, period_hours=24*30),
                QuotaLimit(UsageType.TOKEN_USAGE, -1, period_hours=24*30),
                QuotaLimit(UsageType.MCP_CALL, -1, period_hours=24*30),
                QuotaLimit(UsageType.RAG_QUERY, -1, period_hours=24*30),
                QuotaLimit(UsageType.CONCURRENT_RESEARCH, 20, period_hours=1),
                QuotaLimit(UsageType.DATA_STORAGE, -1, period_hours=24*30),
                QuotaLimit(UsageType.EXPORT_REQUEST, -1, period_hours=24*30),
            ]
        }
    
    async def check_quota(
        self, 
        user_id: str, 
        usage_type: UsageType, 
        amount: int = 1,
        tier: Optional[SubscriptionTier] = None
    ) -> Tuple[bool, QuotaStatus, Dict[str, Any]]:
        """
        Check if usage is within quota limits.
        
        Args:
            user_id: User ID
            usage_type: Type of usage to check
            amount: Amount of usage to check
            tier: User's subscription tier (will fetch if not provided)
            
        Returns:
            Tuple of (allowed, status, details)
        """
        try:
            # Get user's subscription tier if not provided
            if tier is None:
                tier = await self._get_user_tier(user_id)
            
            # Get quota limits for this tier
            limits = self.tier_limits.get(tier, self.tier_limits[SubscriptionTier.FREE])
            quota_limit = next((limit for limit in limits if limit.usage_type == usage_type), None)
            
            if not quota_limit:
                # No limit defined, allow
                return True, QuotaStatus.WITHIN_LIMITS, {"message": "No quota limit defined"}
            
            if quota_limit.limit == -1:
                # Unlimited
                return True, QuotaStatus.UNLIMITED, {"message": "Unlimited usage"}
            
            # Get current usage for the period
            period_start = datetime.now() - timedelta(hours=quota_limit.period_hours)
            current_usage = await self.usage_tracker.get_usage_summary(
                user_id, period_start, datetime.now()
            )
            
            current_amount = current_usage.usage_by_type.get(usage_type, 0)
            projected_usage = current_amount + amount
            
            # Check limits
            if projected_usage > quota_limit.limit:
                return False, QuotaStatus.LIMIT_EXCEEDED, {
                    "current_usage": current_amount,
                    "limit": quota_limit.limit,
                    "requested_amount": amount,
                    "message": f"Quota exceeded: {projected_usage}/{quota_limit.limit}"
                }
            
            # Check warning thresholds
            usage_percentage = projected_usage / quota_limit.limit
            
            if usage_percentage >= quota_limit.critical_threshold:
                status = QuotaStatus.APPROACHING_LIMIT
                message = f"Critical usage: {usage_percentage*100:.1f}% of quota"
            elif usage_percentage >= quota_limit.warning_threshold:
                status = QuotaStatus.APPROACHING_LIMIT
                message = f"High usage: {usage_percentage*100:.1f}% of quota"
            else:
                status = QuotaStatus.WITHIN_LIMITS
                message = f"Normal usage: {usage_percentage*100:.1f}% of quota"
            
            return True, status, {
                "current_usage": current_amount,
                "limit": quota_limit.limit,
                "requested_amount": amount,
                "usage_percentage": usage_percentage * 100,
                "message": message
            }
            
        except Exception as e:
            logger.error(f"Error checking quota for {user_id}: {e}")
            # On error, allow usage but log the issue
            return True, QuotaStatus.WITHIN_LIMITS, {"error": str(e)}
    
    async def _get_user_tier(self, user_id: str) -> SubscriptionTier:
        """Get user's subscription tier."""
        try:
            if BILLING_AVAILABLE:
                # Get from Firebase/Stripe integration
                firebase_integration = FirebasePaymentIntegration()
                subscription = await firebase_integration.get_user_subscription(user_id)
                
                if subscription and subscription.get("tier"):
                    return SubscriptionTier(subscription["tier"])
            
            # Default to free tier
            return SubscriptionTier.FREE
            
        except Exception as e:
            logger.error(f"Error getting user tier for {user_id}: {e}")
            return SubscriptionTier.FREE

class UsageTracker:
    """Comprehensive usage tracking with billing integration."""
    
    def __init__(
        self, 
        redis_client: Optional[redis.Redis] = None,
        stripe_service: Optional[StripeService] = None,
        firebase_integration: Optional[FirebasePaymentIntegration] = None
    ):
        """
        Initialize usage tracker.
        
        Args:
            redis_client: Redis client for real-time tracking
            stripe_service: Stripe service for billing integration
            firebase_integration: Firebase integration for data persistence
        """
        self.redis_client = redis_client
        self.stripe_service = stripe_service
        self.firebase_integration = firebase_integration
        
        # In-memory fallback for when Redis is not available
        self.memory_cache = defaultdict(lambda: deque(maxlen=10000))
        
        # Usage callbacks for real-time notifications
        self.usage_callbacks: List[Callable[[UsageMetric], None]] = []
        
        # Quota enforcer
        self.quota_enforcer = QuotaEnforcer(self)
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()
        
        logger.info("Usage tracker initialized")
    
    async def track_usage(
        self, 
        user_id: str, 
        usage_type: UsageType, 
        amount: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        check_quota: bool = True
    ) -> Dict[str, Any]:
        """
        Track usage and optionally enforce quotas.
        
        Args:
            user_id: User ID
            usage_type: Type of usage
            amount: Amount of usage
            metadata: Additional metadata
            session_id: Session ID for grouping
            request_id: Request ID for tracing
            check_quota: Whether to check quota limits
            
        Returns:
            Tracking result with quota status
        """
        try:
            # Check quota if requested
            quota_result = None
            if check_quota:
                allowed, status, details = await self.quota_enforcer.check_quota(
                    user_id, usage_type, amount
                )
                
                quota_result = {
                    "allowed": allowed,
                    "status": status.value,
                    "details": details
                }
                
                if not allowed:
                    logger.warning(f"Quota exceeded for {user_id}: {usage_type.value}")
                    return {
                        "success": False,
                        "quota": quota_result,
                        "message": "Quota limit exceeded"
                    }
            
            # Create usage metric
            usage_metric = UsageMetric(
                user_id=user_id,
                usage_type=usage_type,
                amount=amount,
                timestamp=datetime.now(),
                metadata=metadata or {},
                session_id=session_id,
                request_id=request_id,
                cost_cents=self._calculate_cost(usage_type, amount)
            )
            
            # Store usage metric
            await self._store_usage_metric(usage_metric)
            
            # Trigger callbacks
            await self._trigger_usage_callbacks(usage_metric)
            
            # Update billing if available
            if BILLING_AVAILABLE and self.stripe_service:
                await self.stripe_service.record_usage(
                    user_id=user_id,
                    usage_type=usage_type.value,
                    amount=amount,
                    metadata=metadata
                )
            
            return {
                "success": True,
                "usage_metric": asdict(usage_metric),
                "quota": quota_result,
                "message": "Usage tracked successfully"
            }
            
        except Exception as e:
            logger.error(f"Error tracking usage for {user_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to track usage"
            }
    
    async def get_usage_summary(
        self, 
        user_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> UsageSummary:
        """
        Get usage summary for a user and time period.
        
        Args:
            user_id: User ID
            start_time: Start of period
            end_time: End of period
            
        Returns:
            Usage summary
        """
        try:
            # Get usage metrics from storage
            usage_metrics = await self._get_usage_metrics(user_id, start_time, end_time)
            
            # Aggregate by usage type
            usage_by_type = defaultdict(int)
            total_cost_cents = 0
            
            for metric in usage_metrics:
                usage_by_type[metric.usage_type] += metric.amount
                if metric.cost_cents:
                    total_cost_cents += metric.cost_cents
            
            # Check quota status for each usage type
            user_tier = await self.quota_enforcer._get_user_tier(user_id)
            quota_status = {}
            warnings = []
            
            for usage_type, amount in usage_by_type.items():
                allowed, status, details = await self.quota_enforcer.check_quota(
                    user_id, usage_type, 0, user_tier  # Check current status
                )
                quota_status[usage_type] = status
                
                if status == QuotaStatus.APPROACHING_LIMIT:
                    warnings.append(f"{usage_type.value}: {details.get('message', 'Approaching limit')}")
                elif status == QuotaStatus.LIMIT_EXCEEDED:
                    warnings.append(f"{usage_type.value}: {details.get('message', 'Limit exceeded')}")
            
            return UsageSummary(
                user_id=user_id,
                period_start=start_time,
                period_end=end_time,
                usage_by_type=dict(usage_by_type),
                total_cost_cents=total_cost_cents,
                quota_status=quota_status,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error getting usage summary for {user_id}: {e}")
            return UsageSummary(
                user_id=user_id,
                period_start=start_time,
                period_end=end_time,
                usage_by_type={},
                total_cost_cents=0,
                quota_status={},
                warnings=[f"Error retrieving usage data: {str(e)}"]
            )
    
    async def get_real_time_usage(self, user_id: str, hours: int = 1) -> Dict[str, Any]:
        """
        Get real-time usage data for the last N hours.
        
        Args:
            user_id: User ID
            hours: Number of hours to look back
            
        Returns:
            Real-time usage data
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Get recent usage metrics
            usage_metrics = await self._get_usage_metrics(user_id, start_time, end_time)
            
            # Group by time buckets (e.g., 5-minute intervals)
            bucket_size_minutes = max(1, (hours * 60) // 20)  # 20 buckets max
            time_buckets = defaultdict(lambda: defaultdict(int))
            
            for metric in usage_metrics:
                bucket_time = metric.timestamp.replace(
                    minute=(metric.timestamp.minute // bucket_size_minutes) * bucket_size_minutes,
                    second=0,
                    microsecond=0
                )
                time_buckets[bucket_time][metric.usage_type] += metric.amount
            
            # Convert to list format
            timeline = []
            for bucket_time in sorted(time_buckets.keys()):
                timeline.append({
                    "timestamp": bucket_time.isoformat(),
                    "usage": dict(time_buckets[bucket_time])
                })
            
            # Get current totals
            current_totals = defaultdict(int)
            for metric in usage_metrics:
                current_totals[metric.usage_type] += metric.amount
            
            return {
                "user_id": user_id,
                "period_hours": hours,
                "timeline": timeline,
                "current_totals": dict(current_totals),
                "bucket_size_minutes": bucket_size_minutes,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time usage for {user_id}: {e}")
            return {
                "user_id": user_id,
                "period_hours": hours,
                "timeline": [],
                "current_totals": {},
                "error": str(e)
            }
    
    async def get_usage_analytics(
        self, 
        user_id: Optional[str] = None, 
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get usage analytics for a user or all users.
        
        Args:
            user_id: User ID (None for all users)
            days: Number of days to analyze
            
        Returns:
            Usage analytics
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            if user_id:
                # Single user analytics
                usage_metrics = await self._get_usage_metrics(user_id, start_time, end_time)
                
                # Daily breakdown
                daily_usage = defaultdict(lambda: defaultdict(int))
                for metric in usage_metrics:
                    day = metric.timestamp.date()
                    daily_usage[day][metric.usage_type] += metric.amount
                
                # Convert to list
                daily_breakdown = []
                for day in sorted(daily_usage.keys()):
                    daily_breakdown.append({
                        "date": day.isoformat(),
                        "usage": dict(daily_usage[day])
                    })
                
                return {
                    "user_id": user_id,
                    "period_days": days,
                    "daily_breakdown": daily_breakdown,
                    "total_metrics": len(usage_metrics)
                }
            else:
                # All users analytics (would require admin permissions)
                return {
                    "period_days": days,
                    "message": "All users analytics not implemented yet"
                }
                
        except Exception as e:
            logger.error(f"Error getting usage analytics: {e}")
            return {"error": str(e)}
    
    def add_usage_callback(self, callback: Callable[[UsageMetric], None]):
        """Add callback for usage events."""
        self.usage_callbacks.append(callback)
    
    def remove_usage_callback(self, callback: Callable[[UsageMetric], None]):
        """Remove usage callback."""
        if callback in self.usage_callbacks:
            self.usage_callbacks.remove(callback)
    
    async def start_background_tasks(self):
        """Start background tasks for data aggregation and cleanup."""
        # Aggregation task
        aggregation_task = asyncio.create_task(self._aggregation_loop())
        self.background_tasks.append(aggregation_task)
        
        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.background_tasks.append(cleanup_task)
        
        logger.info("Usage tracker background tasks started")
    
    async def stop_background_tasks(self):
        """Stop background tasks."""
        self.shutdown_event.set()
        
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        logger.info("Usage tracker background tasks stopped")
    
    async def _store_usage_metric(self, metric: UsageMetric):
        """Store usage metric in Redis and/or Firebase."""
        try:
            # Store in Redis for real-time access
            if self.redis_client:
                key = f"usage:{metric.user_id}:{metric.timestamp.strftime('%Y%m%d')}"
                value = json.dumps(asdict(metric), default=str)
                await self.redis_client.lpush(key, value)
                await self.redis_client.expire(key, 86400 * 7)  # 7 days TTL
            else:
                # Fallback to memory cache
                cache_key = f"{metric.user_id}:{metric.timestamp.strftime('%Y%m%d')}"
                self.memory_cache[cache_key].append(metric)
            
            # Store in Firebase for persistence
            if BILLING_AVAILABLE and self.firebase_integration:
                await self.firebase_integration.record_usage_metrics(
                    user_id=metric.user_id,
                    usage_type=metric.usage_type.value,
                    amount=metric.amount,
                    metadata=metric.metadata
                )
                
        except Exception as e:
            logger.error(f"Error storing usage metric: {e}")
    
    async def _get_usage_metrics(
        self, 
        user_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[UsageMetric]:
        """Get usage metrics from storage."""
        metrics = []
        
        try:
            # Get from Redis first
            if self.redis_client:
                # Generate keys for the date range
                current_date = start_time.date()
                end_date = end_time.date()
                
                while current_date <= end_date:
                    key = f"usage:{user_id}:{current_date.strftime('%Y%m%d')}"
                    values = await self.redis_client.lrange(key, 0, -1)
                    
                    for value in values:
                        try:
                            data = json.loads(value)
                            metric_time = datetime.fromisoformat(data['timestamp'])
                            
                            if start_time <= metric_time <= end_time:
                                # Convert back to UsageMetric
                                data['usage_type'] = UsageType(data['usage_type'])
                                data['timestamp'] = metric_time
                                metrics.append(UsageMetric(**data))
                        except Exception as e:
                            logger.debug(f"Error parsing usage metric: {e}")
                    
                    current_date += timedelta(days=1)
            else:
                # Fallback to memory cache
                for cache_key, cached_metrics in self.memory_cache.items():
                    if user_id in cache_key:
                        for metric in cached_metrics:
                            if start_time <= metric.timestamp <= end_time:
                                metrics.append(metric)
            
            # Sort by timestamp
            metrics.sort(key=lambda m: m.timestamp)
            
        except Exception as e:
            logger.error(f"Error getting usage metrics: {e}")
        
        return metrics
    
    async def _trigger_usage_callbacks(self, metric: UsageMetric):
        """Trigger usage callbacks."""
        for callback in self.usage_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metric)
                else:
                    callback(metric)
            except Exception as e:
                logger.error(f"Error in usage callback: {e}")
    
    def _calculate_cost(self, usage_type: UsageType, amount: int) -> Optional[int]:
        """Calculate cost in cents for usage."""
        # Simple cost calculation - would be more sophisticated in production
        cost_per_unit = {
            UsageType.RESEARCH_REQUEST: 10,  # 10 cents per request
            UsageType.API_CALL: 1,  # 1 cent per API call
            UsageType.TOKEN_USAGE: 0,  # Calculated separately
            UsageType.MCP_CALL: 2,  # 2 cents per MCP call
            UsageType.RAG_QUERY: 5,  # 5 cents per RAG query
            UsageType.SEARCH_QUERY: 3,  # 3 cents per search
            UsageType.DATA_STORAGE: 0,  # Calculated by MB
            UsageType.EXPORT_REQUEST: 25,  # 25 cents per export
        }
        
        unit_cost = cost_per_unit.get(usage_type, 0)
        return unit_cost * amount if unit_cost > 0 else None
    
    async def _aggregation_loop(self):
        """Background loop for data aggregation."""
        while not self.shutdown_event.is_set():
            try:
                # Aggregate daily usage data
                # This would typically move data from Redis to long-term storage
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Error in aggregation loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _cleanup_loop(self):
        """Background loop for data cleanup."""
        while not self.shutdown_event.is_set():
            try:
                # Clean up old data from Redis
                if self.redis_client:
                    # Remove data older than 7 days
                    cutoff_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
                    # Implementation would scan and delete old keys
                
                await asyncio.sleep(86400)  # Run daily
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error
    
    def get_tracker_stats(self) -> Dict[str, Any]:
        """Get usage tracker statistics."""
        return {
            "redis_available": bool(self.redis_client),
            "billing_available": BILLING_AVAILABLE,
            "firebase_available": bool(self.firebase_integration),
            "stripe_available": bool(self.stripe_service),
            "usage_callbacks": len(self.usage_callbacks),
            "background_tasks": len(self.background_tasks),
            "memory_cache_keys": len(self.memory_cache),
            "supported_usage_types": [ut.value for ut in UsageType],
            "quota_enforcer_available": bool(self.quota_enforcer)
        }
