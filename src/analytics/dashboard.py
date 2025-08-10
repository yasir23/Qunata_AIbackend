"""
Real-time Usage Dashboard for analytics and monitoring.

This module provides real-time usage dashboards and metrics visualization
for the Open Deep Research platform.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from .usage_tracker import UsageTracker, UsageType, QuotaStatus

logger = logging.getLogger(__name__)

class DashboardTimeRange(Enum):
    """Dashboard time range options."""
    LAST_HOUR = "1h"
    LAST_6_HOURS = "6h"
    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"

class MetricType(Enum):
    """Types of metrics to display."""
    USAGE_COUNT = "usage_count"
    USAGE_RATE = "usage_rate"
    QUOTA_UTILIZATION = "quota_utilization"
    COST_ANALYSIS = "cost_analysis"
    ERROR_RATE = "error_rate"

@dataclass
class DashboardMetrics:
    """Dashboard metrics data structure."""
    user_id: str
    time_range: DashboardTimeRange
    timestamp: datetime
    
    # Usage metrics
    total_requests: int = 0
    api_calls: int = 0
    tokens_used: int = 0
    mcp_calls: int = 0
    rag_queries: int = 0
    search_queries: int = 0
    
    # Quota metrics
    quota_utilization: Dict[str, float] = None
    quota_warnings: List[str] = None
    
    # Cost metrics
    estimated_cost_cents: int = 0
    cost_breakdown: Dict[str, int] = None
    
    # Performance metrics
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    
    # Timeline data
    timeline: List[Dict[str, Any]] = None

class UsageDashboard:
    """Real-time usage dashboard with analytics and monitoring."""
    
    def __init__(self, usage_tracker: UsageTracker):
        """
        Initialize usage dashboard.
        
        Args:
            usage_tracker: UsageTracker instance
        """
        self.usage_tracker = usage_tracker
        self.cache_ttl_seconds = 60  # Cache dashboard data for 1 minute
        self.dashboard_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Usage dashboard initialized")
    
    async def get_user_dashboard(
        self, 
        user_id: str, 
        time_range: DashboardTimeRange = DashboardTimeRange.LAST_24_HOURS,
        refresh_cache: bool = False
    ) -> DashboardMetrics:
        """
        Get comprehensive dashboard metrics for a user.
        
        Args:
            user_id: User ID
            time_range: Time range for metrics
            refresh_cache: Whether to refresh cached data
            
        Returns:
            Dashboard metrics
        """
        try:
            # Check cache first
            cache_key = f"{user_id}:{time_range.value}"
            if not refresh_cache and cache_key in self.dashboard_cache:
                cached_data = self.dashboard_cache[cache_key]
                if (datetime.now() - cached_data["timestamp"]).total_seconds() < self.cache_ttl_seconds:
                    return DashboardMetrics(**cached_data["metrics"])
            
            # Calculate time range
            end_time = datetime.now()
            hours = self._get_hours_from_range(time_range)
            start_time = end_time - timedelta(hours=hours)
            
            # Get usage summary
            usage_summary = await self.usage_tracker.get_usage_summary(user_id, start_time, end_time)
            
            # Get real-time usage data
            real_time_data = await self.usage_tracker.get_real_time_usage(user_id, hours)
            
            # Calculate metrics
            metrics = await self._calculate_dashboard_metrics(
                user_id, time_range, usage_summary, real_time_data
            )
            
            # Cache the results
            self.dashboard_cache[cache_key] = {
                "timestamp": datetime.now(),
                "metrics": asdict(metrics)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting user dashboard for {user_id}: {e}")
            return DashboardMetrics(
                user_id=user_id,
                time_range=time_range,
                timestamp=datetime.now(),
                quota_warnings=[f"Error loading dashboard: {str(e)}"]
            )
    
    async def get_usage_trends(
        self, 
        user_id: str, 
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get usage trends over time.
        
        Args:
            user_id: User ID
            days: Number of days to analyze
            
        Returns:
            Usage trends data
        """
        try:
            # Get usage analytics
            analytics = await self.usage_tracker.get_usage_analytics(user_id, days)
            
            if "daily_breakdown" not in analytics:
                return {"error": "No trend data available"}
            
            # Calculate trends
            daily_data = analytics["daily_breakdown"]
            
            # Extract trend data for each usage type
            trends = {}
            for usage_type in UsageType:
                daily_values = []
                dates = []
                
                for day_data in daily_data:
                    dates.append(day_data["date"])
                    usage = day_data["usage"].get(usage_type.value, 0)
                    daily_values.append(usage)
                
                # Calculate trend (simple linear regression slope)
                trend_slope = self._calculate_trend_slope(daily_values)
                
                trends[usage_type.value] = {
                    "daily_values": daily_values,
                    "dates": dates,
                    "trend_slope": trend_slope,
                    "trend_direction": "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable",
                    "total": sum(daily_values),
                    "average": sum(daily_values) / len(daily_values) if daily_values else 0
                }
            
            return {
                "user_id": user_id,
                "period_days": days,
                "trends": trends,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting usage trends for {user_id}: {e}")
            return {"error": str(e)}
    
    async def get_quota_dashboard(self, user_id: str) -> Dict[str, Any]:
        """
        Get quota utilization dashboard.
        
        Args:
            user_id: User ID
            
        Returns:
            Quota dashboard data
        """
        try:
            # Get current usage for the month
            end_time = datetime.now()
            start_time = end_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            usage_summary = await self.usage_tracker.get_usage_summary(user_id, start_time, end_time)
            
            # Get user's subscription tier
            user_tier = await self.usage_tracker.quota_enforcer._get_user_tier(user_id)
            
            # Get quota limits for this tier
            tier_limits = self.usage_tracker.quota_enforcer.tier_limits.get(user_tier, [])
            
            # Calculate quota utilization
            quota_data = {}
            warnings = []
            critical_alerts = []
            
            for limit in tier_limits:
                usage_type = limit.usage_type
                current_usage = usage_summary.usage_by_type.get(usage_type, 0)
                
                if limit.limit == -1:
                    # Unlimited
                    utilization = 0.0
                    status = "unlimited"
                else:
                    utilization = (current_usage / limit.limit) * 100 if limit.limit > 0 else 0
                    
                    if utilization >= limit.critical_threshold * 100:
                        status = "critical"
                        critical_alerts.append(f"{usage_type.value}: {utilization:.1f}% used")
                    elif utilization >= limit.warning_threshold * 100:
                        status = "warning"
                        warnings.append(f"{usage_type.value}: {utilization:.1f}% used")
                    else:
                        status = "normal"
                
                quota_data[usage_type.value] = {
                    "current_usage": current_usage,
                    "limit": limit.limit,
                    "utilization_percent": utilization,
                    "status": status,
                    "warning_threshold": limit.warning_threshold * 100,
                    "critical_threshold": limit.critical_threshold * 100,
                    "period_hours": limit.period_hours
                }
            
            return {
                "user_id": user_id,
                "subscription_tier": user_tier.value,
                "period_start": start_time.isoformat(),
                "period_end": end_time.isoformat(),
                "quota_data": quota_data,
                "warnings": warnings,
                "critical_alerts": critical_alerts,
                "overall_status": "critical" if critical_alerts else "warning" if warnings else "normal",
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting quota dashboard for {user_id}: {e}")
            return {"error": str(e)}
    
    async def get_cost_dashboard(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get cost analysis dashboard.
        
        Args:
            user_id: User ID
            days: Number of days to analyze
            
        Returns:
            Cost dashboard data
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            usage_summary = await self.usage_tracker.get_usage_summary(user_id, start_time, end_time)
            
            # Calculate costs by usage type
            cost_breakdown = {}
            total_cost_cents = usage_summary.total_cost_cents
            
            # Get detailed cost breakdown (would be more sophisticated in production)
            cost_per_unit = {
                UsageType.RESEARCH_REQUEST: 10,  # 10 cents per request
                UsageType.API_CALL: 1,  # 1 cent per API call
                UsageType.MCP_CALL: 2,  # 2 cents per MCP call
                UsageType.RAG_QUERY: 5,  # 5 cents per RAG query
                UsageType.SEARCH_QUERY: 3,  # 3 cents per search
                UsageType.EXPORT_REQUEST: 25,  # 25 cents per export
            }
            
            for usage_type, amount in usage_summary.usage_by_type.items():
                unit_cost = cost_per_unit.get(usage_type, 0)
                type_cost = unit_cost * amount
                
                cost_breakdown[usage_type.value] = {
                    "usage_amount": amount,
                    "cost_per_unit_cents": unit_cost,
                    "total_cost_cents": type_cost,
                    "percentage_of_total": (type_cost / total_cost_cents * 100) if total_cost_cents > 0 else 0
                }
            
            # Calculate daily cost trend
            analytics = await self.usage_tracker.get_usage_analytics(user_id, days)
            daily_costs = []
            
            if "daily_breakdown" in analytics:
                for day_data in analytics["daily_breakdown"]:
                    day_cost = 0
                    for usage_type_str, amount in day_data["usage"].items():
                        try:
                            usage_type = UsageType(usage_type_str)
                            unit_cost = cost_per_unit.get(usage_type, 0)
                            day_cost += unit_cost * amount
                        except ValueError:
                            continue
                    
                    daily_costs.append({
                        "date": day_data["date"],
                        "cost_cents": day_cost
                    })
            
            # Calculate projections
            avg_daily_cost = sum(day["cost_cents"] for day in daily_costs) / len(daily_costs) if daily_costs else 0
            projected_monthly_cost = avg_daily_cost * 30
            
            return {
                "user_id": user_id,
                "period_days": days,
                "total_cost_cents": total_cost_cents,
                "total_cost_dollars": total_cost_cents / 100,
                "cost_breakdown": cost_breakdown,
                "daily_costs": daily_costs,
                "avg_daily_cost_cents": avg_daily_cost,
                "projected_monthly_cost_cents": projected_monthly_cost,
                "projected_monthly_cost_dollars": projected_monthly_cost / 100,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting cost dashboard for {user_id}: {e}")
            return {"error": str(e)}
    
    async def get_performance_dashboard(self, user_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance metrics dashboard.
        
        Args:
            user_id: User ID
            hours: Number of hours to analyze
            
        Returns:
            Performance dashboard data
        """
        try:
            # Get real-time usage data
            real_time_data = await self.usage_tracker.get_real_time_usage(user_id, hours)
            
            # Calculate performance metrics from timeline
            timeline = real_time_data.get("timeline", [])
            
            if not timeline:
                return {
                    "user_id": user_id,
                    "period_hours": hours,
                    "message": "No performance data available"
                }
            
            # Calculate request rate (requests per minute)
            total_requests = sum(
                sum(bucket["usage"].values()) for bucket in timeline
            )
            request_rate = total_requests / (hours * 60) if hours > 0 else 0
            
            # Calculate peak usage
            peak_usage = max(
                sum(bucket["usage"].values()) for bucket in timeline
            ) if timeline else 0
            
            # Calculate usage distribution
            usage_distribution = {}
            for bucket in timeline:
                for usage_type, amount in bucket["usage"].items():
                    usage_distribution[usage_type] = usage_distribution.get(usage_type, 0) + amount
            
            return {
                "user_id": user_id,
                "period_hours": hours,
                "total_requests": total_requests,
                "request_rate_per_minute": request_rate,
                "peak_usage_per_bucket": peak_usage,
                "usage_distribution": usage_distribution,
                "timeline_buckets": len(timeline),
                "bucket_size_minutes": real_time_data.get("bucket_size_minutes", 0),
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting performance dashboard for {user_id}: {e}")
            return {"error": str(e)}
    
    async def get_admin_dashboard(self) -> Dict[str, Any]:
        """
        Get admin dashboard with system-wide metrics.
        
        Returns:
            Admin dashboard data
        """
        try:
            # Get usage tracker stats
            tracker_stats = self.usage_tracker.get_tracker_stats()
            
            # Get cache stats
            cache_stats = {
                "cached_dashboards": len(self.dashboard_cache),
                "cache_ttl_seconds": self.cache_ttl_seconds
            }
            
            # System health metrics
            system_health = {
                "redis_available": tracker_stats.get("redis_available", False),
                "billing_available": tracker_stats.get("billing_available", False),
                "firebase_available": tracker_stats.get("firebase_available", False),
                "background_tasks": tracker_stats.get("background_tasks", 0)
            }
            
            return {
                "system_health": system_health,
                "tracker_stats": tracker_stats,
                "cache_stats": cache_stats,
                "supported_time_ranges": [tr.value for tr in DashboardTimeRange],
                "supported_metrics": [mt.value for mt in MetricType],
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting admin dashboard: {e}")
            return {"error": str(e)}
    
    def clear_cache(self, user_id: Optional[str] = None):
        """
        Clear dashboard cache.
        
        Args:
            user_id: User ID to clear cache for (None for all)
        """
        if user_id:
            # Clear cache for specific user
            keys_to_remove = [key for key in self.dashboard_cache.keys() if key.startswith(f"{user_id}:")]
            for key in keys_to_remove:
                del self.dashboard_cache[key]
            logger.info(f"Cleared dashboard cache for user {user_id}")
        else:
            # Clear all cache
            self.dashboard_cache.clear()
            logger.info("Cleared all dashboard cache")
    
    async def _calculate_dashboard_metrics(
        self, 
        user_id: str, 
        time_range: DashboardTimeRange,
        usage_summary: Any,
        real_time_data: Dict[str, Any]
    ) -> DashboardMetrics:
        """Calculate comprehensive dashboard metrics."""
        
        # Extract usage counts
        usage_by_type = usage_summary.usage_by_type
        
        # Calculate quota utilization
        quota_utilization = {}
        for usage_type, status in usage_summary.quota_status.items():
            if status == QuotaStatus.UNLIMITED:
                quota_utilization[usage_type.value] = 0.0
            else:
                # Get current usage percentage (would need more detailed calculation)
                quota_utilization[usage_type.value] = 50.0  # Placeholder
        
        # Calculate cost breakdown
        cost_breakdown = {}
        total_cost = usage_summary.total_cost_cents
        
        # Create timeline from real-time data
        timeline = real_time_data.get("timeline", [])
        
        return DashboardMetrics(
            user_id=user_id,
            time_range=time_range,
            timestamp=datetime.now(),
            total_requests=usage_by_type.get(UsageType.RESEARCH_REQUEST, 0),
            api_calls=usage_by_type.get(UsageType.API_CALL, 0),
            tokens_used=usage_by_type.get(UsageType.TOKEN_USAGE, 0),
            mcp_calls=usage_by_type.get(UsageType.MCP_CALL, 0),
            rag_queries=usage_by_type.get(UsageType.RAG_QUERY, 0),
            search_queries=usage_by_type.get(UsageType.SEARCH_QUERY, 0),
            quota_utilization=quota_utilization,
            quota_warnings=usage_summary.warnings,
            estimated_cost_cents=total_cost,
            cost_breakdown=cost_breakdown,
            timeline=timeline
        )
    
    def _get_hours_from_range(self, time_range: DashboardTimeRange) -> int:
        """Convert time range to hours."""
        range_hours = {
            DashboardTimeRange.LAST_HOUR: 1,
            DashboardTimeRange.LAST_6_HOURS: 6,
            DashboardTimeRange.LAST_24_HOURS: 24,
            DashboardTimeRange.LAST_7_DAYS: 24 * 7,
            DashboardTimeRange.LAST_30_DAYS: 24 * 30
        }
        return range_hours.get(time_range, 24)
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate simple trend slope."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        if n * x2_sum - x_sum * x_sum == 0:
            return 0.0
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        return {
            "cache_size": len(self.dashboard_cache),
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "supported_time_ranges": [tr.value for tr in DashboardTimeRange],
            "supported_metrics": [mt.value for mt in MetricType],
            "usage_tracker_available": bool(self.usage_tracker)
        }
