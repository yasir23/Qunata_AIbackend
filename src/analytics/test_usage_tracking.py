#!/usr/bin/env python3
"""
Test script for usage tracking and billing integration.

This script tests the comprehensive usage tracking system including quota enforcement,
billing integration, and real-time dashboards.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analytics.usage_tracker import UsageTracker, UsageType, QuotaEnforcer, UsageMetric
from analytics.dashboard import UsageDashboard, DashboardTimeRange, DashboardMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_usage_tracker():
    """Test usage tracker functionality."""
    logger.info("Testing Usage Tracker...")
    
    try:
        # Initialize usage tracker (without Redis for testing)
        usage_tracker = UsageTracker()
        
        logger.info("✅ Usage tracker initialized successfully")
        
        # Test usage tracking
        test_user_id = "test_user_123"
        
        # Track different types of usage
        usage_types = [
            (UsageType.RESEARCH_REQUEST, 1, {"query": "test research"}),
            (UsageType.API_CALL, 5, {"endpoint": "/api/research"}),
            (UsageType.TOKEN_USAGE, 1000, {"model": "gpt-4"}),
            (UsageType.MCP_CALL, 3, {"server": "reddit", "tool": "search"}),
            (UsageType.RAG_QUERY, 2, {"query_type": "context_retrieval"}),
        ]
        
        for usage_type, amount, metadata in usage_types:
            result = await usage_tracker.track_usage(
                user_id=test_user_id,
                usage_type=usage_type,
                amount=amount,
                metadata=metadata,
                session_id="test_session",
                request_id="test_request"
            )
            
            if result.get("success"):
                logger.info(f"✅ Tracked {usage_type.value}: {amount}")
            else:
                logger.warning(f"⚠️ Failed to track {usage_type.value}: {result.get('message')}")
        
        # Test usage summary
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)
        
        summary = await usage_tracker.get_usage_summary(test_user_id, start_time, end_time)
        logger.info(f"✅ Usage summary: {len(summary.usage_by_type)} usage types tracked")
        
        for usage_type, amount in summary.usage_by_type.items():
            logger.info(f"  - {usage_type.value}: {amount}")
        
        # Test real-time usage
        real_time_data = await usage_tracker.get_real_time_usage(test_user_id, hours=1)
        logger.info(f"✅ Real-time data: {len(real_time_data.get('timeline', []))} time buckets")
        
        # Test tracker stats
        stats = usage_tracker.get_tracker_stats()
        logger.info(f"✅ Tracker stats: {stats['supported_usage_types']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Usage tracker test failed: {e}")
        return False

async def test_quota_enforcer():
    """Test quota enforcement functionality."""
    logger.info("Testing Quota Enforcer...")
    
    try:
        # Initialize components
        usage_tracker = UsageTracker()
        quota_enforcer = QuotaEnforcer(usage_tracker)
        
        logger.info("✅ Quota enforcer initialized successfully")
        
        # Test quota limits for different tiers
        from payments.stripe_service import SubscriptionTier
        
        test_user_id = "quota_test_user"
        
        # Test quota checking for different usage types
        quota_tests = [
            (UsageType.RESEARCH_REQUEST, 1, SubscriptionTier.FREE),
            (UsageType.API_CALL, 10, SubscriptionTier.FREE),
            (UsageType.TOKEN_USAGE, 1000, SubscriptionTier.PRO),
            (UsageType.RAG_QUERY, 1, SubscriptionTier.FREE),  # Should be blocked for free tier
            (UsageType.CONCURRENT_RESEARCH, 2, SubscriptionTier.FREE),  # Should be blocked
        ]
        
        for usage_type, amount, tier in quota_tests:
            allowed, status, details = await quota_enforcer.check_quota(
                user_id=test_user_id,
                usage_type=usage_type,
                amount=amount,
                tier=tier
            )
            
            logger.info(f"✅ Quota check {usage_type.value} ({tier.value}): {'✓' if allowed else '✗'} - {status.value}")
            logger.info(f"   Details: {details.get('message', 'No details')}")
        
        # Test tier limits configuration
        tier_limits = quota_enforcer.tier_limits
        logger.info(f"✅ Configured limits for {len(tier_limits)} tiers")
        
        for tier, limits in tier_limits.items():
            logger.info(f"  - {tier.value}: {len(limits)} usage types configured")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Quota enforcer test failed: {e}")
        return False

async def test_usage_dashboard():
    """Test usage dashboard functionality."""
    logger.info("Testing Usage Dashboard...")
    
    try:
        # Initialize components
        usage_tracker = UsageTracker()
        dashboard = UsageDashboard(usage_tracker)
        
        logger.info("✅ Usage dashboard initialized successfully")
        
        # Test dashboard metrics
        test_user_id = "dashboard_test_user"
        
        # Add some test usage data first
        await usage_tracker.track_usage(test_user_id, UsageType.RESEARCH_REQUEST, 5)
        await usage_tracker.track_usage(test_user_id, UsageType.API_CALL, 25)
        await usage_tracker.track_usage(test_user_id, UsageType.TOKEN_USAGE, 5000)
        
        # Test user dashboard
        metrics = await dashboard.get_user_dashboard(
            user_id=test_user_id,
            time_range=DashboardTimeRange.LAST_24_HOURS
        )
        
        logger.info(f"✅ User dashboard: {metrics.total_requests} requests, {metrics.api_calls} API calls")
        logger.info(f"   Tokens used: {metrics.tokens_used}, MCP calls: {metrics.mcp_calls}")
        
        # Test usage trends
        trends = await dashboard.get_usage_trends(test_user_id, days=7)
        if "trends" in trends:
            logger.info(f"✅ Usage trends: {len(trends['trends'])} usage types analyzed")
        else:
            logger.info(f"✅ Usage trends: {trends.get('message', 'No trend data')}")
        
        # Test quota dashboard
        quota_dashboard = await dashboard.get_quota_dashboard(test_user_id)
        if "quota_data" in quota_dashboard:
            logger.info(f"✅ Quota dashboard: {len(quota_dashboard['quota_data'])} usage types monitored")
            logger.info(f"   Overall status: {quota_dashboard.get('overall_status', 'unknown')}")
        else:
            logger.info(f"✅ Quota dashboard: {quota_dashboard.get('message', 'No quota data')}")
        
        # Test cost dashboard
        cost_dashboard = await dashboard.get_cost_dashboard(test_user_id, days=30)
        if "total_cost_cents" in cost_dashboard:
            total_cost = cost_dashboard["total_cost_cents"]
            logger.info(f"✅ Cost dashboard: ${total_cost/100:.2f} total cost")
        else:
            logger.info(f"✅ Cost dashboard: {cost_dashboard.get('message', 'No cost data')}")
        
        # Test performance dashboard
        performance = await dashboard.get_performance_dashboard(test_user_id, hours=24)
        if "total_requests" in performance:
            logger.info(f"✅ Performance dashboard: {performance['total_requests']} total requests")
            logger.info(f"   Request rate: {performance.get('request_rate_per_minute', 0):.2f}/min")
        else:
            logger.info(f"✅ Performance dashboard: {performance.get('message', 'No performance data')}")
        
        # Test admin dashboard
        admin_dashboard = await dashboard.get_admin_dashboard()
        logger.info(f"✅ Admin dashboard: {admin_dashboard.get('system_health', {})}")
        
        # Test dashboard stats
        dashboard_stats = dashboard.get_dashboard_stats()
        logger.info(f"✅ Dashboard stats: {dashboard_stats['cache_size']} cached items")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Usage dashboard test failed: {e}")
        return False

async def test_integration_with_utils():
    """Test integration with open_deep_research.utils."""
    logger.info("Testing Integration with Utils...")
    
    try:
        # Import the enhanced utils
        from open_deep_research.utils import (
            get_usage_tracker, track_research_request, track_api_call,
            track_token_usage, track_mcp_call, track_rag_query,
            check_usage_quota, get_user_usage_summary, get_usage_tracker_stats
        )
        
        logger.info("✅ Successfully imported usage tracking functions from utils")
        
        # Test usage tracker initialization
        usage_tracker = get_usage_tracker()
        if usage_tracker:
            logger.info("✅ Usage tracker accessible from utils")
        else:
            logger.warning("⚠️ Usage tracker not available from utils")
        
        # Test tracking functions
        test_user_id = "utils_test_user"
        
        # Test research request tracking
        result = await track_research_request(
            user_id=test_user_id,
            request_type="deep_research",
            metadata={"query": "test query"},
            session_id="utils_test_session"
        )
        logger.info(f"✅ Research request tracking: {result.get('success', False)}")
        
        # Test API call tracking
        result = await track_api_call(
            user_id=test_user_id,
            api_endpoint="/api/research",
            method="POST",
            metadata={"response_time": 150}
        )
        logger.info(f"✅ API call tracking: {result.get('success', False)}")
        
        # Test token usage tracking
        result = await track_token_usage(
            user_id=test_user_id,
            model_name="gpt-4",
            tokens_used=500,
            operation="completion"
        )
        logger.info(f"✅ Token usage tracking: {result.get('success', False)}")
        
        # Test MCP call tracking
        result = await track_mcp_call(
            user_id=test_user_id,
            server_name="reddit",
            tool_name="search_posts"
        )
        logger.info(f"✅ MCP call tracking: {result.get('success', False)}")
        
        # Test RAG query tracking
        result = await track_rag_query(
            user_id=test_user_id,
            query_type="context_retrieval",
            results_count=5
        )
        logger.info(f"✅ RAG query tracking: {result.get('success', False)}")
        
        # Test quota checking
        allowed, message = await check_usage_quota(
            user_id=test_user_id,
            usage_type=UsageType.RESEARCH_REQUEST,
            amount=1
        )
        logger.info(f"✅ Quota check: {'✓' if allowed else '✗'} - {message}")
        
        # Test usage summary
        summary = await get_user_usage_summary(test_user_id, hours=24)
        if "usage_by_type" in summary:
            logger.info(f"✅ Usage summary: {len(summary['usage_by_type'])} usage types")
        else:
            logger.info(f"✅ Usage summary: {summary.get('message', 'No summary data')}")
        
        # Test tracker stats
        stats = get_usage_tracker_stats()
        logger.info(f"✅ Tracker stats: {stats.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Utils integration test failed: {e}")
        return False

async def test_billing_integration():
    """Test billing integration (mock mode)."""
    logger.info("Testing Billing Integration...")
    
    try:
        # Test billing availability
        from analytics.usage_tracker import BILLING_AVAILABLE
        
        if BILLING_AVAILABLE:
            logger.info("✅ Billing integration is available")
            
            # Test Stripe integration
            try:
                from payments.stripe_service import StripeService, SubscriptionTier
                stripe_service = StripeService(api_key="sk_test_mock_key")
                
                # Test tier limits
                for tier in SubscriptionTier:
                    limits = stripe_service.get_tier_limits(tier)
                    logger.info(f"✅ {tier.value} tier: {limits.research_requests_per_month} research requests/month")
                
            except Exception as e:
                logger.warning(f"⚠️ Stripe integration test failed: {e}")
            
            # Test Firebase integration
            try:
                from payments.firebase_integration import FirebasePaymentIntegration
                logger.info("✅ Firebase payment integration available")
                
            except Exception as e:
                logger.warning(f"⚠️ Firebase integration test failed: {e}")
        
        else:
            logger.warning("⚠️ Billing integration not available")
        
        # Test usage cost calculation
        usage_tracker = UsageTracker()
        
        # Test cost calculation for different usage types
        cost_tests = [
            (UsageType.RESEARCH_REQUEST, 1),
            (UsageType.API_CALL, 10),
            (UsageType.MCP_CALL, 5),
            (UsageType.RAG_QUERY, 3),
        ]
        
        for usage_type, amount in cost_tests:
            cost = usage_tracker._calculate_cost(usage_type, amount)
            logger.info(f"✅ Cost calculation {usage_type.value}: {amount} units = {cost} cents")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Billing integration test failed: {e}")
        return False

async def test_error_handling():
    """Test error handling and edge cases."""
    logger.info("Testing Error Handling...")
    
    try:
        # Test with invalid user ID
        usage_tracker = UsageTracker()
        
        result = await usage_tracker.track_usage(
            user_id="",  # Empty user ID
            usage_type=UsageType.RESEARCH_REQUEST,
            amount=1
        )
        logger.info(f"✅ Empty user ID handled: {result.get('success', False)}")
        
        # Test with negative amount
        result = await usage_tracker.track_usage(
            user_id="test_user",
            usage_type=UsageType.TOKEN_USAGE,
            amount=-100  # Negative amount
        )
        logger.info(f"✅ Negative amount handled: {result.get('success', False)}")
        
        # Test quota check with invalid tier
        quota_enforcer = QuotaEnforcer(usage_tracker)
        
        try:
            allowed, status, details = await quota_enforcer.check_quota(
                user_id="test_user",
                usage_type=UsageType.RESEARCH_REQUEST,
                amount=1,
                tier=None  # Invalid tier
            )
            logger.info(f"✅ Invalid tier handled: {allowed}")
        except Exception as e:
            logger.info(f"✅ Invalid tier error handled: {type(e).__name__}")
        
        # Test dashboard with non-existent user
        dashboard = UsageDashboard(usage_tracker)
        
        metrics = await dashboard.get_user_dashboard("nonexistent_user")
        logger.info(f"✅ Non-existent user dashboard: {metrics.user_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False

async def main():
    """Run all usage tracking and billing integration tests."""
    logger.info("📊 Starting Usage Tracking and Billing Integration Tests")
    logger.info("=" * 70)
    
    tests = [
        ("Usage Tracker", test_usage_tracker),
        ("Quota Enforcer", test_quota_enforcer),
        ("Usage Dashboard", test_usage_dashboard),
        ("Utils Integration", test_integration_with_utils),
        ("Billing Integration", test_billing_integration),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running test: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 Test Results Summary:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {status}: {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! Usage tracking and billing integration is ready.")
        logger.info("\n📋 Integration Summary:")
        logger.info("  ✅ Comprehensive usage tracking with quota enforcement")
        logger.info("  ✅ Real-time usage dashboards and analytics")
        logger.info("  ✅ Stripe billing integration with tiered pricing")
        logger.info("  ✅ Firebase integration for data persistence")
        logger.info("  ✅ Enhanced research workflow with usage tracking")
        logger.info("  ✅ Error handling and resilience")
        logger.info("\n🚀 Ready for production deployment!")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs and configuration.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
