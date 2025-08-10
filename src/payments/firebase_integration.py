"""
Firebase integration for Stripe payments and subscription management.

This module provides integration between Stripe payments and Firebase user profiles,
enabling subscription status tracking and user data synchronization.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import asdict

try:
    from firebase_admin import firestore
    from google.cloud.firestore_v1.base_query import FieldFilter
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logging.warning("Firebase not available - install firebase-admin")

from .stripe_service import SubscriptionTier, PaymentStatus, UserSubscription, UsageMetrics

logger = logging.getLogger(__name__)

class FirebasePaymentIntegration:
    """Firebase integration for payment and subscription management."""
    
    def __init__(self, db_client: Optional[Any] = None):
        """
        Initialize Firebase payment integration.
        
        Args:
            db_client: Firestore database client (will create if None)
        """
        if not FIREBASE_AVAILABLE:
            raise ImportError("Firebase Admin SDK not available")
        
        self.db = db_client or firestore.client()
        
        # Collection names
        self.users_collection = "users"
        self.subscriptions_collection = "subscriptions"
        self.usage_metrics_collection = "usage_metrics"
        self.payment_events_collection = "payment_events"
        
        logger.info("Firebase payment integration initialized")
    
    async def create_or_update_user_subscription(
        self,
        user_id: str,
        subscription_data: Dict[str, Any]
    ) -> bool:
        """
        Create or update user subscription in Firebase.
        
        Args:
            user_id: User ID
            subscription_data: Subscription information
            
        Returns:
            Success status
        """
        try:
            # Update user document with subscription info
            user_ref = self.db.collection(self.users_collection).document(user_id)
            
            user_update_data = {
                "subscription": {
                    "tier": subscription_data.get("tier", "free"),
                    "status": subscription_data.get("status", "inactive"),
                    "customer_id": subscription_data.get("customer_id"),
                    "subscription_id": subscription_data.get("subscription_id"),
                    "current_period_start": subscription_data.get("current_period_start"),
                    "current_period_end": subscription_data.get("current_period_end"),
                    "trial_end": subscription_data.get("trial_end"),
                    "cancel_at_period_end": subscription_data.get("cancel_at_period_end", False),
                    "updated_at": firestore.SERVER_TIMESTAMP
                }
            }
            
            # Use merge to avoid overwriting other user data
            user_ref.set(user_update_data, merge=True)
            
            # Create detailed subscription record
            subscription_ref = self.db.collection(self.subscriptions_collection).document(user_id)
            
            detailed_subscription_data = {
                **subscription_data,
                "created_at": firestore.SERVER_TIMESTAMP,
                "updated_at": firestore.SERVER_TIMESTAMP
            }
            
            subscription_ref.set(detailed_subscription_data, merge=True)
            
            logger.info(f"Updated subscription for user {user_id}: {subscription_data.get('tier', 'unknown')} tier")
            return True
            
        except Exception as e:
            logger.error(f"Error updating user subscription in Firebase: {e}")
            return False
    
    async def get_user_subscription(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user subscription information from Firebase.
        
        Args:
            user_id: User ID
            
        Returns:
            Subscription information or None
        """
        try:
            subscription_ref = self.db.collection(self.subscriptions_collection).document(user_id)
            subscription_doc = subscription_ref.get()
            
            if subscription_doc.exists:
                subscription_data = subscription_doc.to_dict()
                
                # Convert timestamps to ISO strings for JSON serialization
                for field in ["current_period_start", "current_period_end", "trial_end", "created_at", "updated_at"]:
                    if field in subscription_data and subscription_data[field]:
                        if hasattr(subscription_data[field], 'isoformat'):
                            subscription_data[field] = subscription_data[field].isoformat()
                        elif hasattr(subscription_data[field], 'timestamp'):
                            subscription_data[field] = datetime.fromtimestamp(subscription_data[field].timestamp()).isoformat()
                
                return subscription_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting user subscription from Firebase: {e}")
            return None
    
    async def record_usage_metrics(
        self,
        user_id: str,
        usage_type: str,
        amount: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record usage metrics in Firebase.
        
        Args:
            user_id: User ID
            usage_type: Type of usage
            amount: Usage amount
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            # Create usage record
            usage_data = {
                "user_id": user_id,
                "usage_type": usage_type,
                "amount": amount,
                "metadata": metadata or {},
                "timestamp": firestore.SERVER_TIMESTAMP,
                "date": datetime.now().strftime("%Y-%m-%d")
            }
            
            # Add to usage metrics collection
            usage_ref = self.db.collection(self.usage_metrics_collection)
            usage_ref.add(usage_data)
            
            # Update daily aggregates
            await self._update_daily_usage_aggregates(user_id, usage_type, amount)
            
            logger.debug(f"Recorded usage for user {user_id}: {usage_type} = {amount}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording usage metrics in Firebase: {e}")
            return False
    
    async def _update_daily_usage_aggregates(
        self,
        user_id: str,
        usage_type: str,
        amount: int
    ) -> bool:
        """Update daily usage aggregates for efficient querying."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            aggregate_doc_id = f"{user_id}_{today}"
            
            aggregate_ref = self.db.collection("daily_usage_aggregates").document(aggregate_doc_id)
            
            # Use transaction to safely increment counters
            @firestore.transactional
            def update_aggregate(transaction, doc_ref):
                doc = doc_ref.get(transaction=transaction)
                
                if doc.exists:
                    current_data = doc.to_dict()
                    current_data[usage_type] = current_data.get(usage_type, 0) + amount
                    current_data["total_events"] = current_data.get("total_events", 0) + 1
                    current_data["last_updated"] = firestore.SERVER_TIMESTAMP
                else:
                    current_data = {
                        "user_id": user_id,
                        "date": today,
                        usage_type: amount,
                        "total_events": 1,
                        "created_at": firestore.SERVER_TIMESTAMP,
                        "last_updated": firestore.SERVER_TIMESTAMP
                    }
                
                transaction.set(doc_ref, current_data)
            
            transaction = self.db.transaction()
            update_aggregate(transaction, aggregate_ref)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating daily usage aggregates: {e}")
            return False
    
    async def get_user_usage_metrics(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get user usage metrics from Firebase.
        
        Args:
            user_id: User ID
            start_date: Start date for metrics
            end_date: End date for metrics
            
        Returns:
            Usage metrics summary
        """
        try:
            # Default to current month if no dates provided
            if not start_date:
                start_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if not end_date:
                end_date = datetime.now()
            
            # Query daily aggregates for the period
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            query = (
                self.db.collection("daily_usage_aggregates")
                .where(filter=FieldFilter("user_id", "==", user_id))
                .where(filter=FieldFilter("date", ">=", start_date_str))
                .where(filter=FieldFilter("date", "<=", end_date_str))
            )
            
            docs = query.stream()
            
            # Aggregate the results
            total_metrics = {
                "research_requests": 0,
                "api_calls": 0,
                "tokens_used": 0,
                "mcp_calls": 0,
                "total_events": 0
            }
            
            daily_breakdown = []
            
            for doc in docs:
                data = doc.to_dict()
                daily_breakdown.append(data)
                
                # Sum up totals
                for metric in total_metrics:
                    total_metrics[metric] += data.get(metric, 0)
            
            return {
                "user_id": user_id,
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
                "total_metrics": total_metrics,
                "daily_breakdown": daily_breakdown,
                "days_with_activity": len(daily_breakdown)
            }
            
        except Exception as e:
            logger.error(f"Error getting user usage metrics from Firebase: {e}")
            return {
                "user_id": user_id,
                "error": str(e),
                "total_metrics": {},
                "daily_breakdown": []
            }
    
    async def record_payment_event(
        self,
        user_id: str,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> bool:
        """
        Record payment event in Firebase.
        
        Args:
            user_id: User ID
            event_type: Type of payment event
            event_data: Event data
            
        Returns:
            Success status
        """
        try:
            payment_event = {
                "user_id": user_id,
                "event_type": event_type,
                "event_data": event_data,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "processed": True
            }
            
            self.db.collection(self.payment_events_collection).add(payment_event)
            
            logger.info(f"Recorded payment event for user {user_id}: {event_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording payment event in Firebase: {e}")
            return False
    
    async def get_users_by_subscription_tier(self, tier: SubscriptionTier) -> List[str]:
        """
        Get list of user IDs with a specific subscription tier.
        
        Args:
            tier: Subscription tier
            
        Returns:
            List of user IDs
        """
        try:
            query = (
                self.db.collection(self.users_collection)
                .where(filter=FieldFilter("subscription.tier", "==", tier.value))
                .where(filter=FieldFilter("subscription.status", "==", "active"))
            )
            
            docs = query.stream()
            user_ids = [doc.id for doc in docs]
            
            logger.info(f"Found {len(user_ids)} users with {tier.value} tier")
            return user_ids
            
        except Exception as e:
            logger.error(f"Error getting users by subscription tier: {e}")
            return []
    
    async def get_subscription_analytics(self) -> Dict[str, Any]:
        """
        Get subscription analytics and metrics.
        
        Returns:
            Analytics data
        """
        try:
            # Get subscription counts by tier
            tier_counts = {}
            for tier in SubscriptionTier:
                users = await self.get_users_by_subscription_tier(tier)
                tier_counts[tier.value] = len(users)
            
            # Get recent payment events
            recent_events_query = (
                self.db.collection(self.payment_events_collection)
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(10)
            )
            
            recent_events = []
            for doc in recent_events_query.stream():
                event_data = doc.to_dict()
                if "timestamp" in event_data and hasattr(event_data["timestamp"], "timestamp"):
                    event_data["timestamp"] = datetime.fromtimestamp(event_data["timestamp"].timestamp()).isoformat()
                recent_events.append(event_data)
            
            return {
                "subscription_tiers": tier_counts,
                "total_subscribers": sum(tier_counts.values()),
                "recent_events": recent_events,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting subscription analytics: {e}")
            return {
                "error": str(e),
                "subscription_tiers": {},
                "total_subscribers": 0,
                "recent_events": []
            }
    
    async def cleanup_expired_data(self, days_to_keep: int = 90) -> Dict[str, int]:
        """
        Clean up expired usage metrics and payment events.
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Cleanup statistics
        """
        try:
            cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            cutoff_date = cutoff_date.replace(day=cutoff_date.day - days_to_keep)
            
            # Clean up old usage metrics
            old_usage_query = (
                self.db.collection(self.usage_metrics_collection)
                .where(filter=FieldFilter("timestamp", "<", cutoff_date))
            )
            
            usage_docs_deleted = 0
            for doc in old_usage_query.stream():
                doc.reference.delete()
                usage_docs_deleted += 1
            
            # Clean up old payment events
            old_events_query = (
                self.db.collection(self.payment_events_collection)
                .where(filter=FieldFilter("timestamp", "<", cutoff_date))
            )
            
            events_docs_deleted = 0
            for doc in old_events_query.stream():
                doc.reference.delete()
                events_docs_deleted += 1
            
            logger.info(f"Cleaned up {usage_docs_deleted} usage records and {events_docs_deleted} payment events")
            
            return {
                "usage_metrics_deleted": usage_docs_deleted,
                "payment_events_deleted": events_docs_deleted,
                "cutoff_date": cutoff_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error cleaning up expired data: {e}")
            return {
                "error": str(e),
                "usage_metrics_deleted": 0,
                "payment_events_deleted": 0
            }
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get Firebase integration statistics."""
        return {
            "firebase_available": FIREBASE_AVAILABLE,
            "collections": {
                "users": self.users_collection,
                "subscriptions": self.subscriptions_collection,
                "usage_metrics": self.usage_metrics_collection,
                "payment_events": self.payment_events_collection
            },
            "database_connected": bool(self.db),
            "supported_operations": [
                "create_or_update_user_subscription",
                "get_user_subscription",
                "record_usage_metrics",
                "get_user_usage_metrics",
                "record_payment_event",
                "get_users_by_subscription_tier",
                "get_subscription_analytics",
                "cleanup_expired_data"
            ]
        }
