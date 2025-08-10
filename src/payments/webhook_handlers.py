"""
Stripe Webhook Handlers for payment events.

This module handles Stripe webhook events for subscription management,
payment processing, and user profile updates.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import stripe
from enum import Enum

from .stripe_service import StripeService, SubscriptionTier, PaymentStatus

logger = logging.getLogger(__name__)

class WebhookEventType(Enum):
    """Supported Stripe webhook event types."""
    CUSTOMER_SUBSCRIPTION_CREATED = "customer.subscription.created"
    CUSTOMER_SUBSCRIPTION_UPDATED = "customer.subscription.updated"
    CUSTOMER_SUBSCRIPTION_DELETED = "customer.subscription.deleted"
    CUSTOMER_SUBSCRIPTION_TRIAL_WILL_END = "customer.subscription.trial_will_end"
    INVOICE_PAYMENT_SUCCEEDED = "invoice.payment_succeeded"
    INVOICE_PAYMENT_FAILED = "invoice.payment_failed"
    CUSTOMER_CREATED = "customer.created"
    CUSTOMER_UPDATED = "customer.updated"
    CUSTOMER_DELETED = "customer.deleted"
    CHECKOUT_SESSION_COMPLETED = "checkout.session.completed"

class StripeWebhookHandler:
    """Handler for Stripe webhook events."""
    
    def __init__(self, stripe_service: StripeService, firebase_integration: Optional[Any] = None):
        """
        Initialize webhook handler.
        
        Args:
            stripe_service: StripeService instance
            firebase_integration: Firebase integration for user profile updates
        """
        self.stripe_service = stripe_service
        self.firebase_integration = firebase_integration
        
        # Event handlers mapping
        self.event_handlers = {
            WebhookEventType.CUSTOMER_SUBSCRIPTION_CREATED: self._handle_subscription_created,
            WebhookEventType.CUSTOMER_SUBSCRIPTION_UPDATED: self._handle_subscription_updated,
            WebhookEventType.CUSTOMER_SUBSCRIPTION_DELETED: self._handle_subscription_deleted,
            WebhookEventType.CUSTOMER_SUBSCRIPTION_TRIAL_WILL_END: self._handle_trial_will_end,
            WebhookEventType.INVOICE_PAYMENT_SUCCEEDED: self._handle_payment_succeeded,
            WebhookEventType.INVOICE_PAYMENT_FAILED: self._handle_payment_failed,
            WebhookEventType.CUSTOMER_CREATED: self._handle_customer_created,
            WebhookEventType.CUSTOMER_UPDATED: self._handle_customer_updated,
            WebhookEventType.CUSTOMER_DELETED: self._handle_customer_deleted,
            WebhookEventType.CHECKOUT_SESSION_COMPLETED: self._handle_checkout_completed,
        }
        
        logger.info("Stripe webhook handler initialized")
    
    async def handle_webhook(self, payload: bytes, sig_header: str) -> Dict[str, Any]:
        """
        Handle incoming Stripe webhook.
        
        Args:
            payload: Raw webhook payload
            sig_header: Stripe signature header
            
        Returns:
            Processing result
        """
        try:
            # Verify and construct the event
            event = self.stripe_service.construct_webhook_event(payload, sig_header)
            
            logger.info(f"Received Stripe webhook: {event['type']} (ID: {event['id']})")
            
            # Get the event type
            event_type = None
            for webhook_type in WebhookEventType:
                if webhook_type.value == event['type']:
                    event_type = webhook_type
                    break
            
            if not event_type:
                logger.warning(f"Unhandled webhook event type: {event['type']}")
                return {
                    "status": "ignored",
                    "event_type": event['type'],
                    "event_id": event['id'],
                    "message": "Event type not handled"
                }
            
            # Get the handler for this event type
            handler = self.event_handlers.get(event_type)
            if not handler:
                logger.warning(f"No handler configured for event type: {event_type.value}")
                return {
                    "status": "ignored",
                    "event_type": event['type'],
                    "event_id": event['id'],
                    "message": "No handler configured"
                }
            
            # Process the event
            result = await handler(event)
            
            logger.info(f"Successfully processed webhook {event['id']}: {result}")
            
            return {
                "status": "processed",
                "event_type": event['type'],
                "event_id": event['id'],
                "result": result
            }
            
        except stripe.error.SignatureVerificationError as e:
            logger.error(f"Webhook signature verification failed: {e}")
            return {
                "status": "error",
                "error": "signature_verification_failed",
                "message": str(e)
            }
        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
            return {
                "status": "error",
                "error": "processing_failed",
                "message": str(e)
            }
    
    async def _handle_subscription_created(self, event: stripe.Event) -> Dict[str, Any]:
        """Handle subscription created event."""
        subscription = event['data']['object']
        
        try:
            # Extract subscription details
            customer_id = subscription['customer']
            subscription_id = subscription['id']
            status = subscription['status']
            tier_name = subscription['metadata'].get('tier', 'unknown')
            
            # Get customer information
            customer = stripe.Customer.retrieve(customer_id)
            user_id = customer.metadata.get('user_id')
            
            if not user_id:
                logger.warning(f"No user_id found in customer {customer_id} metadata")
                return {"action": "skipped", "reason": "no_user_id"}
            
            # Convert tier name to enum
            tier = SubscriptionTier(tier_name) if tier_name in [t.value for t in SubscriptionTier] else SubscriptionTier.FREE
            
            # Update user profile in Firebase
            if self.firebase_integration:
                await self._update_user_subscription_profile(
                    user_id=user_id,
                    customer_id=customer_id,
                    subscription_id=subscription_id,
                    tier=tier,
                    status=PaymentStatus(status),
                    current_period_start=datetime.fromtimestamp(subscription['current_period_start']),
                    current_period_end=datetime.fromtimestamp(subscription['current_period_end']),
                    trial_end=datetime.fromtimestamp(subscription['trial_end']) if subscription.get('trial_end') else None
                )
            
            # Record usage tracking reset
            await self.stripe_service.record_usage(
                user_id=user_id,
                usage_type="subscription_created",
                metadata={
                    "subscription_id": subscription_id,
                    "tier": tier.value,
                    "status": status
                }
            )
            
            logger.info(f"Processed subscription created for user {user_id}: {tier.value}")
            
            return {
                "action": "subscription_created",
                "user_id": user_id,
                "subscription_id": subscription_id,
                "tier": tier.value,
                "status": status
            }
            
        except Exception as e:
            logger.error(f"Error handling subscription created: {e}")
            raise
    
    async def _handle_subscription_updated(self, event: stripe.Event) -> Dict[str, Any]:
        """Handle subscription updated event."""
        subscription = event['data']['object']
        previous_attributes = event['data'].get('previous_attributes', {})
        
        try:
            customer_id = subscription['customer']
            subscription_id = subscription['id']
            status = subscription['status']
            tier_name = subscription['metadata'].get('tier', 'unknown')
            
            # Get customer information
            customer = stripe.Customer.retrieve(customer_id)
            user_id = customer.metadata.get('user_id')
            
            if not user_id:
                logger.warning(f"No user_id found in customer {customer_id} metadata")
                return {"action": "skipped", "reason": "no_user_id"}
            
            tier = SubscriptionTier(tier_name) if tier_name in [t.value for t in SubscriptionTier] else SubscriptionTier.FREE
            
            # Check what changed
            changes = []
            if 'status' in previous_attributes:
                changes.append(f"status: {previous_attributes['status']} -> {status}")
            if 'metadata' in previous_attributes and previous_attributes['metadata'].get('tier') != tier_name:
                changes.append(f"tier: {previous_attributes['metadata'].get('tier')} -> {tier_name}")
            
            # Update user profile in Firebase
            if self.firebase_integration:
                await self._update_user_subscription_profile(
                    user_id=user_id,
                    customer_id=customer_id,
                    subscription_id=subscription_id,
                    tier=tier,
                    status=PaymentStatus(status),
                    current_period_start=datetime.fromtimestamp(subscription['current_period_start']),
                    current_period_end=datetime.fromtimestamp(subscription['current_period_end']),
                    trial_end=datetime.fromtimestamp(subscription['trial_end']) if subscription.get('trial_end') else None,
                    cancel_at_period_end=subscription.get('cancel_at_period_end', False)
                )
            
            logger.info(f"Processed subscription updated for user {user_id}: {', '.join(changes)}")
            
            return {
                "action": "subscription_updated",
                "user_id": user_id,
                "subscription_id": subscription_id,
                "tier": tier.value,
                "status": status,
                "changes": changes
            }
            
        except Exception as e:
            logger.error(f"Error handling subscription updated: {e}")
            raise
    
    async def _handle_subscription_deleted(self, event: stripe.Event) -> Dict[str, Any]:
        """Handle subscription deleted event."""
        subscription = event['data']['object']
        
        try:
            customer_id = subscription['customer']
            subscription_id = subscription['id']
            
            # Get customer information
            customer = stripe.Customer.retrieve(customer_id)
            user_id = customer.metadata.get('user_id')
            
            if not user_id:
                logger.warning(f"No user_id found in customer {customer_id} metadata")
                return {"action": "skipped", "reason": "no_user_id"}
            
            # Downgrade to free tier
            if self.firebase_integration:
                await self._update_user_subscription_profile(
                    user_id=user_id,
                    customer_id=customer_id,
                    subscription_id=None,
                    tier=SubscriptionTier.FREE,
                    status=PaymentStatus.CANCELED,
                    current_period_start=datetime.now(),
                    current_period_end=datetime.now(),
                    cancel_at_period_end=False
                )
            
            logger.info(f"Processed subscription deleted for user {user_id}")
            
            return {
                "action": "subscription_deleted",
                "user_id": user_id,
                "subscription_id": subscription_id,
                "new_tier": SubscriptionTier.FREE.value
            }
            
        except Exception as e:
            logger.error(f"Error handling subscription deleted: {e}")
            raise
    
    async def _handle_trial_will_end(self, event: stripe.Event) -> Dict[str, Any]:
        """Handle trial will end event."""
        subscription = event['data']['object']
        
        try:
            customer_id = subscription['customer']
            subscription_id = subscription['id']
            trial_end = datetime.fromtimestamp(subscription['trial_end'])
            
            # Get customer information
            customer = stripe.Customer.retrieve(customer_id)
            user_id = customer.metadata.get('user_id')
            
            if not user_id:
                logger.warning(f"No user_id found in customer {customer_id} metadata")
                return {"action": "skipped", "reason": "no_user_id"}
            
            # Send trial ending notification (would integrate with email service)
            logger.info(f"Trial ending for user {user_id} on {trial_end}")
            
            # Record event for analytics
            await self.stripe_service.record_usage(
                user_id=user_id,
                usage_type="trial_ending",
                metadata={
                    "subscription_id": subscription_id,
                    "trial_end": trial_end.isoformat()
                }
            )
            
            return {
                "action": "trial_will_end",
                "user_id": user_id,
                "subscription_id": subscription_id,
                "trial_end": trial_end.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error handling trial will end: {e}")
            raise
    
    async def _handle_payment_succeeded(self, event: stripe.Event) -> Dict[str, Any]:
        """Handle successful payment event."""
        invoice = event['data']['object']
        
        try:
            customer_id = invoice['customer']
            subscription_id = invoice.get('subscription')
            amount_paid = invoice['amount_paid']
            
            # Get customer information
            customer = stripe.Customer.retrieve(customer_id)
            user_id = customer.metadata.get('user_id')
            
            if not user_id:
                logger.warning(f"No user_id found in customer {customer_id} metadata")
                return {"action": "skipped", "reason": "no_user_id"}
            
            # Record successful payment
            await self.stripe_service.record_usage(
                user_id=user_id,
                usage_type="payment_succeeded",
                amount=amount_paid,
                metadata={
                    "invoice_id": invoice['id'],
                    "subscription_id": subscription_id,
                    "amount_paid": amount_paid
                }
            )
            
            logger.info(f"Payment succeeded for user {user_id}: ${amount_paid/100:.2f}")
            
            return {
                "action": "payment_succeeded",
                "user_id": user_id,
                "subscription_id": subscription_id,
                "amount_paid": amount_paid
            }
            
        except Exception as e:
            logger.error(f"Error handling payment succeeded: {e}")
            raise
    
    async def _handle_payment_failed(self, event: stripe.Event) -> Dict[str, Any]:
        """Handle failed payment event."""
        invoice = event['data']['object']
        
        try:
            customer_id = invoice['customer']
            subscription_id = invoice.get('subscription')
            amount_due = invoice['amount_due']
            
            # Get customer information
            customer = stripe.Customer.retrieve(customer_id)
            user_id = customer.metadata.get('user_id')
            
            if not user_id:
                logger.warning(f"No user_id found in customer {customer_id} metadata")
                return {"action": "skipped", "reason": "no_user_id"}
            
            # Update subscription status to past_due if applicable
            if subscription_id and self.firebase_integration:
                subscription = stripe.Subscription.retrieve(subscription_id)
                tier_name = subscription['metadata'].get('tier', 'unknown')
                tier = SubscriptionTier(tier_name) if tier_name in [t.value for t in SubscriptionTier] else SubscriptionTier.FREE
                
                await self._update_user_subscription_profile(
                    user_id=user_id,
                    customer_id=customer_id,
                    subscription_id=subscription_id,
                    tier=tier,
                    status=PaymentStatus.PAST_DUE,
                    current_period_start=datetime.fromtimestamp(subscription['current_period_start']),
                    current_period_end=datetime.fromtimestamp(subscription['current_period_end'])
                )
            
            # Record failed payment
            await self.stripe_service.record_usage(
                user_id=user_id,
                usage_type="payment_failed",
                amount=amount_due,
                metadata={
                    "invoice_id": invoice['id'],
                    "subscription_id": subscription_id,
                    "amount_due": amount_due
                }
            )
            
            logger.warning(f"Payment failed for user {user_id}: ${amount_due/100:.2f}")
            
            return {
                "action": "payment_failed",
                "user_id": user_id,
                "subscription_id": subscription_id,
                "amount_due": amount_due
            }
            
        except Exception as e:
            logger.error(f"Error handling payment failed: {e}")
            raise
    
    async def _handle_customer_created(self, event: stripe.Event) -> Dict[str, Any]:
        """Handle customer created event."""
        customer = event['data']['object']
        
        try:
            customer_id = customer['id']
            user_id = customer['metadata'].get('user_id')
            email = customer['email']
            
            if not user_id:
                logger.warning(f"No user_id found in customer {customer_id} metadata")
                return {"action": "skipped", "reason": "no_user_id"}
            
            logger.info(f"Customer created for user {user_id}: {customer_id}")
            
            return {
                "action": "customer_created",
                "user_id": user_id,
                "customer_id": customer_id,
                "email": email
            }
            
        except Exception as e:
            logger.error(f"Error handling customer created: {e}")
            raise
    
    async def _handle_customer_updated(self, event: stripe.Event) -> Dict[str, Any]:
        """Handle customer updated event."""
        customer = event['data']['object']
        
        try:
            customer_id = customer['id']
            user_id = customer['metadata'].get('user_id')
            
            if not user_id:
                logger.warning(f"No user_id found in customer {customer_id} metadata")
                return {"action": "skipped", "reason": "no_user_id"}
            
            logger.info(f"Customer updated for user {user_id}: {customer_id}")
            
            return {
                "action": "customer_updated",
                "user_id": user_id,
                "customer_id": customer_id
            }
            
        except Exception as e:
            logger.error(f"Error handling customer updated: {e}")
            raise
    
    async def _handle_customer_deleted(self, event: stripe.Event) -> Dict[str, Any]:
        """Handle customer deleted event."""
        customer = event['data']['object']
        
        try:
            customer_id = customer['id']
            user_id = customer['metadata'].get('user_id')
            
            if not user_id:
                logger.warning(f"No user_id found in customer {customer_id} metadata")
                return {"action": "skipped", "reason": "no_user_id"}
            
            # Clean up user subscription data
            if self.firebase_integration:
                await self._update_user_subscription_profile(
                    user_id=user_id,
                    customer_id=None,
                    subscription_id=None,
                    tier=SubscriptionTier.FREE,
                    status=PaymentStatus.CANCELED,
                    current_period_start=datetime.now(),
                    current_period_end=datetime.now()
                )
            
            logger.info(f"Customer deleted for user {user_id}: {customer_id}")
            
            return {
                "action": "customer_deleted",
                "user_id": user_id,
                "customer_id": customer_id
            }
            
        except Exception as e:
            logger.error(f"Error handling customer deleted: {e}")
            raise
    
    async def _handle_checkout_completed(self, event: stripe.Event) -> Dict[str, Any]:
        """Handle checkout session completed event."""
        session = event['data']['object']
        
        try:
            customer_id = session['customer']
            subscription_id = session.get('subscription')
            
            # Get customer information
            customer = stripe.Customer.retrieve(customer_id)
            user_id = customer.metadata.get('user_id')
            
            if not user_id:
                logger.warning(f"No user_id found in customer {customer_id} metadata")
                return {"action": "skipped", "reason": "no_user_id"}
            
            logger.info(f"Checkout completed for user {user_id}: subscription {subscription_id}")
            
            return {
                "action": "checkout_completed",
                "user_id": user_id,
                "customer_id": customer_id,
                "subscription_id": subscription_id
            }
            
        except Exception as e:
            logger.error(f"Error handling checkout completed: {e}")
            raise
    
    async def _update_user_subscription_profile(
        self,
        user_id: str,
        customer_id: Optional[str],
        subscription_id: Optional[str],
        tier: SubscriptionTier,
        status: PaymentStatus,
        current_period_start: datetime,
        current_period_end: datetime,
        trial_end: Optional[datetime] = None,
        cancel_at_period_end: bool = False
    ) -> bool:
        """
        Update user subscription profile in Firebase.
        
        Args:
            user_id: User ID
            customer_id: Stripe customer ID
            subscription_id: Stripe subscription ID
            tier: Subscription tier
            status: Payment status
            current_period_start: Current period start
            current_period_end: Current period end
            trial_end: Trial end date
            cancel_at_period_end: Whether subscription cancels at period end
            
        Returns:
            Success status
        """
        try:
            if not self.firebase_integration:
                logger.warning("Firebase integration not available for user profile update")
                return False
            
            # This would integrate with Firebase Firestore
            # For now, we'll log the update
            subscription_data = {
                "user_id": user_id,
                "customer_id": customer_id,
                "subscription_id": subscription_id,
                "tier": tier.value,
                "status": status.value,
                "current_period_start": current_period_start.isoformat(),
                "current_period_end": current_period_end.isoformat(),
                "trial_end": trial_end.isoformat() if trial_end else None,
                "cancel_at_period_end": cancel_at_period_end,
                "updated_at": datetime.now().isoformat()
            }
            
            logger.info(f"Would update Firebase profile for user {user_id}: {subscription_data}")
            
            # In a real implementation, this would:
            # await self.firebase_integration.update_user_subscription(user_id, subscription_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating user subscription profile: {e}")
            return False
    
    def get_supported_events(self) -> List[str]:
        """Get list of supported webhook event types."""
        return [event_type.value for event_type in WebhookEventType]
    
    def get_handler_stats(self) -> Dict[str, Any]:
        """Get webhook handler statistics."""
        return {
            "supported_events": len(self.event_handlers),
            "event_types": self.get_supported_events(),
            "firebase_integration": bool(self.firebase_integration),
            "handlers_configured": {
                event_type.value: bool(handler) 
                for event_type, handler in self.event_handlers.items()
            }
        }

