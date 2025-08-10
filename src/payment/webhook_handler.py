"""
Stripe Webhook Handler for Open Deep Research

This module provides FastAPI endpoints for handling Stripe webhooks
and processing payment events in real-time.
"""

import os
import logging
from typing import Dict, Any
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
import stripe
from supabase import Client, create_client
from .stripe_integration import handle_webhook

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.warning("Supabase credentials not found. Webhook handling may not work properly.")
    supabase_client = None
else:
    supabase_client = create_client(supabase_url, supabase_key)

# Create router for webhook endpoints
webhook_router = APIRouter(prefix="/webhooks", tags=["webhooks"])

def get_supabase_client() -> Client:
    """Dependency to get Supabase client."""
    if not supabase_client:
        raise HTTPException(
            status_code=500,
            detail="Supabase client not initialized. Check environment variables."
        )
    return supabase_client

@webhook_router.post("/stripe")
async def stripe_webhook_endpoint(
    request: Request,
    supabase: Client = Depends(get_supabase_client)
) -> JSONResponse:
    """
    Handle Stripe webhook events.
    
    This endpoint receives webhook events from Stripe and processes them
    to update subscription status, payment history, and user data.
    """
    try:
        # Get the raw payload and signature
        payload = await request.body()
        signature = request.headers.get("stripe-signature")
        
        if not signature:
            logger.error("Missing Stripe signature header")
            raise HTTPException(status_code=400, detail="Missing signature header")
        
        # Process the webhook
        result = handle_webhook(payload, signature, supabase)
        
        logger.info(f"Webhook processed successfully: {result}")
        
        return JSONResponse(
            status_code=200,
            content={"status": "success", "result": result}
        )
        
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Webhook signature verification failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
        
    except ValueError as e:
        logger.error(f"Webhook processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Unexpected webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@webhook_router.get("/stripe/test")
async def test_stripe_webhook() -> JSONResponse:
    """
    Test endpoint to verify webhook configuration.
    
    This endpoint can be used to test that the webhook infrastructure
    is properly set up and accessible.
    """
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    stripe_secret_key = os.getenv("STRIPE_SECRET_KEY")
    
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "webhook_secret_configured": bool(webhook_secret),
            "stripe_key_configured": bool(stripe_secret_key),
            "supabase_configured": bool(supabase_client),
            "message": "Webhook endpoint is ready to receive events"
        }
    )

# Health check endpoint for webhook monitoring
@webhook_router.get("/health")
async def webhook_health_check() -> JSONResponse:
    """Health check endpoint for webhook service monitoring."""
    try:
        # Test Supabase connection
        if supabase_client:
            # Simple query to test connection
            supabase_client.table("subscription_tiers").select("tier").limit(1).execute()
            supabase_status = "healthy"
        else:
            supabase_status = "not_configured"
        
        # Test Stripe configuration
        stripe_status = "healthy" if os.getenv("STRIPE_SECRET_KEY") else "not_configured"
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "services": {
                    "supabase": supabase_status,
                    "stripe": stripe_status
                },
                "timestamp": "2024-01-01T00:00:00Z"  # Will be replaced with actual timestamp
            }
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": "2024-01-01T00:00:00Z"  # Will be replaced with actual timestamp
            }
        )
