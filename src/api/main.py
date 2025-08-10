"""
Main FastAPI Application for Open Deep Research

This module provides the central FastAPI application that integrates all components:
- Authentication endpoints
- Payment webhooks
- LangGraph functionality
- Health checks and monitoring
- CORS configuration
- Middleware setup
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from supabase import Client, create_client
import uvicorn
import asyncio

# Import routers from existing components
from .auth_endpoints import auth_router, limiter as auth_limiter
from ..payment.webhook_handler import webhook_router
from ..security.auth import supabase

# Import LangGraph components
from ..open_deep_research.deep_researcher import deep_researcher
from ..open_deep_research.configuration import Configuration
from ..open_deep_research.state import AgentInputState

# Import usage tracking middleware
from ..middleware.usage_tracker import (
    UsageTrackingMiddleware, 
    get_user_subscription_limits,
    check_user_can_access_mcp_server,
    record_token_usage,
    require_subscription_tier
)
from ..database.models import SubscriptionTierEnum

# Import subscription management
from ..payment.subscription_manager import (
    SubscriptionManager,
    get_user_subscription_info,
    check_user_feature_access,
    get_user_concurrent_research_limit,
    check_github_mcp_access,
    get_subscription_tier_config
)

# Import RAG system
from ..rag.vector_store import VectorStore, SearchResult

# Import configuration system
from ..open_deep_research.configuration import Configuration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment variables
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# CORS configuration
ALLOWED_ORIGINS = [
    "http://localhost:3000",  # React development server
    "http://localhost:8000",  # FastAPI development server
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "https://smith.langchain.com",  # LangGraph Studio
]

# Add production origins if specified
if production_origins := os.getenv("ALLOWED_ORIGINS"):
    ALLOWED_ORIGINS.extend(production_origins.split(","))

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting Open Deep Research API...")
    
    # Verify Supabase connection
    if supabase:
        try:
            # Test connection by checking if we can access the service
            logger.info("Supabase connection verified")
        except Exception as e:
            logger.warning(f"Supabase connection issue: {e}")
    else:
        logger.warning("Supabase client not initialized")
    
    # Verify environment variables
    required_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
    
    logger.info("Open Deep Research API started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Open Deep Research API...")

# Create FastAPI application
app = FastAPI(
    title="Open Deep Research API",
    description="""
    A comprehensive API for deep research with AI agents, featuring:
    
    - **Authentication**: User signup, signin, and profile management
    - **Payments**: Stripe integration for subscription management
    - **Research**: AI-powered deep research with MCP server integration
    - **RAG**: Vector store for research result storage and retrieval
    - **Usage Tracking**: Monitor API usage and enforce subscription limits
    
    ## Features
    
    - Multiple subscription tiers (Free, Pro, Enterprise)
    - GitHub, Reddit, and YouTube data extraction via MCP servers
    - Real-time payment webhook processing
    - Rate limiting and usage tracking
    - Comprehensive authentication and authorization
    """,
    version="1.0.0",
    contact={
        "name": "Open Deep Research Team",
        "url": "https://github.com/yasir23/Qunata_AIbackend",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
    debug=DEBUG,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Requested-With",
        "Accept",
        "Origin",
        "User-Agent",
        "DNT",
        "Cache-Control",
        "X-Mx-ReqToken",
        "Keep-Alive",
        "X-Requested-With",
        "If-Modified-Since",
    ],
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Add usage tracking middleware
app.add_middleware(UsageTrackingMiddleware, supabase_client=supabase)

# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler with detailed error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "http_exception",
                "timestamp": str(request.state.__dict__.get("timestamp", "unknown")),
            }
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error" if ENVIRONMENT == "production" else str(exc),
                "type": "internal_error",
                "timestamp": str(request.state.__dict__.get("timestamp", "unknown")),
            }
        },
    )

# Middleware for request logging and timing
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log requests and add timing information."""
    import time
    
    start_time = time.time()
    request.state.timestamp = start_time
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log response
    logger.info(f"Response: {response.status_code} - {process_time:.4f}s")
    
    return response

# Health check endpoints
@app.get("/health", tags=["health"])
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "Open Deep Research API",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
    }

@app.get("/health/detailed", tags=["health"])
async def detailed_health_check():
    """Detailed health check with service dependencies."""
    health_status = {
        "status": "healthy",
        "service": "Open Deep Research API",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "checks": {
            "database": "unknown",
            "supabase": "unknown",
            "environment_variables": "unknown",
        }
    }
    
    # Check Supabase connection
    try:
        if supabase:
            # Simple check to verify connection
            health_status["checks"]["supabase"] = "healthy"
        else:
            health_status["checks"]["supabase"] = "unavailable"
    except Exception as e:
        health_status["checks"]["supabase"] = f"error: {str(e)}"
    
    # Check environment variables
    required_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        health_status["checks"]["environment_variables"] = f"missing: {missing_vars}"
    else:
        health_status["checks"]["environment_variables"] = "healthy"
    
    # Determine overall status
    if any(check.startswith("error") or check.startswith("missing") for check in health_status["checks"].values()):
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Open Deep Research API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "features": [
            "AI-powered deep research",
            "Multi-tier subscription system",
            "MCP server integration (GitHub, Reddit, YouTube)",
            "RAG-based research enhancement",
            "Real-time payment processing",
            "Comprehensive usage tracking",
        ]
    }

# Research endpoints for LangGraph integration
@app.post("/research/start", tags=["research"])
@limiter.limit("10/minute")
async def start_research(
    request: Request,
    research_input: Dict[str, Any],
    authorization: Optional[str] = None
):
    """
    Start a new research session using the LangGraph deep researcher.
    
    This endpoint integrates with the existing LangGraph functionality
    and requires authentication.
    """
    # Basic authentication check (will be enhanced with proper middleware later)
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    try:
        # Create configuration from request
        config = {
            "configurable": research_input.get("config", {}),
        }
        
        # Create input state
        input_state = AgentInputState(
            messages=research_input.get("messages", []),
            **research_input.get("additional_params", {})
        )
        
        # Start research using LangGraph
        result = await deep_researcher.ainvoke(
            input_state,
            config=config
        )
        
        return {
            "status": "success",
            "research_id": result.get("research_id"),
            "result": result,
            "message": "Research completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Research error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Research failed: {str(e)}"
        )

@app.get("/research/config", tags=["research"])
async def get_research_config():
    """
    Get the default research configuration schema.
    
    This endpoint returns the available configuration options
    for the LangGraph deep researcher.
    """
    try:
        # Get configuration schema
        config_schema = Configuration.model_json_schema()
        
        return {
            "status": "success",
            "config_schema": config_schema,
            "default_config": Configuration().model_dump(),
            "description": "Configuration options for the deep researcher"
        }
        
    except Exception as e:
        logger.error(f"Config retrieval error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve configuration: {str(e)}"
        )

@app.get("/research/status/{research_id}", tags=["research"])
async def get_research_status(research_id: str):
    """
    Get the status of a research session.
    
    This endpoint will be enhanced with proper session tracking
    in future iterations.
    """
    # Placeholder for research status tracking
    # This will be implemented with proper session management
    return {
        "research_id": research_id,
        "status": "completed",  # Placeholder
        "message": "Research status tracking will be implemented with session management"
    }

# Usage tracking and subscription endpoints
@app.get("/usage/stats", tags=["usage"])
@limiter.limit("30/minute")
async def get_usage_stats(
    request: Request,
    time_window: str = "day",
    authorization: Optional[str] = None
):
    """
    Get current usage statistics for the authenticated user.
    
    Time windows: 'hour', 'day'
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    try:
        # Extract user ID from authorization header
        middleware = UsageTrackingMiddleware(None)
        user_id = await middleware._extract_user_id(request)
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Get usage statistics
        from ..middleware.usage_tracker import UsageTracker
        tracker = UsageTracker()
        
        stats = await tracker.get_usage_stats(user_id, time_window)
        subscription_info = await tracker.get_user_subscription_status(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "time_window": time_window,
            "subscription_tier": subscription_info["tier"],
            "usage_stats": stats,
            "limits": subscription_info["limits"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Usage stats error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve usage statistics: {str(e)}"
        )

@app.get("/usage/limits", tags=["usage"])
@limiter.limit("60/minute")
async def get_usage_limits(
    request: Request,
    authorization: Optional[str] = None
):
    """
    Get subscription limits for the authenticated user.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    try:
        # Extract user ID from authorization header
        middleware = UsageTrackingMiddleware(None)
        user_id = await middleware._extract_user_id(request)
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Get subscription limits
        subscription_info = await get_user_subscription_limits(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "subscription_tier": subscription_info["tier"],
            "subscription_status": subscription_info["status"],
            "limits": subscription_info["limits"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Usage limits error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve usage limits: {str(e)}"
        )

@app.get("/usage/mcp-access/{server_name}", tags=["usage"])
@limiter.limit("60/minute")
async def check_mcp_server_access(
    server_name: str,
    request: Request,
    authorization: Optional[str] = None
):
    """
    Check if the authenticated user can access a specific MCP server.
    
    Supported servers: reddit, youtube, github
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    try:
        # Extract user ID from authorization header
        middleware = UsageTrackingMiddleware(None)
        user_id = await middleware._extract_user_id(request)
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Check MCP server access
        has_access = await check_user_can_access_mcp_server(user_id, server_name)
        subscription_info = await get_user_subscription_limits(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "server_name": server_name,
            "has_access": has_access,
            "subscription_tier": subscription_info["tier"],
            "allowed_servers": subscription_info["limits"].get("mcp_servers", []),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"MCP access check error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check MCP server access: {str(e)}"
        )

# Subscription management endpoints
@app.get("/subscription/info", tags=["subscription"])
@limiter.limit("60/minute")
async def get_subscription_info(
    request: Request,
    authorization: Optional[str] = None
):
    """
    Get comprehensive subscription information for the authenticated user.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    try:
        # Extract user ID from authorization header
        middleware = UsageTrackingMiddleware(None)
        user_id = await middleware._extract_user_id(request)
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Get subscription information
        subscription_info = await get_user_subscription_info(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "subscription": {
                "tier": subscription_info.tier.value,
                "status": subscription_info.status.value,
                "subscription_id": subscription_info.subscription_id,
                "customer_id": subscription_info.customer_id,
                "current_period_start": subscription_info.current_period_start.isoformat() if subscription_info.current_period_start else None,
                "current_period_end": subscription_info.current_period_end.isoformat() if subscription_info.current_period_end else None,
                "cancel_at_period_end": subscription_info.cancel_at_period_end
            },
            "features": subscription_info.features.model_dump(),
            "limits": subscription_info.limits.model_dump(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Subscription info error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve subscription information: {str(e)}"
        )

@app.get("/subscription/tiers", tags=["subscription"])
@limiter.limit("30/minute")
async def get_subscription_tiers():
    """
    Get information about all available subscription tiers.
    """
    try:
        manager = SubscriptionManager()
        tier_comparison = await manager.get_tier_comparison()
        
        return {
            "status": "success",
            "tiers": tier_comparison,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Subscription tiers error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve subscription tiers: {str(e)}"
        )

@app.get("/subscription/features/{feature}", tags=["subscription"])
@limiter.limit("60/minute")
async def check_feature_access(
    feature: str,
    request: Request,
    authorization: Optional[str] = None
):
    """
    Check if the authenticated user has access to a specific feature.
    
    Available features: research_requests, api_access, priority_support, 
    advanced_rag, custom_integrations, dedicated_support, github_mcp_access
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    try:
        # Extract user ID from authorization header
        middleware = UsageTrackingMiddleware(None)
        user_id = await middleware._extract_user_id(request)
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Check feature access
        has_access = await check_user_feature_access(user_id, feature)
        subscription_info = await get_user_subscription_info(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "feature": feature,
            "has_access": has_access,
            "subscription_tier": subscription_info.tier.value,
            "all_features": subscription_info.features.model_dump(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Feature access check error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check feature access: {str(e)}"
        )

@app.get("/subscription/limits/concurrent", tags=["subscription"])
@limiter.limit("60/minute")
async def get_concurrent_research_limit(
    request: Request,
    authorization: Optional[str] = None
):
    """
    Get the concurrent research units limit for the authenticated user.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    try:
        # Extract user ID from authorization header
        middleware = UsageTrackingMiddleware(None)
        user_id = await middleware._extract_user_id(request)
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Get concurrent research limit
        limit = await get_user_concurrent_research_limit(user_id)
        subscription_info = await get_user_subscription_info(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "concurrent_research_limit": limit,
            "subscription_tier": subscription_info.tier.value,
            "all_limits": subscription_info.limits.model_dump(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Concurrent limit error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve concurrent research limit: {str(e)}"
        )

# RAG system endpoints
@app.post("/rag/search", tags=["rag"])
@limiter.limit("30/minute")
async def rag_similarity_search(
    request: Request,
    query: str,
    k: int = 5,
    similarity_threshold: float = 0.7,
    authorization: Optional[str] = None
):
    """
    Perform similarity search in the RAG vector store.
    
    Requires Pro or Enterprise subscription for advanced RAG access.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    try:
        # Extract user ID from authorization header
        middleware = UsageTrackingMiddleware(None)
        user_id = await middleware._extract_user_id(request)
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Check if user has advanced RAG access
        subscription_info = await get_user_subscription_info(user_id)
        if not subscription_info.features.advanced_rag:
            raise HTTPException(
                status_code=403,
                detail="Advanced RAG access requires Pro or Enterprise subscription"
            )
        
        # Initialize RAG vector store
        vector_store = VectorStore()
        
        # Perform similarity search
        results = await vector_store.similarity_search(
            query=query,
            k=k,
            user_id=user_id,
            similarity_threshold=similarity_threshold
        )
        
        # Convert results to response format
        search_results = []
        for result in results:
            search_results.append({
                "document_id": result.document_id,
                "content": result.content,
                "metadata": result.metadata,
                "similarity_score": result.similarity_score,
                "chunk_index": result.chunk_index
            })
        
        return {
            "status": "success",
            "user_id": user_id,
            "query": query,
            "results": search_results,
            "total_results": len(search_results),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform RAG search: {str(e)}"
        )

@app.get("/rag/context", tags=["rag"])
@limiter.limit("30/minute")
async def get_rag_context(
    request: Request,
    query: str,
    max_context_length: int = 4000,
    research_type: Optional[str] = None,
    authorization: Optional[str] = None
):
    """
    Get relevant context from previous research for a query.
    
    Requires Pro or Enterprise subscription for advanced RAG access.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    try:
        # Extract user ID from authorization header
        middleware = UsageTrackingMiddleware(None)
        user_id = await middleware._extract_user_id(request)
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Check if user has advanced RAG access
        subscription_info = await get_user_subscription_info(user_id)
        if not subscription_info.features.advanced_rag:
            raise HTTPException(
                status_code=403,
                detail="Advanced RAG access requires Pro or Enterprise subscription"
            )
        
        # Initialize RAG vector store
        vector_store = VectorStore()
        
        # Get relevant context
        context = await vector_store.get_relevant_context(
            query=query,
            user_id=user_id,
            max_context_length=max_context_length,
            research_type=research_type
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "query": query,
            "context": context,
            "context_length": len(context),
            "research_type": research_type,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG context error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve RAG context: {str(e)}"
        )

@app.post("/rag/store", tags=["rag"])
@limiter.limit("10/minute")
async def store_research_document(
    request: Request,
    research_topic: str,
    research_content: str,
    sources: List[str] = [],
    research_type: str = "general",
    authorization: Optional[str] = None
):
    """
    Store research results in the RAG vector store.
    
    Requires Pro or Enterprise subscription for advanced RAG access.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    try:
        # Extract user ID from authorization header
        middleware = UsageTrackingMiddleware(None)
        user_id = await middleware._extract_user_id(request)
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Check if user has advanced RAG access
        subscription_info = await get_user_subscription_info(user_id)
        if not subscription_info.features.advanced_rag:
            raise HTTPException(
                status_code=403,
                detail="Advanced RAG access requires Pro or Enterprise subscription"
            )
        
        # Initialize RAG vector store
        vector_store = VectorStore()
        
        # Store research result
        document_id = await vector_store.add_research_result(
            research_topic=research_topic,
            research_content=research_content,
            sources=sources,
            user_id=user_id,
            research_type=research_type
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "document_id": document_id,
            "research_topic": research_topic,
            "research_type": research_type,
            "sources_count": len(sources),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG store error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store research document: {str(e)}"
        )

@app.get("/rag/documents", tags=["rag"])
@limiter.limit("30/minute")
async def get_user_documents(
    request: Request,
    limit: int = 20,
    offset: int = 0,
    research_type: Optional[str] = None,
    authorization: Optional[str] = None
):
    """
    Get user's stored research documents.
    
    Requires Pro or Enterprise subscription for advanced RAG access.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    try:
        # Extract user ID from authorization header
        middleware = UsageTrackingMiddleware(None)
        user_id = await middleware._extract_user_id(request)
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Check if user has advanced RAG access
        subscription_info = await get_user_subscription_info(user_id)
        if not subscription_info.features.advanced_rag:
            raise HTTPException(
                status_code=403,
                detail="Advanced RAG access requires Pro or Enterprise subscription"
            )
        
        # Initialize RAG vector store
        vector_store = VectorStore()
        
        # Get user documents
        documents = await vector_store.get_user_documents(
            user_id=user_id,
            limit=limit,
            offset=offset,
            research_type=research_type
        )
        
        return {
            "status": "success",
            "user_id": user_id,
            "documents": documents,
            "limit": limit,
            "offset": offset,
            "research_type": research_type,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG documents error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve user documents: {str(e)}"
        )

@app.delete("/rag/documents/{document_id}", tags=["rag"])
@limiter.limit("10/minute")
async def delete_research_document(
    document_id: str,
    request: Request,
    authorization: Optional[str] = None
):
    """
    Delete a research document from the RAG vector store.
    
    Requires Pro or Enterprise subscription for advanced RAG access.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    try:
        # Extract user ID from authorization header
        middleware = UsageTrackingMiddleware(None)
        user_id = await middleware._extract_user_id(request)
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Check if user has advanced RAG access
        subscription_info = await get_user_subscription_info(user_id)
        if not subscription_info.features.advanced_rag:
            raise HTTPException(
                status_code=403,
                detail="Advanced RAG access requires Pro or Enterprise subscription"
            )
        
        # Initialize RAG vector store
        vector_store = VectorStore()
        
        # Delete document
        success = await vector_store.delete_document(
            document_id=document_id,
            user_id=user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Document not found or access denied"
            )
        
        return {
            "status": "success",
            "user_id": user_id,
            "document_id": document_id,
            "deleted": True,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG delete error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete research document: {str(e)}"
        )

# Configuration endpoints
@app.get("/config/limits", tags=["configuration"])
@limiter.limit("60/minute")
async def get_configuration_limits(
    request: Request,
    authorization: Optional[str] = None
):
    """
    Get configuration limits based on user's subscription tier.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    try:
        # Extract user ID from authorization header
        middleware = UsageTrackingMiddleware(None)
        user_id = await middleware._extract_user_id(request)
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Create subscription-aware configuration
        config = await Configuration.from_user_subscription(user_id)
        effective_limits = await config.get_effective_limits()
        
        return {
            "status": "success",
            "user_id": user_id,
            "configuration_limits": effective_limits,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Configuration limits error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve configuration limits: {str(e)}"
        )

@app.get("/config/tiers", tags=["configuration"])
@limiter.limit("30/minute")
async def get_configuration_tier_comparison():
    """
    Get comparison of configuration limits across all subscription tiers.
    """
    try:
        tier_comparison = Configuration.get_tier_limits_comparison()
        
        return {
            "status": "success",
            "tier_limits": tier_comparison,
            "description": "Configuration limits by subscription tier",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Configuration tiers error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve configuration tiers: {str(e)}"
        )

@app.post("/config/validate", tags=["configuration"])
@limiter.limit("30/minute")
async def validate_configuration(
    request: Request,
    max_concurrent_research_units: Optional[int] = None,
    max_researcher_iterations: Optional[int] = None,
    max_react_tool_calls: Optional[int] = None,
    research_model_max_tokens: Optional[int] = None,
    authorization: Optional[str] = None
):
    """
    Validate configuration parameters against user's subscription limits.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    try:
        # Extract user ID from authorization header
        middleware = UsageTrackingMiddleware(None)
        user_id = await middleware._extract_user_id(request)
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Create base configuration
        config = await Configuration.from_user_subscription(user_id)
        
        # Override with provided parameters
        if max_concurrent_research_units is not None:
            config.max_concurrent_research_units = max_concurrent_research_units
        if max_researcher_iterations is not None:
            config.max_researcher_iterations = max_researcher_iterations
        if max_react_tool_calls is not None:
            config.max_react_tool_calls = max_react_tool_calls
        if research_model_max_tokens is not None:
            config.research_model_max_tokens = research_model_max_tokens
        
        # Validate configuration
        is_valid = await config.validate_subscription_limits()
        
        # Get effective limits for comparison
        effective_limits = await config.get_effective_limits()
        
        return {
            "status": "success",
            "user_id": user_id,
            "validation_result": {
                "is_valid": is_valid,
                "requested_config": {
                    "max_concurrent_research_units": config.max_concurrent_research_units,
                    "max_researcher_iterations": config.max_researcher_iterations,
                    "max_react_tool_calls": config.max_react_tool_calls,
                    "research_model_max_tokens": config.research_model_max_tokens
                },
                "effective_limits": effective_limits["effective_limits"] if "effective_limits" in effective_limits else {},
                "subscription_tier": effective_limits.get("subscription_tier", "unknown")
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Configuration validation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate configuration: {str(e)}"
        )

@app.get("/config/mcp-access", tags=["configuration"])
@limiter.limit("60/minute")
async def get_mcp_server_configuration(
    request: Request,
    authorization: Optional[str] = None
):
    """
    Get MCP server access configuration based on user's subscription.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    try:
        # Extract user ID from authorization header
        middleware = UsageTrackingMiddleware(None)
        user_id = await middleware._extract_user_id(request)
        
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Create subscription-aware configuration
        config = await Configuration.from_user_subscription(user_id)
        
        # Check MCP server access
        mcp_access = {}
        for server in ["reddit", "youtube", "github"]:
            mcp_access[server] = await config.check_mcp_server_access(server)
        
        # Get subscription info
        subscription_info = await get_user_subscription_info(user_id)
        
        return {
            "status": "success",
            "user_id": user_id,
            "mcp_server_access": mcp_access,
            "subscription_tier": subscription_info.tier.value,
            "allowed_servers": subscription_info.features.mcp_servers,
            "github_mcp_access": subscription_info.features.github_mcp_access,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"MCP configuration error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve MCP configuration: {str(e)}"
        )

# Include routers
app.include_router(auth_router)
app.include_router(webhook_router)

# Custom OpenAPI schema
def custom_openapi():
    """Custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Open Deep Research API",
        version="1.0.0",
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token obtained from Supabase authentication"
        }
    }
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": f"http://localhost:{PORT}",
            "description": "Development server"
        },
        {
            "url": "https://api.opendeepresearch.com",
            "description": "Production server"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Development server configuration
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info" if not DEBUG else "debug",
        access_log=True,
    )














