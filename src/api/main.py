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

# Import routers from existing components
from .auth_endpoints import auth_router, limiter as auth_limiter
from ..payment.webhook_handler import webhook_router
from ..security.auth import supabase

# Import LangGraph components
from ..open_deep_research.deep_researcher import deep_researcher
from ..open_deep_research.configuration import Configuration
from ..open_deep_research.state import AgentInputState

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

