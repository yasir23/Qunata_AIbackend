#!/usr/bin/env python3
"""
Startup script for the Open Deep Research FastAPI application.

This script provides a simple way to start the FastAPI server with proper configuration.
"""

import os
import sys
import uvicorn

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Start the FastAPI application."""
    try:
        # Import the FastAPI app
        from api.main import app
        
        # Configuration
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        debug = os.getenv("DEBUG", "false").lower() == "true"
        
        print(f"🚀 Starting Open Deep Research API on {host}:{port}")
        print(f"📚 API Documentation: http://{host}:{port}/docs")
        print(f"🔍 Health Check: http://{host}:{port}/health")
        
        # Start the server
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=debug,
            log_level="info" if not debug else "debug",
            access_log=True,
        )
        
    except ImportError as e:
        print(f"❌ Failed to import FastAPI app: {e}")
        print("Make sure all dependencies are installed and the src directory is accessible.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
