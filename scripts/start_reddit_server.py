#!/usr/bin/env python3
"""
Reddit MCP Server Startup Script

This script starts the Reddit MCP server with proper configuration and error handling.
It loads environment variables, validates API credentials, and starts the server on port 8001.

Usage:
    python scripts/start_reddit_server.py [--port PORT] [--debug]

Environment Variables Required:
    REDDIT_CLIENT_ID - Reddit API client ID
    REDDIT_CLIENT_SECRET - Reddit API client secret
    REDDIT_USER_AGENT - User agent string (optional, defaults to DeepResearch/1.0)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_logging(debug: bool = False):
    """Configure logging for the server."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(project_root / 'logs' / 'reddit_server.log', mode='a')
        ]
    )
    
    # Create logs directory if it doesn't exist
    (project_root / 'logs').mkdir(exist_ok=True)

def validate_environment():
    """Validate that required environment variables are set."""
    required_vars = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease set the following environment variables:")
        print("  REDDIT_CLIENT_ID - Your Reddit API client ID")
        print("  REDDIT_CLIENT_SECRET - Your Reddit API client secret")
        print("  REDDIT_USER_AGENT - User agent string (optional)")
        print("\nYou can set these in your .env file or as environment variables.")
        print("\nTo get Reddit API credentials:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Click 'Create App' or 'Create Another App'")
        print("3. Choose 'script' as the app type")
        print("4. Note down your client_id and client_secret")
        return False
    
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import praw
        import mcp
        from mcp.server.fastmcp import FastMCP
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("\nPlease install the required dependencies:")
        print("  pip install praw>=7.7.1 mcp>=1.9.4")
        print("  or")
        print("  uv pip install -r pyproject.toml")
        return False

def test_reddit_connection():
    """Test Reddit API connection."""
    try:
        import praw
        
        client_id = os.getenv("REDDIT_CLIENT_ID")
        client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = os.getenv("REDDIT_USER_AGENT", "DeepResearch/1.0")
        
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            read_only=True
        )
        
        # Test connection by accessing user info
        reddit.user.me()
        print("✅ Reddit API connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Reddit API connection failed: {e}")
        print("\nPlease check your Reddit API credentials:")
        print("  - Verify REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are correct")
        print("  - Ensure your Reddit app is configured as 'script' type")
        print("  - Check that your credentials are not expired")
        return False

def start_server(port: int = 8001, debug: bool = False):
    """Start the Reddit MCP server."""
    try:
        # Import the server module
        sys.path.insert(0, str(project_root / 'mcp_servers'))
        from reddit_server import mcp
        
        print(f"🚀 Starting Reddit MCP Server on port {port}")
        print(f"📡 Server URL: http://localhost:{port}/mcp")
        print("🔄 Server is running... Press Ctrl+C to stop")
        
        # Start the server
        mcp.run(transport="streamable-http", port=port)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def main():
    """Main function to start the Reddit MCP server."""
    parser = argparse.ArgumentParser(
        description="Start the Reddit MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/start_reddit_server.py
  python scripts/start_reddit_server.py --port 8001 --debug
  
Environment Variables:
  REDDIT_CLIENT_ID     Reddit API client ID (required)
  REDDIT_CLIENT_SECRET Reddit API client secret (required)
  REDDIT_USER_AGENT    User agent string (optional)
        """
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=8001,
        help='Port to run the server on (default: 8001)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip connection tests and start server directly'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    print("🔧 Reddit MCP Server Startup")
    print("=" * 40)
    
    # Load environment variables
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"✅ Loaded environment variables from {env_file}")
    else:
        print(f"⚠️  No .env file found at {env_file}")
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Test Reddit connection (unless skipped)
    if not args.skip_tests:
        if not test_reddit_connection():
            print("\n💡 Use --skip-tests to start server without connection test")
            sys.exit(1)
    
    print("=" * 40)
    
    # Start the server
    start_server(args.port, args.debug)

if __name__ == "__main__":
    main()
