#!/usr/bin/env python3
"""
YouTube MCP Server Startup Script

This script starts the YouTube MCP server with proper configuration and error handling.
It loads environment variables, validates API credentials, and starts the server on port 8002.

Usage:
    python scripts/start_youtube_server.py [--port PORT] [--debug]

Environment Variables Required:
    YOUTUBE_API_KEY - YouTube Data API v3 key
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
            logging.FileHandler(project_root / 'logs' / 'youtube_server.log', mode='a')
        ]
    )
    
    # Create logs directory if it doesn't exist
    (project_root / 'logs').mkdir(exist_ok=True)

def validate_environment():
    """Validate that required environment variables are set."""
    required_vars = ['YOUTUBE_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease set the following environment variables:")
        print("  YOUTUBE_API_KEY - Your YouTube Data API v3 key")
        print("\nYou can set these in your .env file or as environment variables.")
        print("\nTo get YouTube API credentials:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a new project or select an existing one")
        print("3. Enable the YouTube Data API v3")
        print("4. Create credentials (API Key)")
        print("5. Note down your API key")
        return False
    
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import googleapiclient
        from googleapiclient.discovery import build
        import mcp
        from mcp.server.fastmcp import FastMCP
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("\nPlease install the required dependencies:")
        print("  pip install google-api-python-client>=2.100.0 mcp>=1.9.4")
        print("  or")
        print("  uv pip install -r pyproject.toml")
        return False

def test_youtube_connection():
    """Test YouTube API connection."""
    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
        
        api_key = os.getenv("YOUTUBE_API_KEY")
        
        # Build YouTube service
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        # Test connection with a simple search
        search_response = youtube.search().list(
            q='test',
            part='id,snippet',
            maxResults=1,
            type='video'
        ).execute()
        
        print("✅ YouTube API connection successful")
        return True
        
    except HttpError as e:
        print(f"❌ YouTube API connection failed: {e}")
        if e.resp.status == 403:
            print("\nPossible issues:")
            print("  - Invalid API key")
            print("  - YouTube Data API v3 not enabled for your project")
            print("  - API key restrictions preventing access")
        elif e.resp.status == 400:
            print("\nPossible issues:")
            print("  - Malformed API key")
            print("  - Invalid request parameters")
        else:
            print(f"\nHTTP Error {e.resp.status}: {e}")
        return False
    except Exception as e:
        print(f"❌ YouTube API connection failed: {e}")
        print("\nPlease check your YouTube API key:")
        print("  - Verify YOUTUBE_API_KEY is correct")
        print("  - Ensure YouTube Data API v3 is enabled in Google Cloud Console")
        print("  - Check that your API key has the necessary permissions")
        return False

def start_server(port: int = 8002, debug: bool = False):
    """Start the YouTube MCP server."""
    try:
        # Import the server module
        sys.path.insert(0, str(project_root / 'mcp_servers'))
        from youtube_server import mcp
        
        print(f"🚀 Starting YouTube MCP Server on port {port}")
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
    """Main function to start the YouTube MCP server."""
    parser = argparse.ArgumentParser(
        description="Start the YouTube MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/start_youtube_server.py
  python scripts/start_youtube_server.py --port 8002 --debug
  
Environment Variables:
  YOUTUBE_API_KEY      YouTube Data API v3 key (required)
        """
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=8002,
        help='Port to run the server on (default: 8002)'
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
    
    print("🔧 YouTube MCP Server Startup")
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
    
    # Test YouTube connection (unless skipped)
    if not args.skip_tests:
        if not test_youtube_connection():
            print("\n💡 Use --skip-tests to start server without connection test")
            sys.exit(1)
    
    print("=" * 40)
    
    # Start the server
    start_server(args.port, args.debug)

if __name__ == "__main__":
    main()
