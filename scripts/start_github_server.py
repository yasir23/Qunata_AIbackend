#!/usr/bin/env python3
"""
GitHub MCP Server Startup Script

This script starts the GitHub MCP server with proper configuration and error handling.
It loads environment variables, validates API credentials, and starts the server on port 8003.

Usage:
    python scripts/start_github_server.py [--port PORT] [--debug]

Environment Variables Required:
    GITHUB_TOKEN - GitHub Personal Access Token
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
            logging.FileHandler(project_root / 'logs' / 'github_server.log', mode='a')
        ]
    )
    
    # Create logs directory if it doesn't exist
    (project_root / 'logs').mkdir(exist_ok=True)

def validate_environment():
    """Validate that required environment variables are set."""
    required_vars = ['GITHUB_TOKEN']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"  {var}")
        
        print("\nPlease set the following environment variables:")
        print("  GITHUB_TOKEN - Your GitHub Personal Access Token")
        
        print("\nTo get GitHub API credentials:")
        print("1. Go to https://github.com/settings/tokens")
        print("2. Click 'Generate new token' -> 'Generate new token (classic)'")
        print("3. Select scopes: 'repo' (for private repos) or 'public_repo' (for public repos only)")
        print("4. Generate and copy your token")
        print("5. Add it to your .env file as GITHUB_TOKEN=your_token_here")
        
        return False
    
    return True

def test_github_connection():
    """Test GitHub API connection."""
    try:
        import requests
        
        token = os.getenv("GITHUB_TOKEN")
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "DeepResearch/1.0"
        }
        
        # Test API connection with a simple request
        response = requests.get("https://api.github.com/user", headers=headers, timeout=10)
        
        if response.status_code == 200:
            user_data = response.json()
            print(f"✅ GitHub API connection successful")
            print(f"   Authenticated as: {user_data.get('login', 'Unknown')}")
            print(f"   Rate limit remaining: {response.headers.get('X-RateLimit-Remaining', 'Unknown')}")
            return True
        else:
            print(f"❌ GitHub API connection failed: HTTP {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except ImportError:
        print("❌ Required package 'requests' not found. Please install it:")
        print("   pip install requests")
        return False
    except Exception as e:
        print(f"❌ GitHub API connection failed: {e}")
        print("\nPlease check your GitHub API credentials:")
        print("1. Verify your GITHUB_TOKEN is correct")
        print("2. Ensure the token has appropriate permissions")
        print("3. Check your internet connection")
        return False

def start_server(port: int = 8003, debug: bool = False):
    """Start the GitHub MCP server."""
    try:
        # Import the server module
        sys.path.insert(0, str(project_root / 'mcp_servers'))
        from github_server import mcp
        
        print(f"🚀 Starting GitHub MCP Server on port {port}")
        print(f"📡 Server URL: http://localhost:{port}/mcp")
        print("🔄 Server is running... Press Ctrl+C to stop")
        
        # Start the server
        mcp.run(port=port, debug=debug)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

def main():
    """Main function to start the GitHub MCP server."""
    parser = argparse.ArgumentParser(
        description="Start the GitHub MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/start_github_server.py
  python scripts/start_github_server.py --port 8003 --debug

Environment Variables:
  GITHUB_TOKEN    GitHub Personal Access Token (required)

For more information on getting GitHub API credentials:
  https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
        """
    )
    
    parser.add_argument(
        '--port', 
        type=int, 
        default=8003,
        help='Port to run the server on (default: 8003)'
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
    
    # Load environment variables
    load_dotenv()
    
    print("🔧 GitHub MCP Server Startup")
    print("=" * 40)
    
    # Setup logging
    setup_logging(args.debug)
    
    # Validate environment variables
    if not validate_environment():
        sys.exit(1)
    
    # Test connection unless skipped
    if not args.skip_tests:
        print("\n🔍 Testing GitHub API connection...")
        if not test_github_connection():
            print("\n💡 Use --skip-tests to start server without connection test")
            sys.exit(1)
        print()
    
    # Start the server
    start_server(args.port, args.debug)

if __name__ == "__main__":
    main()
