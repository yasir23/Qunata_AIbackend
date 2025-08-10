#!/usr/bin/env python3
"""
GitHub MCP Server Startup Script

This script starts the GitHub MCP server with proper configuration and error handling.
It loads environment variables, validates API credentials, and starts the server on port 8003.

Usage:
    python scripts/start_github_server.py [--port PORT] [--debug]

Environment Variables Required:
    GITHUB_TOKEN - GitHub API personal access token
    GITHUB_ORG - GitHub organization (optional, for organization-specific operations)
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
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        print("\nTo get a GitHub token:")
        print("1. Go to GitHub Settings > Developer settings > Personal access tokens")
        print("2. Generate a new token with 'repo' and 'read:org' scopes")
        print("3. Set GITHUB_TOKEN=your_token_here in your .env file")
        return False
    
    return True

def check_github_connection():
    """Test GitHub API connection."""
    try:
        import requests
        
        token = os.getenv('GITHUB_TOKEN')
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'DeepResearch-GitHub-MCP/1.0'
        }
        
        # Test API connection with rate limit check
        response = requests.get('https://api.github.com/rate_limit', headers=headers, timeout=10)
        
        if response.status_code == 200:
            rate_limit_data = response.json()
            remaining = rate_limit_data.get('rate', {}).get('remaining', 0)
            limit = rate_limit_data.get('rate', {}).get('limit', 0)
            
            print(f"✅ GitHub API connection successful")
            print(f"   Rate limit: {remaining}/{limit} requests remaining")
            
            if remaining < 100:
                print(f"⚠️  Warning: Low rate limit remaining ({remaining} requests)")
            
            return True
        else:
            print(f"❌ GitHub API connection failed: HTTP {response.status_code}")
            if response.status_code == 401:
                print("   Check your GITHUB_TOKEN - it may be invalid or expired")
            elif response.status_code == 403:
                print("   Rate limit exceeded or insufficient permissions")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ GitHub API connection failed: {e}")
        return False
    except ImportError:
        print("❌ Required 'requests' library not found. Please install it.")
        return False

def start_server(port: int = 8003, debug: bool = False):
    """Start the GitHub MCP server."""
    try:
        # Import the server module
        from mcp_servers.github_server import mcp
        
        print(f"🚀 Starting GitHub MCP Server on port {port}")
        print(f"   Debug mode: {'enabled' if debug else 'disabled'}")
        print(f"   GitHub organization: {os.getenv('GITHUB_ORG', 'Not specified (will work with any repo)')}")
        print(f"   Log file: {project_root / 'logs' / 'github_server.log'}")
        print("\n📋 Available tools:")
        print("   - fetch_issues: Get issues from repositories")
        print("   - create_issue: Create new issues")
        print("   - search_repositories: Search for repositories")
        print("   - search_issues: Search for issues across repositories")
        print("   - get_project_insights: Get comprehensive project analytics")
        print("\n🔗 Server will be available at: http://localhost:{}/mcp".format(port))
        print("   Use Ctrl+C to stop the server")
        print("=" * 60)
        
        # Run the server
        mcp.run(transport="streamable-http", port=port)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except ImportError as e:
        print(f"❌ Failed to import GitHub server module: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install requests mcp")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start GitHub MCP server: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start the GitHub MCP Server for development workflow integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/start_github_server.py
    python scripts/start_github_server.py --port 8004 --debug
    
Environment Variables:
    GITHUB_TOKEN    - GitHub personal access token (required)
    GITHUB_ORG      - GitHub organization name (optional)
    
GitHub Token Scopes Required:
    - repo (for private repositories)
    - read:org (for organization repositories)
    - public_repo (for public repositories only)
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
    
    args = parser.parse_args()
    
    # Load environment variables
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f"📁 Loaded environment from {env_file}")
    else:
        print(f"⚠️  No .env file found at {env_file}")
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    print("🐙 GitHub MCP Server Startup")
    print("=" * 40)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Test GitHub connection
    print("\n🔍 Testing GitHub API connection...")
    if not check_github_connection():
        print("\n💡 Troubleshooting tips:")
        print("   1. Verify your GITHUB_TOKEN is correct")
        print("   2. Check token permissions (repo, read:org)")
        print("   3. Ensure you have internet connectivity")
        print("   4. Check if GitHub API is accessible from your network")
        sys.exit(1)
    
    # Start the server
    print(f"\n🎯 All checks passed! Starting server...")
    start_server(port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
