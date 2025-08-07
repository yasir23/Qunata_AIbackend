# Reddit and YouTube Research with MCP Servers

This guide demonstrates how to use the Reddit and YouTube MCP servers with the Deep Research application to extract and analyze data from social media platforms.

## Overview

The Deep Research application now supports two specialized MCP servers for social media data extraction:

- **Reddit MCP Server** (Port 8001): Extract posts, comments, and search Reddit content
- **YouTube MCP Server** (Port 8002): Extract video comments, search videos, and get video metadata

## Prerequisites

### 1. API Credentials Setup

#### Reddit API Setup
1. Go to [Reddit App Preferences](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Choose "script" as the app type
4. Note down your `client_id` and `client_secret`

#### YouTube API Setup
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the YouTube Data API v3
4. Create credentials (API Key)
5. Note down your API key

### 2. Environment Configuration

Update your `.env` file with the required API credentials:

```bash
# Reddit API Configuration
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=DeepResearch/1.0

# YouTube API Configuration
YOUTUBE_API_KEY=your_youtube_api_key_here
```

### 3. Install Dependencies

Ensure all required dependencies are installed:

```bash
# Install the project dependencies
uv pip install -r pyproject.toml

# Or if using pip directly
pip install praw>=7.7.1 google-api-python-client>=2.100.0
```

## Starting the MCP Servers

### Option 1: Using Startup Scripts (Recommended)

```bash
# Start Reddit MCP Server
python scripts/start_reddit_server.py

# Start YouTube MCP Server (in another terminal)
python scripts/start_youtube_server.py
```

### Option 2: Direct Server Execution

```bash
# Start Reddit MCP Server
python mcp_servers/reddit_server.py

# Start YouTube MCP Server (in another terminal)
python mcp_servers/youtube_server.py
```

The servers will start on:
- Reddit: `http://localhost:8001/mcp`
- YouTube: `http://localhost:8002/mcp`

## Configuration Examples

### Single Server Configuration

#### Reddit Only
```json
{
  "mcp_config": {
    "url": "http://localhost:8001/mcp",
    "tools": ["get_reddit_posts", "get_reddit_comments", "search_reddit"],
    "auth_required": false
  }
}
```

#### YouTube Only
```json
{
  "mcp_config": {
    "url": "http://localhost:8002/mcp",
    "tools": ["get_youtube_comments", "search_youtube_videos", "get_video_info"],
    "auth_required": false
  }
}
```

### Multiple Server Configuration

For using both Reddit and YouTube servers simultaneously:

```json
{
  "mcp_server_config": {
    "reddit": {
      "url": "http://localhost:8001/mcp",
      "transport": "streamable_http"
    },
    "youtube": {
      "url": "http://localhost:8002/mcp",
      "transport": "streamable_http"
    }
  }
}
```

## Usage Examples

### Research Scenario 1: Analyzing Public Opinion on a Topic

**Research Question**: "What are people saying about artificial intelligence on Reddit and YouTube?"

**LangGraph Studio Configuration**:
1. Set `mcp_config.url` to `http://localhost:8001/mcp` for Reddit analysis
2. Run research query: "Analyze public opinion about artificial intelligence on Reddit"

**Expected Tools Usage**:
- `search_reddit(query="artificial intelligence", limit=20)`
- `get_reddit_posts(subreddit="MachineLearning", limit=15, sort="top")`
- `get_reddit_comments(post_url="[relevant post URL]", limit=30)`

### Research Scenario 2: Video Content Analysis

**Research Question**: "What are the main concerns in YouTube comments about climate change videos?"

**LangGraph Studio Configuration**:
1. Set `mcp_config.url` to `http://localhost:8002/mcp` for YouTube analysis
2. Run research query: "Analyze YouTube comments about climate change concerns"

**Expected Tools Usage**:
- `search_youtube_videos(query="climate change", limit=10)`
- `get_video_info(video_id="[video ID from search]")`
- `get_youtube_comments(video_id="[video ID]", limit=50)`

### Research Scenario 3: Cross-Platform Analysis

**Research Question**: "Compare discussions about electric vehicles on Reddit vs YouTube"

**Configuration**: Use multiple server setup with both Reddit and YouTube

**Expected Tools Usage**:
- `search_reddit(query="electric vehicles", limit=15)`
- `search_youtube_videos(query="electric vehicles", limit=10)`
- `get_reddit_comments(post_url="[Reddit post]", limit=25)`
- `get_youtube_comments(video_id="[YouTube video]", limit=40)`

## Tool Reference

### Reddit MCP Server Tools

#### `get_reddit_posts`
Fetches posts from a specific subreddit.

**Parameters**:
- `subreddit` (str): Subreddit name without "r/" prefix
- `limit` (int, optional): Number of posts (default: 10, max: 100)
- `sort` (str, optional): Sort method - "hot", "new", "top", "rising" (default: "hot")

**Example Usage**:
```
Get the top 20 posts from r/technology sorted by hot
```

#### `get_reddit_comments`
Extracts comments from a specific Reddit post.

**Parameters**:
- `post_url` (str): Full Reddit URL or permalink
- `limit` (int, optional): Number of comments (default: 20, max: 100)

**Example Usage**:
```
Get comments from this Reddit post: https://reddit.com/r/technology/comments/abc123/
```

#### `search_reddit`
Searches Reddit content for specific queries.

**Parameters**:
- `query` (str): Search query
- `subreddit` (str, optional): Limit search to specific subreddit
- `limit` (int, optional): Number of results (default: 10, max: 50)

**Example Usage**:
```
Search for "machine learning" discussions in r/programming
```

### YouTube MCP Server Tools

#### `get_youtube_comments`
Fetches comments from a YouTube video.

**Parameters**:
- `video_id` (str): YouTube video ID or full URL
- `limit` (int, optional): Number of comments (default: 50, max: 100)

**Example Usage**:
```
Get comments from this YouTube video: https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

#### `search_youtube_videos`
Searches for YouTube videos.

**Parameters**:
- `query` (str): Search query
- `limit` (int, optional): Number of results (default: 10, max: 50)

**Example Usage**:
```
Search for videos about "quantum computing"
```

#### `get_video_info`
Gets detailed metadata about a YouTube video.

**Parameters**:
- `video_id` (str): YouTube video ID or full URL

**Example Usage**:
```
Get detailed information about this video: https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

## Best Practices

### 1. Rate Limiting
- Use reasonable limits to avoid API quota exhaustion
- Reddit: Max 100 posts/comments, 50 search results
- YouTube: Max 100 comments, 50 search results/videos

### 2. Query Optimization
- Use specific subreddits for focused Reddit research
- Include relevant keywords in search queries
- Combine multiple tools for comprehensive analysis

### 3. Data Analysis
- Use the extracted data to identify trends and patterns
- Cross-reference findings between platforms
- Consider the context and demographics of each platform

### 4. Error Handling
- Monitor server logs for API errors
- Handle rate limiting gracefully
- Verify API credentials are correctly configured

## Troubleshooting

### Common Issues

#### Server Connection Failed
**Problem**: Cannot connect to MCP servers
**Solution**: 
- Verify servers are running on correct ports (8001, 8002)
- Check firewall settings
- Ensure no other services are using these ports

#### Reddit API Authentication Errors
**Problem**: "Reddit API credentials not found" error
**Solution**:
- Verify `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` are set in `.env`
- Check that the Reddit app is configured as "script" type
- Ensure credentials are valid and not expired

#### YouTube API Quota Exceeded
**Problem**: "YouTube API quota exceeded" error
**Solution**:
- Check your Google Cloud Console quota usage
- Reduce the `limit` parameters in tool calls
- Consider upgrading your YouTube API quota if needed

#### Tool Not Found Errors
**Problem**: "Tool [name] not found" error
**Solution**:
- Verify the MCP server is running and accessible
- Check that tool names match exactly (case-sensitive)
- Ensure the server configuration includes the required tools

### Debug Mode

Enable debug logging by setting the environment variable:
```bash
export PYTHONPATH=/home/daytona/Qunata_AIbackend/src
export DEBUG=true
python mcp_servers/reddit_server.py
```

## Advanced Usage

### Custom Research Workflows

You can create custom research workflows that combine multiple tools:

1. **Trend Analysis**: Use `search_reddit` and `search_youtube_videos` to find trending topics
2. **Sentiment Analysis**: Extract comments using both servers and analyze sentiment
3. **Content Comparison**: Compare discussions across platforms for the same topic
4. **Influencer Research**: Use `get_video_info` to analyze popular content creators

### Integration with LangGraph

The MCP servers integrate seamlessly with LangGraph workflows:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# Configure MCP client
client = MultiServerMCPClient({
    "reddit": {
        "url": "http://localhost:8001/mcp",
        "transport": "streamable_http"
    },
    "youtube": {
        "url": "http://localhost:8002/mcp", 
        "transport": "streamable_http"
    }
})

# Get tools and create agent
tools = await client.get_tools()
agent = create_react_agent("anthropic:claude-3-5-sonnet-latest", tools)

# Run research
response = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Analyze Reddit discussions about AI safety"}]
})
```

## Support

For additional support:
- Check the server logs for detailed error messages
- Review the MCP server configuration in `examples/mcp_servers_config.json`
- Ensure all environment variables are properly set
- Verify API credentials have the necessary permissions

## Contributing

To extend the MCP servers with additional functionality:
1. Add new tools to the respective server files
2. Update the configuration examples
3. Add usage examples to this documentation
4. Test the new functionality thoroughly

---

*This documentation covers the basic usage of Reddit and YouTube MCP servers with the Deep Research application. For more advanced configurations and custom implementations, refer to the LangGraph MCP documentation.*
