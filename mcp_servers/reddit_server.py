#!/usr/bin/env python3
"""
Reddit MCP Server for extracting posts and comments data.

This server provides tools for:
- Fetching posts from subreddits
- Extracting comments from specific posts
- Searching Reddit content

Requires Reddit API credentials:
- REDDIT_CLIENT_ID
- REDDIT_CLIENT_SECRET
- REDDIT_USER_AGENT
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP
import praw
from praw.models import Submission, Comment
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Reddit")

def get_reddit_client() -> praw.Reddit:
    """Initialize and return a Reddit client."""
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "DeepResearch/1.0")
    
    if not client_id or not client_secret:
        raise ValueError("Reddit API credentials not found. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")
    
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        read_only=True
    )

def format_post_data(submission: Submission) -> Dict[str, Any]:
    """Format Reddit post data for output."""
    return {
        "id": submission.id,
        "title": submission.title,
        "author": str(submission.author) if submission.author else "[deleted]",
        "score": submission.score,
        "upvote_ratio": submission.upvote_ratio,
        "num_comments": submission.num_comments,
        "created_utc": submission.created_utc,
        "url": submission.url,
        "permalink": f"https://reddit.com{submission.permalink}",
        "selftext": submission.selftext[:500] + "..." if len(submission.selftext) > 500 else submission.selftext,
        "subreddit": str(submission.subreddit),
        "is_self": submission.is_self,
        "domain": submission.domain,
        "flair_text": submission.link_flair_text
    }

def format_comment_data(comment: Comment, depth: int = 0) -> Dict[str, Any]:
    """Format Reddit comment data for output."""
    return {
        "id": comment.id,
        "author": str(comment.author) if comment.author else "[deleted]",
        "body": comment.body[:1000] + "..." if len(comment.body) > 1000 else comment.body,
        "score": comment.score,
        "created_utc": comment.created_utc,
        "permalink": f"https://reddit.com{comment.permalink}",
        "depth": depth,
        "is_submitter": comment.is_submitter,
        "stickied": comment.stickied,
        "parent_id": comment.parent_id
    }

@mcp.tool()
def get_reddit_posts(subreddit: str, limit: int = 10, sort: str = "hot") -> str:
    """
    Fetch posts from a specific subreddit.
    
    Args:
        subreddit: Name of the subreddit (without r/ prefix)
        limit: Number of posts to fetch (default: 10, max: 100)
        sort: Sort method - 'hot', 'new', 'top', 'rising' (default: 'hot')
    
    Returns:
        Formatted string containing post information
    """
    try:
        reddit = get_reddit_client()
        
        # Validate and limit the number of posts
        limit = min(max(1, limit), 100)
        
        # Get subreddit
        sub = reddit.subreddit(subreddit)
        
        # Get posts based on sort method
        if sort == "hot":
            posts = sub.hot(limit=limit)
        elif sort == "new":
            posts = sub.new(limit=limit)
        elif sort == "top":
            posts = sub.top(limit=limit, time_filter="day")
        elif sort == "rising":
            posts = sub.rising(limit=limit)
        else:
            posts = sub.hot(limit=limit)  # Default to hot
        
        # Format posts
        formatted_posts = []
        for post in posts:
            post_data = format_post_data(post)
            formatted_posts.append(post_data)
        
        # Create formatted output
        output = f"## Posts from r/{subreddit} (sorted by {sort})\n\n"
        
        for i, post in enumerate(formatted_posts, 1):
            output += f"### {i}. {post['title']}\n"
            output += f"**Author:** {post['author']} | **Score:** {post['score']} | **Comments:** {post['num_comments']}\n"
            output += f"**URL:** {post['url']}\n"
            output += f"**Reddit Link:** {post['permalink']}\n"
            if post['selftext']:
                output += f"**Content:** {post['selftext']}\n"
            output += f"**Flair:** {post['flair_text'] or 'None'}\n"
            output += "---\n\n"
        
        return output
        
    except Exception as e:
        logger.error(f"Error fetching Reddit posts: {e}")
        return f"Error fetching posts from r/{subreddit}: {str(e)}"

@mcp.tool()
def get_reddit_comments(post_url: str, limit: int = 20) -> str:
    """
    Fetch comments from a specific Reddit post.
    
    Args:
        post_url: Full URL or Reddit permalink to the post
        limit: Number of top-level comments to fetch (default: 20, max: 100)
    
    Returns:
        Formatted string containing comment information
    """
    try:
        reddit = get_reddit_client()
        
        # Validate and limit the number of comments
        limit = min(max(1, limit), 100)
        
        # Extract submission ID from URL
        if "reddit.com" in post_url:
            # Handle full Reddit URLs
            submission = reddit.submission(url=post_url)
        elif post_url.startswith("/r/"):
            # Handle Reddit permalinks
            submission = reddit.submission(url=f"https://reddit.com{post_url}")
        else:
            # Assume it's a submission ID
            submission = reddit.submission(id=post_url)
        
        # Get post info
        post_data = format_post_data(submission)
        
        # Get comments
        submission.comments.replace_more(limit=0)  # Remove "more comments" objects
        comments = submission.comments.list()[:limit]
        
        # Format output
        output = f"## Comments for: {post_data['title']}\n"
        output += f"**Post by:** {post_data['author']} | **Score:** {post_data['score']} | **Total Comments:** {post_data['num_comments']}\n"
        output += f"**Post URL:** {post_data['permalink']}\n\n"
        
        if not comments:
            output += "No comments found or comments are not accessible.\n"
            return output
        
        output += f"### Top {len(comments)} Comments:\n\n"
        
        for i, comment in enumerate(comments, 1):
            if hasattr(comment, 'body') and comment.body != "[deleted]":
                comment_data = format_comment_data(comment)
                output += f"**{i}. {comment_data['author']}** (Score: {comment_data['score']})\n"
                output += f"{comment_data['body']}\n"
                output += f"*Posted: {comment_data['created_utc']}*\n"
                output += "---\n\n"
        
        return output
        
    except Exception as e:
        logger.error(f"Error fetching Reddit comments: {e}")
        return f"Error fetching comments from {post_url}: {str(e)}"

@mcp.tool()
def search_reddit(query: str, subreddit: Optional[str] = None, limit: int = 10) -> str:
    """
    Search Reddit content for specific queries.
    
    Args:
        query: Search query string
        subreddit: Optional subreddit to limit search to (without r/ prefix)
        limit: Number of results to return (default: 10, max: 50)
    
    Returns:
        Formatted string containing search results
    """
    try:
        reddit = get_reddit_client()
        
        # Validate and limit the number of results
        limit = min(max(1, limit), 50)
        
        # Perform search
        if subreddit:
            # Search within specific subreddit
            sub = reddit.subreddit(subreddit)
            search_results = sub.search(query, limit=limit, sort="relevance")
            search_context = f"r/{subreddit}"
        else:
            # Search all of Reddit
            search_results = reddit.subreddit("all").search(query, limit=limit, sort="relevance")
            search_context = "all of Reddit"
        
        # Format results
        results = []
        for post in search_results:
            post_data = format_post_data(post)
            results.append(post_data)
        
        # Create formatted output
        output = f"## Search Results for '{query}' in {search_context}\n\n"
        
        if not results:
            output += "No results found for your search query.\n"
            return output
        
        output += f"Found {len(results)} results:\n\n"
        
        for i, post in enumerate(results, 1):
            output += f"### {i}. {post['title']}\n"
            output += f"**Subreddit:** r/{post['subreddit']} | **Author:** {post['author']}\n"
            output += f"**Score:** {post['score']} | **Comments:** {post['num_comments']}\n"
            output += f"**URL:** {post['url']}\n"
            output += f"**Reddit Link:** {post['permalink']}\n"
            if post['selftext']:
                output += f"**Content Preview:** {post['selftext'][:200]}...\n"
            output += "---\n\n"
        
        return output
        
    except Exception as e:
        logger.error(f"Error searching Reddit: {e}")
        return f"Error searching Reddit for '{query}': {str(e)}"

if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="streamable-http", port=8001)
