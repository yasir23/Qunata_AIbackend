#!/usr/bin/env python3
"""
YouTube MCP Server for extracting comments and video data.

This server provides tools for:
- Fetching comments from YouTube videos
- Searching for YouTube videos
- Getting video metadata and information

Requires YouTube API credentials:
- YOUTUBE_API_KEY
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging
import re
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("YouTube")

def get_youtube_client():
    """Initialize and return a YouTube API client."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    
    if not api_key:
        raise ValueError("YouTube API key not found. Please set YOUTUBE_API_KEY environment variable.")
    
    return build('youtube', 'v3', developerKey=api_key)

def extract_video_id(video_input: str) -> str:
    """Extract video ID from various YouTube URL formats or return as-is if already an ID."""
    # YouTube URL patterns
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/v/([a-zA-Z0-9_-]{11})',
        r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, video_input)
        if match:
            return match.group(1)
    
    # If no pattern matches, assume it's already a video ID
    if len(video_input) == 11 and re.match(r'^[a-zA-Z0-9_-]+$', video_input):
        return video_input
    
    raise ValueError(f"Invalid YouTube video URL or ID: {video_input}")

def format_video_data(video: Dict[str, Any]) -> Dict[str, Any]:
    """Format YouTube video data for output."""
    snippet = video.get('snippet', {})
    statistics = video.get('statistics', {})
    
    return {
        "id": video.get('id', ''),
        "title": snippet.get('title', ''),
        "channel_title": snippet.get('channelTitle', ''),
        "channel_id": snippet.get('channelId', ''),
        "description": snippet.get('description', '')[:500] + "..." if len(snippet.get('description', '')) > 500 else snippet.get('description', ''),
        "published_at": snippet.get('publishedAt', ''),
        "duration": video.get('contentDetails', {}).get('duration', ''),
        "view_count": statistics.get('viewCount', '0'),
        "like_count": statistics.get('likeCount', '0'),
        "comment_count": statistics.get('commentCount', '0'),
        "thumbnail_url": snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
        "url": f"https://www.youtube.com/watch?v={video.get('id', '')}"
    }

def format_comment_data(comment: Dict[str, Any]) -> Dict[str, Any]:
    """Format YouTube comment data for output."""
    snippet = comment.get('snippet', {})
    top_level_comment = snippet.get('topLevelComment', {}).get('snippet', {}) if 'topLevelComment' in snippet else snippet
    
    return {
        "id": comment.get('id', ''),
        "author": top_level_comment.get('authorDisplayName', ''),
        "author_channel_url": top_level_comment.get('authorChannelUrl', ''),
        "text": top_level_comment.get('textDisplay', ''),
        "like_count": top_level_comment.get('likeCount', 0),
        "published_at": top_level_comment.get('publishedAt', ''),
        "updated_at": top_level_comment.get('updatedAt', ''),
        "reply_count": snippet.get('totalReplyCount', 0) if 'totalReplyCount' in snippet else 0
    }

@mcp.tool()
def get_youtube_comments(video_id: str, limit: int = 50) -> str:
    """
    Fetch comments from a YouTube video.
    
    Args:
        video_id: YouTube video ID or full YouTube URL
        limit: Number of comments to fetch (default: 50, max: 100)
    
    Returns:
        Formatted string containing comment information
    """
    try:
        youtube = get_youtube_client()
        
        # Extract video ID from URL if needed
        video_id = extract_video_id(video_id)
        
        # Validate and limit the number of comments
        limit = min(max(1, limit), 100)
        
        # First, get video information
        video_response = youtube.videos().list(
            part='snippet,statistics',
            id=video_id
        ).execute()
        
        if not video_response['items']:
            return f"Video with ID '{video_id}' not found or is not accessible."
        
        video_data = format_video_data(video_response['items'][0])
        
        # Get comments
        comments_response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=limit,
            order='relevance'
        ).execute()
        
        comments = comments_response.get('items', [])
        
        # Format output
        output = f"## Comments for: {video_data['title']}\n"
        output += f"**Channel:** {video_data['channel_title']}\n"
        output += f"**Views:** {video_data['view_count']} | **Likes:** {video_data['like_count']} | **Total Comments:** {video_data['comment_count']}\n"
        output += f"**Published:** {video_data['published_at']}\n"
        output += f"**Video URL:** {video_data['url']}\n\n"
        
        if not comments:
            output += "No comments found or comments are disabled for this video.\n"
            return output
        
        output += f"### Top {len(comments)} Comments:\n\n"
        
        for i, comment_thread in enumerate(comments, 1):
            comment_data = format_comment_data(comment_thread)
            output += f"**{i}. {comment_data['author']}** (👍 {comment_data['like_count']})\n"
            output += f"{comment_data['text']}\n"
            output += f"*Posted: {comment_data['published_at']}*"
            if comment_data['reply_count'] > 0:
                output += f" | *Replies: {comment_data['reply_count']}*"
            output += "\n---\n\n"
        
        return output
        
    except HttpError as e:
        logger.error(f"YouTube API error: {e}")
        if e.resp.status == 403:
            return f"Error: YouTube API quota exceeded or comments are disabled for video '{video_id}'"
        elif e.resp.status == 404:
            return f"Error: Video '{video_id}' not found"
        else:
            return f"Error fetching comments from video '{video_id}': {str(e)}"
    except Exception as e:
        logger.error(f"Error fetching YouTube comments: {e}")
        return f"Error fetching comments from video '{video_id}': {str(e)}"

@mcp.tool()
def search_youtube_videos(query: str, limit: int = 10) -> str:
    """
    Search for YouTube videos.
    
    Args:
        query: Search query string
        limit: Number of results to return (default: 10, max: 50)
    
    Returns:
        Formatted string containing search results
    """
    try:
        youtube = get_youtube_client()
        
        # Validate and limit the number of results
        limit = min(max(1, limit), 50)
        
        # Search for videos
        search_response = youtube.search().list(
            q=query,
            part='id,snippet',
            maxResults=limit,
            type='video',
            order='relevance'
        ).execute()
        
        videos = search_response.get('items', [])
        
        if not videos:
            output = f"## Search Results for '{query}'\n\n"
            output += "No videos found for your search query.\n"
            return output
        
        # Get detailed video information including statistics
        video_ids = [video['id']['videoId'] for video in videos]
        videos_response = youtube.videos().list(
            part='snippet,statistics,contentDetails',
            id=','.join(video_ids)
        ).execute()
        
        detailed_videos = videos_response.get('items', [])
        
        # Format output
        output = f"## Search Results for '{query}'\n\n"
        output += f"Found {len(detailed_videos)} videos:\n\n"
        
        for i, video in enumerate(detailed_videos, 1):
            video_data = format_video_data(video)
            output += f"### {i}. {video_data['title']}\n"
            output += f"**Channel:** {video_data['channel_title']}\n"
            output += f"**Views:** {video_data['view_count']} | **Likes:** {video_data['like_count']} | **Comments:** {video_data['comment_count']}\n"
            output += f"**Published:** {video_data['published_at']}\n"
            output += f"**URL:** {video_data['url']}\n"
            if video_data['description']:
                output += f"**Description:** {video_data['description']}\n"
            output += "---\n\n"
        
        return output
        
    except HttpError as e:
        logger.error(f"YouTube API error: {e}")
        if e.resp.status == 403:
            return f"Error: YouTube API quota exceeded"
        else:
            return f"Error searching YouTube for '{query}': {str(e)}"
    except Exception as e:
        logger.error(f"Error searching YouTube: {e}")
        return f"Error searching YouTube for '{query}': {str(e)}"

@mcp.tool()
def get_video_info(video_id: str) -> str:
    """
    Get detailed information about a YouTube video.
    
    Args:
        video_id: YouTube video ID or full YouTube URL
    
    Returns:
        Formatted string containing video metadata
    """
    try:
        youtube = get_youtube_client()
        
        # Extract video ID from URL if needed
        video_id = extract_video_id(video_id)
        
        # Get video information
        video_response = youtube.videos().list(
            part='snippet,statistics,contentDetails,status',
            id=video_id
        ).execute()
        
        if not video_response['items']:
            return f"Video with ID '{video_id}' not found or is not accessible."
        
        video = video_response['items'][0]
        video_data = format_video_data(video)
        
        # Additional details
        content_details = video.get('contentDetails', {})
        status = video.get('status', {})
        
        # Format output
        output = f"## Video Information: {video_data['title']}\n\n"
        output += f"**Video ID:** {video_data['id']}\n"
        output += f"**Channel:** {video_data['channel_title']}\n"
        output += f"**Channel ID:** {video_data['channel_id']}\n"
        output += f"**Published:** {video_data['published_at']}\n"
        output += f"**Duration:** {content_details.get('duration', 'N/A')}\n"
        output += f"**Privacy Status:** {status.get('privacyStatus', 'N/A')}\n"
        output += f"**Upload Status:** {status.get('uploadStatus', 'N/A')}\n\n"
        
        output += "### Statistics:\n"
        output += f"- **Views:** {video_data['view_count']}\n"
        output += f"- **Likes:** {video_data['like_count']}\n"
        output += f"- **Comments:** {video_data['comment_count']}\n\n"
        
        output += f"**Video URL:** {video_data['url']}\n"
        output += f"**Thumbnail:** {video_data['thumbnail_url']}\n\n"
        
        if video_data['description']:
            output += f"### Description:\n{video_data['description']}\n\n"
        
        # Additional content details
        if content_details.get('definition'):
            output += f"**Definition:** {content_details['definition']}\n"
        if content_details.get('caption'):
            output += f"**Captions Available:** {content_details['caption']}\n"
        if content_details.get('licensedContent'):
            output += f"**Licensed Content:** {content_details['licensedContent']}\n"
        
        return output
        
    except HttpError as e:
        logger.error(f"YouTube API error: {e}")
        if e.resp.status == 403:
            return f"Error: YouTube API quota exceeded or video '{video_id}' is private"
        elif e.resp.status == 404:
            return f"Error: Video '{video_id}' not found"
        else:
            return f"Error fetching video info for '{video_id}': {str(e)}"
    except Exception as e:
        logger.error(f"Error fetching YouTube video info: {e}")
        return f"Error fetching video info for '{video_id}': {str(e)}"

if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="streamable-http", port=8002)
