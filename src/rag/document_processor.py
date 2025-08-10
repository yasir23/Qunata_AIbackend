"""
Document Processor for RAG system.

This module processes and chunks documents from MCP servers (Reddit, YouTube, GitHub)
for ingestion into the vector database.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes documents from various MCP servers for vector storage."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\'\/]', '', text)
        
        # Remove URLs (they're preserved in metadata)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        return text.strip()
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunks with metadata
        """
        if not text or len(text) <= self.chunk_size:
            return [{
                "content": text,
                "metadata": {**metadata, "chunk_index": 0, "total_chunks": 1}
            }]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at sentence boundaries
            if end < len(text):
                # Look for sentence endings within the last 200 characters
                sentence_end = text.rfind('.', start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind('!', start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind('?', start, end)
                
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_metadata = {
                    **metadata,
                    "chunk_index": chunk_index,
                    "chunk_start": start,
                    "chunk_end": end,
                    "chunk_length": len(chunk_text)
                }
                
                chunks.append({
                    "content": chunk_text,
                    "metadata": chunk_metadata
                })
                
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
        
        # Update total chunks count in all chunks
        for chunk in chunks:
            chunk["metadata"]["total_chunks"] = len(chunks)
        
        return chunks
    
    async def process_reddit_posts(self, posts_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process Reddit posts for vector storage.
        
        Args:
            posts_data: List of Reddit post data from MCP server
            
        Returns:
            List of processed document chunks
        """
        processed_docs = []
        
        for post in posts_data:
            try:
                # Extract and clean content
                title = post.get('title', '')
                selftext = post.get('selftext', '')
                
                # Combine title and content
                content = f"Title: {title}\n\nContent: {selftext}" if selftext else f"Title: {title}"
                content = self._clean_text(content)
                
                if not content.strip():
                    continue
                
                # Create metadata
                metadata = {
                    "source": "reddit",
                    "type": "post",
                    "post_id": post.get('id', ''),
                    "title": title,
                    "author": post.get('author', ''),
                    "subreddit": post.get('subreddit', ''),
                    "score": post.get('score', 0),
                    "num_comments": post.get('num_comments', 0),
                    "created_utc": post.get('created_utc', ''),
                    "url": post.get('url', ''),
                    "permalink": post.get('permalink', ''),
                    "processed_at": datetime.now().isoformat()
                }
                
                # Chunk the content
                chunks = self._chunk_text(content, metadata)
                processed_docs.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing Reddit post {post.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_docs)} chunks from {len(posts_data)} Reddit posts")
        return processed_docs
    
    async def process_reddit_comments(self, comments_data: List[Dict[str, Any]], post_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process Reddit comments for vector storage.
        
        Args:
            comments_data: List of Reddit comment data from MCP server
            post_context: Context about the parent post
            
        Returns:
            List of processed document chunks
        """
        processed_docs = []
        
        for comment in comments_data:
            try:
                # Extract and clean content
                body = comment.get('body', '')
                if not body or body in ['[deleted]', '[removed]']:
                    continue
                
                content = self._clean_text(body)
                if not content.strip():
                    continue
                
                # Create metadata
                metadata = {
                    "source": "reddit",
                    "type": "comment",
                    "comment_id": comment.get('id', ''),
                    "author": comment.get('author', ''),
                    "score": comment.get('score', 0),
                    "created_utc": comment.get('created_utc', ''),
                    "parent_id": comment.get('parent_id', ''),
                    "processed_at": datetime.now().isoformat()
                }
                
                # Add post context if available
                if post_context:
                    metadata.update({
                        "post_title": post_context.get('title', ''),
                        "post_id": post_context.get('id', ''),
                        "subreddit": post_context.get('subreddit', '')
                    })
                
                # Chunk the content
                chunks = self._chunk_text(content, metadata)
                processed_docs.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing Reddit comment {comment.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_docs)} chunks from {len(comments_data)} Reddit comments")
        return processed_docs
    
    async def process_youtube_videos(self, videos_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process YouTube video data for vector storage.
        
        Args:
            videos_data: List of YouTube video data from MCP server
            
        Returns:
            List of processed document chunks
        """
        processed_docs = []
        
        for video in videos_data:
            try:
                # Extract and clean content
                title = video.get('title', '')
                description = video.get('description', '')
                
                # Combine title and description
                content = f"Title: {title}\n\nDescription: {description}" if description else f"Title: {title}"
                content = self._clean_text(content)
                
                if not content.strip():
                    continue
                
                # Create metadata
                metadata = {
                    "source": "youtube",
                    "type": "video",
                    "video_id": video.get('id', ''),
                    "title": title,
                    "channel_title": video.get('channel_title', ''),
                    "channel_id": video.get('channel_id', ''),
                    "published_at": video.get('published_at', ''),
                    "view_count": video.get('view_count', 0),
                    "like_count": video.get('like_count', 0),
                    "comment_count": video.get('comment_count', 0),
                    "duration": video.get('duration', ''),
                    "url": video.get('url', ''),
                    "thumbnail_url": video.get('thumbnail_url', ''),
                    "processed_at": datetime.now().isoformat()
                }
                
                # Chunk the content
                chunks = self._chunk_text(content, metadata)
                processed_docs.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing YouTube video {video.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_docs)} chunks from {len(videos_data)} YouTube videos")
        return processed_docs
    
    async def process_youtube_comments(self, comments_data: List[Dict[str, Any]], video_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process YouTube comments for vector storage.
        
        Args:
            comments_data: List of YouTube comment data from MCP server
            video_context: Context about the parent video
            
        Returns:
            List of processed document chunks
        """
        processed_docs = []
        
        for comment in comments_data:
            try:
                # Extract and clean content
                text = comment.get('text', '')
                if not text:
                    continue
                
                content = self._clean_text(text)
                if not content.strip():
                    continue
                
                # Create metadata
                metadata = {
                    "source": "youtube",
                    "type": "comment",
                    "comment_id": comment.get('id', ''),
                    "author": comment.get('author', ''),
                    "like_count": comment.get('like_count', 0),
                    "published_at": comment.get('published_at', ''),
                    "processed_at": datetime.now().isoformat()
                }
                
                # Add video context if available
                if video_context:
                    metadata.update({
                        "video_title": video_context.get('title', ''),
                        "video_id": video_context.get('id', ''),
                        "channel_title": video_context.get('channel_title', '')
                    })
                
                # Chunk the content
                chunks = self._chunk_text(content, metadata)
                processed_docs.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing YouTube comment {comment.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_docs)} chunks from {len(comments_data)} YouTube comments")
        return processed_docs
    
    async def process_github_issues(self, issues_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process GitHub issues for vector storage.
        
        Args:
            issues_data: List of GitHub issue data from MCP server
            
        Returns:
            List of processed document chunks
        """
        processed_docs = []
        
        for issue in issues_data:
            try:
                # Extract and clean content
                title = issue.get('title', '')
                body = issue.get('body', '')
                
                # Combine title and body
                content = f"Title: {title}\n\nDescription: {body}" if body else f"Title: {title}"
                content = self._clean_text(content)
                
                if not content.strip():
                    continue
                
                # Create metadata
                metadata = {
                    "source": "github",
                    "type": "issue",
                    "issue_id": issue.get('id', ''),
                    "issue_number": issue.get('number', ''),
                    "title": title,
                    "state": issue.get('state', ''),
                    "author": issue.get('user', {}).get('login', '') if issue.get('user') else '',
                    "assignees": [assignee.get('login', '') for assignee in issue.get('assignees', [])],
                    "labels": [label.get('name', '') for label in issue.get('labels', [])],
                    "created_at": issue.get('created_at', ''),
                    "updated_at": issue.get('updated_at', ''),
                    "html_url": issue.get('html_url', ''),
                    "comments": issue.get('comments', 0),
                    "milestone": issue.get('milestone', {}).get('title', '') if issue.get('milestone') else '',
                    "processed_at": datetime.now().isoformat()
                }
                
                # Extract repository info from URL
                if issue.get('html_url'):
                    url_parts = issue['html_url'].split('/')
                    if len(url_parts) >= 5:
                        metadata['repository'] = f"{url_parts[3]}/{url_parts[4]}"
                
                # Chunk the content
                chunks = self._chunk_text(content, metadata)
                processed_docs.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing GitHub issue {issue.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_docs)} chunks from {len(issues_data)} GitHub issues")
        return processed_docs
    
    async def process_github_repositories(self, repos_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process GitHub repository data for vector storage.
        
        Args:
            repos_data: List of GitHub repository data from MCP server
            
        Returns:
            List of processed document chunks
        """
        processed_docs = []
        
        for repo in repos_data:
            try:
                # Extract and clean content
                name = repo.get('name', '')
                description = repo.get('description', '')
                readme = repo.get('readme', '')  # If available
                
                # Combine available content
                content_parts = [f"Repository: {name}"]
                if description:
                    content_parts.append(f"Description: {description}")
                if readme:
                    content_parts.append(f"README: {readme}")
                
                content = "\n\n".join(content_parts)
                content = self._clean_text(content)
                
                if not content.strip():
                    continue
                
                # Create metadata
                metadata = {
                    "source": "github",
                    "type": "repository",
                    "repo_id": repo.get('id', ''),
                    "name": name,
                    "full_name": repo.get('full_name', ''),
                    "description": description,
                    "language": repo.get('language', ''),
                    "stars": repo.get('stargazers_count', 0),
                    "forks": repo.get('forks_count', 0),
                    "watchers": repo.get('watchers_count', 0),
                    "open_issues": repo.get('open_issues_count', 0),
                    "created_at": repo.get('created_at', ''),
                    "updated_at": repo.get('updated_at', ''),
                    "html_url": repo.get('html_url', ''),
                    "clone_url": repo.get('clone_url', ''),
                    "topics": repo.get('topics', []),
                    "license": repo.get('license', {}).get('name', '') if repo.get('license') else '',
                    "processed_at": datetime.now().isoformat()
                }
                
                # Chunk the content
                chunks = self._chunk_text(content, metadata)
                processed_docs.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing GitHub repository {repo.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_docs)} chunks from {len(repos_data)} GitHub repositories")
        return processed_docs
    
    async def process_search_results(self, search_results: List[Dict[str, Any]], source: str = "web") -> List[Dict[str, Any]]:
        """
        Process web search results for vector storage.
        
        Args:
            search_results: List of search result data
            source: Source of the search results
            
        Returns:
            List of processed document chunks
        """
        processed_docs = []
        
        for result in search_results:
            try:
                # Extract and clean content
                title = result.get('title', '')
                content = result.get('content', '')
                snippet = result.get('snippet', '')
                
                # Use content if available, otherwise use snippet
                text_content = content or snippet
                if not text_content:
                    continue
                
                # Combine title and content
                full_content = f"Title: {title}\n\nContent: {text_content}"
                full_content = self._clean_text(full_content)
                
                if not full_content.strip():
                    continue
                
                # Create metadata
                metadata = {
                    "source": source,
                    "type": "search_result",
                    "title": title,
                    "url": result.get('url', ''),
                    "domain": result.get('domain', ''),
                    "published_date": result.get('published_date', ''),
                    "score": result.get('score', 0),
                    "query": result.get('query', ''),
                    "processed_at": datetime.now().isoformat()
                }
                
                # Chunk the content
                chunks = self._chunk_text(full_content, metadata)
                processed_docs.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error processing search result {result.get('url', 'unknown')}: {e}")
                continue
        
        logger.info(f"Processed {len(processed_docs)} chunks from {len(search_results)} search results")
        return processed_docs
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about the document processor configuration."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "supported_sources": ["reddit", "youtube", "github", "web"],
            "supported_types": [
                "reddit_post", "reddit_comment",
                "youtube_video", "youtube_comment", 
                "github_issue", "github_repository",
                "search_result"
            ]
        }
