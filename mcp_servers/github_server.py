#!/usr/bin/env python3
"""
GitHub MCP Server for extracting issues, repositories, and comments data.

This server provides tools for:
- Fetching issues from repositories
- Searching repositories
- Getting issue comments and details
- Repository information and statistics

Requires GitHub API credentials:
- GITHUB_TOKEN
"""

import os
import asyncio
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP
import requests
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("GitHub")

def get_github_headers() -> Dict[str, str]:
    """Get GitHub API headers with authentication."""
    token = os.getenv("GITHUB_TOKEN")
    
    if not token:
        raise ValueError("GitHub API token not found. Please set GITHUB_TOKEN environment variable.")
    
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "DeepResearch/1.0"
    }

def format_issue_data(issue: Dict[str, Any]) -> Dict[str, Any]:
    """Format GitHub issue data for output."""
    return {
        "id": issue.get("id"),
        "number": issue.get("number"),
        "title": issue.get("title", ""),
        "body": (issue.get("body", "") or "")[:1000] + "..." if len(issue.get("body", "") or "") > 1000 else (issue.get("body", "") or ""),
        "state": issue.get("state", ""),
        "author": issue.get("user", {}).get("login", ""),
        "assignees": [assignee.get("login", "") for assignee in issue.get("assignees", [])],
        "labels": [label.get("name", "") for label in issue.get("labels", [])],
        "comments_count": issue.get("comments", 0),
        "created_at": issue.get("created_at", ""),
        "updated_at": issue.get("updated_at", ""),
        "closed_at": issue.get("closed_at"),
        "url": issue.get("html_url", ""),
        "api_url": issue.get("url", ""),
        "milestone": issue.get("milestone", {}).get("title") if issue.get("milestone") else None,
        "locked": issue.get("locked", False)
    }

def format_repository_data(repo: Dict[str, Any]) -> Dict[str, Any]:
    """Format GitHub repository data for output."""
    return {
        "id": repo.get("id"),
        "name": repo.get("name", ""),
        "full_name": repo.get("full_name", ""),
        "description": repo.get("description", ""),
        "owner": repo.get("owner", {}).get("login", ""),
        "private": repo.get("private", False),
        "html_url": repo.get("html_url", ""),
        "clone_url": repo.get("clone_url", ""),
        "language": repo.get("language"),
        "stargazers_count": repo.get("stargazers_count", 0),
        "watchers_count": repo.get("watchers_count", 0),
        "forks_count": repo.get("forks_count", 0),
        "open_issues_count": repo.get("open_issues_count", 0),
        "created_at": repo.get("created_at", ""),
        "updated_at": repo.get("updated_at", ""),
        "pushed_at": repo.get("pushed_at", ""),
        "size": repo.get("size", 0),
        "default_branch": repo.get("default_branch", ""),
        "topics": repo.get("topics", []),
        "license": repo.get("license", {}).get("name") if repo.get("license") else None
    }

def format_comment_data(comment: Dict[str, Any]) -> Dict[str, Any]:
    """Format GitHub comment data for output."""
    return {
        "id": comment.get("id"),
        "author": comment.get("user", {}).get("login", ""),
        "body": (comment.get("body", "") or "")[:1000] + "..." if len(comment.get("body", "") or "") > 1000 else (comment.get("body", "") or ""),
        "created_at": comment.get("created_at", ""),
        "updated_at": comment.get("updated_at", ""),
        "url": comment.get("html_url", ""),
        "api_url": comment.get("url", ""),
        "author_association": comment.get("author_association", "")
    }

@mcp.tool()
def get_repository_issues(owner: str, repo: str, state: str = "open", limit: int = 20, sort: str = "created", direction: str = "desc") -> str:
    """
    Fetch issues from a specific GitHub repository.
    
    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        state: Issue state - 'open', 'closed', or 'all' (default: 'open')
        limit: Number of issues to fetch (default: 20, max: 100)
        sort: Sort by - 'created', 'updated', 'comments' (default: 'created')
        direction: Sort direction - 'asc' or 'desc' (default: 'desc')
    
    Returns:
        Formatted string containing issue information
    """
    try:
        headers = get_github_headers()
        
        # Validate and limit the number of issues
        limit = min(max(1, limit), 100)
        
        # Build API URL
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        params = {
            "state": state,
            "per_page": limit,
            "sort": sort,
            "direction": direction
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        issues = response.json()
        
        # Filter out pull requests (GitHub API includes PRs in issues endpoint)
        issues = [issue for issue in issues if not issue.get("pull_request")]
        
        if not issues:
            return f"No issues found in {owner}/{repo} with state '{state}'"
        
        # Format issues
        formatted_issues = []
        for issue in issues:
            issue_data = format_issue_data(issue)
            formatted_issues.append(issue_data)
        
        # Create formatted output
        output = f"## Issues from {owner}/{repo} (state: {state}, sorted by {sort} {direction})\n\n"
        
        for i, issue in enumerate(formatted_issues, 1):
            output += f"### {i}. #{issue['number']}: {issue['title']}\n"
            output += f"**Author:** {issue['author']} | **State:** {issue['state']} | **Comments:** {issue['comments_count']}\n"
            output += f"**Created:** {issue['created_at']} | **Updated:** {issue['updated_at']}\n"
            if issue['assignees']:
                output += f"**Assignees:** {', '.join(issue['assignees'])}\n"
            if issue['labels']:
                output += f"**Labels:** {', '.join(issue['labels'])}\n"
            if issue['milestone']:
                output += f"**Milestone:** {issue['milestone']}\n"
            output += f"**URL:** {issue['url']}\n"
            if issue['body']:
                output += f"**Description:** {issue['body']}\n"
            output += "---\n\n"
        
        return output
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching GitHub issues: {e}")
        return f"Error fetching issues from {owner}/{repo}: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"Unexpected error fetching issues: {str(e)}"

@mcp.tool()
def get_issue_comments(owner: str, repo: str, issue_number: int, limit: int = 30) -> str:
    """
    Fetch comments from a specific GitHub issue.
    
    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
        issue_number: Issue number
        limit: Number of comments to fetch (default: 30, max: 100)
    
    Returns:
        Formatted string containing comment information
    """
    try:
        headers = get_github_headers()
        
        # Validate and limit the number of comments
        limit = min(max(1, limit), 100)
        
        # First, get issue information
        issue_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        issue_response = requests.get(issue_url, headers=headers)
        issue_response.raise_for_status()
        
        issue_data = format_issue_data(issue_response.json())
        
        # Get comments
        comments_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}/comments"
        params = {"per_page": limit}
        
        comments_response = requests.get(comments_url, headers=headers, params=params)
        comments_response.raise_for_status()
        
        comments = comments_response.json()
        
        # Format output
        output = f"## Comments for Issue #{issue_data['number']}: {issue_data['title']}\n"
        output += f"**Repository:** {owner}/{repo}\n"
        output += f"**Author:** {issue_data['author']} | **State:** {issue_data['state']}\n"
        output += f"**Created:** {issue_data['created_at']} | **Updated:** {issue_data['updated_at']}\n"
        output += f"**Issue URL:** {issue_data['url']}\n"
        if issue_data['body']:
            output += f"**Issue Description:** {issue_data['body']}\n"
        output += f"\n**Total Comments:** {len(comments)}\n\n"
        
        if not comments:
            output += "No comments found for this issue.\n"
            return output
        
        output += f"### Comments:\n\n"
        
        for i, comment in enumerate(comments, 1):
            comment_data = format_comment_data(comment)
            output += f"#### Comment {i} by {comment_data['author']}\n"
            output += f"**Posted:** {comment_data['created_at']}\n"
            if comment_data['updated_at'] != comment_data['created_at']:
                output += f"**Updated:** {comment_data['updated_at']}\n"
            output += f"**Association:** {comment_data['author_association']}\n"
            output += f"**Content:** {comment_data['body']}\n"
            output += f"**URL:** {comment_data['url']}\n"
            output += "---\n\n"
        
        return output
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching issue comments: {e}")
        return f"Error fetching comments for issue #{issue_number} in {owner}/{repo}: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"Unexpected error fetching comments: {str(e)}"

@mcp.tool()
def search_repositories(query: str, sort: str = "stars", order: str = "desc", limit: int = 20) -> str:
    """
    Search for GitHub repositories.
    
    Args:
        query: Search query string
        sort: Sort by - 'stars', 'forks', 'help-wanted-issues', 'updated' (default: 'stars')
        order: Sort order - 'asc' or 'desc' (default: 'desc')
        limit: Number of repositories to return (default: 20, max: 100)
    
    Returns:
        Formatted string containing repository information
    """
    try:
        headers = get_github_headers()
        
        # Validate and limit the number of results
        limit = min(max(1, limit), 100)
        
        # Build API URL
        url = "https://api.github.com/search/repositories"
        params = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": limit
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        repositories = data.get("items", [])
        
        if not repositories:
            return f"No repositories found for query: '{query}'"
        
        # Format repositories
        formatted_repos = []
        for repo in repositories:
            repo_data = format_repository_data(repo)
            formatted_repos.append(repo_data)
        
        # Create formatted output
        output = f"## Repository Search Results for: '{query}'\n"
        output += f"**Total Results:** {data.get('total_count', 0)} (showing top {len(repositories)})\n"
        output += f"**Sorted by:** {sort} ({order})\n\n"
        
        for i, repo in enumerate(formatted_repos, 1):
            output += f"### {i}. {repo['full_name']}\n"
            output += f"**Description:** {repo['description'] or 'No description'}\n"
            output += f"**Language:** {repo['language'] or 'Not specified'}\n"
            output += f"**Stars:** ⭐ {repo['stargazers_count']} | **Forks:** 🍴 {repo['forks_count']} | **Issues:** 🐛 {repo['open_issues_count']}\n"
            output += f"**Owner:** {repo['owner']} | **Private:** {'Yes' if repo['private'] else 'No'}\n"
            output += f"**Created:** {repo['created_at']} | **Updated:** {repo['updated_at']}\n"
            if repo['topics']:
                output += f"**Topics:** {', '.join(repo['topics'])}\n"
            if repo['license']:
                output += f"**License:** {repo['license']}\n"
            output += f"**URL:** {repo['html_url']}\n"
            output += "---\n\n"
        
        return output
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching repositories: {e}")
        return f"Error searching repositories: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"Unexpected error searching repositories: {str(e)}"

@mcp.tool()
def get_repository_info(owner: str, repo: str) -> str:
    """
    Get detailed information about a specific GitHub repository.
    
    Args:
        owner: Repository owner (username or organization)
        repo: Repository name
    
    Returns:
        Formatted string containing detailed repository information
    """
    try:
        headers = get_github_headers()
        
        # Get repository information
        url = f"https://api.github.com/repos/{owner}/{repo}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        repo_data = format_repository_data(response.json())
        
        # Get additional statistics
        contributors_url = f"https://api.github.com/repos/{owner}/{repo}/contributors"
        contributors_response = requests.get(contributors_url, headers=headers, params={"per_page": 5})
        contributors = contributors_response.json() if contributors_response.status_code == 200 else []
        
        releases_url = f"https://api.github.com/repos/{owner}/{repo}/releases"
        releases_response = requests.get(releases_url, headers=headers, params={"per_page": 3})
        releases = releases_response.json() if releases_response.status_code == 200 else []
        
        # Format output
        output = f"## Repository Information: {repo_data['full_name']}\n\n"
        output += f"**Description:** {repo_data['description'] or 'No description provided'}\n"
        output += f"**Owner:** {repo_data['owner']}\n"
        output += f"**Primary Language:** {repo_data['language'] or 'Not specified'}\n"
        output += f"**Visibility:** {'Private' if repo_data['private'] else 'Public'}\n\n"
        
        output += f"### Statistics\n"
        output += f"- **Stars:** ⭐ {repo_data['stargazers_count']:,}\n"
        output += f"- **Watchers:** 👀 {repo_data['watchers_count']:,}\n"
        output += f"- **Forks:** 🍴 {repo_data['forks_count']:,}\n"
        output += f"- **Open Issues:** 🐛 {repo_data['open_issues_count']:,}\n"
        output += f"- **Size:** 📦 {repo_data['size']:,} KB\n\n"
        
        output += f"### Dates\n"
        output += f"- **Created:** {repo_data['created_at']}\n"
        output += f"- **Last Updated:** {repo_data['updated_at']}\n"
        output += f"- **Last Push:** {repo_data['pushed_at']}\n\n"
        
        if repo_data['topics']:
            output += f"### Topics\n"
            output += f"{', '.join(repo_data['topics'])}\n\n"
        
        if repo_data['license']:
            output += f"### License\n"
            output += f"{repo_data['license']}\n\n"
        
        if contributors and isinstance(contributors, list):
            output += f"### Top Contributors\n"
            for contributor in contributors[:5]:
                if isinstance(contributor, dict):
                    output += f"- **{contributor.get('login', 'Unknown')}** ({contributor.get('contributions', 0)} contributions)\n"
            output += "\n"
        
        if releases and isinstance(releases, list):
            output += f"### Recent Releases\n"
            for release in releases[:3]:
                if isinstance(release, dict):
                    output += f"- **{release.get('tag_name', 'Unknown')}** - {release.get('name', 'Unnamed')} ({release.get('published_at', 'Unknown date')})\n"
            output += "\n"
        
        output += f"### URLs\n"
        output += f"- **Repository:** {repo_data['html_url']}\n"
        output += f"- **Clone URL:** {repo_data['clone_url']}\n"
        output += f"- **Default Branch:** {repo_data['default_branch']}\n"
        
        return output
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching repository info: {e}")
        return f"Error fetching repository information for {owner}/{repo}: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"Unexpected error fetching repository information: {str(e)}"

@mcp.tool()
def search_issues(query: str, sort: str = "created", order: str = "desc", limit: int = 20) -> str:
    """
    Search for GitHub issues across all repositories.
    
    Args:
        query: Search query string (can include qualifiers like 'repo:owner/name', 'state:open', 'label:bug')
        sort: Sort by - 'created', 'updated', 'comments' (default: 'created')
        order: Sort order - 'asc' or 'desc' (default: 'desc')
        limit: Number of issues to return (default: 20, max: 100)
    
    Returns:
        Formatted string containing issue information
    """
    try:
        headers = get_github_headers()
        
        # Validate and limit the number of results
        limit = min(max(1, limit), 100)
        
        # Build API URL
        url = "https://api.github.com/search/issues"
        params = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": limit
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        issues = data.get("items", [])
        
        # Filter out pull requests
        issues = [issue for issue in issues if not issue.get("pull_request")]
        
        if not issues:
            return f"No issues found for query: '{query}'"
        
        # Format issues
        formatted_issues = []
        for issue in issues:
            issue_data = format_issue_data(issue)
            # Add repository information
            repo_url = issue.get("repository_url", "")
            if repo_url:
                repo_parts = repo_url.split("/")
                if len(repo_parts) >= 2:
                    issue_data["repository"] = f"{repo_parts[-2]}/{repo_parts[-1]}"
            formatted_issues.append(issue_data)
        
        # Create formatted output
        output = f"## Issue Search Results for: '{query}'\n"
        output += f"**Total Results:** {data.get('total_count', 0)} (showing top {len(issues)})\n"
        output += f"**Sorted by:** {sort} ({order})\n\n"
        
        for i, issue in enumerate(formatted_issues, 1):
            output += f"### {i}. #{issue['number']}: {issue['title']}\n"
            if issue.get('repository'):
                output += f"**Repository:** {issue['repository']}\n"
            output += f"**Author:** {issue['author']} | **State:** {issue['state']} | **Comments:** {issue['comments_count']}\n"
            output += f"**Created:** {issue['created_at']} | **Updated:** {issue['updated_at']}\n"
            if issue['assignees']:
                output += f"**Assignees:** {', '.join(issue['assignees'])}\n"
            if issue['labels']:
                output += f"**Labels:** {', '.join(issue['labels'])}\n"
            output += f"**URL:** {issue['url']}\n"
            if issue['body']:
                output += f"**Description:** {issue['body']}\n"
            output += "---\n\n"
        
        return output
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching issues: {e}")
        return f"Error searching issues: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"Unexpected error searching issues: {str(e)}"

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
