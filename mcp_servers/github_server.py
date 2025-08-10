#!/usr/bin/env python3
"""
GitHub Issues MCP Server for development workflow integration.

This server provides tools for:
- Fetching issues from repositories
- Creating new issues
- Searching repositories and issues
- Extracting project insights and analytics

Requires GitHub API credentials:
- GITHUB_TOKEN
- GITHUB_ORG (optional, for organization-specific operations)
"""

import os
import asyncio
from typing import Optional, List, Dict, Any, Union
from mcp.server.fastmcp import FastMCP
import requests
from datetime import datetime, timedelta
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("GitHub")

class GitHubClient:
    """GitHub API client with authentication and rate limiting."""
    
    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.org = os.getenv("GITHUB_ORG")
        self.base_url = "https://api.github.com"
        
        if not self.token:
            raise ValueError("GitHub API token not found. Please set GITHUB_TOKEN environment variable.")
        
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "DeepResearch-GitHub-MCP/1.0"
        }
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated request to GitHub API with error handling."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=data,
                timeout=30
            )
            
            # Check rate limiting
            if response.status_code == 403 and 'rate limit' in response.text.lower():
                reset_time = response.headers.get('X-RateLimit-Reset', '')
                logger.warning(f"GitHub API rate limit exceeded. Reset time: {reset_time}")
                raise Exception("GitHub API rate limit exceeded. Please try again later.")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"GitHub API request failed: {e}")
            raise Exception(f"GitHub API request failed: {str(e)}")
    
    def get_issues(self, repo: str, state: str = "open", labels: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch issues from a repository."""
        params = {
            "state": state,
            "per_page": min(limit, 100),
            "sort": "updated",
            "direction": "desc"
        }
        
        if labels:
            params["labels"] = labels
        
        endpoint = f"repos/{repo}/issues"
        return self._make_request("GET", endpoint, params=params)
    
    def create_issue(self, repo: str, title: str, body: Optional[str] = None, labels: Optional[List[str]] = None, assignees: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a new issue in a repository."""
        data = {"title": title}
        
        if body:
            data["body"] = body
        if labels:
            data["labels"] = labels
        if assignees:
            data["assignees"] = assignees
        
        endpoint = f"repos/{repo}/issues"
        return self._make_request("POST", endpoint, data=data)
    
    def search_repositories(self, query: str, sort: str = "stars", limit: int = 20) -> Dict[str, Any]:
        """Search for repositories."""
        params = {
            "q": query,
            "sort": sort,
            "order": "desc",
            "per_page": min(limit, 100)
        }
        
        return self._make_request("GET", "search/repositories", params=params)
    
    def search_issues(self, query: str, sort: str = "updated", limit: int = 20) -> Dict[str, Any]:
        """Search for issues across repositories."""
        params = {
            "q": query,
            "sort": sort,
            "order": "desc",
            "per_page": min(limit, 100)
        }
        
        return self._make_request("GET", "search/issues", params=params)
    
    def get_repository_info(self, repo: str) -> Dict[str, Any]:
        """Get detailed information about a repository."""
        return self._make_request("GET", f"repos/{repo}")
    
    def get_repository_stats(self, repo: str) -> Dict[str, Any]:
        """Get repository statistics and insights."""
        repo_info = self.get_repository_info(repo)
        
        # Get additional stats
        contributors = self._make_request("GET", f"repos/{repo}/contributors", params={"per_page": 10})
        languages = self._make_request("GET", f"repos/{repo}/languages")
        
        # Get recent activity (commits, issues, PRs)
        recent_commits = self._make_request("GET", f"repos/{repo}/commits", params={"per_page": 10})
        recent_issues = self._make_request("GET", f"repos/{repo}/issues", params={"per_page": 10, "state": "all"})
        
        return {
            "repository": repo_info,
            "contributors": contributors,
            "languages": languages,
            "recent_commits": recent_commits,
            "recent_issues": recent_issues
        }

# Global GitHub client
github_client = None

def get_github_client() -> GitHubClient:
    """Initialize and return a GitHub client."""
    global github_client
    if github_client is None:
        github_client = GitHubClient()
    return github_client

def format_issue_data(issue: Dict[str, Any]) -> Dict[str, Any]:
    """Format GitHub issue data for output."""
    return {
        "id": issue.get("id"),
        "number": issue.get("number"),
        "title": issue.get("title", ""),
        "body": issue.get("body", "")[:500] + "..." if len(issue.get("body", "")) > 500 else issue.get("body", ""),
        "state": issue.get("state"),
        "author": issue.get("user", {}).get("login", "unknown"),
        "assignees": [assignee.get("login") for assignee in issue.get("assignees", [])],
        "labels": [label.get("name") for label in issue.get("labels", [])],
        "created_at": issue.get("created_at"),
        "updated_at": issue.get("updated_at"),
        "html_url": issue.get("html_url"),
        "comments": issue.get("comments", 0),
        "milestone": issue.get("milestone", {}).get("title") if issue.get("milestone") else None
    }

def format_repository_data(repo: Dict[str, Any]) -> Dict[str, Any]:
    """Format GitHub repository data for output."""
    return {
        "id": repo.get("id"),
        "name": repo.get("name"),
        "full_name": repo.get("full_name"),
        "description": repo.get("description", ""),
        "language": repo.get("language"),
        "stars": repo.get("stargazers_count", 0),
        "forks": repo.get("forks_count", 0),
        "watchers": repo.get("watchers_count", 0),
        "open_issues": repo.get("open_issues_count", 0),
        "created_at": repo.get("created_at"),
        "updated_at": repo.get("updated_at"),
        "html_url": repo.get("html_url"),
        "clone_url": repo.get("clone_url"),
        "topics": repo.get("topics", []),
        "license": repo.get("license", {}).get("name") if repo.get("license") else None
    }

@mcp.tool()
def fetch_issues(repo: str, state: str = "open", labels: Optional[str] = None, limit: int = 20) -> str:
    """
    Fetch issues from a GitHub repository.
    
    Args:
        repo: Repository name in format 'owner/repo'
        state: Issue state ('open', 'closed', 'all')
        labels: Comma-separated list of labels to filter by
        limit: Number of issues to return (default: 20, max: 100)
    
    Returns:
        Formatted string containing issue information
    """
    try:
        client = get_github_client()
        
        # Validate and limit the number of results
        limit = min(max(1, limit), 100)
        
        # Fetch issues
        issues = client.get_issues(repo, state=state, labels=labels, limit=limit)
        
        # Format output
        output = f"## Issues from {repo} (State: {state})\n\n"
        
        if not issues:
            output += f"No {state} issues found"
            if labels:
                output += f" with labels: {labels}"
            output += ".\n"
            return output
        
        output += f"Found {len(issues)} issues:\n\n"
        
        for i, issue in enumerate(issues, 1):
            issue_data = format_issue_data(issue)
            output += f"### {i}. #{issue_data['number']}: {issue_data['title']}\n"
            output += f"**Author:** {issue_data['author']} | **State:** {issue_data['state']}\n"
            output += f"**Created:** {issue_data['created_at']} | **Updated:** {issue_data['updated_at']}\n"
            
            if issue_data['labels']:
                output += f"**Labels:** {', '.join(issue_data['labels'])}\n"
            
            if issue_data['assignees']:
                output += f"**Assignees:** {', '.join(issue_data['assignees'])}\n"
            
            if issue_data['milestone']:
                output += f"**Milestone:** {issue_data['milestone']}\n"
            
            output += f"**Comments:** {issue_data['comments']}\n"
            output += f"**URL:** {issue_data['html_url']}\n"
            
            if issue_data['body']:
                output += f"**Description:** {issue_data['body']}\n"
            
            output += "---\n\n"
        
        return output
        
    except Exception as e:
        logger.error(f"Error fetching GitHub issues: {e}")
        return f"Error fetching issues from {repo}: {str(e)}"

@mcp.tool()
def create_issue(repo: str, title: str, body: Optional[str] = None, labels: Optional[str] = None, assignees: Optional[str] = None) -> str:
    """
    Create a new issue in a GitHub repository.
    
    Args:
        repo: Repository name in format 'owner/repo'
        title: Issue title
        body: Issue description/body (optional)
        labels: Comma-separated list of labels (optional)
        assignees: Comma-separated list of usernames to assign (optional)
    
    Returns:
        Formatted string with created issue information
    """
    try:
        client = get_github_client()
        
        # Parse labels and assignees
        labels_list = [label.strip() for label in labels.split(",")] if labels else None
        assignees_list = [assignee.strip() for assignee in assignees.split(",")] if assignees else None
        
        # Create issue
        issue = client.create_issue(repo, title, body=body, labels=labels_list, assignees=assignees_list)
        issue_data = format_issue_data(issue)
        
        # Format output
        output = f"## ✅ Issue Created Successfully\n\n"
        output += f"**Repository:** {repo}\n"
        output += f"**Issue Number:** #{issue_data['number']}\n"
        output += f"**Title:** {issue_data['title']}\n"
        output += f"**Author:** {issue_data['author']}\n"
        output += f"**State:** {issue_data['state']}\n"
        output += f"**Created:** {issue_data['created_at']}\n"
        output += f"**URL:** {issue_data['html_url']}\n"
        
        if issue_data['labels']:
            output += f"**Labels:** {', '.join(issue_data['labels'])}\n"
        
        if issue_data['assignees']:
            output += f"**Assignees:** {', '.join(issue_data['assignees'])}\n"
        
        if issue_data['body']:
            output += f"\n**Description:**\n{issue_data['body']}\n"
        
        return output
        
    except Exception as e:
        logger.error(f"Error creating GitHub issue: {e}")
        return f"Error creating issue in {repo}: {str(e)}"

@mcp.tool()
def search_repositories(query: str, sort: str = "stars", limit: int = 20) -> str:
    """
    Search for GitHub repositories.
    
    Args:
        query: Search query (can include qualifiers like 'language:python', 'stars:>100')
        sort: Sort by 'stars', 'forks', 'help-wanted-issues', 'updated'
        limit: Number of results to return (default: 20, max: 100)
    
    Returns:
        Formatted string containing repository search results
    """
    try:
        client = get_github_client()
        
        # Validate and limit the number of results
        limit = min(max(1, limit), 100)
        
        # Search repositories
        search_results = client.search_repositories(query, sort=sort, limit=limit)
        repositories = search_results.get("items", [])
        
        # Format output
        output = f"## Repository Search Results for '{query}'\n\n"
        output += f"**Total Results:** {search_results.get('total_count', 0)} (showing {len(repositories)})\n"
        output += f"**Sorted by:** {sort}\n\n"
        
        if not repositories:
            output += "No repositories found for your search query.\n"
            return output
        
        for i, repo in enumerate(repositories, 1):
            repo_data = format_repository_data(repo)
            output += f"### {i}. {repo_data['full_name']}\n"
            output += f"**Description:** {repo_data['description']}\n"
            output += f"**Language:** {repo_data['language']} | **Stars:** ⭐ {repo_data['stars']} | **Forks:** 🍴 {repo_data['forks']}\n"
            output += f"**Open Issues:** {repo_data['open_issues']} | **Updated:** {repo_data['updated_at']}\n"
            output += f"**URL:** {repo_data['html_url']}\n"
            
            if repo_data['topics']:
                output += f"**Topics:** {', '.join(repo_data['topics'])}\n"
            
            if repo_data['license']:
                output += f"**License:** {repo_data['license']}\n"
            
            output += "---\n\n"
        
        return output
        
    except Exception as e:
        logger.error(f"Error searching GitHub repositories: {e}")
        return f"Error searching repositories for '{query}': {str(e)}"

@mcp.tool()
def search_issues(query: str, sort: str = "updated", limit: int = 20) -> str:
    """
    Search for GitHub issues across repositories.
    
    Args:
        query: Search query (can include qualifiers like 'repo:owner/name', 'label:bug', 'state:open')
        sort: Sort by 'comments', 'reactions', 'reactions-+1', 'reactions--1', 'reactions-smile', 'reactions-thinking_face', 'reactions-heart', 'reactions-tada', 'interactions', 'created', 'updated'
        limit: Number of results to return (default: 20, max: 100)
    
    Returns:
        Formatted string containing issue search results
    """
    try:
        client = get_github_client()
        
        # Validate and limit the number of results
        limit = min(max(1, limit), 100)
        
        # Search issues
        search_results = client.search_issues(query, sort=sort, limit=limit)
        issues = search_results.get("items", [])
        
        # Format output
        output = f"## Issue Search Results for '{query}'\n\n"
        output += f"**Total Results:** {search_results.get('total_count', 0)} (showing {len(issues)})\n"
        output += f"**Sorted by:** {sort}\n\n"
        
        if not issues:
            output += "No issues found for your search query.\n"
            return output
        
        for i, issue in enumerate(issues, 1):
            issue_data = format_issue_data(issue)
            repo_name = issue.get("repository_url", "").split("/")[-2:] if issue.get("repository_url") else ["unknown", "repo"]
            repo_full_name = "/".join(repo_name) if len(repo_name) == 2 else "unknown/repo"
            
            output += f"### {i}. {repo_full_name}#{issue_data['number']}: {issue_data['title']}\n"
            output += f"**Repository:** {repo_full_name}\n"
            output += f"**Author:** {issue_data['author']} | **State:** {issue_data['state']}\n"
            output += f"**Created:** {issue_data['created_at']} | **Updated:** {issue_data['updated_at']}\n"
            output += f"**Comments:** {issue_data['comments']}\n"
            
            if issue_data['labels']:
                output += f"**Labels:** {', '.join(issue_data['labels'])}\n"
            
            if issue_data['assignees']:
                output += f"**Assignees:** {', '.join(issue_data['assignees'])}\n"
            
            output += f"**URL:** {issue_data['html_url']}\n"
            
            if issue_data['body']:
                output += f"**Description:** {issue_data['body']}\n"
            
            output += "---\n\n"
        
        return output
        
    except Exception as e:
        logger.error(f"Error searching GitHub issues: {e}")
        return f"Error searching issues for '{query}': {str(e)}"

@mcp.tool()
def get_project_insights(repo: str) -> str:
    """
    Get comprehensive project insights and analytics for a repository.
    
    Args:
        repo: Repository name in format 'owner/repo'
    
    Returns:
        Formatted string containing project insights and statistics
    """
    try:
        client = get_github_client()
        
        # Get comprehensive repository statistics
        stats = client.get_repository_stats(repo)
        repo_info = stats["repository"]
        contributors = stats["contributors"]
        languages = stats["languages"]
        recent_commits = stats["recent_commits"]
        recent_issues = stats["recent_issues"]
        
        # Format output
        output = f"# 📊 Project Insights: {repo}\n\n"
        
        # Basic repository information
        output += f"## 📋 Repository Overview\n"
        output += f"**Description:** {repo_info.get('description', 'No description available')}\n"
        output += f"**Created:** {repo_info.get('created_at')}\n"
        output += f"**Last Updated:** {repo_info.get('updated_at')}\n"
        output += f"**Default Branch:** {repo_info.get('default_branch', 'main')}\n"
        output += f"**Size:** {repo_info.get('size', 0)} KB\n"
        
        if repo_info.get('homepage'):
            output += f"**Homepage:** {repo_info['homepage']}\n"
        
        if repo_info.get('topics'):
            output += f"**Topics:** {', '.join(repo_info['topics'])}\n"
        
        if repo_info.get('license'):
            output += f"**License:** {repo_info['license']['name']}\n"
        
        output += "\n"
        
        # Statistics
        output += f"## 📈 Statistics\n"
        output += f"- **Stars:** ⭐ {repo_info.get('stargazers_count', 0)}\n"
        output += f"- **Forks:** 🍴 {repo_info.get('forks_count', 0)}\n"
        output += f"- **Watchers:** 👀 {repo_info.get('watchers_count', 0)}\n"
        output += f"- **Open Issues:** 🐛 {repo_info.get('open_issues_count', 0)}\n"
        output += f"- **Network Count:** 🌐 {repo_info.get('network_count', 0)}\n"
        output += f"- **Subscribers:** 📧 {repo_info.get('subscribers_count', 0)}\n\n"
        
        # Languages
        if languages:
            output += f"## 💻 Languages\n"
            total_bytes = sum(languages.values())
            for language, bytes_count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                percentage = (bytes_count / total_bytes) * 100 if total_bytes > 0 else 0
                output += f"- **{language}:** {percentage:.1f}% ({bytes_count:,} bytes)\n"
            output += "\n"
        
        # Top contributors
        if contributors:
            output += f"## 👥 Top Contributors\n"
            for i, contributor in enumerate(contributors[:10], 1):
                output += f"{i}. **{contributor.get('login', 'Unknown')}** - {contributor.get('contributions', 0)} contributions\n"
            output += "\n"
        
        # Recent activity
        if recent_commits:
            output += f"## 🔄 Recent Commits (Last 10)\n"
            for commit in recent_commits[:10]:
                commit_info = commit.get('commit', {})
                author = commit_info.get('author', {})
                message = commit_info.get('message', 'No message')[:100]
                if len(commit_info.get('message', '')) > 100:
                    message += "..."
                
                output += f"- **{author.get('name', 'Unknown')}** ({author.get('date', 'Unknown date')}): {message}\n"
            output += "\n"
        
        # Recent issues
        if recent_issues:
            open_issues = [issue for issue in recent_issues if issue.get('state') == 'open']
            closed_issues = [issue for issue in recent_issues if issue.get('state') == 'closed']
            
            output += f"## 🎯 Recent Issues Activity\n"
            output += f"**Recent Open Issues:** {len(open_issues)}\n"
            output += f"**Recent Closed Issues:** {len(closed_issues)}\n\n"
            
            if open_issues:
                output += f"### 🔓 Recent Open Issues:\n"
                for issue in open_issues[:5]:
                    output += f"- #{issue.get('number')}: {issue.get('title', 'No title')[:80]}{'...' if len(issue.get('title', '')) > 80 else ''}\n"
                output += "\n"
        
        # Project health indicators
        output += f"## 🏥 Project Health Indicators\n"
        
        # Calculate activity score based on recent commits and issues
        recent_activity_score = len(recent_commits) + len(recent_issues)
        if recent_activity_score > 15:
            activity_level = "🟢 Very Active"
        elif recent_activity_score > 8:
            activity_level = "🟡 Moderately Active"
        elif recent_activity_score > 3:
            activity_level = "🟠 Low Activity"
        else:
            activity_level = "🔴 Inactive"
        
        output += f"- **Activity Level:** {activity_level}\n"
        
        # Community engagement
        stars = repo_info.get('stargazers_count', 0)
        forks = repo_info.get('forks_count', 0)
        if stars > 1000 or forks > 100:
            engagement = "🟢 High Community Engagement"
        elif stars > 100 or forks > 20:
            engagement = "🟡 Moderate Community Engagement"
        elif stars > 10 or forks > 5:
            engagement = "🟠 Low Community Engagement"
        else:
            engagement = "🔴 Minimal Community Engagement"
        
        output += f"- **Community Engagement:** {engagement}\n"
        
        # Maintenance status
        last_update = datetime.fromisoformat(repo_info.get('updated_at', '').replace('Z', '+00:00'))
        days_since_update = (datetime.now(last_update.tzinfo) - last_update).days
        
        if days_since_update < 7:
            maintenance = "🟢 Actively Maintained"
        elif days_since_update < 30:
            maintenance = "🟡 Recently Maintained"
        elif days_since_update < 90:
            maintenance = "🟠 Infrequently Maintained"
        else:
            maintenance = "🔴 Possibly Unmaintained"
        
        output += f"- **Maintenance Status:** {maintenance} (last updated {days_since_update} days ago)\n"
        
        return output
        
    except Exception as e:
        logger.error(f"Error getting project insights: {e}")
        return f"Error getting project insights for {repo}: {str(e)}"

if __name__ == "__main__":
    # Run the MCP server
    mcp.run(transport="streamable-http", port=8003)
