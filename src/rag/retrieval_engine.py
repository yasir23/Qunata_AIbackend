"""
Retrieval Engine for RAG system.

This module enhances research queries with relevant context from the vector database,
integrating with existing search tools and MCP servers.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

from .vector_store import VectorStore
from .document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class RetrievalEngine:
    """Retrieval engine that enhances research queries with relevant context."""
    
    def __init__(self, vector_store: Optional[VectorStore] = None, document_processor: Optional[DocumentProcessor] = None):
        """
        Initialize the retrieval engine.
        
        Args:
            vector_store: Vector store instance (will create if None)
            document_processor: Document processor instance (will create if None)
        """
        self.vector_store = vector_store or VectorStore()
        self.document_processor = document_processor or DocumentProcessor()
        
        # Configuration
        self.max_context_length = 8000  # Maximum characters for context
        self.similarity_threshold = 0.7  # Minimum similarity score
        self.max_results_per_source = 5  # Maximum results per source type
        
        logger.info("Retrieval engine initialized")
    
    async def enhance_query_with_context(
        self, 
        query: str, 
        sources: Optional[List[str]] = None,
        max_context_items: int = 10
    ) -> Dict[str, Any]:
        """
        Enhance a research query with relevant context from the vector database.
        
        Args:
            query: Original research query
            sources: List of sources to search in (e.g., ['reddit', 'github', 'youtube'])
            max_context_items: Maximum number of context items to retrieve
            
        Returns:
            Dictionary with enhanced query and context information
        """
        try:
            if not self.vector_store.is_initialized():
                logger.warning("Vector store not initialized, returning original query")
                return {
                    "original_query": query,
                    "enhanced_query": query,
                    "context": [],
                    "context_summary": "",
                    "sources_used": []
                }
            
            # Search for relevant context
            context_items = await self._retrieve_relevant_context(
                query, sources, max_context_items
            )
            
            # Generate enhanced query with context
            enhanced_query = await self._generate_enhanced_query(query, context_items)
            
            # Create context summary
            context_summary = await self._create_context_summary(context_items)
            
            # Get sources used
            sources_used = list(set(item.get("metadata", {}).get("source", "unknown") for item in context_items))
            
            return {
                "original_query": query,
                "enhanced_query": enhanced_query,
                "context": context_items,
                "context_summary": context_summary,
                "sources_used": sources_used,
                "context_count": len(context_items),
                "retrieved_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error enhancing query with context: {e}")
            return {
                "original_query": query,
                "enhanced_query": query,
                "context": [],
                "context_summary": "",
                "sources_used": [],
                "error": str(e)
            }
    
    async def _retrieve_relevant_context(
        self, 
        query: str, 
        sources: Optional[List[str]] = None,
        max_items: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant context from the vector database."""
        all_context = []
        
        if sources:
            # Search each source separately to ensure diversity
            for source in sources:
                try:
                    results = await self.vector_store.search_similar(
                        query=query,
                        n_results=self.max_results_per_source,
                        source_filter=source
                    )
                    
                    # Filter by similarity threshold
                    filtered_results = [
                        result for result in results 
                        if result.get("similarity_score", 0) >= self.similarity_threshold
                    ]
                    
                    all_context.extend(filtered_results)
                    
                except Exception as e:
                    logger.error(f"Error searching {source}: {e}")
                    continue
        else:
            # Search all sources
            try:
                results = await self.vector_store.search_similar(
                    query=query,
                    n_results=max_items * 2  # Get more to filter
                )
                
                # Filter by similarity threshold
                filtered_results = [
                    result for result in results 
                    if result.get("similarity_score", 0) >= self.similarity_threshold
                ]
                
                all_context.extend(filtered_results)
                
            except Exception as e:
                logger.error(f"Error searching vector store: {e}")
                return []
        
        # Sort by similarity score and limit results
        all_context.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
        return all_context[:max_items]
    
    async def _generate_enhanced_query(self, original_query: str, context_items: List[Dict[str, Any]]) -> str:
        """Generate an enhanced query with context information."""
        if not context_items:
            return original_query
        
        # Extract key information from context
        context_insights = []
        
        for item in context_items[:5]:  # Use top 5 items for enhancement
            metadata = item.get("metadata", {})
            content = item.get("content", "")[:200]  # First 200 chars
            
            source = metadata.get("source", "unknown")
            item_type = metadata.get("type", "unknown")
            
            if source == "reddit":
                subreddit = metadata.get("subreddit", "")
                context_insights.append(f"Reddit discussion in r/{subreddit}: {content}")
            elif source == "github":
                repo = metadata.get("repository", "")
                context_insights.append(f"GitHub issue in {repo}: {content}")
            elif source == "youtube":
                channel = metadata.get("channel_title", "")
                context_insights.append(f"YouTube content from {channel}: {content}")
            else:
                context_insights.append(f"{source} content: {content}")
        
        if context_insights:
            enhanced_query = f"{original_query}\n\nRelevant context from previous research:\n"
            enhanced_query += "\n".join(f"- {insight}" for insight in context_insights)
            return enhanced_query
        
        return original_query
    
    async def _create_context_summary(self, context_items: List[Dict[str, Any]]) -> str:
        """Create a summary of the retrieved context."""
        if not context_items:
            return "No relevant context found."
        
        # Group by source
        sources = {}
        for item in context_items:
            source = item.get("metadata", {}).get("source", "unknown")
            if source not in sources:
                sources[source] = []
            sources[source].append(item)
        
        summary_parts = []
        summary_parts.append(f"Found {len(context_items)} relevant context items from {len(sources)} sources:")
        
        for source, items in sources.items():
            avg_similarity = sum(item.get("similarity_score", 0) for item in items) / len(items)
            summary_parts.append(f"- {source.title()}: {len(items)} items (avg similarity: {avg_similarity:.2f})")
        
        return "\n".join(summary_parts)
    
    async def ingest_mcp_data(
        self, 
        mcp_data: Dict[str, Any], 
        source: str,
        data_type: str = "auto"
    ) -> Dict[str, Any]:
        """
        Ingest data from MCP servers into the vector database.
        
        Args:
            mcp_data: Data from MCP server
            source: Source name (reddit, youtube, github)
            data_type: Type of data (auto-detect if 'auto')
            
        Returns:
            Ingestion results
        """
        try:
            processed_docs = []
            
            # Process based on source and data type
            if source == "reddit":
                if data_type == "auto":
                    # Auto-detect based on data structure
                    if isinstance(mcp_data, list) and mcp_data:
                        first_item = mcp_data[0]
                        if "subreddit" in first_item:
                            data_type = "posts"
                        elif "parent_id" in first_item:
                            data_type = "comments"
                
                if data_type == "posts":
                    processed_docs = await self.document_processor.process_reddit_posts(mcp_data)
                elif data_type == "comments":
                    processed_docs = await self.document_processor.process_reddit_comments(mcp_data)
                    
            elif source == "youtube":
                if data_type == "auto":
                    if isinstance(mcp_data, list) and mcp_data:
                        first_item = mcp_data[0]
                        if "channel_title" in first_item:
                            data_type = "videos"
                        elif "text" in first_item:
                            data_type = "comments"
                
                if data_type == "videos":
                    processed_docs = await self.document_processor.process_youtube_videos(mcp_data)
                elif data_type == "comments":
                    processed_docs = await self.document_processor.process_youtube_comments(mcp_data)
                    
            elif source == "github":
                if data_type == "auto":
                    if isinstance(mcp_data, list) and mcp_data:
                        first_item = mcp_data[0]
                        if "number" in first_item and "title" in first_item:
                            data_type = "issues"
                        elif "full_name" in first_item:
                            data_type = "repositories"
                
                if data_type == "issues":
                    processed_docs = await self.document_processor.process_github_issues(mcp_data)
                elif data_type == "repositories":
                    processed_docs = await self.document_processor.process_github_repositories(mcp_data)
            
            # Extract documents and metadata
            if processed_docs:
                documents = [doc["content"] for doc in processed_docs]
                metadatas = [doc["metadata"] for doc in processed_docs]
                
                # Add to vector store
                added_ids = await self.vector_store.add_documents(
                    documents=documents,
                    metadatas=metadatas,
                    source=source
                )
                
                return {
                    "success": True,
                    "source": source,
                    "data_type": data_type,
                    "processed_documents": len(processed_docs),
                    "added_documents": len(added_ids),
                    "document_ids": added_ids,
                    "ingested_at": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "source": source,
                    "data_type": data_type,
                    "processed_documents": 0,
                    "added_documents": 0,
                    "error": "No documents processed"
                }
                
        except Exception as e:
            logger.error(f"Error ingesting MCP data from {source}: {e}")
            return {
                "success": False,
                "source": source,
                "data_type": data_type,
                "error": str(e)
            }
    
    async def search_historical_research(
        self, 
        query: str, 
        time_range_days: Optional[int] = None,
        min_similarity: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Search for historical research related to the query.
        
        Args:
            query: Search query
            time_range_days: Limit to documents from last N days
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant historical research items
        """
        try:
            # Build metadata filter for time range
            metadata_filter = {}
            if time_range_days:
                cutoff_date = (datetime.now() - timedelta(days=time_range_days)).isoformat()
                # Note: This would need to be implemented in the vector store
                # metadata_filter["added_at"] = {"$gte": cutoff_date}
            
            # Search for similar documents
            results = await self.vector_store.search_similar(
                query=query,
                n_results=20,
                metadata_filter=metadata_filter if metadata_filter else None
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result.get("similarity_score", 0) >= min_similarity
            ]
            
            # Group by research session or topic
            grouped_results = self._group_research_results(filtered_results)
            
            return grouped_results
            
        except Exception as e:
            logger.error(f"Error searching historical research: {e}")
            return []
    
    def _group_research_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group research results by topic or session."""
        # Simple grouping by source and similarity
        grouped = {}
        
        for result in results:
            metadata = result.get("metadata", {})
            source = metadata.get("source", "unknown")
            
            if source not in grouped:
                grouped[source] = []
            
            grouped[source].append(result)
        
        # Convert to list format
        grouped_results = []
        for source, items in grouped.items():
            grouped_results.append({
                "source": source,
                "items": items,
                "count": len(items),
                "avg_similarity": sum(item.get("similarity_score", 0) for item in items) / len(items)
            })
        
        # Sort by average similarity
        grouped_results.sort(key=lambda x: x["avg_similarity"], reverse=True)
        
        return grouped_results
    
    async def get_context_for_research_brief(self, research_brief: str) -> Dict[str, Any]:
        """
        Get relevant context for a research brief.
        
        Args:
            research_brief: The research brief text
            
        Returns:
            Context information for the research brief
        """
        try:
            # Extract key topics from research brief
            key_topics = self._extract_key_topics(research_brief)
            
            # Search for context on each topic
            all_context = []
            for topic in key_topics:
                context = await self._retrieve_relevant_context(
                    topic, 
                    sources=["reddit", "github", "youtube"],
                    max_items=3
                )
                all_context.extend(context)
            
            # Remove duplicates and sort by relevance
            unique_context = self._deduplicate_context(all_context)
            
            # Create structured context response
            return {
                "research_brief": research_brief,
                "key_topics": key_topics,
                "context_items": unique_context,
                "context_summary": await self._create_context_summary(unique_context),
                "recommendations": self._generate_research_recommendations(unique_context)
            }
            
        except Exception as e:
            logger.error(f"Error getting context for research brief: {e}")
            return {
                "research_brief": research_brief,
                "key_topics": [],
                "context_items": [],
                "context_summary": "Error retrieving context",
                "recommendations": [],
                "error": str(e)
            }
    
    def _extract_key_topics(self, text: str) -> List[str]:
        """Extract key topics from text (simple implementation)."""
        # Simple keyword extraction - could be enhanced with NLP
        import re
        
        # Remove common words and extract meaningful terms
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'
        }
        
        meaningful_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Return top unique words
        return list(set(meaningful_words))[:10]
    
    def _deduplicate_context(self, context_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate context items."""
        seen_content = set()
        unique_items = []
        
        for item in context_items:
            content_hash = hash(item.get("content", "")[:100])  # Hash first 100 chars
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_items.append(item)
        
        return unique_items
    
    def _generate_research_recommendations(self, context_items: List[Dict[str, Any]]) -> List[str]:
        """Generate research recommendations based on context."""
        recommendations = []
        
        if not context_items:
            return ["No specific recommendations - consider broadening search terms"]
        
        # Analyze sources
        sources = set(item.get("metadata", {}).get("source", "") for item in context_items)
        
        if "reddit" in sources:
            recommendations.append("Consider exploring Reddit discussions for community perspectives")
        
        if "github" in sources:
            recommendations.append("Review GitHub issues for technical implementation details")
        
        if "youtube" in sources:
            recommendations.append("Check YouTube content for educational or tutorial material")
        
        # Analyze content types
        types = set(item.get("metadata", {}).get("type", "") for item in context_items)
        
        if "issue" in types:
            recommendations.append("Focus on problem-solving approaches from GitHub issues")
        
        if "post" in types:
            recommendations.append("Leverage community discussions and experiences")
        
        return recommendations or ["Explore the retrieved context for relevant insights"]
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get statistics about the retrieval engine."""
        vector_stats = self.vector_store.get_collection_stats()
        processor_stats = self.document_processor.get_processing_stats()
        
        return {
            "vector_store": vector_stats,
            "document_processor": processor_stats,
            "configuration": {
                "max_context_length": self.max_context_length,
                "similarity_threshold": self.similarity_threshold,
                "max_results_per_source": self.max_results_per_source
            },
            "status": "initialized" if self.vector_store.is_initialized() else "not_initialized"
        }
