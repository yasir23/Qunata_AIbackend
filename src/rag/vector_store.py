"""
Vector Store implementation for Open Deep Research RAG system.

This module provides vector database integration using Supabase's pgvector extension
for storing and retrieving research documents with semantic search capabilities.
"""

import os
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from pydantic import BaseModel, Field
from supabase import Client, create_client
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ResearchDocument:
    """Research document data structure."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class DocumentChunk(BaseModel):
    """Document chunk model for vector storage."""
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    embedding: List[float] = Field(..., description="Vector embedding")
    chunk_index: int = Field(..., description="Chunk position in document")

class SearchResult(BaseModel):
    """Search result model."""
    document_id: str = Field(..., description="Document identifier")
    content: str = Field(..., description="Matching content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    chunk_index: int = Field(..., description="Chunk position")

class VectorStore:
    """
    Vector store implementation using Supabase pgvector for semantic search
    and retrieval of research documents.
    """
    
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        table_name: str = "research_documents",
        chunks_table_name: str = "document_chunks"
    ):
        """
        Initialize the vector store.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase service key
            openai_api_key: OpenAI API key for embeddings
            embedding_model: OpenAI embedding model to use
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
            table_name: Name of the documents table
            chunks_table_name: Name of the chunks table
        """
        # Initialize Supabase client
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and key are required")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize embeddings
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for embeddings")
        
        self.embeddings = OpenAIEmbeddings(
            api_key=self.openai_api_key,
            model=embedding_model
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.table_name = table_name
        self.chunks_table_name = chunks_table_name
        
        logger.info(f"VectorStore initialized with embedding model: {embedding_model}")
    
    def _generate_document_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate a unique document ID based on content and metadata."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        metadata_str = json.dumps(metadata, sort_keys=True)
        metadata_hash = hashlib.sha256(metadata_str.encode()).hexdigest()[:8]
        return f"doc_{content_hash}_{metadata_hash}"
    
    def _generate_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """Generate a unique chunk ID."""
        return f"{document_id}_chunk_{chunk_index}"
    
    async def add_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> str:
        """
        Add a document to the vector store.
        
        Args:
            content: Document content
            metadata: Document metadata
            user_id: User ID for access control
            
        Returns:
            Document ID
        """
        try:
            # Generate document ID
            document_id = self._generate_document_id(content, metadata)
            
            # Check if document already exists
            existing = self.supabase.table(self.table_name).select("id").eq("id", document_id).execute()
            if existing.data:
                logger.info(f"Document {document_id} already exists, skipping")
                return document_id
            
            # Split document into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Generate embeddings for all chunks
            embeddings = await asyncio.to_thread(
                self.embeddings.embed_documents, chunks
            )
            
            # Store document metadata
            document_data = {
                "id": document_id,
                "content": content,
                "metadata": metadata,
                "user_id": user_id,
                "chunk_count": len(chunks),
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            self.supabase.table(self.table_name).insert(document_data).execute()
            
            # Store document chunks with embeddings
            chunk_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = self._generate_chunk_id(document_id, i)
                chunk_data.append({
                    "id": chunk_id,
                    "document_id": document_id,
                    "content": chunk,
                    "metadata": metadata,
                    "embedding": embedding,
                    "chunk_index": i,
                    "user_id": user_id,
                    "created_at": datetime.utcnow().isoformat()
                })
            
            # Insert chunks in batches
            batch_size = 100
            for i in range(0, len(chunk_data), batch_size):
                batch = chunk_data[i:i + batch_size]
                self.supabase.table(self.chunks_table_name).insert(batch).execute()
            
            logger.info(f"Added document {document_id} with {len(chunks)} chunks")
            return document_id
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    async def add_research_result(
        self,
        research_topic: str,
        research_content: str,
        sources: List[str],
        user_id: Optional[str] = None,
        research_type: str = "general"
    ) -> str:
        """
        Add research results to the vector store.
        
        Args:
            research_topic: The research topic/question
            research_content: The research findings/content
            sources: List of sources used
            user_id: User ID for access control
            research_type: Type of research (general, academic, news, etc.)
            
        Returns:
            Document ID
        """
        metadata = {
            "type": "research_result",
            "topic": research_topic,
            "sources": sources,
            "research_type": research_type,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self.add_document(research_content, metadata, user_id)
    
    async def similarity_search(
        self,
        query: str,
        k: int = 5,
        user_id: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.7
    ) -> List[SearchResult]:
        """
        Perform similarity search for relevant documents.
        
        Args:
            query: Search query
            k: Number of results to return
            user_id: User ID for access control
            filter_metadata: Metadata filters
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search results
        """
        try:
            # Generate query embedding
            query_embedding = await asyncio.to_thread(
                self.embeddings.embed_query, query
            )
            
            # Build the RPC call for similarity search
            rpc_params = {
                "query_embedding": query_embedding,
                "match_threshold": similarity_threshold,
                "match_count": k
            }
            
            if user_id:
                rpc_params["user_id_filter"] = user_id
            
            # Perform similarity search using Supabase RPC
            result = self.supabase.rpc("match_document_chunks", rpc_params).execute()
            
            search_results = []
            for row in result.data:
                search_results.append(SearchResult(
                    document_id=row["document_id"],
                    content=row["content"],
                    metadata=row["metadata"],
                    similarity_score=row["similarity"],
                    chunk_index=row["chunk_index"]
                ))
            
            # Apply metadata filters if specified
            if filter_metadata:
                filtered_results = []
                for result in search_results:
                    match = True
                    for key, value in filter_metadata.items():
                        if key not in result.metadata or result.metadata[key] != value:
                            match = False
                            break
                    if match:
                        filtered_results.append(result)
                search_results = filtered_results
            
            logger.info(f"Similarity search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    async def get_relevant_context(
        self,
        query: str,
        user_id: Optional[str] = None,
        max_context_length: int = 4000,
        research_type: Optional[str] = None
    ) -> str:
        """
        Get relevant context for a query from previous research.
        
        Args:
            query: Search query
            user_id: User ID for access control
            max_context_length: Maximum context length in characters
            research_type: Filter by research type
            
        Returns:
            Relevant context string
        """
        try:
            # Build metadata filter
            filter_metadata = {"type": "research_result"}
            if research_type:
                filter_metadata["research_type"] = research_type
            
            # Search for relevant documents
            results = await self.similarity_search(
                query=query,
                k=10,
                user_id=user_id,
                filter_metadata=filter_metadata,
                similarity_threshold=0.6
            )
            
            if not results:
                return ""
            
            # Build context from results
            context_parts = []
            current_length = 0
            
            for result in results:
                # Add topic and content
                topic = result.metadata.get("topic", "Unknown Topic")
                content_part = f"## {topic}\n{result.content}\n"
                
                if current_length + len(content_part) > max_context_length:
                    break
                
                context_parts.append(content_part)
                current_length += len(content_part)
            
            context = "\n".join(context_parts)
            
            if context:
                context = f"# Relevant Previous Research\n\n{context}\n---\n"
                logger.info(f"Retrieved {len(context_parts)} relevant research contexts")
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting relevant context: {e}")
            return ""
    
    async def delete_document(self, document_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a document and its chunks.
        
        Args:
            document_id: Document ID to delete
            user_id: User ID for access control
            
        Returns:
            True if successful
        """
        try:
            # Build query
            query = self.supabase.table(self.table_name).delete().eq("id", document_id)
            if user_id:
                query = query.eq("user_id", user_id)
            
            # Delete document (chunks will be deleted via cascade)
            result = query.execute()
            
            success = len(result.data) > 0
            if success:
                logger.info(f"Deleted document {document_id}")
            else:
                logger.warning(f"Document {document_id} not found or access denied")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    async def get_user_documents(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get documents for a specific user.
        
        Args:
            user_id: User ID
            limit: Maximum number of documents
            offset: Offset for pagination
            
        Returns:
            List of document metadata
        """
        try:
            result = self.supabase.table(self.table_name).select(
                "id, metadata, chunk_count, created_at, updated_at"
            ).eq("user_id", user_id).range(offset, offset + limit - 1).execute()
            
            return result.data
            
        except Exception as e:
            logger.error(f"Error getting user documents: {e}")
            return []
    
    async def update_document_metadata(
        self,
        document_id: str,
        metadata: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> bool:
        """
        Update document metadata.
        
        Args:
            document_id: Document ID
            metadata: New metadata
            user_id: User ID for access control
            
        Returns:
            True if successful
        """
        try:
            # Build query
            query = self.supabase.table(self.table_name).update({
                "metadata": metadata,
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", document_id)
            
            if user_id:
                query = query.eq("user_id", user_id)
            
            result = query.execute()
            
            success = len(result.data) > 0
            if success:
                # Also update chunk metadata
                chunk_query = self.supabase.table(self.chunks_table_name).update({
                    "metadata": metadata
                }).eq("document_id", document_id)
                
                if user_id:
                    chunk_query = chunk_query.eq("user_id", user_id)
                
                chunk_query.execute()
                logger.info(f"Updated metadata for document {document_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating document metadata: {e}")
            return False

    async def create_tables(self) -> bool:
        """
        Create the necessary database tables and functions.
        This should be called during setup/migration.
        
        Returns:
            True if successful
        """
        try:
            # This would typically be done via migration scripts
            # For now, we'll log that tables should be created
            logger.info("Vector store tables should be created via database migrations")
            return True
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            return False

# Utility functions for integration with research workflow

async def get_research_context(
    query: str,
    user_id: Optional[str] = None,
    vector_store: Optional[VectorStore] = None
) -> str:
    """
    Get relevant research context for a query.
    
    Args:
        query: Research query
        user_id: User ID for access control
        vector_store: Vector store instance
        
    Returns:
        Relevant context string
    """
    if not vector_store:
        try:
            vector_store = VectorStore()
        except Exception as e:
            logger.warning(f"Could not initialize vector store: {e}")
            return ""
    
    return await vector_store.get_relevant_context(query, user_id)

async def store_research_result(
    topic: str,
    content: str,
    sources: List[str],
    user_id: Optional[str] = None,
    research_type: str = "general",
    vector_store: Optional[VectorStore] = None
) -> Optional[str]:
    """
    Store research results in the vector store.
    
    Args:
        topic: Research topic
        content: Research content
        sources: List of sources
        user_id: User ID
        research_type: Type of research
        vector_store: Vector store instance
        
    Returns:
        Document ID if successful
    """
    if not vector_store:
        try:
            vector_store = VectorStore()
        except Exception as e:
            logger.warning(f"Could not initialize vector store: {e}")
            return None
    
    try:
        return await vector_store.add_research_result(
            topic, content, sources, user_id, research_type
        )
    except Exception as e:
        logger.error(f"Error storing research result: {e}")
        return None
