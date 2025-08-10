"""
Vector Store implementation for RAG system using ChromaDB.

This module provides vector database operations for storing and retrieving
document embeddings from MCP server data (Reddit, YouTube, GitHub).
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import hashlib
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store implementation using ChromaDB for document embeddings."""
    
    def __init__(self, persist_directory: Optional[str] = None, collection_name: str = "research_documents"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the database (optional)
            collection_name: Name of the collection to use
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or os.path.join(os.getcwd(), "chroma_db")
        
        # Initialize ChromaDB client
        self.client = None
        self.collection = None
        self.embedding_function = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Use OpenAI embeddings if available, otherwise use default
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key,
                    model_name="text-embedding-3-small"
                )
            else:
                # Use default sentence transformer embeddings
                self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Research documents from MCP servers"}
            )
            
            logger.info(f"Vector store initialized with collection: {self.collection_name}")
            logger.info(f"Persist directory: {self.persist_directory}")
            logger.info(f"Collection count: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def _generate_document_id(self, content: str, source: str, metadata: Dict[str, Any]) -> str:
        """Generate a unique document ID based on content and metadata."""
        # Create a hash of content + source + key metadata
        key_metadata = {
            "source": source,
            "url": metadata.get("url", ""),
            "title": metadata.get("title", ""),
            "timestamp": metadata.get("timestamp", "")
        }
        
        content_hash = hashlib.md5(
            (content + json.dumps(key_metadata, sort_keys=True)).encode()
        ).hexdigest()
        
        return f"{source}_{content_hash[:16]}"
    
    async def add_documents(
        self, 
        documents: List[str], 
        metadatas: List[Dict[str, Any]], 
        source: str = "unknown"
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries for each document
            source: Source of the documents (e.g., 'reddit', 'youtube', 'github')
            
        Returns:
            List of document IDs that were added
        """
        if not documents or not metadatas:
            return []
        
        if len(documents) != len(metadatas):
            raise ValueError("Number of documents must match number of metadata entries")
        
        try:
            # Generate document IDs
            document_ids = []
            filtered_documents = []
            filtered_metadatas = []
            
            for doc, metadata in zip(documents, metadatas):
                if not doc.strip():  # Skip empty documents
                    continue
                
                doc_id = self._generate_document_id(doc, source, metadata)
                
                # Check if document already exists
                try:
                    existing = self.collection.get(ids=[doc_id])
                    if existing['ids']:
                        logger.debug(f"Document {doc_id} already exists, skipping")
                        continue
                except Exception:
                    pass  # Document doesn't exist, proceed to add
                
                document_ids.append(doc_id)
                filtered_documents.append(doc)
                
                # Add source and timestamp to metadata
                enhanced_metadata = {
                    **metadata,
                    "source": source,
                    "added_at": datetime.now().isoformat(),
                    "content_length": len(doc)
                }
                filtered_metadatas.append(enhanced_metadata)
            
            if not filtered_documents:
                logger.info("No new documents to add")
                return []
            
            # Add documents to collection in batches
            batch_size = 100
            added_ids = []
            
            for i in range(0, len(filtered_documents), batch_size):
                batch_docs = filtered_documents[i:i + batch_size]
                batch_ids = document_ids[i:i + batch_size]
                batch_metadata = filtered_metadatas[i:i + batch_size]
                
                await asyncio.to_thread(
                    self.collection.add,
                    documents=batch_docs,
                    metadatas=batch_metadata,
                    ids=batch_ids
                )
                
                added_ids.extend(batch_ids)
                logger.debug(f"Added batch of {len(batch_docs)} documents")
            
            logger.info(f"Successfully added {len(added_ids)} documents from {source}")
            return added_ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    async def search_similar(
        self, 
        query: str, 
        n_results: int = 10, 
        source_filter: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the vector store.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            source_filter: Filter by source (e.g., 'reddit', 'youtube', 'github')
            metadata_filter: Additional metadata filters
            
        Returns:
            List of similar documents with metadata and similarity scores
        """
        try:
            # Build where clause for filtering
            where_clause = {}
            if source_filter:
                where_clause["source"] = source_filter
            
            if metadata_filter:
                where_clause.update(metadata_filter)
            
            # Perform similarity search
            results = await asyncio.to_thread(
                self.collection.query,
                query_texts=[query],
                n_results=min(n_results, 100),  # Limit to reasonable number
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity_score": 1 - distance,  # Convert distance to similarity
                        "rank": i + 1
                    })
            
            logger.debug(f"Found {len(formatted_results)} similar documents for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            return []
    
    async def get_documents_by_source(self, source: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get documents by source.
        
        Args:
            source: Source to filter by
            limit: Maximum number of documents to return
            
        Returns:
            List of documents from the specified source
        """
        try:
            results = await asyncio.to_thread(
                self.collection.get,
                where={"source": source},
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            formatted_results = []
            if results['documents']:
                for doc, metadata in zip(results['documents'], results['metadatas']):
                    formatted_results.append({
                        "content": doc,
                        "metadata": metadata
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error getting documents by source: {e}")
            return []
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """
        Delete documents from the vector store.
        
        Args:
            document_ids: List of document IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            await asyncio.to_thread(
                self.collection.delete,
                ids=document_ids
            )
            
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    async def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Research documents from MCP servers"}
            )
            
            logger.info("Cleared all documents from collection")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze sources
            sample_results = self.collection.get(
                limit=min(1000, count),
                include=["metadatas"]
            )
            
            sources = {}
            if sample_results['metadatas']:
                for metadata in sample_results['metadatas']:
                    source = metadata.get('source', 'unknown')
                    sources[source] = sources.get(source, 0) + 1
            
            return {
                "total_documents": count,
                "sources": sources,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {
                "total_documents": 0,
                "sources": {},
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "error": str(e)
            }
    
    def is_initialized(self) -> bool:
        """Check if the vector store is properly initialized."""
        return self.client is not None and self.collection is not None
