#!/usr/bin/env python3
"""
Test script for RAG system integration.

This script tests the RAG system components to ensure they work properly
with the existing research workflow and MCP servers.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.vector_store import VectorStore
from rag.document_processor import DocumentProcessor
from rag.retrieval_engine import RetrievalEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_vector_store():
    """Test vector store initialization and basic operations."""
    logger.info("Testing Vector Store...")
    
    try:
        # Initialize vector store
        vector_store = VectorStore(collection_name="test_collection")
        
        if not vector_store.is_initialized():
            logger.error("❌ Vector store failed to initialize")
            return False
        
        logger.info("✅ Vector store initialized successfully")
        
        # Test adding documents
        test_documents = [
            "This is a test document about machine learning and AI research.",
            "Another document discussing deep learning techniques and neural networks.",
            "A third document about natural language processing and transformers."
        ]
        
        test_metadata = [
            {"source": "test", "type": "document", "title": "ML Research", "url": "test://1"},
            {"source": "test", "type": "document", "title": "Deep Learning", "url": "test://2"},
            {"source": "test", "type": "document", "title": "NLP Research", "url": "test://3"}
        ]
        
        added_ids = await vector_store.add_documents(test_documents, test_metadata, "test")
        
        if added_ids:
            logger.info(f"✅ Added {len(added_ids)} test documents")
        else:
            logger.warning("⚠️ No documents were added (might already exist)")
        
        # Test similarity search
        search_results = await vector_store.search_similar("machine learning research", n_results=2)
        
        if search_results:
            logger.info(f"✅ Found {len(search_results)} similar documents")
            for i, result in enumerate(search_results, 1):
                logger.info(f"  {i}. Similarity: {result.get('similarity_score', 0):.3f} - {result.get('metadata', {}).get('title', 'No title')}")
        else:
            logger.warning("⚠️ No similar documents found")
        
        # Get collection stats
        stats = vector_store.get_collection_stats()
        logger.info(f"✅ Collection stats: {stats['total_documents']} documents, sources: {stats['sources']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {e}")
        return False

async def test_document_processor():
    """Test document processor with sample data."""
    logger.info("Testing Document Processor...")
    
    try:
        processor = DocumentProcessor(chunk_size=500, chunk_overlap=100)
        
        # Test Reddit post processing
        sample_reddit_posts = [
            {
                "id": "test123",
                "title": "How to implement RAG systems effectively",
                "selftext": "I've been working on implementing RAG (Retrieval-Augmented Generation) systems and wanted to share some insights. The key is to have good document chunking and embedding strategies.",
                "author": "test_user",
                "subreddit": "MachineLearning",
                "score": 42,
                "num_comments": 15,
                "created_utc": "2024-01-01T00:00:00Z",
                "url": "https://reddit.com/test",
                "permalink": "/r/MachineLearning/test123"
            }
        ]
        
        reddit_docs = await processor.process_reddit_posts(sample_reddit_posts)
        logger.info(f"✅ Processed {len(reddit_docs)} Reddit document chunks")
        
        # Test GitHub issue processing
        sample_github_issues = [
            {
                "id": 456,
                "number": 123,
                "title": "Implement vector database integration",
                "body": "We need to add vector database support for our RAG system. This should include ChromaDB integration and proper document chunking.",
                "state": "open",
                "user": {"login": "developer"},
                "assignees": [],
                "labels": [{"name": "enhancement"}, {"name": "rag"}],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "html_url": "https://github.com/test/repo/issues/123",
                "comments": 5
            }
        ]
        
        github_docs = await processor.process_github_issues(sample_github_issues)
        logger.info(f"✅ Processed {len(github_docs)} GitHub document chunks")
        
        # Test YouTube video processing
        sample_youtube_videos = [
            {
                "id": "test_video_id",
                "title": "Understanding RAG Systems in AI",
                "description": "This video explains how Retrieval-Augmented Generation works and how to implement it effectively in your AI applications.",
                "channel_title": "AI Education Channel",
                "channel_id": "test_channel",
                "published_at": "2024-01-01T00:00:00Z",
                "view_count": 10000,
                "like_count": 500,
                "comment_count": 50,
                "duration": "PT15M30S",
                "url": "https://youtube.com/watch?v=test",
                "thumbnail_url": "https://img.youtube.com/test.jpg"
            }
        ]
        
        youtube_docs = await processor.process_youtube_videos(sample_youtube_videos)
        logger.info(f"✅ Processed {len(youtube_docs)} YouTube document chunks")
        
        # Test processor stats
        stats = processor.get_processing_stats()
        logger.info(f"✅ Processor supports {len(stats['supported_sources'])} sources and {len(stats['supported_types'])} types")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Document processor test failed: {e}")
        return False

async def test_retrieval_engine():
    """Test retrieval engine with sample data."""
    logger.info("Testing Retrieval Engine...")
    
    try:
        # Initialize components
        vector_store = VectorStore(collection_name="test_rag_engine")
        processor = DocumentProcessor()
        engine = RetrievalEngine(vector_store, processor)
        
        # Add some test data
        sample_data = [
            {
                "id": "rag_post_1",
                "title": "Best practices for RAG implementation",
                "selftext": "When implementing RAG systems, focus on document quality, chunking strategy, and embedding model selection. These are crucial for good retrieval performance.",
                "author": "rag_expert",
                "subreddit": "MachineLearning",
                "score": 100,
                "num_comments": 25,
                "created_utc": "2024-01-01T00:00:00Z",
                "url": "https://reddit.com/rag_post_1",
                "permalink": "/r/MachineLearning/rag_post_1"
            }
        ]
        
        # Ingest sample data
        ingest_result = await engine.ingest_mcp_data(sample_data, "reddit", "posts")
        
        if ingest_result.get("success"):
            logger.info(f"✅ Ingested {ingest_result.get('processed_documents', 0)} documents")
        else:
            logger.warning(f"⚠️ Ingestion result: {ingest_result}")
        
        # Test query enhancement
        test_query = "How to implement RAG systems effectively?"
        enhanced_result = await engine.enhance_query_with_context(
            query=test_query,
            sources=["reddit"],
            max_context_items=3
        )
        
        logger.info(f"✅ Enhanced query with {enhanced_result.get('context_count', 0)} context items")
        logger.info(f"   Sources used: {enhanced_result.get('sources_used', [])}")
        logger.info(f"   Context summary: {enhanced_result.get('context_summary', 'No summary')}")
        
        # Test research brief context
        research_brief = "Research the best practices for implementing RAG systems in production environments"
        context_result = await engine.get_context_for_research_brief(research_brief)
        
        logger.info(f"✅ Research brief context: {len(context_result.get('context_items', []))} items")
        logger.info(f"   Key topics: {context_result.get('key_topics', [])}")
        logger.info(f"   Recommendations: {len(context_result.get('recommendations', []))} recommendations")
        
        # Test engine stats
        stats = engine.get_engine_stats()
        logger.info(f"✅ Engine stats: {stats.get('status', 'unknown')} status")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Retrieval engine test failed: {e}")
        return False

async def test_integration_with_utils():
    """Test integration with open_deep_research.utils."""
    logger.info("Testing Integration with Utils...")
    
    try:
        # Import the enhanced utils
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from open_deep_research.utils import get_rag_engine, get_rag_stats, ingest_mcp_data_to_rag
        
        # Test RAG engine initialization
        rag_engine = get_rag_engine()
        
        if rag_engine:
            logger.info("✅ RAG engine accessible from utils")
        else:
            logger.warning("⚠️ RAG engine not available from utils")
        
        # Test RAG stats
        stats = get_rag_stats()
        logger.info(f"✅ RAG stats: {stats.get('status', 'unknown')}")
        
        # Test MCP data ingestion
        sample_mcp_data = [
            {
                "id": "utils_test",
                "title": "Testing utils integration",
                "selftext": "This is a test of the RAG system integration with the utils module.",
                "author": "test_user",
                "subreddit": "test",
                "score": 1,
                "num_comments": 0,
                "created_utc": "2024-01-01T00:00:00Z",
                "url": "https://test.com",
                "permalink": "/test"
            }
        ]
        
        ingest_result = await ingest_mcp_data_to_rag(sample_mcp_data, "reddit", "posts")
        
        if ingest_result.get("success"):
            logger.info("✅ MCP data ingestion through utils successful")
        else:
            logger.warning(f"⚠️ MCP data ingestion result: {ingest_result}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Utils integration test failed: {e}")
        return False

async def main():
    """Run all RAG system tests."""
    logger.info("🧠 Starting RAG System Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Vector Store", test_vector_store),
        ("Document Processor", test_document_processor),
        ("Retrieval Engine", test_retrieval_engine),
        ("Utils Integration", test_integration_with_utils)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running test: {test_name}")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Results Summary:")
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {status}: {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! RAG system is ready for integration.")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs and configuration.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

