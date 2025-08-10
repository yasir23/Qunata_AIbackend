import os
import aiohttp
import asyncio
import logging
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, List, Literal, Dict, Optional, Any
from langchain_core.tools import BaseTool, StructuredTool, tool, ToolException, InjectedToolArg
from langchain_core.messages import HumanMessage, AIMessage, MessageLikeRepresentation, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model
from tavily import AsyncTavilyClient
from langgraph.config import get_store
from mcp import McpError
from langchain_mcp_adapters.client import MultiServerMCPClient
from open_deep_research.state import Summary, ResearchComplete
from open_deep_research.configuration import SearchAPI, Configuration
from open_deep_research.prompts import summarize_webpage_prompt

# RAG system imports
try:
    from rag.retrieval_engine import RetrievalEngine
    from rag.vector_store import VectorStore
    from rag.document_processor import DocumentProcessor
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    logging.warning("RAG system not available - install chromadb and dependencies")

# Analytics and usage tracking imports
try:
    from analytics.usage_tracker import UsageTracker, UsageType, QuotaEnforcer
    ANALYTICS_AVAILABLE = True
except ImportError:
    ANALYTICS_AVAILABLE = False
    logging.warning("Analytics system not available - install redis and dependencies")


##########################
# Tavily Search Tool Utils
##########################
TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    config: RunnableConfig = None
) -> str:
    """
    Fetches results from Tavily search API.

    Args
        queries (List[str]): List of search queries, you can pass in as many queries as you need.
        max_results (int): Maximum number of results to return
        topic (Literal['general', 'news', 'finance']): Topic to filter results by

    Returns:
        str: A formatted string of search results
    """
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        config=config
    )
    # Format the search results and deduplicate results by URL
    formatted_output = f"Search results: \n\n"
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results:
                unique_results[url] = {**result, "query": response['query']}
    configurable = Configuration.from_runnable_config(config)
    max_char_to_include = 50_000   # NOTE: This can be tuned by the developer. This character count keeps us safely under input token limits for the latest models.
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(
        model=configurable.summarization_model,
        max_tokens=configurable.summarization_model_max_tokens,
        api_key=model_api_key,
        tags=["langsmith:nostream"]
    ).with_structured_output(Summary).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    async def noop():
        return None
    summarization_tasks = [
        noop() if not result.get("raw_content") else summarize_webpage(
            summarization_model, 
            result['raw_content'][:max_char_to_include],
        )
        for result in unique_results.values()
    ]
    summaries = await asyncio.gather(*summarization_tasks)
    summarized_results = {
        url: {'title': result['title'], 'content': result['content'] if summary is None else summary}
        for url, result, summary in zip(unique_results.keys(), unique_results.values(), summaries)
    }
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += f"\n\n--- SOURCE {i+1}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    if summarized_results:
        return formatted_output
    else:
        return "No valid search results found. Please try different search queries or use a different search API."


async def tavily_search_async(search_queries, max_results: int = 5, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = True, config: RunnableConfig = None):
    tavily_async_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    search_tasks = []
    for query in search_queries:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                    topic=topic
                )
            )
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    try:
        summary = await asyncio.wait_for(
            model.ainvoke([HumanMessage(content=summarize_webpage_prompt.format(webpage_content=webpage_content, date=get_today_str()))]),
            timeout=60.0
        )
        return f"""<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"""
    except (asyncio.TimeoutError, Exception) as e:
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content


##########################
# MCP Utils
##########################
async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[Dict[str, Any]]:
    try:
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                base_mcp_url.rstrip("/") + "/oauth/token",
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=form_data,
            ) as token_response:
                if token_response.status == 200:
                    token_data = await token_response.json()
                    return token_data
                else:
                    response_text = await token_response.text()
                    logging.error(f"Token exchange failed: {response_text}")
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")
    return None

async def get_tokens(config: RunnableConfig):
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    expires_in = tokens.value.get("expires_in")  # seconds until expiration
    created_at = tokens.created_at  # datetime of token creation
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    if current_time > expiration_time:
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    await store.aput((user_id, "tokens"), "data", tokens)
    return

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))

    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    old_coroutine = tool.coroutine
    async def wrapped_mcp_coroutine(**kwargs):
        def _find_first_mcp_error_nested(exc: BaseException) -> McpError | None:
            if isinstance(exc, McpError):
                return exc
            if isinstance(exc, ExceptionGroup):
                for sub_exc in exc.exceptions:
                    if found := _find_first_mcp_error_nested(sub_exc):
                        return found
            return None
        try:
            return await old_coroutine(**kwargs)
        except BaseException as e_orig:
            mcp_error = _find_first_mcp_error_nested(e_orig)
            if not mcp_error:
                raise e_orig
            error_details = mcp_error.error
            is_interaction_required = getattr(error_details, "code", None) == -32003
            error_data = getattr(error_details, "data", None) or {}
            if is_interaction_required:
                message_payload = error_data.get("message", {})
                error_message_text = "Required interaction"
                if isinstance(message_payload, dict):
                    error_message_text = (
                        message_payload.get("text") or error_message_text
                    )
                if url := error_data.get("url"):
                    error_message_text = f"{error_message_text} {url}"
                raise ToolException(error_message_text) from e_orig
            raise e_orig
    tool.coroutine = wrapped_mcp_coroutine
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    configurable = Configuration.from_runnable_config(config)
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    if not (configurable.mcp_config and configurable.mcp_config.url and configurable.mcp_config.tools and (mcp_tokens or not configurable.mcp_config.auth_required)):
        return []
    tools = []
    # TODO: When the Multi-MCP Server support is merged in OAP, update this code.
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    mcp_server_config = {
        "server_1":{
            "url": server_url,
            "headers": {"Authorization": f"Bearer {mcp_tokens['access_token']}"} if mcp_tokens else None,
            "transport": "streamable_http"
        }
    }
    try:
        client = MultiServerMCPClient(mcp_server_config)
        mcp_tools = await client.get_tools()
    except Exception as e:
        print(f"Error loading MCP tools: {e}")
        return []
    for tool in mcp_tools:
        if tool.name in existing_tool_names:
            warnings.warn(
                f"Trying to add MCP tool with a name {tool.name} that is already in use - this tool will be ignored."
            )
            continue
        if tool.name not in set(configurable.mcp_config.tools):
            continue
        tools.append(wrap_mcp_authenticate_tool(tool))
    return tools


##########################
# Tool Utils
##########################
async def get_search_tool(search_api: SearchAPI):
    if search_api == SearchAPI.ANTHROPIC:
        return [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    elif search_api == SearchAPI.OPENAI:
        return [{"type": "web_search_preview"}]
    elif search_api == SearchAPI.TAVILY:
        search_tool = tavily_search
        search_tool.metadata = {**(search_tool.metadata or {}), "type": "search", "name": "web_search"}
        return [search_tool]
    elif search_api == SearchAPI.NONE:
        return []
    
async def get_all_tools(config: RunnableConfig):
    tools = [tool(ResearchComplete)]
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    tools.extend(await get_search_tool(search_api))
    existing_tool_names = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search") for tool in tools}
    mcp_tools = await load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    return tools

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


##########################
# RAG-Enhanced Search Tools
##########################

# Global RAG engine instance
_rag_engine = None

def get_rag_engine() -> Optional['RetrievalEngine']:
    """Get or create the global RAG engine instance."""
    global _rag_engine
    if not RAG_AVAILABLE:
        return None
    
    if _rag_engine is None:
        try:
            _rag_engine = RetrievalEngine()
            logging.info("RAG engine initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize RAG engine: {e}")
            return None
    
    return _rag_engine

@tool(description="Enhanced search with RAG context from historical research, GitHub issues, and social media insights")
async def rag_enhanced_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    include_context: Annotated[bool, InjectedToolArg] = True,
    context_sources: Annotated[List[str], InjectedToolArg] = ["reddit", "github", "youtube"],
    config: RunnableConfig = None
) -> str:
    """
    Enhanced search that combines web search with RAG context from MCP servers.
    
    Args:
        queries: List of search queries
        max_results: Maximum number of web search results
        include_context: Whether to include RAG context
        context_sources: Sources to search for context (reddit, github, youtube)
        
    Returns:
        Formatted string with web search results and relevant context
    """
    try:
        # Perform regular web search
        web_results = await tavily_search(queries, max_results, config=config)
        
        if not include_context or not RAG_AVAILABLE:
            return web_results
        
        # Get RAG engine
        rag_engine = get_rag_engine()
        if not rag_engine:
            return web_results
        
        # Enhance queries with RAG context
        enhanced_context = []
        for query in queries:
            try:
                context_result = await rag_engine.enhance_query_with_context(
                    query=query,
                    sources=context_sources,
                    max_context_items=5
                )
                
                if context_result.get("context"):
                    enhanced_context.append({
                        "query": query,
                        "context": context_result["context"],
                        "context_summary": context_result.get("context_summary", ""),
                        "sources_used": context_result.get("sources_used", [])
                    })
            except Exception as e:
                logging.error(f"Error getting RAG context for query '{query}': {e}")
                continue
        
        # Format combined results
        if enhanced_context:
            formatted_output = web_results + "\n\n" + "=" * 80 + "\n"
            formatted_output += "## 🧠 RELEVANT CONTEXT FROM HISTORICAL RESEARCH\n\n"
            
            for ctx in enhanced_context:
                formatted_output += f"### Query: {ctx['query']}\n"
                formatted_output += f"**Sources:** {', '.join(ctx['sources_used'])}\n"
                formatted_output += f"**Summary:** {ctx['context_summary']}\n\n"
                
                for i, item in enumerate(ctx['context'][:3], 1):  # Show top 3 context items
                    metadata = item.get('metadata', {})
                    content = item.get('content', '')[:300] + "..." if len(item.get('content', '')) > 300 else item.get('content', '')
                    
                    formatted_output += f"**Context {i}** (Similarity: {item.get('similarity_score', 0):.2f})\n"
                    formatted_output += f"Source: {metadata.get('source', 'unknown')} | Type: {metadata.get('type', 'unknown')}\n"
                    
                    if metadata.get('title'):
                        formatted_output += f"Title: {metadata['title']}\n"
                    if metadata.get('url'):
                        formatted_output += f"URL: {metadata['url']}\n"
                    
                    formatted_output += f"Content: {content}\n"
                    formatted_output += "---\n\n"
            
            return formatted_output
        else:
            return web_results
            
    except Exception as e:
        logging.error(f"Error in RAG-enhanced search: {e}")
        # Fallback to regular search
        return await tavily_search(queries, max_results, config=config)

async def ingest_mcp_data_to_rag(
    mcp_data: Any,
    source: str,
    data_type: str = "auto"
) -> Dict[str, Any]:
    """
    Ingest data from MCP servers into the RAG system.
    
    Args:
        mcp_data: Data from MCP server
        source: Source name (reddit, youtube, github)
        data_type: Type of data (auto-detect if 'auto')
        
    Returns:
        Ingestion results
    """
    if not RAG_AVAILABLE:
        return {"success": False, "error": "RAG system not available"}
    
    rag_engine = get_rag_engine()
    if not rag_engine:
        return {"success": False, "error": "RAG engine not initialized"}
    
    try:
        result = await rag_engine.ingest_mcp_data(mcp_data, source, data_type)
        return result
    except Exception as e:
        logging.error(f"Error ingesting MCP data to RAG: {e}")
        return {"success": False, "error": str(e)}

async def get_research_context(research_brief: str) -> Dict[str, Any]:
    """
    Get relevant context for a research brief from the RAG system.
    
    Args:
        research_brief: The research brief text
        
    Returns:
        Context information for the research brief
    """
    if not RAG_AVAILABLE:
        return {"context_items": [], "context_summary": "RAG system not available"}
    
    rag_engine = get_rag_engine()
    if not rag_engine:
        return {"context_items": [], "context_summary": "RAG engine not initialized"}
    
    try:
        context = await rag_engine.get_context_for_research_brief(research_brief)
        return context
    except Exception as e:
        logging.error(f"Error getting research context: {e}")
        return {"context_items": [], "context_summary": f"Error: {str(e)}", "error": str(e)}

def get_rag_stats() -> Dict[str, Any]:
    """Get statistics about the RAG system."""
    if not RAG_AVAILABLE:
        return {"status": "not_available", "error": "RAG system not installed"}
    
    rag_engine = get_rag_engine()
    if not rag_engine:
        return {"status": "not_initialized", "error": "RAG engine not initialized"}
    
    try:
        return rag_engine.get_engine_stats()
    except Exception as e:
        logging.error(f"Error getting RAG stats: {e}")
        return {"status": "error", "error": str(e)}


##########################
# Usage Tracking Integration
##########################

# Global usage tracker instance
_usage_tracker = None

def get_usage_tracker() -> Optional['UsageTracker']:
    """Get or create the global usage tracker instance."""
    global _usage_tracker
    if not ANALYTICS_AVAILABLE:
        return None
    
    if _usage_tracker is None:
        try:
            # Initialize with Redis if available
            import redis
            try:
                redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
                redis_client.ping()  # Test connection
            except Exception:
                redis_client = None
                logging.warning("Redis not available, using in-memory cache for usage tracking")
            
            # Initialize Stripe and Firebase integrations if available
            stripe_service = None
            firebase_integration = None
            
            try:
                from payments.stripe_service import StripeService
                from payments.firebase_integration import FirebasePaymentIntegration
                
                stripe_service = StripeService()
                firebase_integration = FirebasePaymentIntegration()
            except Exception as e:
                logging.warning(f"Billing integration not available: {e}")
            
            _usage_tracker = UsageTracker(
                redis_client=redis_client,
                stripe_service=stripe_service,
                firebase_integration=firebase_integration
            )
            logging.info("Usage tracker initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize usage tracker: {e}")
            return None
    
    return _usage_tracker

async def track_research_request(
    user_id: str,
    request_type: str = "research",
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Track a research request with quota enforcement.
    
    Args:
        user_id: User ID
        request_type: Type of research request
        metadata: Additional metadata
        session_id: Session ID for grouping
        request_id: Request ID for tracing
        
    Returns:
        Tracking result with quota status
    """
    if not ANALYTICS_AVAILABLE:
        return {"success": True, "message": "Analytics not available"}
    
    usage_tracker = get_usage_tracker()
    if not usage_tracker:
        return {"success": True, "message": "Usage tracker not available"}
    
    try:
        result = await usage_tracker.track_usage(
            user_id=user_id,
            usage_type=UsageType.RESEARCH_REQUEST,
            amount=1,
            metadata={
                "request_type": request_type,
                **(metadata or {})
            },
            session_id=session_id,
            request_id=request_id,
            check_quota=True
        )
        
        return result
        
    except Exception as e:
        logging.error(f"Error tracking research request: {e}")
        return {"success": True, "error": str(e)}

async def track_api_call(
    user_id: str,
    api_endpoint: str,
    method: str = "POST",
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Track an API call with quota enforcement.
    
    Args:
        user_id: User ID
        api_endpoint: API endpoint called
        method: HTTP method
        metadata: Additional metadata
        session_id: Session ID for grouping
        request_id: Request ID for tracing
        
    Returns:
        Tracking result with quota status
    """
    if not ANALYTICS_AVAILABLE:
        return {"success": True, "message": "Analytics not available"}
    
    usage_tracker = get_usage_tracker()
    if not usage_tracker:
        return {"success": True, "message": "Usage tracker not available"}
    
    try:
        result = await usage_tracker.track_usage(
            user_id=user_id,
            usage_type=UsageType.API_CALL,
            amount=1,
            metadata={
                "api_endpoint": api_endpoint,
                "method": method,
                **(metadata or {})
            },
            session_id=session_id,
            request_id=request_id,
            check_quota=True
        )
        
        return result
        
    except Exception as e:
        logging.error(f"Error tracking API call: {e}")
        return {"success": True, "error": str(e)}

async def track_token_usage(
    user_id: str,
    model_name: str,
    tokens_used: int,
    operation: str = "completion",
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Track token usage with quota enforcement.
    
    Args:
        user_id: User ID
        model_name: Name of the model used
        tokens_used: Number of tokens used
        operation: Type of operation (completion, embedding, etc.)
        metadata: Additional metadata
        session_id: Session ID for grouping
        request_id: Request ID for tracing
        
    Returns:
        Tracking result with quota status
    """
    if not ANALYTICS_AVAILABLE:
        return {"success": True, "message": "Analytics not available"}
    
    usage_tracker = get_usage_tracker()
    if not usage_tracker:
        return {"success": True, "message": "Usage tracker not available"}
    
    try:
        result = await usage_tracker.track_usage(
            user_id=user_id,
            usage_type=UsageType.TOKEN_USAGE,
            amount=tokens_used,
            metadata={
                "model_name": model_name,
                "operation": operation,
                **(metadata or {})
            },
            session_id=session_id,
            request_id=request_id,
            check_quota=True
        )
        
        return result
        
    except Exception as e:
        logging.error(f"Error tracking token usage: {e}")
        return {"success": True, "error": str(e)}

async def track_mcp_call(
    user_id: str,
    server_name: str,
    tool_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Track MCP server call with quota enforcement.
    
    Args:
        user_id: User ID
        server_name: Name of MCP server
        tool_name: Name of tool called
        metadata: Additional metadata
        session_id: Session ID for grouping
        request_id: Request ID for tracing
        
    Returns:
        Tracking result with quota status
    """
    if not ANALYTICS_AVAILABLE:
        return {"success": True, "message": "Analytics not available"}
    
    usage_tracker = get_usage_tracker()
    if not usage_tracker:
        return {"success": True, "message": "Usage tracker not available"}
    
    try:
        result = await usage_tracker.track_usage(
            user_id=user_id,
            usage_type=UsageType.MCP_CALL,
            amount=1,
            metadata={
                "server_name": server_name,
                "tool_name": tool_name,
                **(metadata or {})
            },
            session_id=session_id,
            request_id=request_id,
            check_quota=True
        )
        
        return result
        
    except Exception as e:
        logging.error(f"Error tracking MCP call: {e}")
        return {"success": True, "error": str(e)}

async def track_rag_query(
    user_id: str,
    query_type: str = "context_retrieval",
    results_count: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Track RAG query with quota enforcement.
    
    Args:
        user_id: User ID
        query_type: Type of RAG query
        results_count: Number of results returned
        metadata: Additional metadata
        session_id: Session ID for grouping
        request_id: Request ID for tracing
        
    Returns:
        Tracking result with quota status
    """
    if not ANALYTICS_AVAILABLE:
        return {"success": True, "message": "Analytics not available"}
    
    usage_tracker = get_usage_tracker()
    if not usage_tracker:
        return {"success": True, "message": "Usage tracker not available"}
    
    try:
        result = await usage_tracker.track_usage(
            user_id=user_id,
            usage_type=UsageType.RAG_QUERY,
            amount=1,
            metadata={
                "query_type": query_type,
                "results_count": results_count,
                **(metadata or {})
            },
            session_id=session_id,
            request_id=request_id,
            check_quota=True
        )
        
        return result
        
    except Exception as e:
        logging.error(f"Error tracking RAG query: {e}")
        return {"success": True, "error": str(e)}

async def check_usage_quota(
    user_id: str,
    usage_type: UsageType,
    amount: int = 1
) -> Tuple[bool, str]:
    """
    Check if user can perform an operation within quota limits.
    
    Args:
        user_id: User ID
        usage_type: Type of usage to check
        amount: Amount of usage to check
        
    Returns:
        Tuple of (allowed, message)
    """
    if not ANALYTICS_AVAILABLE:
        return True, "Analytics not available"
    
    usage_tracker = get_usage_tracker()
    if not usage_tracker:
        return True, "Usage tracker not available"
    
    try:
        allowed, status, details = await usage_tracker.quota_enforcer.check_quota(
            user_id=user_id,
            usage_type=usage_type,
            amount=amount
        )
        
        message = details.get("message", f"Quota check: {status.value}")
        return allowed, message
        
    except Exception as e:
        logging.error(f"Error checking usage quota: {e}")
        return True, f"Error checking quota: {str(e)}"

async def get_user_usage_summary(
    user_id: str,
    hours: int = 24
) -> Dict[str, Any]:
    """
    Get usage summary for a user.
    
    Args:
        user_id: User ID
        hours: Number of hours to look back
        
    Returns:
        Usage summary
    """
    if not ANALYTICS_AVAILABLE:
        return {"error": "Analytics not available"}
    
    usage_tracker = get_usage_tracker()
    if not usage_tracker:
        return {"error": "Usage tracker not available"}
    
    try:
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        summary = await usage_tracker.get_usage_summary(user_id, start_time, end_time)
        return asdict(summary)
        
    except Exception as e:
        logging.error(f"Error getting usage summary: {e}")
        return {"error": str(e)}

def get_usage_tracker_stats() -> Dict[str, Any]:
    """Get usage tracker statistics."""
    if not ANALYTICS_AVAILABLE:
        return {"status": "not_available", "error": "Analytics system not installed"}
    
    usage_tracker = get_usage_tracker()
    if not usage_tracker:
        return {"status": "not_initialized", "error": "Usage tracker not initialized"}
    
    try:
        return usage_tracker.get_tracker_stats()
    except Exception as e:
        logging.error(f"Error getting usage tracker stats: {e}")
        return {"status": "error", "error": str(e)}


##########################
# Model Provider Native Websearch Utils
##########################
def anthropic_websearch_called(response):
    try:
        usage = response.response_metadata.get("usage")
        if not usage:
            return False
        server_tool_use = usage.get("server_tool_use")
        if not server_tool_use:
            return False
        web_search_requests = server_tool_use.get("web_search_requests")
        if web_search_requests is None:
            return False
        return web_search_requests > 0
    except (AttributeError, TypeError):
        return False

def openai_websearch_called(response):
    tool_outputs = response.additional_kwargs.get("tool_outputs")
    if tool_outputs:
        for tool_output in tool_outputs:
            if tool_output.get("type") == "web_search_call":
                return True
    return False


##########################
# Token Limit Exceeded Utils
##########################
def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    error_str = str(exception).lower()
    provider = None
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)
    
    return (_check_openai_token_limit(exception, error_str) or
            _check_anthropic_token_limit(exception, error_str) or
            _check_gemini_token_limit(exception, error_str))

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    is_openai_exception = ('openai' in exception_type.lower() or 
                          'openai' in module_name.lower())
    is_bad_request = class_name in ['BadRequestError', 'InvalidRequestError']
    if is_openai_exception and is_bad_request:
        token_keywords = ['token', 'context', 'length', 'maximum context', 'reduce']
        if any(keyword in error_str for keyword in token_keywords):
            return True
    if hasattr(exception, 'code') and hasattr(exception, 'type'):
        if (getattr(exception, 'code', '') == 'context_length_exceeded' or
            getattr(exception, 'type', '') == 'invalid_request_error'):
            return True
    return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    is_anthropic_exception = ('anthropic' in exception_type.lower() or 
                             'anthropic' in module_name.lower())
    is_bad_request = class_name == 'BadRequestError'
    if is_anthropic_exception and is_bad_request:
        if 'prompt is too long' in error_str:
            return True
    return False

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    is_google_exception = ('google' in exception_type.lower() or 'google' in module_name.lower())
    is_resource_exhausted = class_name in ['ResourceExhausted', 'GoogleGenerativeAIFetchError']
    if is_google_exception and is_resource_exhausted:
        return True
    if 'google.api_core.exceptions.resourceexhausted' in exception_type.lower():
        return True
    
    return False

# NOTE: This may be out of date or not applicable to your models. Please update this as needed.
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o4-mini": 200000,
    "openai:o3-mini": 200000,
    "openai:o3": 200000,
    "openai:o3-pro": 200000,
    "openai:o1": 200000,
    "openai:o1-pro": 200000,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-7-sonnet": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-pro": 32768,
    "cohere:command-r-plus": 128000,
    "cohere:command-r": 128000,
    "cohere:command-light": 4096,
    "cohere:command": 4096,
    "mistral:mistral-large": 32768,
    "mistral:mistral-medium": 32768,
    "mistral:mistral-small": 32768,
    "mistral:mistral-7b-instruct": 32768,
    "ollama:codellama": 16384,
    "ollama:llama2:70b": 4096,
    "ollama:llama2:13b": 4096,
    "ollama:llama2": 4096,
    "ollama:mistral": 32768,
}

def get_model_token_limit(model_string):
    for key, token_limit in MODEL_TOKEN_LIMITS.items():
        if key in model_string:
            return token_limit
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            return messages[:i]  # Return everything up to (but not including) the last AI message
    return messages

##########################
# Misc Utils
##########################
def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

def get_config_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    model_name = model_name.lower()
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        if model_name.startswith("openai:"):
            return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return api_keys.get("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return api_keys.get("GOOGLE_API_KEY")
        return None
    else:
        if model_name.startswith("openai:"): 
            return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"):
            return os.getenv("ANTHROPIC_API_KEY")
        elif model_name.startswith("google"):
            return os.getenv("GOOGLE_API_KEY")
        return None

def get_tavily_api_key(config: RunnableConfig):
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if not api_keys:
            return None
        return api_keys.get("TAVILY_API_KEY")
    else:
        return os.getenv("TAVILY_API_KEY")


##########################
# Tool Loading Utils
##########################

async def _load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """Load MCP tools from configured servers."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.mcp_config or not configurable.mcp_config.url:
        return []

    # Convert single MCPConfig to MultiServerMCPClient format
    mcp_server_config = {
        "default": {
            "url": configurable.mcp_config.url,
            "transport": "streamable_http",
        }
    }
    
    try:
        client = MultiServerMCPClient(mcp_server_config)
        mcp_tools = await client.get_tools()
        filtered_mcp_tools: list[BaseTool] = []
        
        for tool in mcp_tools:
            # Skip tools with conflicting names
            if tool.name in existing_tool_names:
                warnings.warn(
                    f"Trying to add MCP tool with a name {tool.name} that is already in use - this tool will be ignored."
                )
                continue

            # Filter tools if specific tools are configured
            if configurable.mcp_config.tools and tool.name not in configurable.mcp_config.tools:
                continue

            filtered_mcp_tools.append(tool)

        return filtered_mcp_tools
    except Exception as e:
        warnings.warn(f"Failed to load MCP tools: {e}")
        return []


async def get_all_tools(config: RunnableConfig) -> list[BaseTool]:
    """Get all available tools based on configuration.
    
    This function loads search tools (like tavily_search) and MCP tools
    based on the provided configuration.
    
    Args:
        config: RunnableConfig containing the configuration settings
        
    Returns:
        List of BaseTool instances available for use
    """
    configurable = Configuration.from_runnable_config(config)
    tools: list[BaseTool] = []
    
    # Add search tools based on configuration
    if configurable.search_api == SearchAPI.TAVILY:
        tools.append(tavily_search)
    # Note: Other search APIs can be added here as needed
    
    # Get existing tool names to avoid conflicts
    existing_tool_names = {tool.name for tool in tools}
    
    # Load MCP tools if configured
    mcp_tools = await _load_mcp_tools(config, existing_tool_names)
    tools.extend(mcp_tools)
    
    return tools




