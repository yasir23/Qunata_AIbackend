from pydantic import BaseModel, Field
from typing import Any, List, Optional
from langchain_core.runnables import RunnableConfig
import os
import asyncio
import logging
from enum import Enum

# Import subscription management components
from ..payment.subscription_manager import (
    SubscriptionManager,
    get_user_subscription_info,
    get_user_concurrent_research_limit,
    check_github_mcp_access,
    get_tier_based_concurrent_limit,
    get_tier_based_mcp_servers
)
from ..database.models import SubscriptionTierEnum

# Configure logging
logger = logging.getLogger(__name__)

class SearchAPI(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    url: Optional[str] = Field(
        default=None,
        optional=True,
    )
    """The URL of the MCP server"""
    tools: Optional[List[str]] = Field(
        default=None,
        optional=True,
    )
    """The tools to make available to the LLM"""
    auth_required: Optional[bool] = Field(
        default=False,
        optional=True,
    )
    """Whether the MCP server requires authentication"""

class Configuration(BaseModel):
    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 3,
                "min": 1,
                "max": 10,
                "description": "Maximum number of retries for structured output calls from models"
            }
        }
    )
    allow_clarification: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to allow the researcher to ask the user clarifying questions before starting research"
            }
        }
    )
    max_concurrent_research_units: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 20,
                "step": 1,
                "description": "Maximum number of research units to run concurrently. This will allow the researcher to use multiple sub-agents to conduct research. Note: with more concurrency, you may run into rate limits. This value is automatically adjusted based on your subscription tier."
            }
        }
    )
    
    # Subscription-based configuration fields
    user_id: Optional[str] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "hidden",
                "description": "User ID for subscription-based configuration"
            }
        }
    )
    
    subscription_tier: Optional[SubscriptionTierEnum] = Field(
        default=None,
        metadata={
            "x_oap_ui_config": {
                "type": "hidden",
                "description": "User subscription tier"
            }
        }
    )
    
    enforce_subscription_limits: bool = Field(
        default=True,
        metadata={
            "x_oap_ui_config": {
                "type": "boolean",
                "default": True,
                "description": "Whether to enforce subscription-based limits on configuration"
            }
        }
    )
    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        metadata={
            "x_oap_ui_config": {
                "type": "select",
                "default": "tavily",
                "description": "Search API to use for research. NOTE: Make sure your Researcher Model supports the selected search API.",
                "options": [
                    {"label": "Tavily", "value": SearchAPI.TAVILY.value},
                    {"label": "OpenAI Native Web Search", "value": SearchAPI.OPENAI.value},
                    {"label": "Anthropic Native Web Search", "value": SearchAPI.ANTHROPIC.value},
                    {"label": "None", "value": SearchAPI.NONE.value}
                ]
            }
        }
    )
    max_researcher_iterations: int = Field(
        default=3,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 3,
                "min": 1,
                "max": 10,
                "step": 1,
                "description": "Maximum number of research iterations for the Research Supervisor. This is the number of times the Research Supervisor will reflect on the research and ask follow-up questions."
            }
        }
    )
    max_react_tool_calls: int = Field(
        default=5,
        metadata={
            "x_oap_ui_config": {
                "type": "slider",
                "default": 5,
                "min": 1,
                "max": 30,
                "step": 1,
                "description": "Maximum number of tool calling iterations to make in a single researcher step."
            }
        }
    )
    # Model Configuration
    summarization_model: str = Field(
        default="openai:gpt-4.1-nano",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-nano",
                "description": "Model for summarizing research results from Tavily search results"
            }
        }
    )
    summarization_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for summarization model"
            }
        }
    )
    research_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for conducting research. NOTE: Make sure your Researcher Model supports the selected search API."
            }
        }
    )
    research_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for research model"
            }
        }
    )
    compression_model: str = Field(
        default="openai:gpt-4.1-mini",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1-mini",
                "description": "Model for compressing research findings from sub-agents. NOTE: Make sure your Compression Model supports the selected search API."
            }
        }
    )
    compression_model_max_tokens: int = Field(
        default=8192,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 8192,
                "description": "Maximum output tokens for compression model"
            }
        }
    )
    final_report_model: str = Field(
        default="openai:gpt-4.1",
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "default": "openai:gpt-4.1",
                "description": "Model for writing the final report from all research findings"
            }
        }
    )
    final_report_model_max_tokens: int = Field(
        default=10000,
        metadata={
            "x_oap_ui_config": {
                "type": "number",
                "default": 10000,
                "description": "Maximum output tokens for final report model"
            }
        }
    )
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "mcp",
                "description": "MCP server configuration"
            }
        }
    )
    mcp_prompt: Optional[str] = Field(
        default=None,
        optional=True,
        metadata={
            "x_oap_ui_config": {
                "type": "text",
                "description": "Any additional instructions to pass along to the Agent regarding the MCP tools that are available to it."
            }
        }
    )


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})
    
    @classmethod
    async def from_user_subscription(
        cls, 
        user_id: str, 
        config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance with subscription-based limits."""
        try:
            # Get base configuration
            base_config = cls.from_runnable_config(config)
            
            # Get user subscription information
            subscription_info = await get_user_subscription_info(user_id)
            
            # Apply subscription-based limits
            if base_config.enforce_subscription_limits:
                # Update concurrent research units based on subscription
                subscription_limit = subscription_info.limits.concurrent_research_units
                base_config.max_concurrent_research_units = min(
                    base_config.max_concurrent_research_units,
                    subscription_limit
                )
                
                # Apply tier-based limits for research iterations
                tier_limits = cls._get_tier_based_limits(subscription_info.tier)
                
                # Limit research iterations based on subscription tier
                base_config.max_researcher_iterations = min(
                    base_config.max_researcher_iterations,
                    tier_limits["max_researcher_iterations"]
                )
                
                # Limit tool calls based on subscription tier
                base_config.max_react_tool_calls = min(
                    base_config.max_react_tool_calls,
                    tier_limits["max_react_tool_calls"]
                )
                
                # Apply token limits based on subscription tier
                base_config.research_model_max_tokens = min(
                    base_config.research_model_max_tokens,
                    tier_limits["max_tokens_per_request"]
                )
                
                base_config.final_report_model_max_tokens = min(
                    base_config.final_report_model_max_tokens,
                    tier_limits["max_tokens_per_request"]
                )
                
                # Set subscription information
                base_config.user_id = user_id
                base_config.subscription_tier = subscription_info.tier
                
                logger.info(f"Applied subscription limits for user {user_id}: "
                          f"tier={subscription_info.tier.value}, "
                          f"concurrent_units={base_config.max_concurrent_research_units}, "
                          f"iterations={base_config.max_researcher_iterations}, "
                          f"tool_calls={base_config.max_react_tool_calls}")
            
            return base_config
            
        except Exception as e:
            logger.error(f"Error applying subscription limits for user {user_id}: {e}")
            # Return base configuration on error
            base_config = cls.from_runnable_config(config)
            base_config.user_id = user_id
            return base_config
    
    @staticmethod
    def _get_tier_based_limits(tier: SubscriptionTierEnum) -> dict[str, int]:
        """Get tier-based configuration limits."""
        tier_limits = {
            SubscriptionTierEnum.FREE: {
                "max_researcher_iterations": 2,
                "max_react_tool_calls": 3,
                "max_tokens_per_request": 4000,
                "max_structured_output_retries": 2
            },
            SubscriptionTierEnum.PRO: {
                "max_researcher_iterations": 5,
                "max_react_tool_calls": 10,
                "max_tokens_per_request": 8000,
                "max_structured_output_retries": 3
            },
            SubscriptionTierEnum.ENTERPRISE: {
                "max_researcher_iterations": 10,
                "max_react_tool_calls": 30,
                "max_tokens_per_request": 16000,
                "max_structured_output_retries": 5
            }
        }
        return tier_limits.get(tier, tier_limits[SubscriptionTierEnum.FREE])

    async def validate_subscription_limits(self) -> bool:
        """Validate that current configuration respects subscription limits."""
        if not self.enforce_subscription_limits or not self.user_id:
            return True
        
        try:
            # Get current subscription limits
            subscription_info = await get_user_subscription_info(self.user_id)
            tier_limits = self._get_tier_based_limits(subscription_info.tier)
            
            # Check concurrent research units limit
            max_allowed_concurrent = subscription_info.limits.concurrent_research_units
            if self.max_concurrent_research_units > max_allowed_concurrent:
                logger.warning(f"Configuration exceeds concurrent research units limit: "
                             f"requested={self.max_concurrent_research_units}, "
                             f"allowed={max_allowed_concurrent}")
                return False
            
            # Check researcher iterations limit
            max_allowed_iterations = tier_limits["max_researcher_iterations"]
            if self.max_researcher_iterations > max_allowed_iterations:
                logger.warning(f"Configuration exceeds researcher iterations limit: "
                             f"requested={self.max_researcher_iterations}, "
                             f"allowed={max_allowed_iterations}")
                return False
            
            # Check tool calls limit
            max_allowed_tool_calls = tier_limits["max_react_tool_calls"]
            if self.max_react_tool_calls > max_allowed_tool_calls:
                logger.warning(f"Configuration exceeds tool calls limit: "
                             f"requested={self.max_react_tool_calls}, "
                             f"allowed={max_allowed_tool_calls}")
                return False
            
            # Check token limits
            max_allowed_tokens = tier_limits["max_tokens_per_request"]
            if (self.research_model_max_tokens > max_allowed_tokens or 
                self.final_report_model_max_tokens > max_allowed_tokens):
                logger.warning(f"Configuration exceeds token limits: "
                             f"research_tokens={self.research_model_max_tokens}, "
                             f"report_tokens={self.final_report_model_max_tokens}, "
                             f"allowed={max_allowed_tokens}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating subscription limits: {e}")
            return True  # Allow on error to avoid blocking users
    
    async def apply_subscription_limits(self) -> "Configuration":
        """Apply subscription limits to current configuration."""
        if not self.enforce_subscription_limits or not self.user_id:
            return self
        
        try:
            # Get subscription information
            subscription_info = await get_user_subscription_info(self.user_id)
            tier_limits = self._get_tier_based_limits(subscription_info.tier)
            
            # Apply concurrent research units limit
            max_allowed_concurrent = subscription_info.limits.concurrent_research_units
            if self.max_concurrent_research_units > max_allowed_concurrent:
                self.max_concurrent_research_units = max_allowed_concurrent
                logger.info(f"Applied concurrent research units limit: {max_allowed_concurrent}")
            
            # Apply researcher iterations limit
            max_allowed_iterations = tier_limits["max_researcher_iterations"]
            if self.max_researcher_iterations > max_allowed_iterations:
                self.max_researcher_iterations = max_allowed_iterations
                logger.info(f"Applied researcher iterations limit: {max_allowed_iterations}")
            
            # Apply tool calls limit
            max_allowed_tool_calls = tier_limits["max_react_tool_calls"]
            if self.max_react_tool_calls > max_allowed_tool_calls:
                self.max_react_tool_calls = max_allowed_tool_calls
                logger.info(f"Applied tool calls limit: {max_allowed_tool_calls}")
            
            # Apply token limits
            max_allowed_tokens = tier_limits["max_tokens_per_request"]
            if self.research_model_max_tokens > max_allowed_tokens:
                self.research_model_max_tokens = max_allowed_tokens
                logger.info(f"Applied research model token limit: {max_allowed_tokens}")
            
            if self.final_report_model_max_tokens > max_allowed_tokens:
                self.final_report_model_max_tokens = max_allowed_tokens
                logger.info(f"Applied final report model token limit: {max_allowed_tokens}")
            
            if self.compression_model_max_tokens > max_allowed_tokens:
                self.compression_model_max_tokens = max_allowed_tokens
                logger.info(f"Applied compression model token limit: {max_allowed_tokens}")
            
            if self.summarization_model_max_tokens > max_allowed_tokens:
                self.summarization_model_max_tokens = max_allowed_tokens
                logger.info(f"Applied summarization model token limit: {max_allowed_tokens}")
            
            # Apply structured output retries limit
            max_allowed_retries = tier_limits["max_structured_output_retries"]
            if self.max_structured_output_retries > max_allowed_retries:
                self.max_structured_output_retries = max_allowed_retries
                logger.info(f"Applied structured output retries limit: {max_allowed_retries}")
            
            # Update subscription tier
            self.subscription_tier = subscription_info.tier
            
            logger.info(f"Applied all subscription limits for user {self.user_id} (tier: {subscription_info.tier.value})")
            return self
            
        except Exception as e:
            logger.error(f"Error applying subscription limits: {e}")
            return self
    
    async def check_mcp_server_access(self, server_name: str) -> bool:
        """Check if user has access to a specific MCP server based on subscription."""
        if not self.user_id:
            # Default access for unauthenticated users (Reddit/YouTube only)
            return server_name.lower() in ["reddit", "youtube"]
        
        try:
            # Check GitHub MCP access specifically
            if server_name.lower() == "github":
                return await check_github_mcp_access(self.user_id)
            
            # Check general MCP server access
            subscription_info = await get_user_subscription_info(self.user_id)
            allowed_servers = subscription_info.features.mcp_servers
            return server_name.lower() in [s.lower() for s in allowed_servers]
            
        except Exception as e:
            logger.error(f"Error checking MCP server access: {e}")
            # Default to basic access on error
            return server_name.lower() in ["reddit", "youtube"]
    
    def get_subscription_summary(self) -> dict[str, Any]:
        """Get comprehensive summary of subscription-based configuration."""
        tier_limits = {}
        if self.subscription_tier:
            tier_limits = self._get_tier_based_limits(self.subscription_tier)
        
        return {
            "user_id": self.user_id,
            "subscription_tier": self.subscription_tier.value if self.subscription_tier else "unknown",
            "enforce_limits": self.enforce_subscription_limits,
            "subscription_aware": self.user_id is not None,
            "current_limits": {
                "max_concurrent_research_units": self.max_concurrent_research_units,
                "max_researcher_iterations": self.max_researcher_iterations,
                "max_react_tool_calls": self.max_react_tool_calls,
                "research_model_max_tokens": self.research_model_max_tokens,
                "final_report_model_max_tokens": self.final_report_model_max_tokens,
                "compression_model_max_tokens": self.compression_model_max_tokens,
                "summarization_model_max_tokens": self.summarization_model_max_tokens,
                "max_structured_output_retries": self.max_structured_output_retries
            },
            "tier_limits": tier_limits,
            "mcp_config": {
                "url": self.mcp_config.url if self.mcp_config else None,
                "tools": self.mcp_config.tools if self.mcp_config else None,
                "auth_required": self.mcp_config.auth_required if self.mcp_config else None
            }
        }
    
    async def get_effective_limits(self) -> dict[str, Any]:
        """Get effective limits after applying subscription restrictions."""
        if not self.user_id or not self.enforce_subscription_limits:
            return self.get_subscription_summary()
        
        try:
            # Get subscription information
            subscription_info = await get_user_subscription_info(self.user_id)
            tier_limits = self._get_tier_based_limits(subscription_info.tier)
            
            # Calculate effective limits (minimum of configured and subscription limits)
            effective_limits = {
                "max_concurrent_research_units": min(
                    self.max_concurrent_research_units,
                    subscription_info.limits.concurrent_research_units
                ),
                "max_researcher_iterations": min(
                    self.max_researcher_iterations,
                    tier_limits["max_researcher_iterations"]
                ),
                "max_react_tool_calls": min(
                    self.max_react_tool_calls,
                    tier_limits["max_react_tool_calls"]
                ),
                "max_tokens_per_request": tier_limits["max_tokens_per_request"],
                "max_structured_output_retries": min(
                    self.max_structured_output_retries,
                    tier_limits["max_structured_output_retries"]
                )
            }
            
            return {
                "user_id": self.user_id,
                "subscription_tier": subscription_info.tier.value,
                "subscription_status": subscription_info.status.value,
                "enforce_limits": self.enforce_subscription_limits,
                "effective_limits": effective_limits,
                "subscription_features": subscription_info.features.model_dump(),
                "mcp_server_access": {
                    "allowed_servers": subscription_info.features.mcp_servers,
                    "github_access": subscription_info.features.github_mcp_access,
                    "advanced_rag": subscription_info.features.advanced_rag
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting effective limits: {e}")
            return self.get_subscription_summary()
    
    @classmethod
    def get_tier_limits_comparison(cls) -> dict[str, Any]:
        """Get comparison of limits across all subscription tiers."""
        return {
            tier.value: cls._get_tier_based_limits(tier)
            for tier in SubscriptionTierEnum
        }

    class Config:
        arbitrary_types_allowed = True






