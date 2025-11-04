"""OpenAI API integration for TMR framework.

This module provides a complete integration with OpenAI's API, including:
- Client wrapper with rate limiting and error handling
- Prompt templates for different reasoning tasks
- Response parsing and validation
- Cost tracking
- TMR verification support

Example:
    >>> from tmr.integrations import TMROpenAI
    >>> client = TMROpenAI(api_key="your-key")
    >>> response = client.chat_completion(
    ...     messages=[{"role": "user", "content": "Solve: 2x + 5 = 13"}],
    ...     verify=True
    ... )
    >>> print(response.content)
"""

# Main client
from .openai_client import (
    TMROpenAI,
    create_client,
    quick_completion,
    quick_completion_async,
)

# Configuration
from .config import (
    OpenAIConfig,
    ModelConfig,
    MODEL_CONFIGS,
    get_model_config,
    calculate_cost,
)

# Prompt templates
from .prompt_templates import (
    PromptType,
    PromptTemplate,
    PromptBuilder,
    get_template,
    create_messages,
    TEMPLATES,
    SYSTEM_PROMPTS,
)

# Response parsing
from .response_parsing import (
    ParsedResponse,
    ReasoningStep,
    ResponseParser,
    CostTracker,
)

# Rate limiting
from .rate_limiting import (
    RateLimiter,
    RateLimitConfig,
    TokenEstimator,
    ExponentialBackoff,
)

# Error handling
from .error_handling import (
    TMRAPIError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    APIConnectionError,
    APITimeoutError,
    ServiceUnavailableError,
    ModelNotFoundError,
    ContextLengthExceededError,
    ContentFilterError,
    ErrorHandler,
    with_retry,
    ErrorContext,
)

__all__ = [
    # Main client
    "TMROpenAI",
    "create_client",
    "quick_completion",
    "quick_completion_async",
    # Configuration
    "OpenAIConfig",
    "ModelConfig",
    "MODEL_CONFIGS",
    "get_model_config",
    "calculate_cost",
    # Prompt templates
    "PromptType",
    "PromptTemplate",
    "PromptBuilder",
    "get_template",
    "create_messages",
    "TEMPLATES",
    "SYSTEM_PROMPTS",
    # Response parsing
    "ParsedResponse",
    "ReasoningStep",
    "ResponseParser",
    "CostTracker",
    # Rate limiting
    "RateLimiter",
    "RateLimitConfig",
    "TokenEstimator",
    "ExponentialBackoff",
    # Error handling
    "TMRAPIError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidRequestError",
    "APIConnectionError",
    "APITimeoutError",
    "ServiceUnavailableError",
    "ModelNotFoundError",
    "ContextLengthExceededError",
    "ContentFilterError",
    "ErrorHandler",
    "with_retry",
    "ErrorContext",
]
