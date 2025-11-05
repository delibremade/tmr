"""OpenAI API client wrapper with TMR integration."""

import logging
from typing import Optional, Dict, Any, List, Union
import asyncio

try:
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    AsyncOpenAI = None

from .config import OpenAIConfig, calculate_cost
from .prompt_templates import PromptBuilder, PromptType, create_messages
from .response_parsing import ResponseParser, ParsedResponse, CostTracker
from .rate_limiting import RateLimiter, RateLimitConfig, TokenEstimator
from .error_handling import (
    with_retry,
    handle_openai_error,
    ErrorHandler,
    TMRAPIError,
)

logger = logging.getLogger(__name__)


class TMROpenAI:
    """OpenAI API client with TMR framework integration.

    Provides a wrapper around OpenAI's API with additional features:
    - Automatic rate limiting (RPM and TPM)
    - Error handling and retries with exponential backoff
    - Response parsing and validation
    - Cost tracking
    - Prompt templates
    - Optional TMR verification

    Example:
        >>> client = TMROpenAI(api_key="your-key")
        >>> response = client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     verify=True
        ... )
        >>> print(response.content)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[OpenAIConfig] = None,
        enable_rate_limiting: bool = True,
        enable_cost_tracking: bool = True,
    ):
        """Initialize TMR OpenAI client.

        Args:
            api_key: OpenAI API key (if not provided, uses config or env)
            config: OpenAI configuration
            enable_rate_limiting: Whether to enable rate limiting
            enable_cost_tracking: Whether to track API costs

        Raises:
            ImportError: If openai package is not installed
            ValueError: If no API key is provided
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package is not installed. "
                "Install it with: pip install openai"
            )

        # Load configuration
        if config is None:
            if api_key:
                config = OpenAIConfig(api_key=api_key)
            else:
                config = OpenAIConfig.from_env()

        self.config = config

        # Initialize OpenAI clients
        self.client = OpenAI(
            api_key=self.config.api_key,
            organization=self.config.org_id,
            timeout=self.config.timeout,
            max_retries=0,  # We handle retries ourselves
        )

        self.async_client = AsyncOpenAI(
            api_key=self.config.api_key,
            organization=self.config.org_id,
            timeout=self.config.timeout,
            max_retries=0,  # We handle retries ourselves
        )

        # Initialize rate limiter
        self.rate_limiter: Optional[RateLimiter] = None
        if enable_rate_limiting:
            rate_limit_config = RateLimitConfig(
                requests_per_minute=self.config.get_effective_rpm(),
                tokens_per_minute=self.config.get_effective_tpm(),
                max_retries=self.config.max_retries,
            )
            self.rate_limiter = RateLimiter(rate_limit_config)

        # Initialize utilities
        self.parser = ResponseParser()
        self.cost_tracker = CostTracker() if enable_cost_tracking else None

        logger.info(
            f"Initialized TMR OpenAI client with model={self.config.model}, "
            f"rate_limiting={enable_rate_limiting}, cost_tracking={enable_cost_tracking}"
        )

    @with_retry(max_retries=3)
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        verify: bool = False,
        parse_response: bool = True,
        **kwargs,
    ) -> Union[Any, ParsedResponse]:
        """Create a chat completion.

        Args:
            messages: List of message dictionaries
            model: Model to use (default: from config)
            temperature: Sampling temperature (default: from config)
            max_tokens: Maximum tokens in response (default: from config)
            verify: Whether to apply TMR verification (default: False)
            parse_response: Whether to parse response (default: True)
            **kwargs: Additional parameters for OpenAI API

        Returns:
            ParsedResponse if parse_response=True, else raw OpenAI response

        Raises:
            TMRAPIError: On API errors
        """
        # Use config defaults if not specified
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        # Estimate tokens for rate limiting
        estimated_tokens = 0
        if self.rate_limiter:
            estimated_tokens = TokenEstimator.estimate_message_tokens(messages, model)
            estimated_tokens += max_tokens  # Add estimated response tokens
            self.rate_limiter.acquire(estimated_tokens)

        try:
            # Make API call
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Update rate limiter with actual tokens
            if self.rate_limiter and hasattr(response, "usage"):
                actual_tokens = response.usage.total_tokens
                self.rate_limiter.release(actual_tokens)

            # Track cost
            if self.cost_tracker and hasattr(response, "usage"):
                cost = calculate_cost(
                    model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
                self.cost_tracker.add_usage(
                    model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    cost,
                )

            # Parse response if requested
            if parse_response:
                parsed = self.parser.parse(response)

                # Apply TMR verification if requested
                if verify:
                    parsed = self._verify_response(parsed)

                return parsed

            return response

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            handle_openai_error(e, "chat completion")

    @with_retry(max_retries=3)
    async def chat_completion_async(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        verify: bool = False,
        parse_response: bool = True,
        **kwargs,
    ) -> Union[Any, ParsedResponse]:
        """Create a chat completion (async).

        Args:
            messages: List of message dictionaries
            model: Model to use (default: from config)
            temperature: Sampling temperature (default: from config)
            max_tokens: Maximum tokens in response (default: from config)
            verify: Whether to apply TMR verification (default: False)
            parse_response: Whether to parse response (default: True)
            **kwargs: Additional parameters for OpenAI API

        Returns:
            ParsedResponse if parse_response=True, else raw OpenAI response

        Raises:
            TMRAPIError: On API errors
        """
        # Use config defaults if not specified
        model = model or self.config.model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens

        # Estimate tokens for rate limiting
        estimated_tokens = 0
        if self.rate_limiter:
            estimated_tokens = TokenEstimator.estimate_message_tokens(messages, model)
            estimated_tokens += max_tokens  # Add estimated response tokens
            await self.rate_limiter.acquire_async(estimated_tokens)

        try:
            # Make API call
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Update rate limiter with actual tokens
            if self.rate_limiter and hasattr(response, "usage"):
                actual_tokens = response.usage.total_tokens
                self.rate_limiter.release(actual_tokens)

            # Track cost
            if self.cost_tracker and hasattr(response, "usage"):
                cost = calculate_cost(
                    model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                )
                self.cost_tracker.add_usage(
                    model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    cost,
                )

            # Parse response if requested
            if parse_response:
                parsed = self.parser.parse(response)

                # Apply TMR verification if requested
                if verify:
                    parsed = await self._verify_response_async(parsed)

                return parsed

            return response

        except Exception as e:
            logger.error(f"Error in async chat completion: {e}")
            handle_openai_error(e, "async chat completion")

    def create_with_template(
        self,
        prompt_type: PromptType,
        verify: bool = False,
        **template_kwargs,
    ) -> ParsedResponse:
        """Create a completion using a prompt template.

        Args:
            prompt_type: Type of prompt template to use
            verify: Whether to apply TMR verification
            **template_kwargs: Variables for template formatting

        Returns:
            ParsedResponse with structured information

        Raises:
            TMRAPIError: On API errors
        """
        messages = create_messages(prompt_type, **template_kwargs)
        return self.chat_completion(messages, verify=verify)

    async def create_with_template_async(
        self,
        prompt_type: PromptType,
        verify: bool = False,
        **template_kwargs,
    ) -> ParsedResponse:
        """Create a completion using a prompt template (async).

        Args:
            prompt_type: Type of prompt template to use
            verify: Whether to apply TMR verification
            **template_kwargs: Variables for template formatting

        Returns:
            ParsedResponse with structured information

        Raises:
            TMRAPIError: On API errors
        """
        messages = create_messages(prompt_type, **template_kwargs)
        return await self.chat_completion_async(messages, verify=verify)

    def _verify_response(self, parsed: ParsedResponse) -> ParsedResponse:
        """Apply TMR verification to parsed response.

        Args:
            parsed: Parsed response to verify

        Returns:
            ParsedResponse with verification metadata added
        """
        try:
            from ..fundamentals import FundamentalsLayer

            # Initialize fundamentals layer if not already done
            if not hasattr(self, "_fundamentals"):
                self._fundamentals = FundamentalsLayer()

            # Prepare reasoning chain for verification
            if parsed.reasoning_steps:
                reasoning_chain = {
                    "steps": [
                        {
                            "content": step.description,
                            "dependencies": step.dependencies,
                        }
                        for step in parsed.reasoning_steps
                    ],
                    "conclusion": parsed.conclusion or parsed.content,
                }

                # Validate using fundamentals layer
                validation_result = self._fundamentals.validate(reasoning_chain)

                # Add verification metadata
                parsed.metadata["tmr_verification"] = {
                    "valid": validation_result["valid"],
                    "confidence": validation_result["confidence"],
                    "details": validation_result["details"],
                }

                # Update overall confidence if verification provided one
                if validation_result["confidence"] is not None:
                    parsed.confidence = validation_result["confidence"]

            return parsed

        except Exception as e:
            logger.warning(f"TMR verification failed: {e}")
            parsed.metadata["tmr_verification_error"] = str(e)
            return parsed

    async def _verify_response_async(self, parsed: ParsedResponse) -> ParsedResponse:
        """Apply TMR verification to parsed response (async).

        Args:
            parsed: Parsed response to verify

        Returns:
            ParsedResponse with verification metadata added
        """
        # Run synchronous verification in thread pool
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._verify_response, parsed)

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics including rate limiting and costs.

        Returns:
            Dictionary with statistics
        """
        stats = {}

        if self.rate_limiter:
            stats["rate_limiting"] = {
                "current_usage": self.rate_limiter.get_current_usage(),
                "statistics": self.rate_limiter.get_statistics(),
            }

        if self.cost_tracker:
            stats["costs"] = self.cost_tracker.get_statistics()

        return stats

    def reset_statistics(self):
        """Reset all statistics."""
        if self.rate_limiter:
            self.rate_limiter.reset_statistics()

        if self.cost_tracker:
            self.cost_tracker.reset()

        logger.info("Statistics reset")


# Convenience functions
def create_client(
    api_key: Optional[str] = None,
    config: Optional[OpenAIConfig] = None,
    **kwargs,
) -> TMROpenAI:
    """Create a TMR OpenAI client.

    Args:
        api_key: OpenAI API key
        config: OpenAI configuration
        **kwargs: Additional arguments for TMROpenAI

    Returns:
        TMROpenAI client instance
    """
    return TMROpenAI(api_key=api_key, config=config, **kwargs)


def quick_completion(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    verify: bool = False,
) -> str:
    """Quick convenience function for simple completions.

    Args:
        prompt: User prompt
        api_key: OpenAI API key
        model: Model to use
        verify: Whether to apply TMR verification

    Returns:
        Response content as string
    """
    client = create_client(api_key=api_key)
    messages = [{"role": "user", "content": prompt}]
    response = client.chat_completion(messages, model=model, verify=verify)
    return response.content


async def quick_completion_async(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    verify: bool = False,
) -> str:
    """Quick convenience function for simple completions (async).

    Args:
        prompt: User prompt
        api_key: OpenAI API key
        model: Model to use
        verify: Whether to apply TMR verification

    Returns:
        Response content as string
    """
    client = create_client(api_key=api_key)
    messages = [{"role": "user", "content": prompt}]
    response = await client.chat_completion_async(messages, model=model, verify=verify)
    return response.content
