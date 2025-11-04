"""Error handling for OpenAI API integration."""

import logging
from typing import Optional, Callable, Any, Type
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)


class TMRAPIError(Exception):
    """Base exception for TMR API errors."""

    def __init__(
        self,
        message: str,
        original_error: Optional[Exception] = None,
        retry_after: Optional[float] = None,
    ):
        """Initialize API error.

        Args:
            message: Error message
            original_error: Original exception that caused this error
            retry_after: Suggested retry delay in seconds
        """
        super().__init__(message)
        self.original_error = original_error
        self.retry_after = retry_after


class AuthenticationError(TMRAPIError):
    """Authentication failed (invalid API key, etc.)."""

    pass


class RateLimitError(TMRAPIError):
    """Rate limit exceeded."""

    pass


class InvalidRequestError(TMRAPIError):
    """Invalid request parameters."""

    pass


class APIConnectionError(TMRAPIError):
    """Failed to connect to API."""

    pass


class APITimeoutError(TMRAPIError):
    """Request timed out."""

    pass


class ServiceUnavailableError(TMRAPIError):
    """API service is unavailable."""

    pass


class ModelNotFoundError(TMRAPIError):
    """Requested model not found."""

    pass


class ContextLengthExceededError(TMRAPIError):
    """Input exceeds model's context length."""

    pass


class ContentFilterError(TMRAPIError):
    """Content was filtered by OpenAI's content policy."""

    pass


class ErrorHandler:
    """Handler for OpenAI API errors with retry logic."""

    # Mapping of OpenAI error types to custom exceptions
    ERROR_MAP = {
        "invalid_api_key": AuthenticationError,
        "invalid_request_error": InvalidRequestError,
        "rate_limit_error": RateLimitError,
        "connection_error": APIConnectionError,
        "timeout": APITimeoutError,
        "service_unavailable": ServiceUnavailableError,
        "model_not_found": ModelNotFoundError,
        "context_length_exceeded": ContextLengthExceededError,
        "content_filter": ContentFilterError,
    }

    # Which errors are retryable
    RETRYABLE_ERRORS = {
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
        ServiceUnavailableError,
    }

    @staticmethod
    def classify_error(error: Exception) -> Type[TMRAPIError]:
        """Classify an exception into a TMR error type.

        Args:
            error: Original exception

        Returns:
            TMR error class
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # Check for authentication errors
        if "api_key" in error_str or "authentication" in error_str or "unauthorized" in error_str:
            return AuthenticationError

        # Check for rate limit errors
        if "rate" in error_str and "limit" in error_str or "429" in error_str:
            return RateLimitError

        # Check for invalid request errors
        if "invalid" in error_str or "400" in error_str:
            return InvalidRequestError

        # Check for connection errors
        if "connection" in error_str or "network" in error_str:
            return APIConnectionError

        # Check for timeout errors
        if "timeout" in error_str or "timed out" in error_str:
            return APITimeoutError

        # Check for service unavailable
        if "503" in error_str or "502" in error_str or "unavailable" in error_str:
            return ServiceUnavailableError

        # Check for model errors
        if "model" in error_str and ("not found" in error_str or "does not exist" in error_str):
            return ModelNotFoundError

        # Check for context length errors
        if "context" in error_str and "length" in error_str or "maximum context" in error_str:
            return ContextLengthExceededError

        # Check for content filter errors
        if "content" in error_str and "filter" in error_str or "content_policy" in error_str:
            return ContentFilterError

        # Default to base API error
        return TMRAPIError

    @staticmethod
    def handle_error(
        error: Exception,
        context: Optional[str] = None,
    ) -> TMRAPIError:
        """Handle an error and convert to TMR error type.

        Args:
            error: Original exception
            context: Additional context about where error occurred

        Returns:
            TMRAPIError subclass instance
        """
        # Classify the error
        error_class = ErrorHandler.classify_error(error)

        # Extract retry_after if present (for rate limits)
        retry_after = None
        if hasattr(error, "retry_after"):
            retry_after = error.retry_after
        elif "retry-after" in str(error).lower():
            # Try to extract retry-after from error message
            import re
            match = re.search(r"retry.*?(\d+)\s*(?:second|sec|s)", str(error).lower())
            if match:
                retry_after = float(match.group(1))

        # Create message
        message = str(error)
        if context:
            message = f"{context}: {message}"

        # Create and return the appropriate error
        return error_class(message, original_error=error, retry_after=retry_after)

    @staticmethod
    def is_retryable(error: Exception) -> bool:
        """Check if an error is retryable.

        Args:
            error: Exception to check

        Returns:
            True if error is retryable
        """
        if isinstance(error, TMRAPIError):
            return type(error) in ErrorHandler.RETRYABLE_ERRORS

        # Check the original error
        error_class = ErrorHandler.classify_error(error)
        return error_class in ErrorHandler.RETRYABLE_ERRORS


def with_retry(
    max_retries: int = 3,
    backoff_strategy: Optional[Any] = None,
    retryable_exceptions: Optional[tuple] = None,
):
    """Decorator to add retry logic to a function.

    Args:
        max_retries: Maximum number of retry attempts
        backoff_strategy: Backoff strategy instance (default: ExponentialBackoff)
        retryable_exceptions: Tuple of exceptions to retry (default: RETRYABLE_ERRORS)

    Returns:
        Decorated function
    """
    if retryable_exceptions is None:
        retryable_exceptions = tuple(ErrorHandler.RETRYABLE_ERRORS)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            from .rate_limiting import ExponentialBackoff

            backoff = backoff_strategy or ExponentialBackoff()
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    # Convert to TMR error if needed
                    if not isinstance(e, TMRAPIError):
                        e = ErrorHandler.handle_error(e)

                    # Check if retryable
                    if not ErrorHandler.is_retryable(e) or attempt >= max_retries:
                        logger.error(f"Non-retryable error or max retries reached: {e}")
                        raise

                    # Log retry
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. Retrying..."
                    )

                    # Use custom retry_after if provided
                    if hasattr(e, "retry_after") and e.retry_after:
                        import time
                        time.sleep(e.retry_after)
                    else:
                        backoff.sleep(attempt)

            # Should never reach here, but just in case
            raise last_error

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            from .rate_limiting import ExponentialBackoff

            backoff = backoff_strategy or ExponentialBackoff()
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e

                    # Convert to TMR error if needed
                    if not isinstance(e, TMRAPIError):
                        e = ErrorHandler.handle_error(e)

                    # Check if retryable
                    if not ErrorHandler.is_retryable(e) or attempt >= max_retries:
                        logger.error(f"Non-retryable error or max retries reached: {e}")
                        raise

                    # Log retry
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. Retrying..."
                    )

                    # Use custom retry_after if provided
                    if hasattr(e, "retry_after") and e.retry_after:
                        await asyncio.sleep(e.retry_after)
                    else:
                        await backoff.sleep_async(attempt)

            # Should never reach here, but just in case
            raise last_error

        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def handle_openai_error(error: Exception, operation: str = "API call") -> TMRAPIError:
    """Handle OpenAI-specific errors.

    Args:
        error: OpenAI exception
        operation: Description of the operation that failed

    Returns:
        TMRAPIError instance

    Raises:
        TMRAPIError: Always raises after handling
    """
    # Try to extract error details from OpenAI error
    error_message = str(error)
    error_type = None
    error_code = None

    # Check if it's an OpenAI error with structured data
    if hasattr(error, "error"):
        error_data = error.error
        if isinstance(error_data, dict):
            error_message = error_data.get("message", error_message)
            error_type = error_data.get("type")
            error_code = error_data.get("code")

    # Handle specific error codes
    if error_code == "invalid_api_key":
        raise AuthenticationError(
            f"Invalid API key during {operation}",
            original_error=error,
        )
    elif error_code == "rate_limit_exceeded":
        # Try to extract retry_after
        retry_after = None
        if hasattr(error, "headers") and "retry-after" in error.headers:
            try:
                retry_after = float(error.headers["retry-after"])
            except (ValueError, TypeError):
                pass

        raise RateLimitError(
            f"Rate limit exceeded during {operation}",
            original_error=error,
            retry_after=retry_after,
        )
    elif error_code == "context_length_exceeded":
        raise ContextLengthExceededError(
            f"Context length exceeded during {operation}: {error_message}",
            original_error=error,
        )
    elif error_code == "model_not_found":
        raise ModelNotFoundError(
            f"Model not found during {operation}: {error_message}",
            original_error=error,
        )
    else:
        # Use generic error handling
        tmr_error = ErrorHandler.handle_error(error, context=operation)
        raise tmr_error


class ErrorContext:
    """Context manager for handling errors with additional context."""

    def __init__(self, operation: str, reraise: bool = True):
        """Initialize error context.

        Args:
            operation: Description of the operation
            reraise: Whether to reraise the error after handling
        """
        self.operation = operation
        self.reraise = reraise
        self.error: Optional[Exception] = None

    def __enter__(self):
        """Enter context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and handle any errors."""
        if exc_val is not None:
            self.error = ErrorHandler.handle_error(exc_val, context=self.operation)
            logger.error(f"Error in {self.operation}: {self.error}")

            if self.reraise:
                raise self.error

            return True  # Suppress exception

        return False

    async def __aenter__(self):
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and handle any errors."""
        return self.__exit__(exc_type, exc_val, exc_tb)
