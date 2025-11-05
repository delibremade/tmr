"""Rate limiting for OpenAI API requests."""

import asyncio
import time
from typing import Optional, Dict, Any, Tuple
from collections import deque
from dataclasses import dataclass
from threading import Lock
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting.

    Attributes:
        requests_per_minute: Maximum requests per minute
        tokens_per_minute: Maximum tokens per minute
        max_retries: Maximum retry attempts
        retry_delay: Base delay between retries in seconds
        exponential_backoff: Whether to use exponential backoff
    """

    requests_per_minute: int
    tokens_per_minute: int
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True


@dataclass
class RequestRecord:
    """Record of a single request.

    Attributes:
        timestamp: Unix timestamp when request was made
        tokens: Number of tokens used
    """

    timestamp: float
    tokens: int


class RateLimiter:
    """Rate limiter for API requests with token and request tracking.

    Implements sliding window rate limiting for both requests per minute (RPM)
    and tokens per minute (TPM).
    """

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.request_history: deque = deque()
        self.token_history: deque = deque()
        self.lock = Lock()

        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_wait_time = 0.0
        self.rate_limit_hits = 0

    def _clean_history(self, current_time: float):
        """Remove records older than 1 minute.

        Args:
            current_time: Current Unix timestamp
        """
        cutoff_time = current_time - 60.0

        # Clean request history
        while self.request_history and self.request_history[0].timestamp < cutoff_time:
            self.request_history.popleft()

        # Clean token history
        while self.token_history and self.token_history[0].timestamp < cutoff_time:
            self.token_history.popleft()

    def _get_current_counts(self, current_time: float) -> Tuple[int, int]:
        """Get current request and token counts in the sliding window.

        Args:
            current_time: Current Unix timestamp

        Returns:
            Tuple of (request_count, token_count)
        """
        self._clean_history(current_time)

        request_count = len(self.request_history)
        token_count = sum(record.tokens for record in self.token_history)

        return request_count, token_count

    def _calculate_wait_time(
        self, current_time: float, estimated_tokens: int
    ) -> float:
        """Calculate how long to wait before making a request.

        Args:
            current_time: Current Unix timestamp
            estimated_tokens: Estimated tokens for the request

        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        request_count, token_count = self._get_current_counts(current_time)

        wait_times = []

        # Check if we're at request limit
        if request_count >= self.config.requests_per_minute:
            # Find the oldest request and calculate when it will expire
            oldest_request = self.request_history[0]
            time_until_expire = (oldest_request.timestamp + 60.0) - current_time
            wait_times.append(max(0, time_until_expire))

        # Check if we're at token limit
        if token_count + estimated_tokens > self.config.tokens_per_minute:
            # Find how many tokens we need to free up
            tokens_to_free = (token_count + estimated_tokens) - self.config.tokens_per_minute

            # Calculate when enough tokens will be freed
            tokens_freed = 0
            for record in self.token_history:
                tokens_freed += record.tokens
                if tokens_freed >= tokens_to_free:
                    time_until_expire = (record.timestamp + 60.0) - current_time
                    wait_times.append(max(0, time_until_expire))
                    break

        return max(wait_times) if wait_times else 0.0

    def acquire(self, estimated_tokens: int = 0) -> float:
        """Acquire permission to make a request (blocking).

        Args:
            estimated_tokens: Estimated number of tokens for the request

        Returns:
            Time waited in seconds
        """
        with self.lock:
            current_time = time.time()
            wait_time = self._calculate_wait_time(current_time, estimated_tokens)

            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                self.rate_limit_hits += 1
                self.total_wait_time += wait_time
                time.sleep(wait_time)
                current_time = time.time()

            # Record the request
            record = RequestRecord(timestamp=current_time, tokens=estimated_tokens)
            self.request_history.append(record)
            if estimated_tokens > 0:
                self.token_history.append(record)

            self.total_requests += 1
            self.total_tokens += estimated_tokens

            return wait_time

    async def acquire_async(self, estimated_tokens: int = 0) -> float:
        """Acquire permission to make a request (async).

        Args:
            estimated_tokens: Estimated number of tokens for the request

        Returns:
            Time waited in seconds
        """
        # Calculate wait time
        current_time = time.time()
        with self.lock:
            wait_time = self._calculate_wait_time(current_time, estimated_tokens)

        # Wait asynchronously if needed
        if wait_time > 0:
            logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
            self.rate_limit_hits += 1
            self.total_wait_time += wait_time
            await asyncio.sleep(wait_time)
            current_time = time.time()

        # Record the request
        with self.lock:
            record = RequestRecord(timestamp=current_time, tokens=estimated_tokens)
            self.request_history.append(record)
            if estimated_tokens > 0:
                self.token_history.append(record)

            self.total_requests += 1
            self.total_tokens += estimated_tokens

        return wait_time

    def release(self, actual_tokens: int):
        """Update token count with actual usage after request completes.

        Args:
            actual_tokens: Actual number of tokens used
        """
        with self.lock:
            if self.token_history:
                # Update the most recent record with actual token count
                last_record = self.token_history[-1]
                token_diff = actual_tokens - last_record.tokens
                last_record.tokens = actual_tokens

                # Update statistics
                self.total_tokens += token_diff

    def get_current_usage(self) -> Dict[str, Any]:
        """Get current rate limit usage.

        Returns:
            Dictionary with current usage information
        """
        with self.lock:
            current_time = time.time()
            request_count, token_count = self._get_current_counts(current_time)

            return {
                "current_rpm": request_count,
                "max_rpm": self.config.requests_per_minute,
                "rpm_utilization": request_count / self.config.requests_per_minute,
                "current_tpm": token_count,
                "max_tpm": self.config.tokens_per_minute,
                "tpm_utilization": token_count / self.config.tokens_per_minute,
                "available_requests": max(0, self.config.requests_per_minute - request_count),
                "available_tokens": max(0, self.config.tokens_per_minute - token_count),
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics.

        Returns:
            Dictionary with statistics
        """
        with self.lock:
            return {
                "total_requests": self.total_requests,
                "total_tokens": self.total_tokens,
                "total_wait_time_seconds": round(self.total_wait_time, 2),
                "rate_limit_hits": self.rate_limit_hits,
                "average_wait_time": (
                    round(self.total_wait_time / self.rate_limit_hits, 2)
                    if self.rate_limit_hits > 0
                    else 0.0
                ),
            }

    def reset_statistics(self):
        """Reset statistics (not history)."""
        with self.lock:
            self.total_requests = 0
            self.total_tokens = 0
            self.total_wait_time = 0.0
            self.rate_limit_hits = 0


class TokenEstimator:
    """Estimate token count for text.

    Uses a simple approximation: ~4 characters per token for English text.
    For more accurate estimation, consider using tiktoken library.
    """

    # Average characters per token for different models
    CHARS_PER_TOKEN = {
        "gpt-4": 4.0,
        "gpt-4-turbo": 4.0,
        "gpt-3.5-turbo": 4.0,
        "default": 4.0,
    }

    @classmethod
    def estimate_tokens(cls, text: str, model: str = "default") -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate tokens for
            model: Model name (affects estimation)

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        chars_per_token = cls.CHARS_PER_TOKEN.get(model, cls.CHARS_PER_TOKEN["default"])
        return max(1, int(len(text) / chars_per_token))

    @classmethod
    def estimate_message_tokens(
        cls, messages: list, model: str = "default"
    ) -> int:
        """Estimate token count for a list of messages.

        Args:
            messages: List of message dictionaries
            model: Model name

        Returns:
            Estimated token count
        """
        # Base tokens for message formatting
        base_tokens = 3  # Every message follows <im_start>{role/name}\n{content}<im_end>\n

        total_tokens = 0
        for message in messages:
            total_tokens += base_tokens
            for key, value in message.items():
                total_tokens += cls.estimate_tokens(str(value), model)

        total_tokens += 2  # Every reply is primed with <im_start>assistant

        return total_tokens


class ExponentialBackoff:
    """Exponential backoff strategy for retries."""

    def __init__(
        self,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
    ):
        """Initialize exponential backoff.

        Args:
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential calculation
            jitter: Whether to add random jitter
        """
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = min(
            self.base_delay * (self.exponential_base ** attempt),
            self.max_delay,
        )

        if self.jitter:
            import random
            # Add jitter: random value between 0 and delay
            delay = delay * (0.5 + random.random() * 0.5)

        return delay

    def sleep(self, attempt: int):
        """Sleep for the calculated delay.

        Args:
            attempt: Attempt number (0-indexed)
        """
        delay = self.calculate_delay(attempt)
        logger.debug(f"Backing off for {delay:.2f}s (attempt {attempt + 1})")
        time.sleep(delay)

    async def sleep_async(self, attempt: int):
        """Sleep for the calculated delay (async).

        Args:
            attempt: Attempt number (0-indexed)
        """
        delay = self.calculate_delay(attempt)
        logger.debug(f"Backing off for {delay:.2f}s (attempt {attempt + 1})")
        await asyncio.sleep(delay)
