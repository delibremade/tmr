"""Tests for rate limiting."""

import pytest
import time
from tmr.integrations.rate_limiting import (
    RateLimiter,
    RateLimitConfig,
    TokenEstimator,
    ExponentialBackoff,
)


def test_rate_limiter_basic():
    """Test basic rate limiting."""
    config = RateLimitConfig(
        requests_per_minute=10,
        tokens_per_minute=1000,
    )
    limiter = RateLimiter(config)

    # First request should go through immediately
    wait_time = limiter.acquire(estimated_tokens=100)
    assert wait_time == 0.0

    # Get current usage
    usage = limiter.get_current_usage()
    assert usage["current_rpm"] == 1
    assert usage["current_tpm"] == 100


def test_rate_limiter_request_limit():
    """Test request limit enforcement."""
    config = RateLimitConfig(
        requests_per_minute=2,  # Very low limit
        tokens_per_minute=10000,
    )
    limiter = RateLimiter(config)

    # Make 2 requests
    limiter.acquire(estimated_tokens=10)
    limiter.acquire(estimated_tokens=10)

    # Third request should wait
    start = time.time()
    wait_time = limiter.acquire(estimated_tokens=10)
    elapsed = time.time() - start

    assert wait_time > 0
    assert elapsed >= wait_time * 0.9  # Allow some timing variance


def test_token_estimator():
    """Test token estimation."""
    # Basic estimation
    tokens = TokenEstimator.estimate_tokens("Hello, world!")
    assert tokens > 0
    assert tokens < 10  # ~3 tokens

    # Empty text
    tokens = TokenEstimator.estimate_tokens("")
    assert tokens == 0

    # Message estimation
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    tokens = TokenEstimator.estimate_message_tokens(messages)
    assert tokens > 0


def test_exponential_backoff():
    """Test exponential backoff calculation."""
    backoff = ExponentialBackoff(
        base_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=False,  # Disable jitter for deterministic tests
    )

    # Test exponential growth
    delay0 = backoff.calculate_delay(0)
    delay1 = backoff.calculate_delay(1)
    delay2 = backoff.calculate_delay(2)

    assert delay0 == 1.0
    assert delay1 == 2.0
    assert delay2 == 4.0

    # Test max delay cap
    delay10 = backoff.calculate_delay(10)
    assert delay10 == 60.0  # Capped at max_delay


def test_rate_limiter_statistics():
    """Test rate limiter statistics tracking."""
    config = RateLimitConfig(
        requests_per_minute=100,
        tokens_per_minute=10000,
    )
    limiter = RateLimiter(config)

    # Make several requests
    limiter.acquire(estimated_tokens=100)
    limiter.acquire(estimated_tokens=200)
    limiter.acquire(estimated_tokens=300)

    stats = limiter.get_statistics()
    assert stats["total_requests"] == 3
    assert stats["total_tokens"] == 600

    # Reset statistics
    limiter.reset_statistics()
    stats = limiter.get_statistics()
    assert stats["total_requests"] == 0
    assert stats["total_tokens"] == 0


def test_rate_limiter_release():
    """Test updating token count after request."""
    config = RateLimitConfig(
        requests_per_minute=100,
        tokens_per_minute=10000,
    )
    limiter = RateLimiter(config)

    # Acquire with estimated tokens
    limiter.acquire(estimated_tokens=100)

    # Release with actual tokens
    limiter.release(actual_tokens=150)

    # Check that statistics were updated
    stats = limiter.get_statistics()
    assert stats["total_tokens"] == 150  # Updated to actual


@pytest.mark.asyncio
async def test_rate_limiter_async():
    """Test async rate limiting."""
    config = RateLimitConfig(
        requests_per_minute=10,
        tokens_per_minute=1000,
    )
    limiter = RateLimiter(config)

    # First request should go through immediately
    wait_time = await limiter.acquire_async(estimated_tokens=100)
    assert wait_time == 0.0

    # Get current usage
    usage = limiter.get_current_usage()
    assert usage["current_rpm"] == 1
    assert usage["current_tpm"] == 100
