"""Tests for integration configuration."""

import pytest
from tmr.integrations.config import (
    OpenAIConfig,
    ModelConfig,
    get_model_config,
    calculate_cost,
)


def test_openai_config_validation():
    """Test OpenAI config validation."""
    # Valid config
    config = OpenAIConfig(api_key="test-key", model="gpt-4")
    assert config.api_key == "test-key"
    assert config.model == "gpt-4"

    # Invalid temperature
    with pytest.raises(ValueError, match="temperature"):
        OpenAIConfig(api_key="test-key", temperature=3.0)

    # Invalid top_p
    with pytest.raises(ValueError, match="top_p"):
        OpenAIConfig(api_key="test-key", top_p=1.5)

    # Invalid max_tokens
    with pytest.raises(ValueError, match="max_tokens"):
        OpenAIConfig(api_key="test-key", max_tokens=-1)


def test_openai_config_effective_limits():
    """Test effective rate limits with buffer."""
    config = OpenAIConfig(
        api_key="test-key",
        rpm=1000,
        tpm=50000,
        rate_limit_buffer=0.9
    )

    assert config.get_effective_rpm() == 900
    assert config.get_effective_tpm() == 45000


def test_get_model_config():
    """Test getting model configuration."""
    # Standard models
    gpt4_config = get_model_config("gpt-4")
    assert gpt4_config.name == "gpt-4"
    assert gpt4_config.max_context == 8192

    gpt35_config = get_model_config("gpt-3.5-turbo")
    assert gpt35_config.name == "gpt-3.5-turbo"
    assert gpt35_config.max_context == 16385

    # Model name variations
    gpt4_turbo = get_model_config("gpt-4-turbo-preview")
    assert gpt4_turbo.name == "gpt-4-turbo"


def test_calculate_cost():
    """Test cost calculation."""
    # GPT-4 costs
    cost = calculate_cost("gpt-4", 1000, 1000)
    assert cost == pytest.approx(0.09, rel=1e-6)  # (1000/1000)*0.03 + (1000/1000)*0.06

    # GPT-3.5-turbo costs
    cost = calculate_cost("gpt-3.5-turbo", 1000, 1000)
    assert cost == pytest.approx(0.002, rel=1e-6)  # Much cheaper


def test_config_to_dict():
    """Test config serialization."""
    config = OpenAIConfig(
        api_key="test-key",
        model="gpt-4",
        temperature=0.7
    )

    config_dict = config.to_dict()
    assert config_dict["api_key"] == "***"  # Masked
    assert config_dict["model"] == "gpt-4"
    assert config_dict["temperature"] == 0.7
