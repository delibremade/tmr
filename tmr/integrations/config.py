"""OpenAI integration configuration."""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI API integration.

    Attributes:
        api_key: OpenAI API key
        org_id: OpenAI organization ID (optional)
        model: Model name (e.g., gpt-4, gpt-3.5-turbo)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens in response
        top_p: Nucleus sampling parameter
        frequency_penalty: Frequency penalty (-2.0 to 2.0)
        presence_penalty: Presence penalty (-2.0 to 2.0)
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        rpm: Requests per minute limit
        tpm: Tokens per minute limit
        rate_limit_buffer: Buffer multiplier for rate limits (0.0-1.0)
        enable_streaming: Enable streaming responses
        enable_cost_tracking: Track API usage costs
        default_system_prompt: Default system prompt
        metadata: Additional metadata
    """

    # API credentials
    api_key: str
    org_id: Optional[str] = None

    # Model configuration
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Request configuration
    timeout: int = 30
    max_retries: int = 3

    # Rate limiting
    rpm: int = 3500  # Requests per minute
    tpm: int = 90000  # Tokens per minute
    rate_limit_buffer: float = 0.9  # Use 90% of limit

    # Feature flags
    enable_streaming: bool = False
    enable_cost_tracking: bool = True

    # Prompting
    default_system_prompt: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration values."""
        if not self.api_key:
            raise ValueError("api_key is required")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be between 0.0 and 2.0, got {self.temperature}"
            )

        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError(
                f"top_p must be between 0.0 and 1.0, got {self.top_p}"
            )

        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError(
                f"frequency_penalty must be between -2.0 and 2.0, "
                f"got {self.frequency_penalty}"
            )

        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError(
                f"presence_penalty must be between -2.0 and 2.0, "
                f"got {self.presence_penalty}"
            )

        if self.max_tokens <= 0:
            raise ValueError(
                f"max_tokens must be positive, got {self.max_tokens}"
            )

        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")

        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}"
            )

        if self.rpm <= 0:
            raise ValueError(f"rpm must be positive, got {self.rpm}")

        if self.tpm <= 0:
            raise ValueError(f"tpm must be positive, got {self.tpm}")

        if not 0.0 <= self.rate_limit_buffer <= 1.0:
            raise ValueError(
                f"rate_limit_buffer must be between 0.0 and 1.0, "
                f"got {self.rate_limit_buffer}"
            )

    @classmethod
    def from_env(cls, env_file: Optional[str] = None) -> "OpenAIConfig":
        """Create configuration from environment variables.

        Args:
            env_file: Path to .env file (optional)

        Returns:
            OpenAIConfig instance

        Raises:
            ValueError: If required environment variables are missing
        """
        if env_file:
            from dotenv import load_dotenv
            load_dotenv(env_file)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Set it or provide api_key directly."
            )

        return cls(
            api_key=api_key,
            org_id=os.getenv("OPENAI_ORG_ID"),
            model=os.getenv("OPENAI_MODEL", "gpt-4"),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
            top_p=float(os.getenv("OPENAI_TOP_P", "1.0")),
            frequency_penalty=float(os.getenv("OPENAI_FREQUENCY_PENALTY", "0.0")),
            presence_penalty=float(os.getenv("OPENAI_PRESENCE_PENALTY", "0.0")),
            timeout=int(os.getenv("TMR_TIMEOUT", "30")),
            max_retries=int(os.getenv("TMR_MAX_RETRIES", "3")),
            rpm=int(os.getenv("OPENAI_RPM", "3500")),
            tpm=int(os.getenv("OPENAI_TPM", "90000")),
            rate_limit_buffer=float(os.getenv("OPENAI_RATE_LIMIT_BUFFER", "0.9")),
            enable_streaming=os.getenv("TMR_ENABLE_STREAMING", "false").lower()
            == "true",
            enable_cost_tracking=os.getenv("TMR_ENABLE_COST_TRACKING", "true").lower()
            == "true",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "api_key": "***" if self.api_key else None,  # Mask API key
            "org_id": self.org_id,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "rpm": self.rpm,
            "tpm": self.tpm,
            "rate_limit_buffer": self.rate_limit_buffer,
            "enable_streaming": self.enable_streaming,
            "enable_cost_tracking": self.enable_cost_tracking,
            "default_system_prompt": self.default_system_prompt,
            "metadata": self.metadata,
        }

    def get_effective_rpm(self) -> int:
        """Get effective requests per minute considering buffer.

        Returns:
            Effective RPM limit
        """
        return int(self.rpm * self.rate_limit_buffer)

    def get_effective_tpm(self) -> int:
        """Get effective tokens per minute considering buffer.

        Returns:
            Effective TPM limit
        """
        return int(self.tpm * self.rate_limit_buffer)


@dataclass
class ModelConfig:
    """Configuration for specific OpenAI models with pricing.

    Attributes:
        name: Model name
        input_cost_per_1k: Cost per 1k input tokens in USD
        output_cost_per_1k: Cost per 1k output tokens in USD
        max_context: Maximum context window size
        supports_functions: Whether model supports function calling
        supports_vision: Whether model supports vision
    """

    name: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    max_context: int
    supports_functions: bool = True
    supports_vision: bool = False


# Model pricing and capabilities
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "gpt-4": ModelConfig(
        name="gpt-4",
        input_cost_per_1k=0.03,
        output_cost_per_1k=0.06,
        max_context=8192,
        supports_functions=True,
        supports_vision=False,
    ),
    "gpt-4-32k": ModelConfig(
        name="gpt-4-32k",
        input_cost_per_1k=0.06,
        output_cost_per_1k=0.12,
        max_context=32768,
        supports_functions=True,
        supports_vision=False,
    ),
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo",
        input_cost_per_1k=0.01,
        output_cost_per_1k=0.03,
        max_context=128000,
        supports_functions=True,
        supports_vision=True,
    ),
    "gpt-4-turbo-preview": ModelConfig(
        name="gpt-4-turbo-preview",
        input_cost_per_1k=0.01,
        output_cost_per_1k=0.03,
        max_context=128000,
        supports_functions=True,
        supports_vision=False,
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
        max_context=16385,
        supports_functions=True,
        supports_vision=False,
    ),
    "gpt-3.5-turbo-16k": ModelConfig(
        name="gpt-3.5-turbo-16k",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.004,
        max_context=16385,
        supports_functions=True,
        supports_vision=False,
    ),
}


def get_model_config(model_name: str) -> ModelConfig:
    """Get configuration for a specific model.

    Args:
        model_name: Name of the model

    Returns:
        ModelConfig for the specified model

    Raises:
        ValueError: If model is not recognized
    """
    # Handle model name variations
    if model_name.startswith("gpt-4-turbo") or model_name.startswith("gpt-4-1106"):
        base_name = "gpt-4-turbo"
    elif model_name.startswith("gpt-4-32k"):
        base_name = "gpt-4-32k"
    elif model_name.startswith("gpt-4"):
        base_name = "gpt-4"
    elif model_name.startswith("gpt-3.5-turbo-16k"):
        base_name = "gpt-3.5-turbo-16k"
    elif model_name.startswith("gpt-3.5-turbo"):
        base_name = "gpt-3.5-turbo"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return MODEL_CONFIGS.get(base_name, MODEL_CONFIGS["gpt-4"])


def calculate_cost(
    model_name: str, input_tokens: int, output_tokens: int
) -> float:
    """Calculate cost for a given model and token usage.

    Args:
        model_name: Name of the model
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Total cost in USD
    """
    config = get_model_config(model_name)
    input_cost = (input_tokens / 1000) * config.input_cost_per_1k
    output_cost = (output_tokens / 1000) * config.output_cost_per_1k
    return input_cost + output_cost
