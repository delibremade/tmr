"""Core configuration classes for TMR framework."""

from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


class VerificationDepth(Enum):
    """Verification depth levels for TMR framework."""

    QUICK = "quick"           # Fast, basic validation
    ADAPTIVE = "adaptive"     # Smart depth selection based on context
    DEEP = "deep"            # Comprehensive validation

    def __str__(self) -> str:
        return self.value


@dataclass
class TMRConfig:
    """Configuration for Trinity Meta-Reasoning Framework.

    Attributes:
        verification_depth: Level of verification to apply
        confidence_threshold: Minimum confidence score (0.0-1.0)
        max_retries: Maximum retry attempts for failed operations
        timeout: Operation timeout in seconds
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        cache_enabled: Whether to enable result caching
        cache_size: Maximum cache size
        cache_ttl: Cache time-to-live in seconds
        enable_async: Enable async operations
        enable_cost_tracking: Track API usage costs
        metadata: Additional configuration metadata
    """

    # Core verification settings
    verification_depth: VerificationDepth = VerificationDepth.ADAPTIVE
    confidence_threshold: float = 0.7
    max_retries: int = 3
    timeout: int = 30

    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Cache settings
    cache_enabled: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600

    # Feature flags
    enable_async: bool = True
    enable_cost_tracking: bool = True

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, "
                f"got {self.confidence_threshold}"
            )

        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}"
            )

        if self.timeout <= 0:
            raise ValueError(
                f"timeout must be positive, got {self.timeout}"
            )

        if self.cache_size <= 0:
            raise ValueError(
                f"cache_size must be positive, got {self.cache_size}"
            )

        if self.cache_ttl <= 0:
            raise ValueError(
                f"cache_ttl must be positive, got {self.cache_ttl}"
            )

        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(
                f"log_level must be one of {valid_log_levels}, "
                f"got {self.log_level}"
            )

        # Ensure verification_depth is a VerificationDepth enum
        if isinstance(self.verification_depth, str):
            self.verification_depth = VerificationDepth(self.verification_depth.lower())

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "verification_depth": str(self.verification_depth),
            "confidence_threshold": self.confidence_threshold,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "cache_enabled": self.cache_enabled,
            "cache_size": self.cache_size,
            "cache_ttl": self.cache_ttl,
            "enable_async": self.enable_async,
            "enable_cost_tracking": self.enable_cost_tracking,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TMRConfig":
        """Create configuration from dictionary.

        Args:
            data: Dictionary with configuration values

        Returns:
            TMRConfig instance
        """
        # Handle verification_depth conversion
        if "verification_depth" in data:
            if isinstance(data["verification_depth"], str):
                data["verification_depth"] = VerificationDepth(
                    data["verification_depth"].lower()
                )

        return cls(**data)

    def update(self, **kwargs) -> "TMRConfig":
        """Create a new configuration with updated values.

        Args:
            **kwargs: Configuration values to update

        Returns:
            New TMRConfig instance with updated values
        """
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return TMRConfig.from_dict(config_dict)
