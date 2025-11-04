"""
Trinity Meta-Reasoning Framework

A three-layer architecture for augmenting Large Language Models with verified reasoning.
"""

__version__ = "0.1.0-alpha"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core framework (only import what's implemented)
from .core.config import TMRConfig, VerificationDepth
from .fundamentals import FundamentalsLayer

# Future imports (not yet implemented)
# from .core.framework import TMRFramework
# from .nuance import NuanceLayer
# from .execution import ExecutionLayer

# OpenAI integration
from .integrations import TMROpenAI, OpenAIConfig

__all__ = [
    # Core
    "TMRConfig",
    "VerificationDepth",
    "FundamentalsLayer",
    # Integrations
    "TMROpenAI",
    "OpenAIConfig",
    # Future (not yet implemented)
    # "TMRFramework",
    # "NuanceLayer",
    # "ExecutionLayer",
]

# Module level constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_VERIFICATION_DEPTH = "adaptive"
DEFAULT_MAX_RETRIES = 3