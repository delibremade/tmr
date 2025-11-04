"""
Trinity Meta-Reasoning Framework

A three-layer architecture for augmenting Large Language Models with verified reasoning.
"""

__version__ = "0.1.0-alpha"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import available layers
from .fundamentals import FundamentalsLayer

# Import execution layer
from .execution import (
    ExecutionSynthesizer,
    DepthSelector,
    VerificationDepth,
    DepthProfile,
    OutputFormat,
    VerificationResult,
    SynthesisContext
)

# Core and full framework not yet implemented
# from .core.framework import TMRFramework
# from .core.config import TMRConfig

# Nuance layer placeholder
# from .nuance import NuanceLayer

__all__ = [
    # Fundamentals Layer
    "FundamentalsLayer",

    # Execution Layer
    "ExecutionSynthesizer",
    "DepthSelector",
    "VerificationDepth",
    "DepthProfile",
    "OutputFormat",
    "VerificationResult",
    "SynthesisContext",

    # Not yet implemented
    # "TMRFramework",
    # "TMRConfig",
    # "NuanceLayer",
]

# Module level constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_VERIFICATION_DEPTH = "adaptive"
DEFAULT_MAX_RETRIES = 3