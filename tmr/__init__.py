"""
Trinity Meta-Reasoning Framework

A three-layer architecture for augmenting Large Language Models with verified reasoning.
"""

__version__ = "0.1.0-alpha"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core modules (not yet implemented)
# from .core.framework import TMRFramework
# from .core.config import TMRConfig, VerificationDepth

# Layer implementations
from .fundamentals import FundamentalsLayer
# from .nuance import NuanceLayer  # Not yet implemented
# from .execution import ExecutionLayer  # Not yet implemented

__all__ = [
    # "TMRFramework",  # Not yet implemented
    # "TMRConfig",  # Not yet implemented
    # "VerificationDepth",  # Not yet implemented
    "FundamentalsLayer",
    # "NuanceLayer",  # Not yet implemented
    # "ExecutionLayer",  # Not yet implemented
]

# Module level constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_VERIFICATION_DEPTH = "adaptive"
DEFAULT_MAX_RETRIES = 3