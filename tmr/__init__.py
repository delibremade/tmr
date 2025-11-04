"""
Trinity Meta-Reasoning Framework

A three-layer architecture for augmenting Large Language Models with verified reasoning.
"""

__version__ = "0.1.0-alpha"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import available layers
from .fundamentals import FundamentalsLayer
from .nuance import NuanceLayer

# Core and Execution layers not yet implemented
# from .core.framework import TMRFramework
# from .core.config import TMRConfig, VerificationDepth
# from .execution import ExecutionLayer

__all__ = [
    "FundamentalsLayer",
    "NuanceLayer",
    # "TMRFramework",
    # "TMRConfig",
    # "VerificationDepth",
    # "ExecutionLayer",
]

# Module level constants
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_VERIFICATION_DEPTH = "adaptive"
DEFAULT_MAX_RETRIES = 3