"""
Execution Layer - Context-aware Verified Synthesis

Layer 3 of the Trinity Meta-Reasoning Framework.
This layer orchestrates verification, scales depth adaptively, and formats output.
"""

__version__ = "0.1.0-alpha"

from .depth_selector import (
    DepthSelector,
    VerificationDepth,
    DepthProfile
)

from .synthesizer import (
    ExecutionSynthesizer,
    OutputFormat,
    VerificationResult,
    SynthesisContext
)

__all__ = [
    # Depth Scaling
    "DepthSelector",
    "VerificationDepth",
    "DepthProfile",

    # Synthesis
    "ExecutionSynthesizer",
    "OutputFormat",
    "VerificationResult",
    "SynthesisContext",
]
