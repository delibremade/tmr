"""
Nuance Layer Module

Layer 2: Domain-Specific Pattern Recognition

This module provides:
- Pattern extraction from domain-specific content
- Domain classification (math, code, logic)
- Pattern storage and retrieval
- Pattern matching and application
"""

from .layer import NuanceLayer
from .patterns import (
    Pattern,
    PatternMetadata,
    PatternMatch,
    DomainType,
    PatternComplexity
)
from .domains import DomainClassifier
from .extractors import (
    PatternExtractor,
    MathPatternExtractor,
    CodePatternExtractor,
    LogicPatternExtractor,
    PatternExtractorFactory
)
from .storage import PatternLibrary

__all__ = [
    # Main layer class
    "NuanceLayer",

    # Pattern classes
    "Pattern",
    "PatternMetadata",
    "PatternMatch",
    "DomainType",
    "PatternComplexity",

    # Domain classification
    "DomainClassifier",

    # Pattern extraction
    "PatternExtractor",
    "MathPatternExtractor",
    "CodePatternExtractor",
    "LogicPatternExtractor",
    "PatternExtractorFactory",

    # Storage
    "PatternLibrary",
]
