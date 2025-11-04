"""
Fundamentals Layer - Immutable Logical Principles

This layer implements the core logical principles that remain constant across all domains.
"""

from .principles import (
    LogicalPrinciples,
    IdentityPrinciple,
    NonContradictionPrinciple,
    ExcludedMiddlePrinciple,
    CausalityPrinciple,
    ConservationPrinciple
)

from .validators import (
    PrincipleValidator,
    LogicalValidator,
    MathematicalValidator,
    CausalValidator,
    ConsistencyValidator
)

from .layer import FundamentalsLayer

__all__ = [
    "FundamentalsLayer",
    "LogicalPrinciples",
    "IdentityPrinciple",
    "NonContradictionPrinciple",
    "ExcludedMiddlePrinciple",
    "CausalityPrinciple",
    "ConservationPrinciple",
    "PrincipleValidator",
    "LogicalValidator",
    "MathematicalValidator",
    "CausalValidator",
    "ConsistencyValidator",
]