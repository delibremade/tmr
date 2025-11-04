"""
Depth Selector for Adaptive Verification Depth Scaling

This module implements adaptive depth scaling to determine the appropriate
level of verification based on context, complexity, and confidence requirements.
"""

from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class VerificationDepth(Enum):
    """Verification depth levels."""
    MINIMAL = "minimal"      # Basic structural checks only
    QUICK = "quick"          # Essential validations
    STANDARD = "standard"    # Standard verification suite
    THOROUGH = "thorough"    # Comprehensive validation
    EXHAUSTIVE = "exhaustive"  # All possible checks
    ADAPTIVE = "adaptive"    # Automatically determine depth


@dataclass
class DepthProfile:
    """Profile defining verification depth requirements."""
    depth: VerificationDepth
    validators: List[str]
    principles: List[str]
    max_time_ms: Optional[float] = None
    min_confidence: float = 0.7
    description: str = ""


class DepthSelector:
    """
    Adaptive Depth Selector

    Determines the appropriate verification depth based on:
    - Input complexity
    - Domain type
    - Required confidence level
    - Performance constraints
    - Historical success patterns
    """

    # Predefined depth profiles
    DEPTH_PROFILES = {
        VerificationDepth.MINIMAL: DepthProfile(
            depth=VerificationDepth.MINIMAL,
            validators=["logical"],
            principles=["identity", "non_contradiction"],
            max_time_ms=50,
            min_confidence=0.5,
            description="Minimal checks for quick validation"
        ),
        VerificationDepth.QUICK: DepthProfile(
            depth=VerificationDepth.QUICK,
            validators=["logical"],
            principles=["identity", "non_contradiction", "excluded_middle"],
            max_time_ms=100,
            min_confidence=0.6,
            description="Quick validation for simple tasks"
        ),
        VerificationDepth.STANDARD: DepthProfile(
            depth=VerificationDepth.STANDARD,
            validators=["logical", "mathematical"],
            principles=["identity", "non_contradiction", "excluded_middle", "causality"],
            max_time_ms=500,
            min_confidence=0.7,
            description="Standard verification for most tasks"
        ),
        VerificationDepth.THOROUGH: DepthProfile(
            depth=VerificationDepth.THOROUGH,
            validators=["logical", "mathematical", "causal"],
            principles=["identity", "non_contradiction", "excluded_middle", "causality", "conservation"],
            max_time_ms=2000,
            min_confidence=0.85,
            description="Thorough verification for complex reasoning"
        ),
        VerificationDepth.EXHAUSTIVE: DepthProfile(
            depth=VerificationDepth.EXHAUSTIVE,
            validators=["logical", "mathematical", "causal", "consistency"],
            principles=["identity", "non_contradiction", "excluded_middle", "causality", "conservation"],
            max_time_ms=10000,
            min_confidence=0.95,
            description="Exhaustive verification for critical tasks"
        )
    }

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the depth selector.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

        # Historical performance tracking
        self.depth_performance = {
            depth: {"successes": 0, "failures": 0, "avg_time_ms": 0.0}
            for depth in VerificationDepth
            if depth != VerificationDepth.ADAPTIVE
        }

        # Context-based depth adjustments
        self.domain_preferences = {
            "mathematical": VerificationDepth.THOROUGH,
            "logical": VerificationDepth.STANDARD,
            "causal": VerificationDepth.THOROUGH,
            "mixed": VerificationDepth.EXHAUSTIVE,
            "simple": VerificationDepth.QUICK,
            "unknown": VerificationDepth.STANDARD
        }

        logger.info("DepthSelector initialized")

    def select_depth(self,
                    input_data: Any,
                    domain: Optional[str] = None,
                    required_confidence: Optional[float] = None,
                    time_budget_ms: Optional[float] = None,
                    user_preference: Optional[VerificationDepth] = None) -> DepthProfile:
        """
        Select appropriate verification depth.

        Args:
            input_data: The input to be verified
            domain: Optional domain hint
            required_confidence: Minimum required confidence (0.0-1.0)
            time_budget_ms: Maximum time budget in milliseconds
            user_preference: User-specified depth preference

        Returns:
            DepthProfile with selected depth and configuration
        """
        # If user specifies non-adaptive depth, use it directly
        if user_preference and user_preference != VerificationDepth.ADAPTIVE:
            logger.info(f"Using user-specified depth: {user_preference.value}")
            return self.DEPTH_PROFILES[user_preference]

        # Calculate complexity score
        complexity = self._calculate_complexity(input_data)

        # Infer domain if not provided
        if not domain:
            domain = self._infer_domain(input_data)

        # Get base depth from domain preferences
        base_depth = self.domain_preferences.get(domain, VerificationDepth.STANDARD)

        # Adjust based on complexity
        adjusted_depth = self._adjust_for_complexity(base_depth, complexity)

        # Adjust for confidence requirement
        if required_confidence:
            adjusted_depth = self._adjust_for_confidence(adjusted_depth, required_confidence)

        # Adjust for time budget
        if time_budget_ms:
            adjusted_depth = self._adjust_for_time_budget(adjusted_depth, time_budget_ms)

        # Get the profile
        profile = self.DEPTH_PROFILES[adjusted_depth]

        # Customize profile if needed
        if required_confidence:
            profile = self._customize_profile(profile, min_confidence=required_confidence)

        logger.info(f"Selected depth: {adjusted_depth.value} "
                   f"(complexity={complexity:.2f}, domain={domain})")

        return profile

    def _calculate_complexity(self, input_data: Any) -> float:
        """
        Calculate complexity score (0.0 = simple, 1.0 = very complex).

        Args:
            input_data: Input to analyze

        Returns:
            Complexity score between 0.0 and 1.0
        """
        complexity = 0.0

        # String analysis
        if isinstance(input_data, str):
            # Length contribution
            length_score = min(len(input_data) / 1000.0, 0.3)
            complexity += length_score

            # Mathematical operators
            math_ops = sum(input_data.count(op) for op in ['+', '-', '*', '/', '^', '='])
            math_score = min(math_ops / 20.0, 0.2)
            complexity += math_score

            # Logical connectives
            logical_keywords = ['if', 'then', 'else', 'and', 'or', 'not', 'because', 'therefore']
            logic_count = sum(input_data.lower().count(kw) for kw in logical_keywords)
            logic_score = min(logic_count / 10.0, 0.2)
            complexity += logic_score

            # Nested structures (parentheses, brackets)
            nesting_count = input_data.count('(') + input_data.count('[') + input_data.count('{')
            nesting_score = min(nesting_count / 10.0, 0.15)
            complexity += nesting_score

        # Dictionary analysis
        elif isinstance(input_data, dict):
            # Number of keys
            key_score = min(len(input_data) / 20.0, 0.2)
            complexity += key_score

            # Nested structures
            if any(isinstance(v, (dict, list)) for v in input_data.values()):
                complexity += 0.3

            # Check for specific complex patterns
            if "steps" in input_data or "chain" in input_data:
                steps = input_data.get("steps", [])
                if isinstance(steps, list):
                    complexity += min(len(steps) / 10.0, 0.3)

            # Multiple reasoning types
            reasoning_types = ["logical", "mathematical", "causal"]
            present_types = sum(1 for rt in reasoning_types if rt in input_data)
            complexity += present_types * 0.15

        # List analysis
        elif isinstance(input_data, list):
            list_score = min(len(input_data) / 15.0, 0.4)
            complexity += list_score

            # Check for nested structures
            if any(isinstance(item, (dict, list)) for item in input_data):
                complexity += 0.3

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, complexity))

    def _infer_domain(self, input_data: Any) -> str:
        """
        Infer the reasoning domain from input data.

        Args:
            input_data: Input to analyze

        Returns:
            Domain name
        """
        # Check for explicit domain markers in dict
        if isinstance(input_data, dict):
            if "domain" in input_data:
                return input_data["domain"]

            # Check for domain-specific keys
            if any(k in input_data for k in ["equation", "calculation", "result"]):
                return "mathematical"
            if any(k in input_data for k in ["cause", "effect", "events", "temporal"]):
                return "causal"
            if any(k in input_data for k in ["steps", "chain", "propositions"]):
                return "logical"
            if len(set(["logical", "mathematical", "causal"]) & set(input_data.keys())) > 1:
                return "mixed"

        # String analysis
        if isinstance(input_data, str):
            text = input_data.lower()

            # Mathematical indicators
            math_indicators = ['=', '+', '-', '*', '/', 'equation', 'calculate', 'solve']
            math_count = sum(text.count(ind) for ind in math_indicators)

            # Causal indicators
            causal_indicators = ['because', 'therefore', 'cause', 'effect', 'leads to', 'results in']
            causal_count = sum(text.count(ind) for ind in causal_indicators)

            # Logical indicators
            logical_indicators = ['if', 'then', 'and', 'or', 'not', 'implies', 'thus']
            logical_count = sum(text.count(ind) for ind in logical_indicators)

            # Determine dominant type
            counts = {
                "mathematical": math_count,
                "causal": causal_count,
                "logical": logical_count
            }

            max_count = max(counts.values())
            if max_count > 0:
                # Check if multiple domains are involved
                active_domains = [d for d, c in counts.items() if c > max_count * 0.6]
                if len(active_domains) > 1:
                    return "mixed"
                return max(counts, key=counts.get)

            # Check length as complexity indicator
            if len(text) < 50:
                return "simple"

        return "unknown"

    def _adjust_for_complexity(self, base_depth: VerificationDepth,
                               complexity: float) -> VerificationDepth:
        """Adjust depth based on complexity score."""
        depth_order = [
            VerificationDepth.MINIMAL,
            VerificationDepth.QUICK,
            VerificationDepth.STANDARD,
            VerificationDepth.THOROUGH,
            VerificationDepth.EXHAUSTIVE
        ]

        current_index = depth_order.index(base_depth)

        # Adjust based on complexity
        if complexity < 0.2:
            # Very simple - can reduce depth
            adjustment = -1
        elif complexity < 0.4:
            # Simple - slight reduction ok
            adjustment = 0
        elif complexity < 0.6:
            # Moderate - keep as is
            adjustment = 0
        elif complexity < 0.8:
            # Complex - increase depth
            adjustment = 1
        else:
            # Very complex - significantly increase depth
            adjustment = 2

        new_index = max(0, min(len(depth_order) - 1, current_index + adjustment))
        return depth_order[new_index]

    def _adjust_for_confidence(self, current_depth: VerificationDepth,
                               required_confidence: float) -> VerificationDepth:
        """Adjust depth to meet confidence requirement."""
        depth_order = [
            VerificationDepth.MINIMAL,
            VerificationDepth.QUICK,
            VerificationDepth.STANDARD,
            VerificationDepth.THOROUGH,
            VerificationDepth.EXHAUSTIVE
        ]

        # Map confidence to minimum depth
        if required_confidence >= 0.95:
            min_depth = VerificationDepth.EXHAUSTIVE
        elif required_confidence >= 0.85:
            min_depth = VerificationDepth.THOROUGH
        elif required_confidence >= 0.7:
            min_depth = VerificationDepth.STANDARD
        elif required_confidence >= 0.6:
            min_depth = VerificationDepth.QUICK
        else:
            min_depth = VerificationDepth.MINIMAL

        # Return the higher of current and minimum required depth
        current_index = depth_order.index(current_depth)
        min_index = depth_order.index(min_depth)

        return depth_order[max(current_index, min_index)]

    def _adjust_for_time_budget(self, current_depth: VerificationDepth,
                               time_budget_ms: float) -> VerificationDepth:
        """Adjust depth to fit within time budget."""
        depth_order = [
            VerificationDepth.MINIMAL,
            VerificationDepth.QUICK,
            VerificationDepth.STANDARD,
            VerificationDepth.THOROUGH,
            VerificationDepth.EXHAUSTIVE
        ]

        # Find the maximum depth that fits the budget
        max_depth = VerificationDepth.MINIMAL
        for depth in depth_order:
            profile = self.DEPTH_PROFILES[depth]
            if profile.max_time_ms and profile.max_time_ms <= time_budget_ms:
                max_depth = depth
            else:
                break

        # Return the lower of current and maximum allowed depth
        current_index = depth_order.index(current_depth)
        max_index = depth_order.index(max_depth)

        return depth_order[min(current_index, max_index)]

    def _customize_profile(self, profile: DepthProfile,
                          min_confidence: Optional[float] = None,
                          validators: Optional[List[str]] = None,
                          principles: Optional[List[str]] = None) -> DepthProfile:
        """Create a customized copy of a depth profile."""
        return DepthProfile(
            depth=profile.depth,
            validators=validators if validators else profile.validators.copy(),
            principles=principles if principles else profile.principles.copy(),
            max_time_ms=profile.max_time_ms,
            min_confidence=min_confidence if min_confidence else profile.min_confidence,
            description=profile.description
        )

    def record_performance(self, depth: VerificationDepth, success: bool, time_ms: float):
        """
        Record performance data for adaptive learning.

        Args:
            depth: The depth level used
            success: Whether verification succeeded
            time_ms: Time taken in milliseconds
        """
        if depth == VerificationDepth.ADAPTIVE:
            return

        stats = self.depth_performance[depth]

        if success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1

        # Update average time (exponential moving average)
        alpha = 0.1
        stats["avg_time_ms"] = alpha * time_ms + (1 - alpha) * stats["avg_time_ms"]

    def get_performance_stats(self) -> Dict[str, Dict]:
        """Get performance statistics for all depth levels."""
        stats = {}
        for depth, perf in self.depth_performance.items():
            total = perf["successes"] + perf["failures"]
            success_rate = perf["successes"] / total if total > 0 else 0.0

            stats[depth.value] = {
                "success_rate": success_rate,
                "total_uses": total,
                "avg_time_ms": perf["avg_time_ms"]
            }

        return stats

    def get_recommended_depth(self, domain: str) -> VerificationDepth:
        """
        Get recommended depth for a domain based on historical performance.

        Args:
            domain: The reasoning domain

        Returns:
            Recommended verification depth
        """
        return self.domain_preferences.get(domain, VerificationDepth.STANDARD)

    def __repr__(self) -> str:
        """String representation of the depth selector."""
        return f"DepthSelector(profiles={len(self.DEPTH_PROFILES)})"
