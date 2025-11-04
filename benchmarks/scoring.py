"""
Scoring System

This module implements a comprehensive scoring system for evaluating TMR
performance on benchmark problems, including correctness, confidence, and
efficiency metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import logging
import statistics

logger = logging.getLogger(__name__)


class ScoreComponent(Enum):
    """Components of the overall score"""
    CORRECTNESS = "correctness"           # Binary or partial correctness
    CONFIDENCE = "confidence"             # Confidence score accuracy
    EFFICIENCY = "efficiency"             # Time and resource usage
    CONSISTENCY = "consistency"           # Reproducibility
    ROBUSTNESS = "robustness"            # Error handling


@dataclass
class Score:
    """
    Comprehensive score for a benchmark result.

    Attributes:
        correctness: Correctness score (0.0-1.0)
        confidence_accuracy: How close confidence was to expected (0.0-1.0)
        efficiency: Efficiency score based on time/resources (0.0-1.0)
        consistency: Consistency score across runs (0.0-1.0)
        robustness: Robustness score (0.0-1.0)
        overall: Weighted overall score (0.0-1.0)
        details: Additional scoring details
    """
    correctness: float
    confidence_accuracy: float
    efficiency: float
    consistency: float = 1.0
    robustness: float = 1.0
    overall: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate overall score after initialization"""
        if self.overall == 0.0:
            self.overall = self.calculate_overall()

    def calculate_overall(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate weighted overall score.

        Args:
            weights: Optional custom weights for components

        Returns:
            Overall weighted score (0.0-1.0)
        """
        default_weights = {
            "correctness": 0.40,      # Most important
            "confidence": 0.25,       # Important for trust
            "efficiency": 0.15,       # Performance matters
            "consistency": 0.10,      # Reproducibility
            "robustness": 0.10,       # Error handling
        }

        weights = weights or default_weights

        overall = (
            self.correctness * weights["correctness"] +
            self.confidence_accuracy * weights["confidence"] +
            self.efficiency * weights["efficiency"] +
            self.consistency * weights["consistency"] +
            self.robustness * weights["robustness"]
        )

        return max(0.0, min(1.0, overall))

    def to_dict(self) -> Dict[str, Any]:
        """Convert score to dictionary"""
        return {
            "correctness": self.correctness,
            "confidence_accuracy": self.confidence_accuracy,
            "efficiency": self.efficiency,
            "consistency": self.consistency,
            "robustness": self.robustness,
            "overall": self.overall,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Score":
        """Create score from dictionary"""
        return cls(
            correctness=data.get("correctness", 0.0),
            confidence_accuracy=data.get("confidence_accuracy", 0.0),
            efficiency=data.get("efficiency", 0.0),
            consistency=data.get("consistency", 1.0),
            robustness=data.get("robustness", 1.0),
            overall=data.get("overall", 0.0),
            details=data.get("details", {}),
        )


class ScoringSystem:
    """
    Unified scoring system for benchmark evaluation.

    This system evaluates TMR performance across multiple dimensions:
    - Correctness: Did it get the right answer?
    - Confidence: Was the confidence score appropriate?
    - Efficiency: How fast/efficient was the verification?
    - Consistency: Reproducible results across runs?
    - Robustness: How well does it handle edge cases?
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize scoring system.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.weights = self.config.get("weights", {})
        self.time_thresholds = self.config.get("time_thresholds", {
            "excellent": 100,   # < 100ms
            "good": 500,        # < 500ms
            "acceptable": 2000, # < 2s
            "poor": 10000,      # < 10s
        })
        self.confidence_tolerance = self.config.get("confidence_tolerance", 0.15)

    def score_result(
        self,
        problem: Any,
        result: Any,
        execution_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Score:
        """
        Score a benchmark result.

        Args:
            problem: The benchmark problem
            result: The verification result from TMR
            execution_time_ms: Execution time in milliseconds
            metadata: Optional additional metadata

        Returns:
            Comprehensive score for the result
        """
        metadata = metadata or {}

        # Calculate component scores
        correctness = self._score_correctness(problem, result)
        confidence_accuracy = self._score_confidence(problem, result)
        efficiency = self._score_efficiency(execution_time_ms, problem)
        consistency = metadata.get("consistency_score", 1.0)
        robustness = self._score_robustness(result, metadata)

        # Create detailed score object
        score = Score(
            correctness=correctness,
            confidence_accuracy=confidence_accuracy,
            efficiency=efficiency,
            consistency=consistency,
            robustness=robustness,
            details={
                "execution_time_ms": execution_time_ms,
                "expected_valid": getattr(problem, "expected_valid", None),
                "actual_valid": self._extract_valid(result),
                "expected_confidence": getattr(problem, "expected_confidence", None),
                "actual_confidence": self._extract_confidence(result),
                "problem_id": getattr(problem, "id", "unknown"),
                "problem_complexity": getattr(problem, "complexity", None),
            }
        )

        return score

    def _score_correctness(self, problem: Any, result: Any) -> float:
        """
        Score correctness of the result.

        Args:
            problem: The benchmark problem
            result: The verification result

        Returns:
            Correctness score (0.0-1.0)
        """
        # Use problem's validator if available
        if hasattr(problem, "validate_result"):
            try:
                if problem.validate_result(result):
                    return 1.0
                else:
                    return 0.0
            except Exception as e:
                logger.error(f"Error in custom validator: {e}")
                return 0.0

        # Default validation
        expected_valid = getattr(problem, "expected_valid", None)
        if expected_valid is None:
            return 0.5  # Unknown expected outcome

        actual_valid = self._extract_valid(result)
        if actual_valid == expected_valid:
            return 1.0
        else:
            return 0.0

    def _score_confidence(self, problem: Any, result: Any) -> float:
        """
        Score confidence accuracy.

        Args:
            problem: The benchmark problem
            result: The verification result

        Returns:
            Confidence accuracy score (0.0-1.0)
        """
        expected_confidence = getattr(problem, "expected_confidence", None)
        if expected_confidence is None:
            return 1.0  # No expected confidence to compare

        actual_confidence = self._extract_confidence(result)
        if actual_confidence is None:
            return 0.0  # No confidence provided

        # Calculate difference
        diff = abs(expected_confidence - actual_confidence)

        # Score based on difference (with tolerance)
        if diff <= self.confidence_tolerance:
            # Within tolerance: linear score
            return 1.0 - (diff / self.confidence_tolerance)
        else:
            # Outside tolerance: exponential decay
            return max(0.0, 0.5 * (1.0 - diff))

    def _score_efficiency(self, execution_time_ms: float, problem: Any) -> float:
        """
        Score efficiency based on execution time.

        Args:
            execution_time_ms: Execution time in milliseconds
            problem: The benchmark problem (may have complexity hints)

        Returns:
            Efficiency score (0.0-1.0)
        """
        # Adjust thresholds based on problem complexity
        complexity = getattr(problem, "complexity", None)
        multiplier = 1.0

        if complexity:
            complexity_multipliers = {
                "trivial": 0.5,
                "simple": 0.75,
                "moderate": 1.0,
                "complex": 1.5,
                "advanced": 2.0,
            }
            complexity_str = complexity.value if hasattr(complexity, "value") else str(complexity)
            multiplier = complexity_multipliers.get(complexity_str, 1.0)

        # Apply multiplier to thresholds
        excellent = self.time_thresholds["excellent"] * multiplier
        good = self.time_thresholds["good"] * multiplier
        acceptable = self.time_thresholds["acceptable"] * multiplier
        poor = self.time_thresholds["poor"] * multiplier

        # Score based on time
        if execution_time_ms < excellent:
            return 1.0
        elif execution_time_ms < good:
            # Linear interpolation between 1.0 and 0.8
            return 1.0 - 0.2 * (execution_time_ms - excellent) / (good - excellent)
        elif execution_time_ms < acceptable:
            # Linear interpolation between 0.8 and 0.5
            return 0.8 - 0.3 * (execution_time_ms - good) / (acceptable - good)
        elif execution_time_ms < poor:
            # Linear interpolation between 0.5 and 0.2
            return 0.5 - 0.3 * (execution_time_ms - acceptable) / (poor - acceptable)
        else:
            # Very slow: exponential decay
            return max(0.0, 0.2 * (1.0 - (execution_time_ms - poor) / poor))

    def _score_robustness(self, result: Any, metadata: Dict[str, Any]) -> float:
        """
        Score robustness (error handling, edge cases).

        Args:
            result: The verification result
            metadata: Additional metadata

        Returns:
            Robustness score (0.0-1.0)
        """
        score = 1.0

        # Check if result indicates an error
        if isinstance(result, dict):
            if "error" in result:
                # Error occurred, but was handled gracefully
                score = 0.5
            if result.get("valid") is None:
                # Missing validity flag
                score *= 0.8

        # Check metadata for error indicators
        if metadata.get("has_errors", False):
            score *= 0.6
        if metadata.get("has_warnings", False):
            score *= 0.9

        return score

    def _extract_valid(self, result: Any) -> Optional[bool]:
        """Extract valid flag from result"""
        if isinstance(result, dict):
            return result.get("valid")
        return None

    def _extract_confidence(self, result: Any) -> Optional[float]:
        """Extract confidence score from result"""
        if isinstance(result, dict):
            # Try multiple common keys
            for key in ["confidence", "confidence_score", "score"]:
                if key in result:
                    return result[key]
        return None

    def aggregate_scores(self, scores: List[Score]) -> Dict[str, Any]:
        """
        Aggregate multiple scores into summary statistics.

        Args:
            scores: List of scores to aggregate

        Returns:
            Dictionary with aggregate statistics
        """
        if not scores:
            return {
                "count": 0,
                "mean": {},
                "median": {},
                "std_dev": {},
                "min": {},
                "max": {},
            }

        components = ["correctness", "confidence_accuracy", "efficiency", "consistency", "robustness", "overall"]

        aggregates = {
            "count": len(scores),
            "mean": {},
            "median": {},
            "std_dev": {},
            "min": {},
            "max": {},
        }

        for component in components:
            values = [getattr(score, component) for score in scores]

            aggregates["mean"][component] = statistics.mean(values)
            aggregates["median"][component] = statistics.median(values)
            aggregates["std_dev"][component] = statistics.stdev(values) if len(values) > 1 else 0.0
            aggregates["min"][component] = min(values)
            aggregates["max"][component] = max(values)

        return aggregates

    def compare_scores(self, scores_a: List[Score], scores_b: List[Score]) -> Dict[str, Any]:
        """
        Compare two sets of scores.

        Args:
            scores_a: First set of scores (e.g., baseline)
            scores_b: Second set of scores (e.g., new system)

        Returns:
            Dictionary with comparison statistics
        """
        agg_a = self.aggregate_scores(scores_a)
        agg_b = self.aggregate_scores(scores_b)

        components = ["correctness", "confidence_accuracy", "efficiency", "consistency", "robustness", "overall"]

        comparison = {
            "baseline": agg_a,
            "comparison": agg_b,
            "improvements": {},
            "regressions": {},
        }

        for component in components:
            mean_a = agg_a["mean"].get(component, 0.0)
            mean_b = agg_b["mean"].get(component, 0.0)
            diff = mean_b - mean_a
            percent_change = (diff / mean_a * 100) if mean_a > 0 else 0.0

            comparison["improvements" if diff > 0 else "regressions"][component] = {
                "absolute_change": diff,
                "percent_change": percent_change,
                "baseline_mean": mean_a,
                "comparison_mean": mean_b,
            }

        return comparison
