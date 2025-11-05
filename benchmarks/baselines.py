"""
Baseline Generation

This module generates baseline performance metrics for comparison with TMR.
Baselines include different verification depths and strategies to measure
improvements.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
import logging
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.metrics import PerformanceMetrics, MetricsTracker, BenchmarkResult
from benchmarks.scoring import ScoringSystem, Score

logger = logging.getLogger(__name__)


class BaselineType(Enum):
    """Types of baseline comparisons"""
    NO_VERIFICATION = "no_verification"           # No verification at all
    FUNDAMENTALS_ONLY = "fundamentals_only"       # Layer 1 only
    WITH_NUANCE = "with_nuance"                   # Layers 1+2
    FULL_TMR = "full_tmr"                         # All 3 layers (complete TMR)
    MINIMAL_DEPTH = "minimal_depth"               # Minimal verification depth
    STANDARD_DEPTH = "standard_depth"             # Standard verification depth
    EXHAUSTIVE_DEPTH = "exhaustive_depth"         # Exhaustive verification


@dataclass
class BaselineConfig:
    """
    Configuration for baseline generation.

    Attributes:
        baseline_type: Type of baseline to generate
        use_caching: Whether to use caching
        max_time_ms: Maximum time per problem
        depth_profile: Verification depth profile
        enable_logging: Whether to enable detailed logging
        metadata: Additional configuration metadata
    """
    baseline_type: BaselineType
    use_caching: bool = True
    max_time_ms: float = 10000.0
    depth_profile: Optional[str] = None
    enable_logging: bool = False
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "baseline_type": self.baseline_type.value,
            "use_caching": self.use_caching,
            "max_time_ms": self.max_time_ms,
            "depth_profile": self.depth_profile,
            "enable_logging": self.enable_logging,
            "metadata": self.metadata,
        }


class BaselineGenerator:
    """
    Generates baseline performance metrics for comparison.

    This class creates different baseline configurations (e.g., no verification,
    fundamentals only, full TMR) to measure performance improvements and
    understand the contribution of each layer.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize baseline generator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.scoring_system = ScoringSystem(config=self.config.get("scoring", {}))
        self.baselines: Dict[str, PerformanceMetrics] = {}

    def generate_baseline(
        self,
        problems: List[Any],
        baseline_type: BaselineType,
        baseline_config: Optional[BaselineConfig] = None
    ) -> PerformanceMetrics:
        """
        Generate a baseline for a set of problems.

        Args:
            problems: List of benchmark problems
            baseline_type: Type of baseline to generate
            baseline_config: Optional baseline configuration

        Returns:
            PerformanceMetrics for the baseline
        """
        if baseline_config is None:
            baseline_config = BaselineConfig(baseline_type=baseline_type)

        logger.info(f"Generating baseline: {baseline_type.value}")

        # Create metrics tracker
        tracker = MetricsTracker(config=self.config.get("metrics", {}))
        tracker.start_tracking()

        # Process each problem
        for problem in problems:
            try:
                result = self._process_problem(problem, baseline_config)
                tracker.add_result(result)
            except Exception as e:
                logger.error(f"Error processing problem {getattr(problem, 'id', 'unknown')}: {e}")
                # Add failed result
                tracker.add_result(BenchmarkResult(
                    problem_id=getattr(problem, "id", "unknown"),
                    domain=getattr(problem, "domain", "unknown").value if hasattr(getattr(problem, "domain", "unknown"), "value") else str(getattr(problem, "domain", "unknown")),
                    complexity=getattr(problem, "complexity", "unknown").value if hasattr(getattr(problem, "complexity", "unknown"), "value") else str(getattr(problem, "complexity", "unknown")),
                    success=False,
                    execution_time_ms=0.0,
                    result=None,
                    error=str(e),
                ))

        tracker.stop_tracking()
        metrics = tracker.compute_metrics()

        # Store baseline
        self.baselines[baseline_type.value] = metrics

        logger.info(f"Baseline {baseline_type.value} completed: {metrics.success_rate:.1%} success rate")

        return metrics

    def _process_problem(
        self,
        problem: Any,
        baseline_config: BaselineConfig
    ) -> BenchmarkResult:
        """
        Process a single problem with the specified baseline configuration.

        Args:
            problem: Benchmark problem
            baseline_config: Baseline configuration

        Returns:
            BenchmarkResult for the problem
        """
        start_time = time.time()

        try:
            # Execute verification based on baseline type
            result = self._execute_verification(problem, baseline_config)

            execution_time_ms = (time.time() - start_time) * 1000

            # Score the result
            score = self.scoring_system.score_result(
                problem=problem,
                result=result,
                execution_time_ms=execution_time_ms,
            )

            return BenchmarkResult(
                problem_id=getattr(problem, "id", "unknown"),
                domain=getattr(problem, "domain", "unknown").value if hasattr(getattr(problem, "domain", "unknown"), "value") else str(getattr(problem, "domain", "unknown")),
                complexity=getattr(problem, "complexity", "unknown").value if hasattr(getattr(problem, "complexity", "unknown"), "value") else str(getattr(problem, "complexity", "unknown")),
                success=True,
                execution_time_ms=execution_time_ms,
                result=result,
                score=score,
                metadata={"baseline_type": baseline_config.baseline_type.value},
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Error executing verification for {getattr(problem, 'id', 'unknown')}: {e}")

            return BenchmarkResult(
                problem_id=getattr(problem, "id", "unknown"),
                domain=getattr(problem, "domain", "unknown").value if hasattr(getattr(problem, "domain", "unknown"), "value") else str(getattr(problem, "domain", "unknown")),
                complexity=getattr(problem, "complexity", "unknown").value if hasattr(getattr(problem, "complexity", "unknown"), "value") else str(getattr(problem, "complexity", "unknown")),
                success=False,
                execution_time_ms=execution_time_ms,
                result=None,
                error=str(e),
                metadata={"baseline_type": baseline_config.baseline_type.value},
            )

    def _execute_verification(
        self,
        problem: Any,
        baseline_config: BaselineConfig
    ) -> Dict[str, Any]:
        """
        Execute verification based on baseline type.

        Args:
            problem: Benchmark problem
            baseline_config: Baseline configuration

        Returns:
            Verification result
        """
        baseline_type = baseline_config.baseline_type

        if baseline_type == BaselineType.NO_VERIFICATION:
            return self._no_verification(problem)
        elif baseline_type == BaselineType.FUNDAMENTALS_ONLY:
            return self._fundamentals_only(problem, baseline_config)
        elif baseline_type == BaselineType.WITH_NUANCE:
            return self._with_nuance(problem, baseline_config)
        elif baseline_type == BaselineType.FULL_TMR:
            return self._full_tmr(problem, baseline_config)
        elif baseline_type == BaselineType.MINIMAL_DEPTH:
            return self._depth_based(problem, baseline_config, "MINIMAL")
        elif baseline_type == BaselineType.STANDARD_DEPTH:
            return self._depth_based(problem, baseline_config, "STANDARD")
        elif baseline_type == BaselineType.EXHAUSTIVE_DEPTH:
            return self._depth_based(problem, baseline_config, "EXHAUSTIVE")
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

    def _no_verification(self, problem: Any) -> Dict[str, Any]:
        """
        No verification - just return expected result.

        This baseline represents the "best case" where we assume all
        reasoning is correct without verification.
        """
        return {
            "valid": True,
            "confidence": 1.0,
            "verification_depth": "none",
            "details": "No verification performed",
        }

    def _fundamentals_only(
        self,
        problem: Any,
        baseline_config: BaselineConfig
    ) -> Dict[str, Any]:
        """
        Layer 1 only - fundamentals verification.

        Uses only the fundamentals layer with logical principles.
        """
        try:
            from fundamentals.layer import FundamentalsLayer

            layer = FundamentalsLayer(config={
                "use_cache": baseline_config.use_caching,
            })

            statement = getattr(problem, "input_statement", "")
            domain_str = getattr(problem, "domain", "unknown")
            if hasattr(domain_str, "value"):
                domain_str = domain_str.value

            result = layer.validate(
                statement=statement,
                domain=domain_str,
                validation_type="comprehensive",
            )

            return result

        except Exception as e:
            logger.error(f"Error in fundamentals_only: {e}")
            return {
                "valid": False,
                "confidence": 0.0,
                "error": str(e),
            }

    def _with_nuance(
        self,
        problem: Any,
        baseline_config: BaselineConfig
    ) -> Dict[str, Any]:
        """
        Layers 1+2 - fundamentals + nuance.

        Uses fundamentals and nuance layers for pattern-aware verification.
        """
        try:
            from fundamentals.layer import FundamentalsLayer
            from nuance.layer import NuanceLayer

            fundamentals = FundamentalsLayer(config={
                "use_cache": baseline_config.use_caching,
            })

            nuance = NuanceLayer(config={
                "load_core_patterns": True,
            })

            statement = getattr(problem, "input_statement", "")
            domain_str = getattr(problem, "domain", "unknown")
            if hasattr(domain_str, "value"):
                domain_str = domain_str.value

            # Extract patterns
            patterns = nuance.extract_patterns(statement, domain_str)

            # Validate with fundamentals
            result = fundamentals.validate(
                statement=statement,
                domain=domain_str,
                validation_type="comprehensive",
            )

            # Enhance with pattern information
            result["patterns_extracted"] = len(patterns)
            result["verification_depth"] = "fundamentals_with_nuance"

            return result

        except Exception as e:
            logger.error(f"Error in with_nuance: {e}")
            return {
                "valid": False,
                "confidence": 0.0,
                "error": str(e),
            }

    def _full_tmr(
        self,
        problem: Any,
        baseline_config: BaselineConfig
    ) -> Dict[str, Any]:
        """
        Full TMR - all 3 layers.

        Uses complete TMR system with execution layer synthesis.
        """
        try:
            from execution.synthesizer import ExecutionSynthesizer

            synthesizer = ExecutionSynthesizer(config={
                "fundamentals": {
                    "use_cache": baseline_config.use_caching,
                },
                "depth_selector": {
                    "default_depth": baseline_config.depth_profile or "STANDARD",
                },
            })

            statement = getattr(problem, "input_statement", "")

            result = synthesizer.synthesize(
                input_data=statement,
                output_format="STANDARD",
            )

            return result

        except Exception as e:
            logger.error(f"Error in full_tmr: {e}")
            return {
                "valid": False,
                "confidence": 0.0,
                "error": str(e),
            }

    def _depth_based(
        self,
        problem: Any,
        baseline_config: BaselineConfig,
        depth: str
    ) -> Dict[str, Any]:
        """
        Depth-based verification.

        Uses full TMR with specific verification depth.
        """
        try:
            from execution.synthesizer import ExecutionSynthesizer

            synthesizer = ExecutionSynthesizer(config={
                "fundamentals": {
                    "use_cache": baseline_config.use_caching,
                },
                "depth_selector": {
                    "default_depth": depth,
                },
            })

            statement = getattr(problem, "input_statement", "")

            result = synthesizer.synthesize(
                input_data=statement,
                output_format="STANDARD",
            )

            return result

        except Exception as e:
            logger.error(f"Error in depth_based ({depth}): {e}")
            return {
                "valid": False,
                "confidence": 0.0,
                "error": str(e),
            }

    def generate_all_baselines(
        self,
        problems: List[Any]
    ) -> Dict[str, PerformanceMetrics]:
        """
        Generate all baseline types.

        Args:
            problems: List of benchmark problems

        Returns:
            Dictionary mapping baseline type to metrics
        """
        baselines = {}

        for baseline_type in BaselineType:
            try:
                metrics = self.generate_baseline(problems, baseline_type)
                baselines[baseline_type.value] = metrics
            except Exception as e:
                logger.error(f"Error generating baseline {baseline_type.value}: {e}")

        return baselines

    def compare_baselines(
        self,
        baseline_a: str,
        baseline_b: str
    ) -> Dict[str, Any]:
        """
        Compare two baselines.

        Args:
            baseline_a: First baseline name
            baseline_b: Second baseline name

        Returns:
            Comparison dictionary
        """
        if baseline_a not in self.baselines:
            raise ValueError(f"Baseline {baseline_a} not found")
        if baseline_b not in self.baselines:
            raise ValueError(f"Baseline {baseline_b} not found")

        metrics_a = self.baselines[baseline_a]
        metrics_b = self.baselines[baseline_b]

        comparison = {
            "baseline_a": {
                "name": baseline_a,
                "metrics": metrics_a.to_dict(),
            },
            "baseline_b": {
                "name": baseline_b,
                "metrics": metrics_b.to_dict(),
            },
            "improvements": {},
        }

        # Compare key metrics
        metrics_to_compare = {
            "success_rate": "higher_is_better",
            "avg_time_ms": "lower_is_better",
            "median_time_ms": "lower_is_better",
        }

        for metric, direction in metrics_to_compare.items():
            value_a = getattr(metrics_a, metric, 0.0)
            value_b = getattr(metrics_b, metric, 0.0)

            if direction == "higher_is_better":
                change = value_b - value_a
                improvement = change > 0
            else:
                change = value_a - value_b
                improvement = change > 0

            percent_change = (change / value_a * 100) if value_a > 0 else 0.0

            comparison["improvements"][metric] = {
                "baseline_a_value": value_a,
                "baseline_b_value": value_b,
                "absolute_change": change,
                "percent_change": percent_change,
                "improvement": improvement,
            }

        return comparison

    def save_baselines(self, filepath: str) -> None:
        """
        Save all baselines to file.

        Args:
            filepath: Path to save baselines
        """
        import json

        data = {
            baseline_name: metrics.to_dict()
            for baseline_name, metrics in self.baselines.items()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved baselines to {filepath}")

    def load_baselines(self, filepath: str) -> None:
        """
        Load baselines from file.

        Args:
            filepath: Path to load baselines from
        """
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        self.baselines = {
            baseline_name: PerformanceMetrics.from_dict(metrics_data)
            for baseline_name, metrics_data in data.items()
        }

        logger.info(f"Loaded baselines from {filepath}")
