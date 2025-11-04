"""
Performance Metrics Tracking

This module tracks and analyzes performance metrics for TMR benchmark validation,
including success rates, timing, resource usage, and domain-specific performance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import statistics
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for benchmark evaluation.

    Attributes:
        total_problems: Total number of problems evaluated
        successful: Number of successful evaluations
        failed: Number of failed evaluations
        success_rate: Success rate (0.0-1.0)
        total_time_ms: Total execution time in milliseconds
        avg_time_ms: Average execution time per problem
        median_time_ms: Median execution time
        min_time_ms: Minimum execution time
        max_time_ms: Maximum execution time
        domain_metrics: Metrics broken down by domain
        complexity_metrics: Metrics broken down by complexity
        confidence_metrics: Confidence score statistics
        timestamp: When metrics were recorded
        metadata: Additional metadata
    """
    total_problems: int = 0
    successful: int = 0
    failed: int = 0
    success_rate: float = 0.0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    median_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0
    domain_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    complexity_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    confidence_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "total_problems": self.total_problems,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": self.success_rate,
            "total_time_ms": self.total_time_ms,
            "avg_time_ms": self.avg_time_ms,
            "median_time_ms": self.median_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "domain_metrics": self.domain_metrics,
            "complexity_metrics": self.complexity_metrics,
            "confidence_metrics": self.confidence_metrics,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetrics":
        """Create metrics from dictionary"""
        return cls(
            total_problems=data.get("total_problems", 0),
            successful=data.get("successful", 0),
            failed=data.get("failed", 0),
            success_rate=data.get("success_rate", 0.0),
            total_time_ms=data.get("total_time_ms", 0.0),
            avg_time_ms=data.get("avg_time_ms", 0.0),
            median_time_ms=data.get("median_time_ms", 0.0),
            min_time_ms=data.get("min_time_ms", 0.0),
            max_time_ms=data.get("max_time_ms", 0.0),
            domain_metrics=data.get("domain_metrics", {}),
            complexity_metrics=data.get("complexity_metrics", {}),
            confidence_metrics=data.get("confidence_metrics", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class BenchmarkResult:
    """
    Result of a single benchmark problem execution.

    Attributes:
        problem_id: Problem identifier
        domain: Problem domain
        complexity: Complexity level
        success: Whether execution was successful
        execution_time_ms: Execution time in milliseconds
        result: The actual verification result
        score: Score object (if scored)
        error: Error message (if failed)
        metadata: Additional metadata
    """
    problem_id: str
    domain: str
    complexity: str
    success: bool
    execution_time_ms: float
    result: Any
    score: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "problem_id": self.problem_id,
            "domain": self.domain,
            "complexity": self.complexity,
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
            "result": self.result,
            "score": self.score.to_dict() if hasattr(self.score, "to_dict") else self.score,
            "error": self.error,
            "metadata": self.metadata,
        }


class MetricsTracker:
    """
    Tracks and analyzes performance metrics for benchmark execution.

    This class collects results, computes statistics, and provides analysis
    capabilities for benchmark validation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize metrics tracker.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.results: List[BenchmarkResult] = []
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def start_tracking(self) -> None:
        """Start tracking session"""
        self.start_time = datetime.now()
        self.results = []
        logger.info("Started metrics tracking")

    def stop_tracking(self) -> None:
        """Stop tracking session"""
        self.end_time = datetime.now()
        logger.info("Stopped metrics tracking")

    def add_result(self, result: BenchmarkResult) -> None:
        """
        Add a benchmark result.

        Args:
            result: Benchmark result to add
        """
        self.results.append(result)

    def compute_metrics(self) -> PerformanceMetrics:
        """
        Compute comprehensive performance metrics from collected results.

        Returns:
            PerformanceMetrics object with all statistics
        """
        if not self.results:
            return PerformanceMetrics()

        # Basic metrics
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful
        success_rate = successful / total if total > 0 else 0.0

        # Time metrics
        times = [r.execution_time_ms for r in self.results]
        total_time = sum(times)
        avg_time = statistics.mean(times)
        median_time = statistics.median(times)
        min_time = min(times)
        max_time = max(times)

        # Domain-specific metrics
        domain_metrics = self._compute_domain_metrics()

        # Complexity-specific metrics
        complexity_metrics = self._compute_complexity_metrics()

        # Confidence metrics
        confidence_metrics = self._compute_confidence_metrics()

        # Session duration
        session_duration_s = 0.0
        if self.start_time and self.end_time:
            session_duration_s = (self.end_time - self.start_time).total_seconds()

        return PerformanceMetrics(
            total_problems=total,
            successful=successful,
            failed=failed,
            success_rate=success_rate,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            median_time_ms=median_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            domain_metrics=domain_metrics,
            complexity_metrics=complexity_metrics,
            confidence_metrics=confidence_metrics,
            metadata={
                "session_duration_s": session_duration_s,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
            }
        )

    def _compute_domain_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Compute metrics broken down by domain"""
        domain_results: Dict[str, List[BenchmarkResult]] = {}

        # Group results by domain
        for result in self.results:
            domain = result.domain
            if domain not in domain_results:
                domain_results[domain] = []
            domain_results[domain].append(result)

        # Compute metrics for each domain
        domain_metrics = {}
        for domain, results in domain_results.items():
            total = len(results)
            successful = sum(1 for r in results if r.success)
            times = [r.execution_time_ms for r in results]

            # Extract scores if available
            scores = [r.score for r in results if r.score is not None]
            avg_overall_score = None
            if scores and hasattr(scores[0], "overall"):
                avg_overall_score = statistics.mean([s.overall for s in scores])

            domain_metrics[domain] = {
                "total": total,
                "successful": successful,
                "failed": total - successful,
                "success_rate": successful / total if total > 0 else 0.0,
                "avg_time_ms": statistics.mean(times),
                "median_time_ms": statistics.median(times),
                "avg_overall_score": avg_overall_score,
            }

        return domain_metrics

    def _compute_complexity_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Compute metrics broken down by complexity"""
        complexity_results: Dict[str, List[BenchmarkResult]] = {}

        # Group results by complexity
        for result in self.results:
            complexity = result.complexity
            if complexity not in complexity_results:
                complexity_results[complexity] = []
            complexity_results[complexity].append(result)

        # Compute metrics for each complexity level
        complexity_metrics = {}
        for complexity, results in complexity_results.items():
            total = len(results)
            successful = sum(1 for r in results if r.success)
            times = [r.execution_time_ms for r in results]

            # Extract scores if available
            scores = [r.score for r in results if r.score is not None]
            avg_overall_score = None
            if scores and hasattr(scores[0], "overall"):
                avg_overall_score = statistics.mean([s.overall for s in scores])

            complexity_metrics[complexity] = {
                "total": total,
                "successful": successful,
                "failed": total - successful,
                "success_rate": successful / total if total > 0 else 0.0,
                "avg_time_ms": statistics.mean(times),
                "median_time_ms": statistics.median(times),
                "avg_overall_score": avg_overall_score,
            }

        return complexity_metrics

    def _compute_confidence_metrics(self) -> Dict[str, float]:
        """Compute confidence score metrics"""
        confidence_scores = []

        for result in self.results:
            if result.score and hasattr(result.score, "confidence_accuracy"):
                confidence_scores.append(result.score.confidence_accuracy)

        if not confidence_scores:
            return {}

        return {
            "avg_confidence_accuracy": statistics.mean(confidence_scores),
            "median_confidence_accuracy": statistics.median(confidence_scores),
            "min_confidence_accuracy": min(confidence_scores),
            "max_confidence_accuracy": max(confidence_scores),
            "std_dev_confidence_accuracy": statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0.0,
        }

    def get_failed_problems(self) -> List[BenchmarkResult]:
        """Get list of failed problems"""
        return [r for r in self.results if not r.success]

    def get_slow_problems(self, threshold_ms: float = 1000.0) -> List[BenchmarkResult]:
        """
        Get problems that exceeded time threshold.

        Args:
            threshold_ms: Time threshold in milliseconds

        Returns:
            List of results exceeding threshold
        """
        return [r for r in self.results if r.execution_time_ms > threshold_ms]

    def get_low_score_problems(self, threshold: float = 0.5) -> List[BenchmarkResult]:
        """
        Get problems with low scores.

        Args:
            threshold: Score threshold (0.0-1.0)

        Returns:
            List of results below threshold
        """
        return [
            r for r in self.results
            if r.score and hasattr(r.score, "overall") and r.score.overall < threshold
        ]

    def compare_with_baseline(
        self,
        baseline_metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """
        Compare current metrics with baseline.

        Args:
            baseline_metrics: Baseline metrics to compare against

        Returns:
            Dictionary with comparison results
        """
        current_metrics = self.compute_metrics()

        comparison = {
            "baseline": baseline_metrics.to_dict(),
            "current": current_metrics.to_dict(),
            "changes": {},
        }

        # Compare key metrics
        metrics_to_compare = [
            "success_rate",
            "avg_time_ms",
            "median_time_ms",
        ]

        for metric in metrics_to_compare:
            baseline_value = getattr(baseline_metrics, metric, 0.0)
            current_value = getattr(current_metrics, metric, 0.0)

            if metric in ["success_rate"]:
                # Higher is better
                change = current_value - baseline_value
                percent_change = (change / baseline_value * 100) if baseline_value > 0 else 0.0
                improvement = change > 0
            else:
                # Lower is better (time metrics)
                change = baseline_value - current_value
                percent_change = (change / baseline_value * 100) if baseline_value > 0 else 0.0
                improvement = change > 0

            comparison["changes"][metric] = {
                "baseline": baseline_value,
                "current": current_value,
                "absolute_change": change,
                "percent_change": percent_change,
                "improvement": improvement,
            }

        return comparison

    def save_results(self, filepath: str) -> None:
        """
        Save results to file.

        Args:
            filepath: Path to save results
        """
        data = {
            "metrics": self.compute_metrics().to_dict(),
            "results": [r.to_dict() for r in self.results],
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved results to {filepath}")

    def load_results(self, filepath: str) -> PerformanceMetrics:
        """
        Load results from file.

        Args:
            filepath: Path to load results from

        Returns:
            PerformanceMetrics object
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct results
        self.results = []
        for result_data in data.get("results", []):
            result = BenchmarkResult(
                problem_id=result_data["problem_id"],
                domain=result_data["domain"],
                complexity=result_data["complexity"],
                success=result_data["success"],
                execution_time_ms=result_data["execution_time_ms"],
                result=result_data["result"],
                score=result_data.get("score"),
                error=result_data.get("error"),
                metadata=result_data.get("metadata", {}),
            )
            self.results.append(result)

        logger.info(f"Loaded results from {filepath}")
        return PerformanceMetrics.from_dict(data.get("metrics", {}))

    def generate_summary(self) -> str:
        """
        Generate a human-readable summary of metrics.

        Returns:
            Summary string
        """
        metrics = self.compute_metrics()

        summary = [
            "=" * 60,
            "BENCHMARK METRICS SUMMARY",
            "=" * 60,
            "",
            f"Total Problems:    {metrics.total_problems}",
            f"Successful:        {metrics.successful} ({metrics.success_rate:.1%})",
            f"Failed:            {metrics.failed}",
            "",
            "Timing Statistics:",
            f"  Total Time:      {metrics.total_time_ms:.2f} ms",
            f"  Average Time:    {metrics.avg_time_ms:.2f} ms",
            f"  Median Time:     {metrics.median_time_ms:.2f} ms",
            f"  Min Time:        {metrics.min_time_ms:.2f} ms",
            f"  Max Time:        {metrics.max_time_ms:.2f} ms",
            "",
        ]

        # Domain breakdown
        if metrics.domain_metrics:
            summary.append("Domain Breakdown:")
            for domain, dm in metrics.domain_metrics.items():
                summary.append(f"  {domain}:")
                summary.append(f"    Success Rate: {dm['success_rate']:.1%}")
                summary.append(f"    Avg Time:     {dm['avg_time_ms']:.2f} ms")
                if dm.get("avg_overall_score") is not None:
                    summary.append(f"    Avg Score:    {dm['avg_overall_score']:.3f}")
            summary.append("")

        # Complexity breakdown
        if metrics.complexity_metrics:
            summary.append("Complexity Breakdown:")
            for complexity, cm in metrics.complexity_metrics.items():
                summary.append(f"  {complexity}:")
                summary.append(f"    Success Rate: {cm['success_rate']:.1%}")
                summary.append(f"    Avg Time:     {cm['avg_time_ms']:.2f} ms")
                if cm.get("avg_overall_score") is not None:
                    summary.append(f"    Avg Score:    {cm['avg_overall_score']:.3f}")
            summary.append("")

        # Confidence metrics
        if metrics.confidence_metrics:
            summary.append("Confidence Metrics:")
            for key, value in metrics.confidence_metrics.items():
                summary.append(f"  {key}: {value:.3f}")
            summary.append("")

        summary.append("=" * 60)

        return "\n".join(summary)
