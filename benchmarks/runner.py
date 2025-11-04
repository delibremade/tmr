"""
Benchmark Runner

This module provides the main orchestration for running benchmarks,
coordinating problem execution, scoring, metrics collection, and reporting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
import logging
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.problems import (
    BenchmarkProblem,
    ProblemSet,
    get_all_problem_sets,
    get_all_problems,
    get_problems_by_domain,
    get_problems_by_complexity,
)
from benchmarks.scoring import ScoringSystem, Score
from benchmarks.metrics import MetricsTracker, BenchmarkResult, PerformanceMetrics
from benchmarks.baselines import BaselineGenerator, BaselineType

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmark execution.

    Attributes:
        run_all_domains: Run all domains
        domains: Specific domains to run (if not all)
        complexities: Specific complexity levels to run
        max_time_per_problem_ms: Maximum time per problem
        use_caching: Enable caching
        verification_depth: Verification depth profile
        generate_baselines: Whether to generate baselines
        baseline_types: Baseline types to generate
        enable_logging: Enable detailed logging
        output_dir: Directory for output files
        metadata: Additional configuration metadata
    """
    run_all_domains: bool = True
    domains: List[str] = field(default_factory=list)
    complexities: List[str] = field(default_factory=list)
    max_time_per_problem_ms: float = 10000.0
    use_caching: bool = True
    verification_depth: str = "STANDARD"
    generate_baselines: bool = True
    baseline_types: List[str] = field(default_factory=list)
    enable_logging: bool = True
    output_dir: str = "./benchmark_results"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Set default baseline types if not specified
        if not self.baseline_types and self.generate_baselines:
            self.baseline_types = [
                "fundamentals_only",
                "with_nuance",
                "full_tmr",
                "standard_depth",
            ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "run_all_domains": self.run_all_domains,
            "domains": self.domains,
            "complexities": self.complexities,
            "max_time_per_problem_ms": self.max_time_per_problem_ms,
            "use_caching": self.use_caching,
            "verification_depth": self.verification_depth,
            "generate_baselines": self.generate_baselines,
            "baseline_types": self.baseline_types,
            "enable_logging": self.enable_logging,
            "output_dir": self.output_dir,
            "metadata": self.metadata,
        }


class BenchmarkRunner:
    """
    Main orchestrator for benchmark execution.

    This class coordinates:
    - Problem selection and filtering
    - Verification execution
    - Scoring and metrics collection
    - Baseline generation
    - Results reporting
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize benchmark runner.

        Args:
            config: Optional benchmark configuration
        """
        self.config = config or BenchmarkConfig()

        # Initialize components
        self.scoring_system = ScoringSystem(config={})
        self.metrics_tracker = MetricsTracker(config={})
        self.baseline_generator = BaselineGenerator(config={})

        # Setup logging
        if self.config.enable_logging:
            self._setup_logging()

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        logger.info("Benchmark runner initialized")

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )

    def get_problems(self) -> List[BenchmarkProblem]:
        """
        Get problems based on configuration.

        Returns:
            List of filtered problems
        """
        all_problems = get_all_problems()

        # Filter by domain if specified
        if not self.config.run_all_domains and self.config.domains:
            from benchmarks.problems import ProblemDomain
            domain_map = {d.value: d for d in ProblemDomain}
            filtered_problems = []
            for domain_str in self.config.domains:
                if domain_str in domain_map:
                    filtered_problems.extend(get_problems_by_domain(domain_map[domain_str]))
            all_problems = filtered_problems

        # Filter by complexity if specified
        if self.config.complexities:
            from benchmarks.problems import ComplexityLevel
            complexity_map = {c.value: c for c in ComplexityLevel}
            filtered_problems = []
            for complexity_str in self.config.complexities:
                if complexity_str in complexity_map:
                    filtered_problems.extend(get_problems_by_complexity(complexity_map[complexity_str]))
            # Intersect with current problems
            all_problems = [p for p in all_problems if p in filtered_problems]

        return all_problems

    def run_benchmarks(
        self,
        problems: Optional[List[BenchmarkProblem]] = None
    ) -> Dict[str, Any]:
        """
        Run benchmarks on problems.

        Args:
            problems: Optional list of problems (uses config if not provided)

        Returns:
            Dictionary with benchmark results
        """
        if problems is None:
            problems = self.get_problems()

        logger.info(f"Running benchmarks on {len(problems)} problems")

        # Start metrics tracking
        self.metrics_tracker.start_tracking()

        # Execute each problem
        for i, problem in enumerate(problems, 1):
            logger.info(f"Processing problem {i}/{len(problems)}: {problem.id}")

            try:
                result = self._execute_problem(problem)
                self.metrics_tracker.add_result(result)

                if result.success:
                    logger.info(f"  ✓ Success (score: {result.score.overall:.3f}, time: {result.execution_time_ms:.2f}ms)")
                else:
                    logger.warning(f"  ✗ Failed: {result.error}")

            except Exception as e:
                logger.error(f"  ✗ Error: {e}")
                self.metrics_tracker.add_result(BenchmarkResult(
                    problem_id=problem.id,
                    domain=problem.domain.value,
                    complexity=problem.complexity.value,
                    success=False,
                    execution_time_ms=0.0,
                    result=None,
                    error=str(e),
                ))

        # Stop tracking
        self.metrics_tracker.stop_tracking()

        # Compute metrics
        metrics = self.metrics_tracker.compute_metrics()

        logger.info(f"Benchmarks completed: {metrics.success_rate:.1%} success rate")

        return {
            "metrics": metrics,
            "results": self.metrics_tracker.results,
        }

    def _execute_problem(self, problem: BenchmarkProblem) -> BenchmarkResult:
        """
        Execute a single problem.

        Args:
            problem: Benchmark problem to execute

        Returns:
            BenchmarkResult
        """
        start_time = time.time()

        try:
            # Execute verification
            result = self._verify_statement(problem)

            execution_time_ms = (time.time() - start_time) * 1000

            # Score the result
            score = self.scoring_system.score_result(
                problem=problem,
                result=result,
                execution_time_ms=execution_time_ms,
            )

            return BenchmarkResult(
                problem_id=problem.id,
                domain=problem.domain.value,
                complexity=problem.complexity.value,
                success=True,
                execution_time_ms=execution_time_ms,
                result=result,
                score=score,
                metadata={"verification_depth": self.config.verification_depth},
            )

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Error executing problem {problem.id}: {e}")

            return BenchmarkResult(
                problem_id=problem.id,
                domain=problem.domain.value,
                complexity=problem.complexity.value,
                success=False,
                execution_time_ms=execution_time_ms,
                result=None,
                error=str(e),
            )

    def _verify_statement(self, problem: BenchmarkProblem) -> Dict[str, Any]:
        """
        Verify a statement using TMR.

        Args:
            problem: Benchmark problem

        Returns:
            Verification result
        """
        try:
            from execution.synthesizer import ExecutionSynthesizer

            synthesizer = ExecutionSynthesizer(config={
                "fundamentals": {
                    "use_cache": self.config.use_caching,
                },
                "depth_selector": {
                    "default_depth": self.config.verification_depth,
                },
            })

            result = synthesizer.synthesize(
                input_data=problem.input_statement,
                output_format="STANDARD",
            )

            return result

        except Exception as e:
            logger.error(f"Error in verification: {e}")
            return {
                "valid": False,
                "confidence": 0.0,
                "error": str(e),
            }

    def run_with_baselines(
        self,
        problems: Optional[List[BenchmarkProblem]] = None
    ) -> Dict[str, Any]:
        """
        Run benchmarks and generate baselines.

        Args:
            problems: Optional list of problems

        Returns:
            Dictionary with benchmark and baseline results
        """
        if problems is None:
            problems = self.get_problems()

        logger.info("Running benchmarks with baseline generation")

        # Run main benchmarks
        main_results = self.run_benchmarks(problems)

        # Generate baselines if configured
        baselines = {}
        if self.config.generate_baselines:
            logger.info("Generating baselines...")

            for baseline_type_str in self.config.baseline_types:
                try:
                    # Convert string to enum
                    baseline_type = BaselineType(baseline_type_str)
                    logger.info(f"  Generating baseline: {baseline_type_str}")

                    metrics = self.baseline_generator.generate_baseline(
                        problems=problems,
                        baseline_type=baseline_type,
                    )

                    baselines[baseline_type_str] = metrics

                    logger.info(f"  ✓ {baseline_type_str}: {metrics.success_rate:.1%} success rate")

                except Exception as e:
                    logger.error(f"  ✗ Error generating baseline {baseline_type_str}: {e}")

        return {
            "main": main_results,
            "baselines": baselines,
        }

    def generate_report(
        self,
        results: Dict[str, Any],
        output_format: str = "text"
    ) -> str:
        """
        Generate a report from results.

        Args:
            results: Results dictionary from run_benchmarks or run_with_baselines
            output_format: Output format (text, json, html)

        Returns:
            Report string
        """
        if output_format == "text":
            return self._generate_text_report(results)
        elif output_format == "json":
            return self._generate_json_report(results)
        elif output_format == "html":
            return self._generate_html_report(results)
        else:
            raise ValueError(f"Unknown output format: {output_format}")

    def _generate_text_report(self, results: Dict[str, Any]) -> str:
        """Generate text report"""
        lines = []
        lines.append("=" * 80)
        lines.append("TMR BENCHMARK VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Main results
        if "main" in results:
            main_data = results["main"]
            metrics = main_data["metrics"]

            lines.append("MAIN BENCHMARK RESULTS")
            lines.append("-" * 80)
            lines.append(f"Total Problems:    {metrics.total_problems}")
            lines.append(f"Successful:        {metrics.successful} ({metrics.success_rate:.1%})")
            lines.append(f"Failed:            {metrics.failed}")
            lines.append("")
            lines.append("Timing Statistics:")
            lines.append(f"  Total Time:      {metrics.total_time_ms:.2f} ms")
            lines.append(f"  Average Time:    {metrics.avg_time_ms:.2f} ms")
            lines.append(f"  Median Time:     {metrics.median_time_ms:.2f} ms")
            lines.append(f"  Min Time:        {metrics.min_time_ms:.2f} ms")
            lines.append(f"  Max Time:        {metrics.max_time_ms:.2f} ms")
            lines.append("")

            # Domain breakdown
            if metrics.domain_metrics:
                lines.append("Domain Breakdown:")
                for domain, dm in metrics.domain_metrics.items():
                    lines.append(f"  {domain}:")
                    lines.append(f"    Total:        {dm['total']}")
                    lines.append(f"    Success Rate: {dm['success_rate']:.1%}")
                    lines.append(f"    Avg Time:     {dm['avg_time_ms']:.2f} ms")
                    if dm.get("avg_overall_score") is not None:
                        lines.append(f"    Avg Score:    {dm['avg_overall_score']:.3f}")
                lines.append("")

            # Complexity breakdown
            if metrics.complexity_metrics:
                lines.append("Complexity Breakdown:")
                for complexity, cm in metrics.complexity_metrics.items():
                    lines.append(f"  {complexity}:")
                    lines.append(f"    Total:        {cm['total']}")
                    lines.append(f"    Success Rate: {cm['success_rate']:.1%}")
                    lines.append(f"    Avg Time:     {cm['avg_time_ms']:.2f} ms")
                    if cm.get("avg_overall_score") is not None:
                        lines.append(f"    Avg Score:    {cm['avg_overall_score']:.3f}")
                lines.append("")

        # Baseline comparisons
        if "baselines" in results and results["baselines"]:
            lines.append("BASELINE COMPARISONS")
            lines.append("-" * 80)

            for baseline_name, baseline_metrics in results["baselines"].items():
                lines.append(f"{baseline_name}:")
                lines.append(f"  Success Rate: {baseline_metrics.success_rate:.1%}")
                lines.append(f"  Avg Time:     {baseline_metrics.avg_time_ms:.2f} ms")
                lines.append("")

            # Compare with main if available
            if "main" in results:
                main_metrics = results["main"]["metrics"]
                lines.append("Comparison with Main:")
                for baseline_name, baseline_metrics in results["baselines"].items():
                    success_rate_diff = main_metrics.success_rate - baseline_metrics.success_rate
                    time_diff = baseline_metrics.avg_time_ms - main_metrics.avg_time_ms
                    lines.append(f"  vs {baseline_name}:")
                    lines.append(f"    Success Rate: {success_rate_diff:+.1%} difference")
                    lines.append(f"    Avg Time:     {time_diff:+.2f} ms difference")
                lines.append("")

        lines.append("=" * 80)

        return "\n".join(lines)

    def _generate_json_report(self, results: Dict[str, Any]) -> str:
        """Generate JSON report"""
        import json

        report_data = {}

        if "main" in results:
            report_data["main"] = {
                "metrics": results["main"]["metrics"].to_dict(),
            }

        if "baselines" in results:
            report_data["baselines"] = {
                name: metrics.to_dict()
                for name, metrics in results["baselines"].items()
            }

        return json.dumps(report_data, indent=2)

    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report"""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("<title>TMR Benchmark Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("h1 { color: #333; }")
        html.append("h2 { color: #666; border-bottom: 2px solid #ccc; }")
        html.append("table { border-collapse: collapse; width: 100%; margin: 20px 0; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #4CAF50; color: white; }")
        html.append("tr:nth-child(even) { background-color: #f2f2f2; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        html.append("<h1>TMR Benchmark Validation Report</h1>")

        if "main" in results:
            metrics = results["main"]["metrics"]
            html.append("<h2>Main Results</h2>")
            html.append("<table>")
            html.append("<tr><th>Metric</th><th>Value</th></tr>")
            html.append(f"<tr><td>Total Problems</td><td>{metrics.total_problems}</td></tr>")
            html.append(f"<tr><td>Success Rate</td><td>{metrics.success_rate:.1%}</td></tr>")
            html.append(f"<tr><td>Average Time</td><td>{metrics.avg_time_ms:.2f} ms</td></tr>")
            html.append("</table>")

        if "baselines" in results and results["baselines"]:
            html.append("<h2>Baseline Comparisons</h2>")
            html.append("<table>")
            html.append("<tr><th>Baseline</th><th>Success Rate</th><th>Avg Time (ms)</th></tr>")
            for baseline_name, baseline_metrics in results["baselines"].items():
                html.append(f"<tr><td>{baseline_name}</td><td>{baseline_metrics.success_rate:.1%}</td><td>{baseline_metrics.avg_time_ms:.2f}</td></tr>")
            html.append("</table>")

        html.append("</body>")
        html.append("</html>")

        return "\n".join(html)

    def save_report(
        self,
        results: Dict[str, Any],
        filename: Optional[str] = None,
        output_format: str = "text"
    ) -> str:
        """
        Save report to file.

        Args:
            results: Results dictionary
            filename: Output filename (auto-generated if None)
            output_format: Output format (text, json, html)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            ext = {"text": "txt", "json": "json", "html": "html"}[output_format]
            filename = f"benchmark_report_{timestamp}.{ext}"

        filepath = os.path.join(self.config.output_dir, filename)

        report = self.generate_report(results, output_format)

        with open(filepath, 'w') as f:
            f.write(report)

        logger.info(f"Report saved to {filepath}")

        return filepath

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Run all benchmarks with baselines and generate report.

        Returns:
            Complete results dictionary
        """
        logger.info("Starting complete benchmark run")

        # Run benchmarks with baselines
        results = self.run_with_baselines()

        # Save results
        self.save_report(results, output_format="text")
        self.save_report(results, output_format="json")
        self.save_report(results, output_format="html")

        # Print summary
        print(self.generate_report(results, output_format="text"))

        logger.info("Complete benchmark run finished")

        return results
