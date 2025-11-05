"""
TMR Benchmark Validation Suite

This module provides a comprehensive benchmarking framework for validating
the Trinity Meta-Reasoning (TMR) framework across multiple domains and
complexity levels.

Components:
    - problems: Benchmark problem definitions across MATH, CODE, LOGIC domains
    - scoring: Unified scoring system for evaluation
    - metrics: Performance metrics tracking and analysis
    - baselines: Baseline generation for comparison
    - runner: Benchmark orchestration and execution
    - reporting: Results visualization and reporting

Usage:
    from benchmarks import BenchmarkRunner

    runner = BenchmarkRunner()
    results = runner.run_all_benchmarks()
    runner.generate_report(results)
"""

__version__ = "0.1.0"
__author__ = "TMR Development Team"

from .problems import (
    BenchmarkProblem,
    ProblemSet,
    ProblemDomain,
    ComplexityLevel,
    get_all_problem_sets,
    get_all_problems,
    get_problems_by_domain,
    get_problems_by_complexity,
    get_benchmark_statistics,
)
from .scoring import Score, ScoringSystem, ScoreComponent
from .metrics import PerformanceMetrics, MetricsTracker, BenchmarkResult
from .baselines import BaselineGenerator, BaselineType, BaselineConfig
from .runner import BenchmarkRunner, BenchmarkConfig
from .reporting import ReportGenerator, ReportFormat, ReportSection, ChartType

__all__ = [
    # Problems
    "BenchmarkProblem",
    "ProblemSet",
    "ProblemDomain",
    "ComplexityLevel",
    "get_all_problem_sets",
    "get_all_problems",
    "get_problems_by_domain",
    "get_problems_by_complexity",
    "get_benchmark_statistics",
    # Scoring
    "Score",
    "ScoringSystem",
    "ScoreComponent",
    # Metrics
    "PerformanceMetrics",
    "MetricsTracker",
    "BenchmarkResult",
    # Baselines
    "BaselineGenerator",
    "BaselineType",
    "BaselineConfig",
    # Runner
    "BenchmarkRunner",
    "BenchmarkConfig",
    # Reporting
    "ReportGenerator",
    "ReportFormat",
    "ReportSection",
    "ChartType",
]
