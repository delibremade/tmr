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

from .problems import BenchmarkProblem, ProblemSet, get_all_problem_sets
from .scoring import Score, ScoringSystem
from .metrics import PerformanceMetrics, MetricsTracker
from .baselines import BaselineGenerator, BaselineType
from .runner import BenchmarkRunner, BenchmarkConfig
from .reporting import ReportGenerator, ReportFormat

__all__ = [
    "BenchmarkProblem",
    "ProblemSet",
    "get_all_problem_sets",
    "Score",
    "ScoringSystem",
    "PerformanceMetrics",
    "MetricsTracker",
    "BaselineGenerator",
    "BaselineType",
    "BenchmarkRunner",
    "BenchmarkConfig",
    "ReportGenerator",
    "ReportFormat",
]
