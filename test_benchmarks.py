"""
Tests for TMR Benchmark Validation Suite

This module tests the benchmark framework components including problems,
scoring, metrics, baselines, and reporting.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.problems import (
    BenchmarkProblem,
    ProblemSet,
    ComplexityLevel,
    ProblemDomain,
    get_all_problem_sets,
    get_all_problems,
    get_problems_by_domain,
    get_problems_by_complexity,
    get_benchmark_statistics,
)
from benchmarks.scoring import Score, ScoringSystem, ScoreComponent
from benchmarks.metrics import (
    PerformanceMetrics,
    MetricsTracker,
    BenchmarkResult,
)
from benchmarks.baselines import BaselineGenerator, BaselineType, BaselineConfig
from benchmarks.runner import BenchmarkRunner, BenchmarkConfig
from benchmarks.reporting import ReportGenerator, ReportFormat, ReportSection, ChartType


# ============================================================================
# Test Problems Module
# ============================================================================

class TestBenchmarkProblem:
    """Test BenchmarkProblem class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.problem = BenchmarkProblem(
            id="TEST-001",
            domain=ProblemDomain.MATH,
            complexity=ComplexityLevel.SIMPLE,
            title="Test Problem",
            description="Test description",
            input_statement="2 + 2 = 4",
            expected_valid=True,
            expected_confidence=0.95,
            ground_truth="Basic arithmetic",
        )

    def test_problem_creation(self):
        """Test problem can be created"""
        assert self.problem.id == "TEST-001"
        assert self.problem.domain == ProblemDomain.MATH
        assert self.problem.complexity == ComplexityLevel.SIMPLE
        assert self.problem.expected_valid is True

    def test_validate_result_default(self):
        """Test default result validation"""
        result = {"valid": True}
        assert self.problem.validate_result(result) is True

        result = {"valid": False}
        assert self.problem.validate_result(result) is False

    def test_validate_result_custom(self):
        """Test custom validator"""
        def custom_validator(result):
            return result.get("custom_field") == "expected"

        problem = BenchmarkProblem(
            id="TEST-002",
            domain=ProblemDomain.CODE,
            complexity=ComplexityLevel.MODERATE,
            title="Custom Validator",
            description="Test",
            input_statement="test",
            expected_valid=True,
            expected_confidence=0.8,
            ground_truth="Test",
            validator=custom_validator,
        )

        assert problem.validate_result({"custom_field": "expected"}) is True
        assert problem.validate_result({"custom_field": "other"}) is False


class TestProblemSet:
    """Test ProblemSet class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.problem_set = ProblemSet(
            name="Test Set",
            domain=ProblemDomain.MATH,
            description="Test problem set"
        )

    def test_add_problem(self):
        """Test adding problems to set"""
        problem = BenchmarkProblem(
            id="TEST-001",
            domain=ProblemDomain.MATH,
            complexity=ComplexityLevel.SIMPLE,
            title="Test",
            description="Test",
            input_statement="test",
            expected_valid=True,
            expected_confidence=0.9,
            ground_truth="Test",
        )

        self.problem_set.add_problem(problem)
        assert len(self.problem_set.problems) == 1

    def test_get_problems_by_complexity(self):
        """Test filtering by complexity"""
        for i, complexity in enumerate([ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE, ComplexityLevel.SIMPLE]):
            problem = BenchmarkProblem(
                id=f"TEST-{i:03d}",
                domain=ProblemDomain.MATH,
                complexity=complexity,
                title=f"Test {i}",
                description="Test",
                input_statement="test",
                expected_valid=True,
                expected_confidence=0.9,
                ground_truth="Test",
            )
            self.problem_set.add_problem(problem)

        simple = self.problem_set.get_problems_by_complexity(ComplexityLevel.SIMPLE)
        assert len(simple) == 2

        moderate = self.problem_set.get_problems_by_complexity(ComplexityLevel.MODERATE)
        assert len(moderate) == 1

    def test_get_statistics(self):
        """Test statistics generation"""
        problem = BenchmarkProblem(
            id="TEST-001",
            domain=ProblemDomain.MATH,
            complexity=ComplexityLevel.SIMPLE,
            title="Test",
            description="Test",
            input_statement="test",
            expected_valid=True,
            expected_confidence=0.9,
            ground_truth="Test",
        )
        self.problem_set.add_problem(problem)

        stats = self.problem_set.get_statistics()
        assert stats["total_problems"] == 1
        assert stats["expected_valid_rate"] == 1.0
        assert stats["avg_expected_confidence"] == 0.9


class TestProblemGeneration:
    """Test problem generation functions"""

    def test_get_all_problem_sets(self):
        """Test getting all problem sets"""
        problem_sets = get_all_problem_sets()
        assert len(problem_sets) == 4  # MATH, CODE, LOGIC, MIXED
        assert all(isinstance(ps, ProblemSet) for ps in problem_sets)

    def test_get_all_problems(self):
        """Test getting all problems"""
        problems = get_all_problems()
        assert len(problems) == 33  # 10 + 10 + 10 + 3
        assert all(isinstance(p, BenchmarkProblem) for p in problems)

    def test_get_problems_by_domain(self):
        """Test filtering by domain"""
        math_problems = get_problems_by_domain(ProblemDomain.MATH)
        assert len(math_problems) == 10
        assert all(p.domain == ProblemDomain.MATH for p in math_problems)

    def test_get_problems_by_complexity(self):
        """Test filtering by complexity"""
        simple_problems = get_problems_by_complexity(ComplexityLevel.SIMPLE)
        assert all(p.complexity == ComplexityLevel.SIMPLE for p in simple_problems)

    def test_get_benchmark_statistics(self):
        """Test benchmark statistics"""
        stats = get_benchmark_statistics()
        assert stats["total_problems"] == 33
        assert stats["problem_sets"] == 4
        assert "by_domain" in stats
        assert "by_complexity" in stats


# ============================================================================
# Test Scoring Module
# ============================================================================

class TestScore:
    """Test Score class"""

    def test_score_creation(self):
        """Test score creation"""
        score = Score(
            correctness=1.0,
            confidence_accuracy=0.9,
            efficiency=0.8,
        )
        assert score.correctness == 1.0
        assert score.overall > 0.0

    def test_score_calculation(self):
        """Test overall score calculation"""
        score = Score(
            correctness=1.0,
            confidence_accuracy=1.0,
            efficiency=1.0,
            consistency=1.0,
            robustness=1.0,
        )
        # Perfect score should be 1.0
        assert score.overall == 1.0

    def test_score_to_dict(self):
        """Test score serialization"""
        score = Score(
            correctness=1.0,
            confidence_accuracy=0.9,
            efficiency=0.8,
        )
        data = score.to_dict()
        assert "correctness" in data
        assert "overall" in data

    def test_score_from_dict(self):
        """Test score deserialization"""
        data = {
            "correctness": 1.0,
            "confidence_accuracy": 0.9,
            "efficiency": 0.8,
        }
        score = Score.from_dict(data)
        assert score.correctness == 1.0


class TestScoringSystem:
    """Test ScoringSystem class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.scoring = ScoringSystem()
        self.problem = BenchmarkProblem(
            id="TEST-001",
            domain=ProblemDomain.MATH,
            complexity=ComplexityLevel.SIMPLE,
            title="Test",
            description="Test",
            input_statement="2 + 2 = 4",
            expected_valid=True,
            expected_confidence=0.95,
            ground_truth="Test",
        )

    def test_score_result_correct(self):
        """Test scoring a correct result"""
        result = {"valid": True, "confidence": 0.95}
        score = self.scoring.score_result(
            problem=self.problem,
            result=result,
            execution_time_ms=100.0,
        )
        assert score.correctness == 1.0

    def test_score_result_incorrect(self):
        """Test scoring an incorrect result"""
        result = {"valid": False, "confidence": 0.5}
        score = self.scoring.score_result(
            problem=self.problem,
            result=result,
            execution_time_ms=100.0,
        )
        assert score.correctness == 0.0

    def test_score_efficiency(self):
        """Test efficiency scoring"""
        # Fast execution should get high score
        result = {"valid": True, "confidence": 0.95}
        score_fast = self.scoring.score_result(
            problem=self.problem,
            result=result,
            execution_time_ms=50.0,
        )

        # Slow execution should get lower score
        score_slow = self.scoring.score_result(
            problem=self.problem,
            result=result,
            execution_time_ms=5000.0,
        )

        assert score_fast.efficiency > score_slow.efficiency

    def test_aggregate_scores(self):
        """Test score aggregation"""
        scores = [
            Score(correctness=1.0, confidence_accuracy=0.9, efficiency=0.8),
            Score(correctness=0.8, confidence_accuracy=0.85, efficiency=0.9),
            Score(correctness=0.9, confidence_accuracy=0.95, efficiency=0.85),
        ]

        aggregates = self.scoring.aggregate_scores(scores)
        assert aggregates["count"] == 3
        assert "mean" in aggregates
        assert "median" in aggregates


# ============================================================================
# Test Metrics Module
# ============================================================================

class TestMetricsTracker:
    """Test MetricsTracker class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.tracker = MetricsTracker()

    def test_start_stop_tracking(self):
        """Test tracking session"""
        self.tracker.start_tracking()
        assert self.tracker.start_time is not None

        self.tracker.stop_tracking()
        assert self.tracker.end_time is not None

    def test_add_result(self):
        """Test adding results"""
        result = BenchmarkResult(
            problem_id="TEST-001",
            domain="math",
            complexity="simple",
            success=True,
            execution_time_ms=100.0,
            result={"valid": True},
        )

        self.tracker.add_result(result)
        assert len(self.tracker.results) == 1

    def test_compute_metrics(self):
        """Test metrics computation"""
        self.tracker.start_tracking()

        # Add successful result
        self.tracker.add_result(BenchmarkResult(
            problem_id="TEST-001",
            domain="math",
            complexity="simple",
            success=True,
            execution_time_ms=100.0,
            result={"valid": True},
        ))

        # Add failed result
        self.tracker.add_result(BenchmarkResult(
            problem_id="TEST-002",
            domain="math",
            complexity="simple",
            success=False,
            execution_time_ms=50.0,
            result=None,
            error="Test error",
        ))

        self.tracker.stop_tracking()

        metrics = self.tracker.compute_metrics()
        assert metrics.total_problems == 2
        assert metrics.successful == 1
        assert metrics.failed == 1
        assert metrics.success_rate == 0.5

    def test_get_failed_problems(self):
        """Test getting failed problems"""
        self.tracker.add_result(BenchmarkResult(
            problem_id="TEST-001",
            domain="math",
            complexity="simple",
            success=True,
            execution_time_ms=100.0,
            result={"valid": True},
        ))

        self.tracker.add_result(BenchmarkResult(
            problem_id="TEST-002",
            domain="math",
            complexity="simple",
            success=False,
            execution_time_ms=50.0,
            result=None,
            error="Error",
        ))

        failed = self.tracker.get_failed_problems()
        assert len(failed) == 1
        assert failed[0].problem_id == "TEST-002"


# ============================================================================
# Test Reporting Module
# ============================================================================

class TestReportGenerator:
    """Test ReportGenerator class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.generator = ReportGenerator()

    def test_add_section(self):
        """Test adding sections"""
        section = ReportSection(
            title="Test Section",
            content="Test content"
        )
        self.generator.add_section(section)
        assert len(self.generator.sections) == 1

    def test_clear_sections(self):
        """Test clearing sections"""
        section = ReportSection(
            title="Test",
            content="Test"
        )
        self.generator.add_section(section)
        self.generator.clear_sections()
        assert len(self.generator.sections) == 0

    def test_format_as_text(self):
        """Test text formatting"""
        section = ReportSection(
            title="Test Section",
            content="Test content"
        )
        self.generator.add_section(section)
        text = self.generator._format_as_text()
        assert "TEST SECTION" in text  # Title is uppercased in text format
        assert "Test content" in text

    def test_format_as_json(self):
        """Test JSON formatting"""
        section = ReportSection(
            title="Test Section",
            content="Test content"
        )
        self.generator.add_section(section)
        json_str = self.generator._format_as_json()
        assert "Test Section" in json_str

    def test_format_as_html(self):
        """Test HTML formatting"""
        section = ReportSection(
            title="Test Section",
            content="Test content"
        )
        self.generator.add_section(section)
        html = self.generator._format_as_html()
        assert "<!DOCTYPE html>" in html
        assert "Test Section" in html


# ============================================================================
# Test Runner Module
# ============================================================================

class TestBenchmarkConfig:
    """Test BenchmarkConfig class"""

    def test_default_config(self):
        """Test default configuration"""
        config = BenchmarkConfig()
        assert config.run_all_domains is True
        assert config.use_caching is True
        assert config.generate_baselines is True

    def test_custom_config(self):
        """Test custom configuration"""
        config = BenchmarkConfig(
            domains=["math"],
            complexities=["simple"],
            use_caching=False,
        )
        assert config.domains == ["math"]
        assert config.use_caching is False

    def test_config_to_dict(self):
        """Test configuration serialization"""
        config = BenchmarkConfig()
        data = config.to_dict()
        assert "run_all_domains" in data
        assert "verification_depth" in data


# ============================================================================
# Main Test Runner
# ============================================================================

def run_tests():
    """Run all tests"""
    import traceback

    test_classes = [
        TestBenchmarkProblem,
        TestProblemSet,
        TestProblemGeneration,
        TestScore,
        TestScoringSystem,
        TestMetricsTracker,
        TestReportGenerator,
        TestBenchmarkConfig,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    print("=" * 70)
    print("TMR BENCHMARK SUITE TESTS")
    print("=" * 70)
    print()

    for test_class in test_classes:
        print(f"Running {test_class.__name__}...")

        # Get test methods
        test_methods = [
            method for method in dir(test_class)
            if method.startswith("test_")
        ]

        for method_name in test_methods:
            total_tests += 1

            try:
                # Create instance and run setup
                instance = test_class()
                if hasattr(instance, "setup_method"):
                    instance.setup_method()

                # Run test
                method = getattr(instance, method_name)
                method()

                print(f"  ✓ {method_name}")
                passed_tests += 1

            except AssertionError as e:
                print(f"  ✗ {method_name}")
                print(f"    AssertionError: {e}")
                failed_tests += 1

            except Exception as e:
                print(f"  ✗ {method_name}")
                print(f"    Error: {e}")
                traceback.print_exc()
                failed_tests += 1

        print()

    print("=" * 70)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    if failed_tests > 0:
        print(f"FAILED: {failed_tests} tests failed")
    print("=" * 70)

    return failed_tests == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
