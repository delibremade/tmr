#!/usr/bin/env python3
"""
TMR Benchmark Suite - Quick Example

This script demonstrates how to use the benchmark suite programmatically.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks import (
    get_problems_by_domain,
    get_problems_by_complexity,
    ProblemDomain,
    ComplexityLevel,
    BenchmarkRunner,
    BenchmarkConfig,
    ScoringSystem,
)


def example_1_run_simple_benchmark():
    """Example 1: Run a simple benchmark on math problems only"""
    print("=" * 70)
    print("EXAMPLE 1: Run Math Domain Benchmarks")
    print("=" * 70)
    print()

    # Get math problems
    math_problems = get_problems_by_domain(ProblemDomain.MATH)
    print(f"Found {len(math_problems)} math problems")
    print()

    # Create configuration
    config = BenchmarkConfig(
        run_all_domains=False,
        domains=["math"],
        generate_baselines=False,  # Skip for speed
        use_caching=True,
        verification_depth="STANDARD",
    )

    # Create runner
    runner = BenchmarkRunner(config=config)

    # Run benchmarks
    print("Running benchmarks...")
    results = runner.run_benchmarks(problems=math_problems[:3])  # Just first 3 for demo

    # Show summary
    print()
    print("Results:")
    metrics = results["metrics"]
    print(f"  Total: {metrics.total_problems}")
    print(f"  Success Rate: {metrics.success_rate:.1%}")
    print(f"  Avg Time: {metrics.avg_time_ms:.2f} ms")
    print()


def example_2_score_custom_result():
    """Example 2: Score a custom verification result"""
    print("=" * 70)
    print("EXAMPLE 2: Score a Custom Result")
    print("=" * 70)
    print()

    # Create a mock problem
    from benchmarks import BenchmarkProblem

    problem = BenchmarkProblem(
        id="DEMO-001",
        domain=ProblemDomain.MATH,
        complexity=ComplexityLevel.SIMPLE,
        title="Demo Problem",
        description="Example for scoring",
        input_statement="2 + 2 = 4",
        expected_valid=True,
        expected_confidence=0.95,
        ground_truth="Basic arithmetic",
    )

    # Mock verification result
    result = {
        "valid": True,
        "confidence": 0.93,
        "details": "Verified successfully"
    }

    # Score the result
    scoring = ScoringSystem()
    score = scoring.score_result(
        problem=problem,
        result=result,
        execution_time_ms=125.5,
    )

    print(f"Problem: {problem.title}")
    print(f"Expected: valid={problem.expected_valid}, confidence={problem.expected_confidence}")
    print(f"Actual: valid={result['valid']}, confidence={result['confidence']}")
    print()
    print("Scores:")
    print(f"  Correctness: {score.correctness:.3f}")
    print(f"  Confidence Accuracy: {score.confidence_accuracy:.3f}")
    print(f"  Efficiency: {score.efficiency:.3f}")
    print(f"  Overall: {score.overall:.3f}")
    print()


def example_3_filter_by_complexity():
    """Example 3: Work with problems by complexity"""
    print("=" * 70)
    print("EXAMPLE 3: Filter Problems by Complexity")
    print("=" * 70)
    print()

    # Get simple problems
    simple_problems = get_problems_by_complexity(ComplexityLevel.SIMPLE)

    print(f"Found {len(simple_problems)} SIMPLE problems:")
    for problem in simple_problems[:5]:  # Show first 5
        print(f"  - {problem.id}: {problem.title} [{problem.domain.value}]")
    print()

    # Get advanced problems
    advanced_problems = get_problems_by_complexity(ComplexityLevel.ADVANCED)

    print(f"Found {len(advanced_problems)} ADVANCED problems:")
    for problem in advanced_problems[:5]:  # Show first 5
        print(f"  - {problem.id}: {problem.title} [{problem.domain.value}]")
    print()


def example_4_custom_scoring_weights():
    """Example 4: Use custom scoring weights"""
    print("=" * 70)
    print("EXAMPLE 4: Custom Scoring Weights")
    print("=" * 70)
    print()

    # Create scoring system with custom weights
    scoring = ScoringSystem(config={
        "weights": {
            "correctness": 0.50,    # Increase correctness weight
            "confidence": 0.20,
            "efficiency": 0.15,
            "consistency": 0.10,
            "robustness": 0.05,
        }
    })

    print("Custom weights:")
    print("  Correctness: 50%")
    print("  Confidence: 20%")
    print("  Efficiency: 15%")
    print("  Consistency: 10%")
    print("  Robustness: 5%")
    print()

    # These weights prioritize correctness over other factors
    print("This configuration prioritizes getting the right answer above all else.")
    print()


def example_5_problem_statistics():
    """Example 5: Analyze problem set statistics"""
    print("=" * 70)
    print("EXAMPLE 5: Problem Set Statistics")
    print("=" * 70)
    print()

    from benchmarks import get_all_problem_sets

    problem_sets = get_all_problem_sets()

    for problem_set in problem_sets:
        print(f"{problem_set.name} ({problem_set.domain.value}):")
        stats = problem_set.get_statistics()
        print(f"  Total Problems: {stats['total_problems']}")
        print(f"  Expected Valid Rate: {stats['expected_valid_rate']:.1%}")
        print(f"  Avg Expected Confidence: {stats['avg_expected_confidence']:.2f}")

        print("  By Complexity:")
        for complexity, count in stats['by_complexity'].items():
            if count > 0:
                print(f"    {complexity}: {count}")
        print()


def main():
    """Run all examples"""
    import sys

    # Check if running non-interactively
    non_interactive = "--non-interactive" in sys.argv or not sys.stdin.isatty()

    print()
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "TMR BENCHMARK SUITE EXAMPLES" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    examples = [
        example_5_problem_statistics,
        example_3_filter_by_complexity,
        example_2_score_custom_result,
        example_4_custom_scoring_weights,
    ]

    for i, example in enumerate(examples, 1):
        example()
        if i < len(examples):
            if not non_interactive:
                try:
                    input("Press Enter to continue to next example...")
                except (EOFError, KeyboardInterrupt):
                    print("\n\nSkipping to end...")
                    break
            print("\n" * 2)

    print("=" * 70)
    print("Examples completed!")
    print()
    print("To run full benchmarks, use:")
    print("  python run_benchmarks.py")
    print()
    print("For more examples, see benchmarks/README.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
