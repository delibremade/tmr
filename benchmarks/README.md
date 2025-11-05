# TMR Benchmark Validation Suite

A comprehensive benchmarking framework for validating the Trinity Meta-Reasoning (TMR) framework across multiple domains and complexity levels.

## Overview

The TMR Benchmark Validation Suite provides:

- **Problem Datasets**: 33 benchmark problems across MATH, CODE, LOGIC, and MIXED domains
- **Scoring System**: Multi-dimensional scoring (correctness, confidence, efficiency, consistency, robustness)
- **Performance Metrics**: Detailed tracking of success rates, timing, and domain-specific performance
- **Baseline Generation**: Multiple baseline configurations for comparison
- **Comprehensive Reporting**: Text, JSON, HTML, Markdown, and CSV output formats
- **CLI Interface**: Easy-to-use command-line runner

## Quick Start

### Running All Benchmarks

```bash
python run_benchmarks.py
```

This will:
1. Run all 33 benchmark problems
2. Generate baseline comparisons
3. Create reports in multiple formats
4. Save results to `./benchmark_results/`

### Running Specific Domains

```bash
# Run only math domain
python run_benchmarks.py --domain math

# Run multiple domains
python run_benchmarks.py --domain math --domain code
```

### Running Specific Complexity Levels

```bash
# Run only simple problems
python run_benchmarks.py --complexity simple

# Run multiple complexity levels
python run_benchmarks.py --complexity simple --complexity moderate
```

### Customizing Verification

```bash
# Use minimal verification depth (faster)
python run_benchmarks.py --depth MINIMAL

# Use exhaustive verification depth (more thorough)
python run_benchmarks.py --depth EXHAUSTIVE

# Disable caching
python run_benchmarks.py --no-cache
```

### Baseline Comparisons

```bash
# Skip baseline generation (faster)
python run_benchmarks.py --no-baselines

# Generate specific baselines only
python run_benchmarks.py --baseline fundamentals_only --baseline full_tmr
```

### View Statistics

```bash
python run_benchmarks.py --stats
```

## Architecture

```
benchmarks/
├── __init__.py          # Package initialization
├── problems.py          # Benchmark problem definitions (33 problems)
├── scoring.py           # Scoring system
├── metrics.py           # Performance metrics tracking
├── baselines.py         # Baseline generation
├── runner.py            # Benchmark orchestration
└── reporting.py         # Visualization and reporting
```

## Problem Domains

### Mathematical Domain (10 problems)
- **Trivial**: Basic arithmetic (2+2=4)
- **Simple**: Algebraic identities, quadratic equations
- **Moderate**: Calculus (derivatives, integrals)
- **Complex**: Chain rule, exponential growth
- **Advanced**: Proof by induction, limits

### Code Domain (10 problems)
- **Trivial**: Variable assignment
- **Simple**: Function evaluation, list indexing
- **Moderate**: Loops, recursion
- **Complex**: Algorithm analysis (binary search, sorting)
- **Advanced**: Dynamic programming, thread safety

### Logical Domain (10 problems)
- **Trivial**: Basic logical operators (AND, NOT)
- **Simple**: Modus ponens, modus tollens
- **Moderate**: Disjunctive syllogism, hypothetical syllogism
- **Complex**: Proof by contradiction, universal instantiation
- **Advanced**: De Morgan's laws, existential generalization

### Mixed Domain (3 problems)
- **Moderate**: Mathematical code reasoning
- **Complex**: Logical code reasoning
- **Advanced**: Mathematical proof in code

## Scoring System

Each benchmark result is scored across 5 dimensions:

1. **Correctness** (40% weight): Did it produce the right answer?
2. **Confidence Accuracy** (25% weight): Was the confidence score appropriate?
3. **Efficiency** (15% weight): How fast was the verification?
4. **Consistency** (10% weight): Reproducible results?
5. **Robustness** (10% weight): How well did it handle edge cases?

**Overall Score**: Weighted combination of all components (0.0-1.0)

## Baseline Types

The suite supports multiple baseline configurations:

1. **No Verification**: Assumes all reasoning is correct (upper bound)
2. **Fundamentals Only**: Layer 1 only (logical principles)
3. **With Nuance**: Layers 1+2 (principles + patterns)
4. **Full TMR**: All 3 layers (complete system)
5. **Minimal Depth**: Minimal verification depth
6. **Standard Depth**: Standard verification depth
7. **Exhaustive Depth**: Exhaustive verification depth

## Performance Metrics

The suite tracks:

- **Success Rates**: Overall and by domain/complexity
- **Timing Statistics**: Total, average, median, min, max
- **Confidence Metrics**: Accuracy of confidence scores
- **Domain Performance**: Breakdown by MATH, CODE, LOGIC, MIXED
- **Complexity Performance**: Breakdown by difficulty level

## Report Formats

### Text Report
Human-readable plain text summary

```bash
python run_benchmarks.py --format text
```

### JSON Report
Machine-readable structured data

```bash
python run_benchmarks.py --format json
```

### HTML Report
Interactive web-based report with charts

```bash
python run_benchmarks.py --format html
```

### Markdown Report
GitHub-friendly markdown format

```bash
python run_benchmarks.py --format markdown
```

### CSV Report
Spreadsheet-compatible detailed results

```bash
python run_benchmarks.py --format csv
```

## Programmatic Usage

### Basic Usage

```python
from benchmarks import BenchmarkRunner, BenchmarkConfig

# Create configuration
config = BenchmarkConfig(
    run_all_domains=True,
    verification_depth="STANDARD",
    generate_baselines=True,
)

# Run benchmarks
runner = BenchmarkRunner(config=config)
results = runner.run_all_benchmarks()
```

### Custom Problem Selection

```python
from benchmarks import get_problems_by_domain, ProblemDomain

# Get specific problems
math_problems = get_problems_by_domain(ProblemDomain.MATH)

# Run on subset
runner = BenchmarkRunner()
results = runner.run_benchmarks(problems=math_problems)
```

### Baseline Comparison

```python
from benchmarks import BaselineGenerator, BaselineType

# Generate baselines
generator = BaselineGenerator()
baseline = generator.generate_baseline(
    problems=all_problems,
    baseline_type=BaselineType.FUNDAMENTALS_ONLY
)

# Compare baselines
comparison = generator.compare_baselines(
    "fundamentals_only",
    "full_tmr"
)
```

### Custom Scoring

```python
from benchmarks import ScoringSystem

# Create scoring system with custom weights
scoring = ScoringSystem(config={
    "weights": {
        "correctness": 0.5,
        "confidence": 0.3,
        "efficiency": 0.2,
    }
})

# Score a result
score = scoring.score_result(
    problem=problem,
    result=verification_result,
    execution_time_ms=150.0
)
```

## Expected Performance

Based on TMR's theoretical projections:

- **Success Rate**: 85-95% across all domains
- **Average Time**: 100-500ms per problem (STANDARD depth)
- **Confidence Accuracy**: ±15% of expected confidence
- **Cross-Domain Transfer**: High success rates across MATH, CODE, LOGIC

## Output Structure

```
benchmark_results/
├── benchmark_report_YYYYMMDD_HHMMSS.txt      # Text report
├── benchmark_report_YYYYMMDD_HHMMSS.json     # JSON report
├── benchmark_report_YYYYMMDD_HHMMSS.html     # HTML report
├── benchmark_report_YYYYMMDD_HHMMSS.md       # Markdown report
└── benchmark_report_YYYYMMDD_HHMMSS.csv      # CSV results
```

## Adding Custom Problems

```python
from benchmarks.problems import BenchmarkProblem, ProblemDomain, ComplexityLevel

# Create custom problem
custom_problem = BenchmarkProblem(
    id="CUSTOM-001",
    domain=ProblemDomain.MATH,
    complexity=ComplexityLevel.MODERATE,
    title="Custom Math Problem",
    description="Test custom reasoning",
    input_statement="If x^2 = 9, then x = 3 or x = -3",
    expected_valid=True,
    expected_confidence=0.90,
    ground_truth="Square root has two solutions: ±3",
    metadata={"category": "algebra"}
)

# Run benchmark
runner.run_benchmarks(problems=[custom_problem])
```

## CLI Options Reference

```
usage: run_benchmarks.py [-h] [--domain {math,code,logic,mixed}]
                         [--complexity {trivial,simple,moderate,complex,advanced}]
                         [--no-baselines]
                         [--baseline {fundamentals_only,with_nuance,full_tmr,minimal_depth,standard_depth,exhaustive_depth}]
                         [--max-time MAX_TIME] [--no-cache]
                         [--depth {MINIMAL,QUICK,STANDARD,THOROUGH,EXHAUSTIVE}]
                         [--output-dir OUTPUT_DIR]
                         [--format {text,json,html,markdown,csv}]
                         [--verbose] [--quiet] [--stats]

Options:
  --domain              Run specific domain(s)
  --complexity          Run specific complexity level(s)
  --no-baselines        Skip baseline generation
  --baseline            Generate specific baseline(s)
  --max-time            Maximum time per problem (ms)
  --no-cache            Disable caching
  --depth               Verification depth
  --output-dir          Output directory
  --format              Output format(s)
  --verbose, -v         Verbose logging
  --quiet, -q           Suppress output
  --stats               Show statistics
```

## Troubleshooting

### Import Errors

Make sure you're running from the TMR root directory:

```bash
cd /path/to/tmr
python run_benchmarks.py
```

### Slow Performance

Try using minimal depth or disabling baselines:

```bash
python run_benchmarks.py --depth MINIMAL --no-baselines
```

### Memory Issues

Run domains separately:

```bash
python run_benchmarks.py --domain math
python run_benchmarks.py --domain code
python run_benchmarks.py --domain logic
```

## Development

### Adding New Problems

1. Edit `benchmarks/problems.py`
2. Add problem to appropriate domain function
3. Run benchmarks to validate

### Adding New Metrics

1. Edit `benchmarks/metrics.py`
2. Add metric computation in `MetricsTracker`
3. Update reporting in `benchmarks/reporting.py`

### Adding New Baselines

1. Edit `benchmarks/baselines.py`
2. Add new `BaselineType` enum value
3. Implement execution method in `BaselineGenerator`

## License

Part of the Trinity Meta-Reasoning (TMR) Framework.

## Contact

For issues or questions, please refer to the main TMR documentation.
