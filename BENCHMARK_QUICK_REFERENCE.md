# TMR Benchmark Suite - Quick Reference

## 📊 Quick Stats

- **33 problems** across 4 domains (MATH: 10, CODE: 10, LOGIC: 10, MIXED: 3)
- **5 complexity levels** (trivial → advanced)
- **7 baseline types** for comparison
- **5 report formats** (text, JSON, HTML, markdown, CSV)

## 🚀 Common Commands

```bash
# Run all benchmarks
python run_benchmarks.py

# Run specific domain
python run_benchmarks.py --domain math

# Run specific complexity
python run_benchmarks.py --complexity simple

# Multiple filters
python run_benchmarks.py --domain math --domain code --complexity moderate

# Fast run (skip baselines)
python run_benchmarks.py --no-baselines

# Exhaustive verification
python run_benchmarks.py --depth EXHAUSTIVE

# Custom output
python run_benchmarks.py --output-dir ./my_results --format html

# View statistics
python run_benchmarks.py --stats

# Run examples
python example_benchmarks.py
```

## 📝 Programmatic Usage

### Basic Usage

```python
from benchmarks import BenchmarkRunner, BenchmarkConfig

# Simple run
runner = BenchmarkRunner()
results = runner.run_all_benchmarks()
```

### Domain-Specific

```python
from benchmarks import get_problems_by_domain, ProblemDomain

math_problems = get_problems_by_domain(ProblemDomain.MATH)
runner = BenchmarkRunner()
results = runner.run_benchmarks(problems=math_problems)
```

### Custom Configuration

```python
config = BenchmarkConfig(
    domains=["math", "code"],
    complexities=["simple", "moderate"],
    verification_depth="THOROUGH",
    generate_baselines=True,
    baseline_types=["fundamentals_only", "full_tmr"],
    output_dir="./results"
)
runner = BenchmarkRunner(config=config)
results = runner.run_with_baselines()
```

### Custom Scoring

```python
from benchmarks import ScoringSystem

scoring = ScoringSystem(config={
    "weights": {
        "correctness": 0.50,
        "confidence": 0.30,
        "efficiency": 0.20,
    }
})

score = scoring.score_result(problem, result, execution_time_ms)
```

### Baseline Generation

```python
from benchmarks import BaselineGenerator, BaselineType

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

### Metrics Analysis

```python
from benchmarks import MetricsTracker

tracker = MetricsTracker()
tracker.start_tracking()

# ... run problems ...

tracker.stop_tracking()
metrics = tracker.compute_metrics()

print(f"Success rate: {metrics.success_rate:.1%}")
print(f"Avg time: {metrics.avg_time_ms:.2f} ms")
```

### Report Generation

```python
from benchmarks import ReportGenerator, ReportFormat

generator = ReportGenerator()
report = generator.generate_comprehensive_report(
    results,
    output_format=ReportFormat.HTML
)
```

## 🎯 Problem Domains

| Domain | Count | Examples |
|--------|-------|----------|
| **MATH** | 10 | Arithmetic, Algebra, Calculus, Proofs |
| **CODE** | 10 | Variables, Functions, Algorithms, Concurrency |
| **LOGIC** | 10 | Boolean ops, Inference, Quantifiers |
| **MIXED** | 3 | Cross-domain reasoning |

## 📈 Complexity Levels

| Level | Count | Description |
|-------|-------|-------------|
| **Trivial** | 6 | Basic operations (2+2=4) |
| **Simple** | 6 | Simple reasoning (algebraic identity) |
| **Moderate** | 7 | Multi-step reasoning (calculus) |
| **Complex** | 7 | Advanced reasoning (algorithm analysis) |
| **Advanced** | 7 | Expert-level (proofs, optimization) |

## 🔧 Verification Depths

- **MINIMAL**: Basic checks (~50ms, 0.5 confidence)
- **QUICK**: Essential validations (~100ms, 0.6 confidence)
- **STANDARD**: Standard suite (~500ms, 0.7 confidence) ⭐ Default
- **THOROUGH**: Comprehensive (~2s, 0.85 confidence)
- **EXHAUSTIVE**: All checks (~10s, 0.95 confidence)

## 📊 Scoring Components

| Component | Weight | Description |
|-----------|--------|-------------|
| **Correctness** | 40% | Right answer? |
| **Confidence** | 25% | Accurate confidence score? |
| **Efficiency** | 15% | Fast execution? |
| **Consistency** | 10% | Reproducible? |
| **Robustness** | 10% | Handles edge cases? |

## 🎲 Baseline Types

1. **no_verification** - Assume all correct (upper bound)
2. **fundamentals_only** - Layer 1 only
3. **with_nuance** - Layers 1+2
4. **full_tmr** - All 3 layers ⭐ Default
5. **minimal_depth** - Minimal verification
6. **standard_depth** - Standard verification
7. **exhaustive_depth** - Exhaustive verification

## 📁 Output Files

```
benchmark_results/
├── benchmark_report_YYYYMMDD_HHMMSS.txt    # Human-readable
├── benchmark_report_YYYYMMDD_HHMMSS.json   # Machine-readable
├── benchmark_report_YYYYMMDD_HHMMSS.html   # Interactive web
├── benchmark_report_YYYYMMDD_HHMMSS.md     # GitHub-friendly
└── benchmark_report_YYYYMMDD_HHMMSS.csv    # Spreadsheet data
```

## 🧪 Testing

```bash
# Run test suite
python test_benchmarks.py

# Expected: 31/31 tests passed
```

## 📚 Key Files

| File | Purpose |
|------|---------|
| `run_benchmarks.py` | CLI entry point |
| `example_benchmarks.py` | Usage examples |
| `test_benchmarks.py` | Test suite |
| `benchmarks/__init__.py` | Package exports |
| `benchmarks/problems.py` | Problem definitions |
| `benchmarks/scoring.py` | Scoring system |
| `benchmarks/metrics.py` | Metrics tracking |
| `benchmarks/baselines.py` | Baseline generation |
| `benchmarks/runner.py` | Orchestration |
| `benchmarks/reporting.py` | Report generation |
| `benchmarks/README.md` | Full documentation |

## 🐛 Troubleshooting

**Import errors?**
```bash
cd /path/to/tmr
python run_benchmarks.py
```

**Too slow?**
```bash
python run_benchmarks.py --depth MINIMAL --no-baselines
```

**Memory issues?**
```bash
python run_benchmarks.py --domain math  # Run one at a time
```

**Need help?**
```bash
python run_benchmarks.py --help
```

## 💡 Tips

- Use `--no-baselines` for faster iteration during development
- Use `--depth MINIMAL` for quick smoke tests
- Use `--format json` for integration with other tools
- Use `--verbose` to debug issues
- Check `benchmarks/README.md` for detailed documentation

## 🎯 Expected Performance

Based on TMR's design goals:
- **Success Rate**: 85-95%
- **Avg Time**: 100-500ms per problem
- **Confidence Accuracy**: ±15%

## 📞 Getting Help

1. Check `benchmarks/README.md` for detailed docs
2. Run `python run_benchmarks.py --help` for CLI options
3. Run `python example_benchmarks.py` for usage examples
4. Run `python test_benchmarks.py` to verify installation

---

**Version**: 0.1.0
**Status**: Ready for use
**Tests**: 31/31 passing ✅
