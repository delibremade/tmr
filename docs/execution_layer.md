# Execution Layer Documentation

## Overview

The Execution Layer is Layer 3 of the Trinity Meta-Reasoning Framework (TMR). It provides context-aware verification synthesis with adaptive depth scaling and flexible output formatting. The execution layer orchestrates the verification process by intelligently selecting validators, scaling depth based on input complexity, and formatting results appropriately.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│              Execution Layer (Layer 3)               │
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────────┐    ┌─────────────────────┐    │
│  │  DepthSelector  │    │ ExecutionSynthesizer │    │
│  │                 │    │                      │    │
│  │ • Complexity    │───▶│ • Orchestration      │    │
│  │   Analysis      │    │ • Validator          │    │
│  │ • Domain        │    │   Selection          │    │
│  │   Inference     │    │ • Output             │    │
│  │ • Adaptive      │    │   Formatting         │    │
│  │   Scaling       │    │                      │    │
│  └─────────────────┘    └──────────┬───────────┘    │
│                                    │                │
└────────────────────────────────────┼────────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │  Fundamentals Layer │
                          │    (Layer 1)        │
                          └─────────────────────┘
```

## Core Components

### 1. DepthSelector

The `DepthSelector` class implements adaptive depth scaling to determine the appropriate level of verification.

#### Verification Depth Levels

| Depth | Time Budget | Min Confidence | Validators | Use Case |
|-------|-------------|----------------|------------|----------|
| **MINIMAL** | 50ms | 50% | 1 | Quick validation, non-critical |
| **QUICK** | 100ms | 60% | 1 | Simple tasks, rapid feedback |
| **STANDARD** | 500ms | 70% | 2 | Most common tasks |
| **THOROUGH** | 2000ms | 85% | 3 | Complex reasoning |
| **EXHAUSTIVE** | 10000ms | 95% | 4 | Critical validations |
| **ADAPTIVE** | - | - | - | Auto-determine depth |

#### Complexity Calculation

The depth selector calculates input complexity based on:

- **Length**: Text length contribution (normalized to 0-0.3)
- **Mathematical operators**: Count of +, -, *, /, ^, = (normalized to 0-0.2)
- **Logical connectives**: if, then, else, and, or, not, etc. (normalized to 0-0.2)
- **Nested structures**: Parentheses, brackets, braces (normalized to 0-0.15)
- **Dictionary complexity**: Number of keys, nested structures
- **List complexity**: List length, nested items

Total complexity score: 0.0 (simple) to 1.0 (very complex)

#### Domain Inference

The depth selector automatically infers the reasoning domain:

- **Mathematical**: Contains equations, operators, calculations
- **Logical**: Contains logical connectives, propositions, reasoning chains
- **Causal**: Contains cause/effect relationships, temporal ordering
- **Mixed**: Multiple domain indicators present
- **Simple**: Short, straightforward statements
- **Unknown**: Cannot determine domain

#### Usage Example

```python
from tmr import DepthSelector, VerificationDepth

selector = DepthSelector()

# Automatic depth selection
profile = selector.select_depth(
    input_data="Complex mathematical proof with multiple steps",
    domain="mathematical",
    required_confidence=0.85,
    time_budget_ms=1000
)

print(f"Selected depth: {profile.depth.value}")
print(f"Validators: {profile.validators}")
print(f"Principles: {profile.principles}")
```

### 2. ExecutionSynthesizer

The `ExecutionSynthesizer` class orchestrates the complete verification process.

#### Key Features

1. **Context-Aware Validator Selection**
   - Analyzes input structure and content
   - Selects appropriate validators (logical, mathematical, causal, consistency)
   - Adapts to multi-domain reasoning

2. **Principle-Based Validation**
   - Validates against fundamental logical principles
   - Configurable principle sets based on depth
   - Integration with FundamentalsLayer

3. **Flexible Output Formatting**
   - Multiple format options for different use cases
   - Comprehensive result metadata
   - Warnings and suggestions generation

4. **Statistics and Monitoring**
   - Performance tracking by depth and domain
   - Success rate calculation
   - Health check capabilities

#### Usage Example

```python
from tmr import ExecutionSynthesizer, SynthesisContext, OutputFormat, VerificationDepth

synthesizer = ExecutionSynthesizer()

# Simple usage
result = synthesizer.synthesize("2 + 2 = 4")

# Advanced usage with full context
context = SynthesisContext(
    input_data={
        "equation": "x^2 - 5x + 6 = 0",
        "steps": ["(x-2)(x-3) = 0", "x = 2 or x = 3"]
    },
    domain="mathematical",
    required_confidence=0.8,
    verification_depth=VerificationDepth.THOROUGH,
    output_format=OutputFormat.DETAILED
)

result = synthesizer.synthesize(context)
print(f"Valid: {result.valid}")
print(f"Confidence: {result.confidence:.2%}")
```

### 3. Output Formats

#### MINIMAL
Essential results only:
```python
{
    "valid": True,
    "confidence": 0.85
}
```

#### STANDARD (Default)
Balanced detail with warnings and suggestions:
```python
{
    "valid": True,
    "confidence": 0.85,
    "depth_used": "standard",
    "domain": "mathematical",
    "validators_used": ["logical", "mathematical"],
    "warnings": [],
    "suggestions": []
}
```

#### DETAILED
Complete validation information including all principle checks and validator details.

#### JSON
Machine-readable structured output for programmatic consumption.

#### HUMAN_READABLE
Formatted text summary:
```
==============================================================
VERIFICATION SUMMARY
==============================================================

Status: ✓ PASSED
Confidence: 85.0%
Domain: mathematical
Depth: standard
Processing Time: 12.45ms

Validators: logical, mathematical
Principles: identity, non_contradiction, excluded_middle, causality

==============================================================
```

## Integration with Fundamentals Layer

The execution layer seamlessly integrates with the fundamentals layer:

```python
from tmr import ExecutionSynthesizer, FundamentalsLayer

# Execution layer automatically creates fundamentals layer
synthesizer = ExecutionSynthesizer()

# Or provide your own configured fundamentals layer
fundamentals = FundamentalsLayer(config={"cache_size": 2000})
synthesizer = ExecutionSynthesizer(fundamentals_layer=fundamentals)
```

## Performance Tracking

The execution layer tracks performance metrics:

```python
synthesizer = ExecutionSynthesizer()

# Perform verifications...
for data in dataset:
    synthesizer.synthesize(data)

# Get statistics
stats = synthesizer.get_statistics()

print(f"Total syntheses: {stats['total_syntheses']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Avg time: {stats['avg_processing_time_ms']:.2f}ms")

# By depth
for depth, depth_stats in stats['by_depth'].items():
    print(f"{depth}: {depth_stats['count']} uses")

# By domain
for domain, domain_stats in stats['by_domain'].items():
    print(f"{domain}: {domain_stats['count']} uses")
```

## Health Monitoring

Check system health:

```python
health = synthesizer.health_check()

print(f"Status: {health['status']}")  # healthy, degraded, or unhealthy

if health['issues']:
    print("Issues detected:")
    for issue in health['issues']:
        print(f"  - {issue}")

print("\nMetrics:")
for key, value in health['metrics'].items():
    print(f"  {key}: {value}")
```

## Advanced Features

### Custom Validators

Specify custom validators for specific tasks:

```python
context = SynthesisContext(
    input_data=data,
    custom_validators=["mathematical", "causal"]
)
result = synthesizer.synthesize(context)
```

### Time Budget Constraints

Enforce time budgets for verification:

```python
context = SynthesisContext(
    input_data=data,
    time_budget_ms=500,  # Max 500ms
    verification_depth=VerificationDepth.ADAPTIVE
)
result = synthesizer.synthesize(context)
```

### Confidence Requirements

Set minimum confidence thresholds:

```python
context = SynthesisContext(
    input_data=data,
    required_confidence=0.95,  # 95% confidence required
    verification_depth=VerificationDepth.ADAPTIVE
)
result = synthesizer.synthesize(context)
# Automatically selects EXHAUSTIVE depth for 95% confidence
```

## API Reference

### DepthSelector

#### Methods

- `select_depth(input_data, domain=None, required_confidence=None, time_budget_ms=None, user_preference=None) -> DepthProfile`
  - Select appropriate verification depth

- `record_performance(depth, success, time_ms)`
  - Record performance data for adaptive learning

- `get_performance_stats() -> Dict`
  - Get performance statistics for all depth levels

- `get_recommended_depth(domain) -> VerificationDepth`
  - Get recommended depth for a domain

### ExecutionSynthesizer

#### Methods

- `synthesize(context) -> VerificationResult`
  - Main synthesis method

- `get_statistics() -> Dict`
  - Get comprehensive statistics

- `reset_statistics()`
  - Reset all statistics

- `health_check() -> Dict`
  - Perform system health check

### VerificationResult

#### Attributes

- `valid: bool` - Whether verification passed
- `confidence: float` - Confidence score (0.0-1.0)
- `depth_used: VerificationDepth` - Depth level used
- `domain: str` - Detected domain
- `validators_used: List[str]` - Validators applied
- `principles_checked: List[str]` - Principles validated
- `details: Dict` - Detailed validation results
- `timestamp: str` - ISO timestamp
- `processing_time_ms: float` - Processing time
- `warnings: List[str]` - Warnings generated
- `suggestions: List[str]` - Suggestions for improvement
- `metadata: Dict` - Additional metadata

## Best Practices

### 1. Use Adaptive Depth for Production

```python
context = SynthesisContext(
    input_data=data,
    verification_depth=VerificationDepth.ADAPTIVE,
    required_confidence=0.7
)
```

### 2. Set Appropriate Confidence Thresholds

- 0.5-0.6: Non-critical, exploratory
- 0.7-0.8: Standard production use
- 0.85-0.9: High-stakes decisions
- 0.95+: Critical, safety-critical systems

### 3. Monitor Performance Metrics

Regularly check statistics to optimize:
```python
stats = synthesizer.get_statistics()
if stats['success_rate'] < 0.7:
    # Consider adjusting depth profiles
    pass
```

### 4. Use Appropriate Output Formats

- **MINIMAL**: High-throughput batch processing
- **STANDARD**: Default for most applications
- **DETAILED**: Debugging and development
- **HUMAN_READABLE**: User-facing reports

### 5. Handle Warnings and Suggestions

```python
result = synthesizer.synthesize(data)

if result.warnings:
    logger.warning(f"Verification warnings: {result.warnings}")

if result.confidence < 0.7:
    for suggestion in result.suggestions:
        print(f"Suggestion: {suggestion}")
```

## Testing

The execution layer includes comprehensive tests:

```bash
python -m tests.test_execution_layer
```

Test coverage includes:
- Depth selector (11 tests)
- Execution synthesizer (14 tests)
- Integration tests (3 tests)

All 28 tests passing ✓

## Examples

See `examples/execution_layer_demo.py` for a comprehensive demonstration of all features:

```bash
python -m examples.execution_layer_demo
```

## Future Enhancements

Planned improvements:
1. **Machine Learning Integration**: Learn optimal depth profiles from historical data
2. **Parallel Validation**: Run multiple validators concurrently
3. **Caching**: Cache verification results for repeated inputs
4. **Custom Depth Profiles**: User-defined depth configurations
5. **Streaming Results**: Support for real-time verification feedback

## Related Documentation

- [Fundamentals Layer](fundamentals_layer.md)
- [Nuance Layer](nuance_layer.md) (planned)
- [API Reference](api.md)
- [Architecture Overview](architecture.md)

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/delibremade/tmr/issues
- Documentation: https://docs.tmr.ai (planned)
