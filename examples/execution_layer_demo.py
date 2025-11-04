"""
Execution Layer Demonstration

This script demonstrates the capabilities of the Execution Layer synthesis framework,
including adaptive depth scaling, context-aware verification, and output formatting.
"""

from tmr import (
    ExecutionSynthesizer,
    DepthSelector,
    VerificationDepth,
    OutputFormat,
    SynthesisContext,
    FundamentalsLayer
)


def demo_simple_verification():
    """Demonstrate simple verification with default settings."""
    print("\n" + "="*70)
    print("DEMO 1: Simple Verification")
    print("="*70)

    synthesizer = ExecutionSynthesizer()

    # Simple mathematical statement
    result = synthesizer.synthesize("2 + 2 = 4")

    print(f"Input: '2 + 2 = 4'")
    print(f"Valid: {result.valid}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Depth Used: {result.depth_used.value}")
    print(f"Domain: {result.domain}")
    print(f"Processing Time: {result.processing_time_ms:.2f}ms")


def demo_adaptive_depth_scaling():
    """Demonstrate adaptive depth scaling based on complexity."""
    print("\n" + "="*70)
    print("DEMO 2: Adaptive Depth Scaling")
    print("="*70)

    synthesizer = ExecutionSynthesizer()

    # Simple input
    simple_input = "A = A"
    simple_result = synthesizer.synthesize(
        SynthesisContext(
            input_data=simple_input,
            verification_depth=VerificationDepth.ADAPTIVE
        )
    )

    # Complex input
    complex_input = {
        "steps": [
            {"statement": "Given: All humans are mortal"},
            {"statement": "Given: Socrates is a human"},
            {"statement": "Therefore: Socrates is mortal"},
        ],
        "logical": {"chain": "syllogism"},
        "mathematical": {"proof_type": "deductive"}
    }
    complex_result = synthesizer.synthesize(
        SynthesisContext(
            input_data=complex_input,
            verification_depth=VerificationDepth.ADAPTIVE
        )
    )

    print(f"\nSimple Input: '{simple_input}'")
    print(f"  → Depth Selected: {simple_result.depth_used.value}")
    print(f"  → Processing Time: {simple_result.processing_time_ms:.2f}ms")

    print(f"\nComplex Input: Multi-domain reasoning")
    print(f"  → Depth Selected: {complex_result.depth_used.value}")
    print(f"  → Processing Time: {complex_result.processing_time_ms:.2f}ms")
    print(f"  → Validators Used: {', '.join(complex_result.validators_used)}")


def demo_confidence_requirements():
    """Demonstrate verification with different confidence requirements."""
    print("\n" + "="*70)
    print("DEMO 3: Confidence-Based Depth Selection")
    print("="*70)

    synthesizer = ExecutionSynthesizer()
    test_input = "If x > 5, then x > 3"

    confidence_levels = [0.6, 0.8, 0.95]

    for confidence in confidence_levels:
        result = synthesizer.synthesize(
            SynthesisContext(
                input_data=test_input,
                required_confidence=confidence,
                verification_depth=VerificationDepth.ADAPTIVE
            )
        )
        print(f"\nRequired Confidence: {confidence:.0%}")
        print(f"  → Depth Selected: {result.depth_used.value}")
        print(f"  → Actual Confidence: {result.confidence:.2%}")
        print(f"  → Principles Checked: {len(result.principles_checked)}")


def demo_context_aware_validation():
    """Demonstrate context-aware validator selection."""
    print("\n" + "="*70)
    print("DEMO 4: Context-Aware Validator Selection")
    print("="*70)

    synthesizer = ExecutionSynthesizer()

    # Mathematical reasoning
    math_input = {
        "equation": "x^2 - 5x + 6 = 0",
        "steps": [
            "x^2 - 5x + 6 = 0",
            "(x - 2)(x - 3) = 0",
            "x = 2 or x = 3"
        ],
        "result": [2, 3]
    }

    math_result = synthesizer.synthesize(math_input)

    print("\nMathematical Input:")
    print(f"  Domain Detected: {math_result.domain}")
    print(f"  Validators Selected: {', '.join(math_result.validators_used)}")

    # Causal reasoning
    causal_input = {
        "events": [
            {"id": 1, "description": "Temperature drops below 0°C", "timestamp": 100},
            {"id": 2, "description": "Water freezes", "timestamp": 120}
        ],
        "relationships": [
            {"cause_id": 1, "effect_id": 2, "confidence": 0.95}
        ]
    }

    causal_result = synthesizer.synthesize(causal_input)

    print("\nCausal Input:")
    print(f"  Domain Detected: {causal_result.domain}")
    print(f"  Validators Selected: {', '.join(causal_result.validators_used)}")


def demo_output_formats():
    """Demonstrate different output formatting options."""
    print("\n" + "="*70)
    print("DEMO 5: Output Formatting Options")
    print("="*70)

    synthesizer = ExecutionSynthesizer()
    test_input = "All squares are rectangles"

    # Minimal format
    minimal = synthesizer.synthesize(
        SynthesisContext(
            input_data=test_input,
            output_format=OutputFormat.MINIMAL
        )
    )

    print("\n1. MINIMAL Format:")
    print(f"   Valid: {minimal.valid}, Confidence: {minimal.confidence:.2%}")
    print(f"   Details: {len(minimal.details)} items")
    print(f"   Warnings: {len(minimal.warnings)} items")

    # Standard format
    standard = synthesizer.synthesize(
        SynthesisContext(
            input_data=test_input,
            output_format=OutputFormat.STANDARD
        )
    )

    print("\n2. STANDARD Format:")
    print(f"   Valid: {standard.valid}, Confidence: {standard.confidence:.2%}")
    print(f"   Details: {len(standard.details)} items")
    print(f"   Warnings: {len(standard.warnings)} items")
    print(f"   Suggestions: {len(standard.suggestions)} items")

    # Human-readable format
    human = synthesizer.synthesize(
        SynthesisContext(
            input_data=test_input,
            output_format=OutputFormat.HUMAN_READABLE
        )
    )

    print("\n3. HUMAN_READABLE Format:")
    if "formatted_summary" in human.metadata:
        print(human.metadata["formatted_summary"])


def demo_statistics_and_monitoring():
    """Demonstrate statistics tracking and health monitoring."""
    print("\n" + "="*70)
    print("DEMO 6: Statistics and Health Monitoring")
    print("="*70)

    synthesizer = ExecutionSynthesizer()

    # Perform several syntheses
    test_inputs = [
        "2 + 2 = 4",
        "If A then B",
        {"equation": "x = 5"},
        {"cause": "rain", "effect": "wet ground"},
        "Complex reasoning with multiple steps and domains"
    ]

    for inp in test_inputs:
        synthesizer.synthesize(inp)

    # Get statistics
    stats = synthesizer.get_statistics()

    print(f"\nStatistics after {stats['total_syntheses']} syntheses:")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
    print(f"  Avg Processing Time: {stats['avg_processing_time_ms']:.2f}ms")

    print("\n  By Depth:")
    for depth, depth_stats in stats['by_depth'].items():
        success_rate = depth_stats['successes'] / depth_stats['count']
        print(f"    {depth}: {depth_stats['count']} uses, {success_rate:.1%} success")

    print("\n  By Domain:")
    for domain, domain_stats in stats['by_domain'].items():
        success_rate = domain_stats['successes'] / domain_stats['count']
        print(f"    {domain}: {domain_stats['count']} uses, {success_rate:.1%} success")

    # Health check
    health = synthesizer.health_check()
    print(f"\nSystem Health: {health['status'].upper()}")
    if health['issues']:
        print(f"  Issues: {', '.join(health['issues'])}")
    else:
        print("  No issues detected")


def demo_depth_profiles():
    """Demonstrate manual depth profile selection."""
    print("\n" + "="*70)
    print("DEMO 7: Manual Depth Profile Selection")
    print("="*70)

    synthesizer = ExecutionSynthesizer()
    test_input = "A logical statement to verify"

    depths = [
        VerificationDepth.QUICK,
        VerificationDepth.STANDARD,
        VerificationDepth.THOROUGH
    ]

    print("\nSame input with different depth profiles:\n")
    for depth in depths:
        result = synthesizer.synthesize(
            SynthesisContext(
                input_data=test_input,
                verification_depth=depth
            )
        )

        print(f"{depth.value.upper()}:")
        print(f"  Validators: {', '.join(result.validators_used)}")
        print(f"  Principles: {', '.join(result.principles_checked)}")
        print(f"  Processing Time: {result.processing_time_ms:.2f}ms")
        print(f"  Confidence: {result.confidence:.2%}")
        print()


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("EXECUTION LAYER SYNTHESIS FRAMEWORK - DEMONSTRATION")
    print("="*70)
    print("\nThis demo showcases the key features of the execution layer:")
    print("  • Adaptive depth scaling")
    print("  • Context-aware verification selection")
    print("  • Multiple output formats")
    print("  • Statistics and health monitoring")

    try:
        demo_simple_verification()
        demo_adaptive_depth_scaling()
        demo_confidence_requirements()
        demo_context_aware_validation()
        demo_output_formats()
        demo_statistics_and_monitoring()
        demo_depth_profiles()

        print("\n" + "="*70)
        print("DEMONSTRATION COMPLETE")
        print("="*70)
        print("\nThe execution layer successfully demonstrated:")
        print("  ✓ Adaptive depth scaling based on complexity")
        print("  ✓ Context-aware validator selection")
        print("  ✓ Flexible output formatting")
        print("  ✓ Comprehensive statistics tracking")
        print("  ✓ Health monitoring capabilities")
        print("  ✓ Integration with fundamentals layer")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
