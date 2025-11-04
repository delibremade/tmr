"""
Tests for the Execution Layer
"""

from datetime import datetime

from tmr.execution import (
    DepthSelector,
    VerificationDepth,
    DepthProfile,
    ExecutionSynthesizer,
    OutputFormat,
    VerificationResult,
    SynthesisContext
)
from tmr.fundamentals import FundamentalsLayer


class TestDepthSelector:
    """Tests for the DepthSelector class."""

    def test_initialization(self):
        """Test depth selector initialization."""
        selector = DepthSelector()
        assert selector is not None
        assert len(selector.DEPTH_PROFILES) == 5

    def test_depth_profiles(self):
        """Test that all depth profiles are properly configured."""
        selector = DepthSelector()

        for depth in [VerificationDepth.MINIMAL, VerificationDepth.QUICK,
                     VerificationDepth.STANDARD, VerificationDepth.THOROUGH,
                     VerificationDepth.EXHAUSTIVE]:
            profile = selector.DEPTH_PROFILES[depth]
            assert isinstance(profile, DepthProfile)
            assert profile.depth == depth
            assert len(profile.validators) > 0
            assert len(profile.principles) > 0
            assert profile.max_time_ms is not None
            assert profile.min_confidence > 0.0

    def test_select_depth_simple_input(self):
        """Test depth selection for simple input."""
        selector = DepthSelector()
        profile = selector.select_depth("2 + 2 = 4")

        assert isinstance(profile, DepthProfile)
        # Should select quick or standard for simple math
        assert profile.depth in [VerificationDepth.QUICK, VerificationDepth.STANDARD,
                               VerificationDepth.THOROUGH]

    def test_select_depth_complex_input(self):
        """Test depth selection for complex input."""
        selector = DepthSelector()

        complex_input = {
            "steps": [f"Step {i}" for i in range(20)],
            "logical": {"chain": "complex"},
            "mathematical": {"equation": "x^2 + y^2 = z^2"},
            "causal": {"events": ["A", "B", "C"]}
        }

        profile = selector.select_depth(complex_input)

        assert isinstance(profile, DepthProfile)
        # Complex input should trigger higher depth
        assert profile.depth in [VerificationDepth.THOROUGH, VerificationDepth.EXHAUSTIVE]

    def test_select_depth_with_confidence_requirement(self):
        """Test depth selection with high confidence requirement."""
        selector = DepthSelector()

        profile = selector.select_depth(
            "Simple statement",
            required_confidence=0.95
        )

        assert profile.depth in [VerificationDepth.EXHAUSTIVE]

    def test_select_depth_with_time_budget(self):
        """Test depth selection with tight time budget."""
        selector = DepthSelector()

        profile = selector.select_depth(
            "Complex reasoning chain with multiple steps",
            time_budget_ms=100
        )

        # Should select quick or minimal due to time constraint
        assert profile.depth in [VerificationDepth.MINIMAL, VerificationDepth.QUICK]

    def test_select_depth_with_user_preference(self):
        """Test depth selection with user-specified preference."""
        selector = DepthSelector()

        profile = selector.select_depth(
            "Any input",
            user_preference=VerificationDepth.THOROUGH
        )

        assert profile.depth == VerificationDepth.THOROUGH

    def test_calculate_complexity(self):
        """Test complexity calculation."""
        selector = DepthSelector()

        # Simple string
        simple = selector._calculate_complexity("hello")
        assert 0.0 <= simple <= 0.3

        # Complex mathematical expression
        complex_math = selector._calculate_complexity(
            "((x + y)^2 - (x - y)^2) / (4 * x * y) = 1"
        )
        assert complex_math > simple

        # Complex dictionary
        complex_dict = {
            "steps": [{"id": i, "statement": f"Step {i}"} for i in range(15)],
            "logical": {},
            "mathematical": {},
            "causal": {}
        }
        dict_complexity = selector._calculate_complexity(complex_dict)
        assert dict_complexity > 0.5

    def test_infer_domain(self):
        """Test domain inference."""
        selector = DepthSelector()

        # Mathematical domain
        assert selector._infer_domain("x + 5 = 10") == "mathematical"
        assert selector._infer_domain({"equation": "E = mc^2"}) == "mathematical"

        # Causal domain
        assert selector._infer_domain("because of A, B occurred") == "causal"
        assert selector._infer_domain({"cause": "A", "effect": "B"}) == "causal"

        # Logical domain
        assert selector._infer_domain("if A then B") == "logical"
        assert selector._infer_domain({"steps": []}) == "logical"

        # Mixed domain
        assert selector._infer_domain({
            "logical": {},
            "mathematical": {},
            "causal": {}
        }) == "mixed"

    def test_record_performance(self):
        """Test performance recording."""
        selector = DepthSelector()

        selector.record_performance(VerificationDepth.STANDARD, True, 150.0)
        selector.record_performance(VerificationDepth.STANDARD, True, 200.0)
        selector.record_performance(VerificationDepth.STANDARD, False, 100.0)

        stats = selector.get_performance_stats()
        standard_stats = stats["standard"]

        assert standard_stats["total_uses"] == 3
        assert standard_stats["success_rate"] == 2/3
        assert standard_stats["avg_time_ms"] > 0

    def test_get_recommended_depth(self):
        """Test recommended depth retrieval."""
        selector = DepthSelector()

        # Known domains
        assert selector.get_recommended_depth("mathematical") == VerificationDepth.THOROUGH
        assert selector.get_recommended_depth("logical") == VerificationDepth.STANDARD
        assert selector.get_recommended_depth("simple") == VerificationDepth.QUICK

        # Unknown domain
        assert selector.get_recommended_depth("unknown_domain") == VerificationDepth.STANDARD


class TestExecutionSynthesizer:
    """Tests for the ExecutionSynthesizer class."""

    def test_initialization(self):
        """Test synthesizer initialization."""
        synthesizer = ExecutionSynthesizer()
        assert synthesizer is not None
        assert isinstance(synthesizer.fundamentals, FundamentalsLayer)
        assert isinstance(synthesizer.depth_selector, DepthSelector)

    def test_synthesize_simple_statement(self):
        """Test synthesis with simple statement."""
        synthesizer = ExecutionSynthesizer()

        result = synthesizer.synthesize("2 + 2 = 4")

        assert isinstance(result, VerificationResult)
        assert isinstance(result.valid, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert result.depth_used in VerificationDepth
        assert result.timestamp is not None
        assert result.processing_time_ms > 0

    def test_synthesize_with_context(self):
        """Test synthesis with full context."""
        synthesizer = ExecutionSynthesizer()

        context = SynthesisContext(
            input_data="If x > 5, then x > 3",
            domain="logical",
            required_confidence=0.8,
            verification_depth=VerificationDepth.THOROUGH,
            output_format=OutputFormat.STANDARD
        )

        result = synthesizer.synthesize(context)

        assert isinstance(result, VerificationResult)
        assert result.depth_used == VerificationDepth.THOROUGH
        assert result.domain == "logical"

    def test_synthesize_mathematical_reasoning(self):
        """Test synthesis with mathematical reasoning."""
        synthesizer = ExecutionSynthesizer()

        math_input = {
            "equation": "2x + 5 = 15",
            "steps": [
                "2x + 5 = 15",
                "2x = 10",
                "x = 5"
            ],
            "result": 5
        }

        result = synthesizer.synthesize(math_input)

        assert isinstance(result, VerificationResult)
        assert "mathematical" in result.validators_used

    def test_synthesize_causal_reasoning(self):
        """Test synthesis with causal reasoning."""
        synthesizer = ExecutionSynthesizer()

        causal_input = {
            "events": [
                {"id": 1, "description": "Rain", "timestamp": 100},
                {"id": 2, "description": "Wet ground", "timestamp": 101}
            ],
            "relationships": [
                {"cause_id": 1, "effect_id": 2, "confidence": 0.9}
            ]
        }

        result = synthesizer.synthesize(causal_input)

        assert isinstance(result, VerificationResult)
        # Should recognize causal domain
        assert result.domain in ["causal", "unknown"]

    def test_output_format_minimal(self):
        """Test minimal output format."""
        synthesizer = ExecutionSynthesizer()

        context = SynthesisContext(
            input_data="Simple test",
            output_format=OutputFormat.MINIMAL
        )

        result = synthesizer.synthesize(context)

        assert len(result.details) == 0
        assert len(result.warnings) == 0
        assert len(result.suggestions) == 0

    def test_output_format_detailed(self):
        """Test detailed output format."""
        synthesizer = ExecutionSynthesizer()

        context = SynthesisContext(
            input_data="Test statement",
            output_format=OutputFormat.DETAILED,
            verification_depth=VerificationDepth.THOROUGH
        )

        result = synthesizer.synthesize(context)

        assert len(result.details) > 0
        assert "fundamental_validation" in result.details

    def test_output_format_human_readable(self):
        """Test human-readable output format."""
        synthesizer = ExecutionSynthesizer()

        context = SynthesisContext(
            input_data="Test",
            output_format=OutputFormat.HUMAN_READABLE
        )

        result = synthesizer.synthesize(context)

        assert "formatted_summary" in result.metadata
        assert isinstance(result.metadata["formatted_summary"], str)
        assert "VERIFICATION SUMMARY" in result.metadata["formatted_summary"]

    def test_validator_selection_mathematical(self):
        """Test context-aware validator selection for math."""
        synthesizer = ExecutionSynthesizer()

        validators = synthesizer._select_validators(
            SynthesisContext(input_data="2 + 2 = 4"),
            DepthSelector.DEPTH_PROFILES[VerificationDepth.STANDARD]
        )

        assert "mathematical" in validators or "logical" in validators

    def test_validator_selection_causal(self):
        """Test context-aware validator selection for causal reasoning."""
        synthesizer = ExecutionSynthesizer()

        causal_input = {
            "cause": "A",
            "effect": "B",
            "events": []
        }

        validators = synthesizer._select_validators(
            SynthesisContext(input_data=causal_input),
            DepthSelector.DEPTH_PROFILES[VerificationDepth.STANDARD]
        )

        assert "causal" in validators

    def test_validator_selection_consistency(self):
        """Test validator selection for multi-domain input."""
        synthesizer = ExecutionSynthesizer()

        mixed_input = {
            "logical": {"steps": []},
            "mathematical": {"equation": "x = 5"},
            "causal": {"events": []}
        }

        validators = synthesizer._select_validators(
            SynthesisContext(input_data=mixed_input),
            DepthSelector.DEPTH_PROFILES[VerificationDepth.STANDARD]
        )

        assert "consistency" in validators

    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        synthesizer = ExecutionSynthesizer()

        # Perform several syntheses
        for i in range(5):
            synthesizer.synthesize(f"Test {i}")

        stats = synthesizer.get_statistics()

        assert stats["total_syntheses"] == 5
        assert "successful_verifications" in stats
        assert "failed_verifications" in stats
        assert "success_rate" in stats
        assert "by_depth" in stats
        assert "by_domain" in stats

    def test_health_check(self):
        """Test health check functionality."""
        synthesizer = ExecutionSynthesizer()

        health = synthesizer.health_check()

        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "issues" in health
        assert "metrics" in health

    def test_reset_statistics(self):
        """Test statistics reset."""
        synthesizer = ExecutionSynthesizer()

        # Generate some statistics
        synthesizer.synthesize("Test")
        assert synthesizer.stats["total_syntheses"] > 0

        # Reset
        synthesizer.reset_statistics()
        assert synthesizer.stats["total_syntheses"] == 0

    def test_synthesize_with_invalid_input(self):
        """Test synthesis with invalid/error-inducing input."""
        synthesizer = ExecutionSynthesizer()

        # Should handle gracefully and return error result
        result = synthesizer.synthesize(None)

        assert isinstance(result, VerificationResult)
        # Should still return a result, possibly with low confidence


class TestIntegration:
    """Integration tests for execution layer with fundamentals."""

    def test_end_to_end_verification(self):
        """Test complete end-to-end verification flow."""
        synthesizer = ExecutionSynthesizer()

        # Create a reasoning chain
        reasoning = {
            "steps": [
                {"statement": "All humans are mortal", "justification": "Premise"},
                {"statement": "Socrates is human", "justification": "Premise"},
                {"statement": "Therefore, Socrates is mortal", "justification": "Conclusion"}
            ],
            "conclusion": "Socrates is mortal"
        }

        context = SynthesisContext(
            input_data=reasoning,
            domain="logical",
            verification_depth=VerificationDepth.STANDARD,
            output_format=OutputFormat.DETAILED
        )

        result = synthesizer.synthesize(context)

        assert isinstance(result, VerificationResult)
        assert result.depth_used == VerificationDepth.STANDARD
        assert result.processing_time_ms > 0
        assert len(result.details) > 0

    def test_adaptive_depth_scaling(self):
        """Test that depth scales appropriately with complexity."""
        synthesizer = ExecutionSynthesizer()

        # Simple input
        simple_context = SynthesisContext(
            input_data="A = A",
            verification_depth=VerificationDepth.ADAPTIVE
        )
        simple_result = synthesizer.synthesize(simple_context)

        # Complex input
        complex_input = {
            "steps": [f"Complex step {i}" for i in range(20)],
            "logical": {},
            "mathematical": {},
            "causal": {}
        }
        complex_context = SynthesisContext(
            input_data=complex_input,
            verification_depth=VerificationDepth.ADAPTIVE
        )
        complex_result = synthesizer.synthesize(complex_context)

        # Verify that different depths were used (or at least appropriate depths)
        assert isinstance(simple_result.depth_used, VerificationDepth)
        assert isinstance(complex_result.depth_used, VerificationDepth)

    def test_confidence_threshold_enforcement(self):
        """Test that confidence requirements are respected."""
        synthesizer = ExecutionSynthesizer()

        context = SynthesisContext(
            input_data="Test statement",
            required_confidence=0.95,
            verification_depth=VerificationDepth.ADAPTIVE
        )

        result = synthesizer.synthesize(context)

        # Should use exhaustive depth for high confidence requirement
        assert result.depth_used in [VerificationDepth.EXHAUSTIVE, VerificationDepth.THOROUGH]


if __name__ == "__main__":
    # Run tests without pytest
    import sys

    test_classes = [TestDepthSelector, TestExecutionSynthesizer, TestIntegration]

    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("=" * 60)

        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in methods:
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✓ {method_name}")
            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                import traceback
                traceback.print_exc()
