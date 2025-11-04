"""
Comprehensive edge case tests for the fundamentals layer.

Tests edge cases, boundary conditions, and error scenarios across
all components of the fundamentals layer.
"""

import pytest
import sys
from tmr.fundamentals.layer import FundamentalsLayer
from tmr.fundamentals.principles import (
    IdentityPrinciple,
    NonContradictionPrinciple,
    ExcludedMiddlePrinciple,
    CausalityPrinciple,
    ConservationPrinciple,
    LogicalPrinciples,
    PrincipleType,
)
from tmr.fundamentals.validators import (
    LogicalValidator,
    MathematicalValidator,
    CausalValidator,
    ConsistencyValidator,
    ReasoningChain,
    ReasoningStep,
)


@pytest.mark.edge_case
class TestPrincipleEdgeCases:
    """Edge cases for principle validation."""

    def test_identity_with_nan(self):
        """Test identity principle with NaN values."""
        principle = IdentityPrinciple()

        # NaN is not equal to itself in standard Python
        result = principle.validate(float('nan'))
        assert isinstance(result.valid, bool)

    def test_identity_with_infinity(self):
        """Test identity principle with infinity."""
        principle = IdentityPrinciple()

        result = principle.validate(float('inf'))
        assert result.valid is True
        assert result.confidence == 1.0

    def test_identity_with_negative_infinity(self):
        """Test identity principle with negative infinity."""
        principle = IdentityPrinciple()

        result = principle.validate(float('-inf'))
        assert result.valid is True

    def test_identity_with_complex_numbers(self):
        """Test identity principle with complex numbers."""
        principle = IdentityPrinciple()

        result = principle.validate(complex(1, 2))
        assert result.valid is True

    def test_identity_with_empty_string(self):
        """Test identity principle with empty string."""
        principle = IdentityPrinciple()

        result = principle.validate("")
        assert result.valid is True

    def test_identity_with_deeply_nested_structure(self):
        """Test identity with deeply nested data structures."""
        principle = IdentityPrinciple()

        nested = {"a": {"b": {"c": {"d": {"e": "value"}}}}}
        result = principle.validate(nested)
        assert isinstance(result.valid, bool)

    def test_causality_with_equal_timestamps(self):
        """Test causality with equal timestamps (simultaneous events)."""
        principle = CausalityPrinciple()

        statement = {
            "cause": "A",
            "effect": "B",
            "timestamp_cause": 100,
            "timestamp_effect": 100
        }
        result = principle.validate(statement)
        # Should be valid (not necessarily causal, but not violating)
        assert isinstance(result.valid, bool)

    def test_causality_with_negative_timestamps(self):
        """Test causality with negative timestamps."""
        principle = CausalityPrinciple()

        statement = {
            "events": [
                {"name": "A", "timestamp": -100},
                {"name": "B", "timestamp": -50},
            ]
        }
        result = principle.validate(statement)
        assert result.valid is True

    def test_causality_with_float_timestamps(self):
        """Test causality with floating-point timestamps."""
        principle = CausalityPrinciple()

        statement = {
            "cause": "A",
            "effect": "B",
            "timestamp_cause": 100.5,
            "timestamp_effect": 100.6
        }
        result = principle.validate(statement)
        assert result.valid is True

    def test_conservation_with_zero_values(self):
        """Test conservation with zero values."""
        principle = ConservationPrinciple()

        statement = {
            "before": 0,
            "after": 0
        }
        result = principle.validate(statement)
        assert result.valid is True

    def test_conservation_with_negative_values(self):
        """Test conservation with negative values."""
        principle = ConservationPrinciple()

        statement = {
            "before": -100,
            "after": -100
        }
        result = principle.validate(statement)
        assert result.valid is True

    def test_conservation_with_very_small_differences(self):
        """Test conservation near floating-point precision limits."""
        principle = ConservationPrinciple()

        statement = {
            "before": 1.0,
            "after": 1.0 + 1e-10
        }
        result = principle.validate(statement)
        # Should be valid due to tolerance
        assert result.valid is True

    def test_conservation_with_empty_lists(self):
        """Test conservation with empty lists."""
        principle = ConservationPrinciple()

        statement = {
            "before": [],
            "after": [],
            "conserved_quantity": "count"
        }
        result = principle.validate(statement)
        assert result.valid is True

    def test_excluded_middle_with_none_truth_value(self):
        """Test excluded middle with None truth value."""
        principle = ExcludedMiddlePrinciple()

        statement = {
            "proposition": "test",
            "truth_value": None
        }
        result = principle.validate(statement)
        assert result.valid is False

    def test_non_contradiction_with_empty_propositions(self):
        """Test non-contradiction with empty propositions."""
        principle = NonContradictionPrinciple()

        statement = {
            "propositions": []
        }
        result = principle.validate(statement)
        assert result.valid is True


@pytest.mark.edge_case
class TestValidatorEdgeCases:
    """Edge cases for validators."""

    def test_logical_validator_empty_chain(self):
        """Test logical validator with empty chain."""
        validator = LogicalValidator()
        chain = ReasoningChain(steps=[])

        valid, confidence, details = validator.validate(chain)
        assert isinstance(valid, bool)
        assert details["num_steps"] == 0

    def test_logical_validator_single_step(self):
        """Test logical validator with single step."""
        validator = LogicalValidator()
        chain = ReasoningChain(
            steps=[ReasoningStep(step_id=0, statement="A")]
        )

        valid, confidence, details = validator.validate(chain)
        assert isinstance(valid, bool)

    def test_logical_validator_circular_dependencies(self):
        """Test logical validator with circular dependencies."""
        validator = LogicalValidator()

        # This creates an invalid dependency structure
        steps = [
            ReasoningStep(step_id=0, statement="A", dependencies=[1]),
            ReasoningStep(step_id=1, statement="B", dependencies=[0]),
        ]
        chain = ReasoningChain(steps=steps)

        valid, confidence, details = validator.validate(chain)
        # Should detect invalid dependencies
        assert details["consistency_valid"] is False

    def test_logical_validator_very_long_statements(self):
        """Test logical validator with very long statements."""
        validator = LogicalValidator()

        long_statement = "A" * 10000
        steps = [ReasoningStep(step_id=0, statement=long_statement)]
        chain = ReasoningChain(steps=steps)

        valid, confidence, details = validator.validate(chain)
        assert isinstance(valid, bool)

    def test_mathematical_validator_no_equation(self):
        """Test mathematical validator with missing equation."""
        validator = MathematicalValidator()

        statement = {
            "steps": ["x = 5"],
            "result": 5
        }

        valid, confidence, details = validator.validate(statement)
        assert isinstance(valid, bool)

    def test_mathematical_validator_empty_steps(self):
        """Test mathematical validator with no steps."""
        validator = MathematicalValidator()

        statement = {
            "equation": "x = 5",
            "steps": [],
            "result": 5
        }

        valid, confidence, details = validator.validate(statement)
        assert details["num_steps"] == 0

    def test_mathematical_validator_unbalanced_parens(self):
        """Test mathematical validator with unbalanced parentheses."""
        validator = MathematicalValidator()

        statement = {
            "equation": "((x + 5) = 10",
            "steps": [],
            "result": None
        }

        valid, confidence, details = validator.validate(statement)
        assert details["structure_valid"] is False

    def test_mathematical_validator_multiple_equals(self):
        """Test mathematical validator with multiple equals signs."""
        validator = MathematicalValidator()

        statement = {
            "equation": "x = y = 5",
            "steps": ["x = y = 5"],
            "result": 5
        }

        valid, confidence, details = validator.validate(statement)
        assert isinstance(valid, bool)

    def test_causal_validator_no_timestamps(self):
        """Test causal validator with events lacking timestamps."""
        validator = CausalValidator()

        chain = {
            "events": [
                {"id": 1, "name": "A"},
                {"id": 2, "name": "B"},
            ],
            "relationships": [{"cause_id": 1, "effect_id": 2}]
        }

        valid, confidence, details = validator.validate(chain)
        assert isinstance(valid, bool)

    def test_causal_validator_duplicate_event_ids(self):
        """Test causal validator with duplicate event IDs."""
        validator = CausalValidator()

        chain = {
            "events": [
                {"id": 1, "name": "A"},
                {"id": 1, "name": "B"},  # Duplicate ID
            ],
            "relationships": []
        }

        valid, confidence, details = validator.validate(chain)
        assert isinstance(valid, bool)

    def test_causal_validator_self_causation(self):
        """Test causal validator with self-causation."""
        validator = CausalValidator()

        chain = {
            "events": [{"id": 1, "name": "A"}],
            "relationships": [{"cause_id": 1, "effect_id": 1}]  # Self-loop
        }

        valid, confidence, details = validator.validate(chain)
        # Self-causation might be detected as circular
        assert isinstance(valid, bool)

    def test_causal_validator_missing_relationship_ids(self):
        """Test causal validator with incomplete relationship data."""
        validator = CausalValidator()

        chain = {
            "events": [{"id": 1, "name": "A"}],
            "relationships": [{"cause_id": None, "effect_id": None}]
        }

        valid, confidence, details = validator.validate(chain)
        assert isinstance(valid, bool)

    def test_consistency_validator_all_empty(self):
        """Test consistency validator with all empty components."""
        validator = ConsistencyValidator()

        composite = {
            "logical": ReasoningChain(steps=[]),
            "mathematical": {"equation": "", "steps": [], "result": None},
            "causal": {"events": [], "relationships": []}
        }

        valid, confidence, details = validator.validate(composite)
        assert isinstance(valid, bool)

    def test_consistency_validator_single_component(self):
        """Test consistency validator with only one component."""
        validator = ConsistencyValidator()

        composite = {
            "logical": ReasoningChain(
                steps=[ReasoningStep(step_id=0, statement="A")]
            )
        }

        valid, confidence, details = validator.validate(composite)
        assert details["num_components"] == 1


@pytest.mark.edge_case
class TestLayerEdgeCases:
    """Edge cases for FundamentalsLayer."""

    def test_layer_with_zero_cache_size(self):
        """Test layer with cache disabled (size 0)."""
        layer = FundamentalsLayer({"cache_size": 0})

        result1 = layer.validate("test")
        result2 = layer.validate("test")

        # Should not cache
        assert layer.stats["cache_hits"] == 0

    def test_layer_with_very_large_cache(self):
        """Test layer with very large cache size."""
        layer = FundamentalsLayer({"cache_size": 1000000})

        assert layer.cache_size == 1000000

    def test_layer_validate_with_all_none(self):
        """Test validation with None values for all parameters."""
        layer = FundamentalsLayer()

        result = layer.validate(None, domain=None, validation_type=None)
        assert isinstance(result, dict)
        assert "valid" in result

    def test_layer_cache_key_generation_consistency(self):
        """Test that cache keys are consistent."""
        layer = FundamentalsLayer()

        # Generate keys for same statement
        key1 = layer._generate_cache_key("test", "math", "mathematical")
        key2 = layer._generate_cache_key("test", "math", "mathematical")

        assert key1 == key2

    def test_layer_cache_key_generation_differences(self):
        """Test that different statements produce different keys."""
        layer = FundamentalsLayer()

        key1 = layer._generate_cache_key("test1", "math", "mathematical")
        key2 = layer._generate_cache_key("test2", "math", "mathematical")

        assert key1 != key2

    def test_layer_type_inference_with_ambiguous_statement(self):
        """Test type inference with ambiguous statement."""
        layer = FundamentalsLayer()

        # Statement that could be multiple types
        statement = {"data": "ambiguous"}

        inferred = layer._infer_validation_type(statement, None)
        assert inferred in ["mathematical", "logical", "causal", "consistency"]

    def test_layer_validation_with_circular_reference(self):
        """Test validation with circular data structure."""
        layer = FundamentalsLayer()

        # Create circular reference
        statement = {"a": None}
        statement["a"] = statement  # Circular reference

        # Should handle without infinite loop
        try:
            result = layer.validate(str(statement))
            assert isinstance(result, dict)
        except RecursionError:
            pytest.fail("Circular reference caused infinite recursion")

    def test_layer_statistics_with_no_validations(self):
        """Test statistics when no validations have been performed."""
        layer = FundamentalsLayer()

        stats = layer.get_statistics()
        assert stats["success_rate"] == 0.0
        assert stats["cache_hit_rate"] == 0.0

    def test_layer_export_stats_empty(self):
        """Test exporting statistics when empty."""
        layer = FundamentalsLayer()

        json_str = layer.export_stats()
        assert isinstance(json_str, str)
        assert "total_validations" in json_str

    def test_layer_multiple_resets(self):
        """Test multiple consecutive resets."""
        layer = FundamentalsLayer()

        layer.validate("test")
        layer.reset_statistics()
        layer.reset_statistics()
        layer.reset_statistics()

        stats = layer.get_statistics()
        assert stats["total_validations"] == 0

    def test_layer_health_check_after_errors(self):
        """Test health check after validation errors."""
        layer = FundamentalsLayer()

        # Cause some errors
        try:
            layer.validate({"malformed": "data"})
        except:
            pass

        health = layer.health_check()
        assert "status" in health


@pytest.mark.edge_case
class TestBoundaryConditions:
    """Test boundary conditions and limits."""

    def test_very_large_number_conservation(self):
        """Test conservation with very large numbers."""
        principle = ConservationPrinciple()

        statement = {
            "before": sys.maxsize,
            "after": sys.maxsize
        }
        result = principle.validate(statement)
        assert result.valid is True

    def test_very_small_number_conservation(self):
        """Test conservation with very small numbers."""
        principle = ConservationPrinciple()

        statement = {
            "before": sys.float_info.min,
            "after": sys.float_info.min
        }
        result = principle.validate(statement)
        assert result.valid is True

    def test_maximum_recursion_depth(self):
        """Test handling of deeply nested structures."""
        layer = FundamentalsLayer()

        # Create deeply nested structure
        nested = {"value": 1}
        current = nested
        for i in range(100):
            current["next"] = {"value": i}
            current = current["next"]

        # Should handle without stack overflow
        result = layer.validate(nested)
        assert isinstance(result, dict)

    def test_very_long_event_sequence(self):
        """Test causal validation with very long event sequence."""
        validator = CausalValidator()

        events = [{"id": i, "name": f"event{i}", "timestamp": i} for i in range(1000)]
        chain = {
            "events": events,
            "relationships": []
        }

        valid, confidence, details = validator.validate(chain)
        assert details["num_events"] == 1000

    def test_zero_confidence_handling(self):
        """Test handling of zero confidence results."""
        layer = FundamentalsLayer()

        result = layer.validate({"equation": "invalid"}, domain="mathematical")

        assert "confidence" in result
        assert result["confidence"] >= 0.0


@pytest.mark.edge_case
class TestSpecialCharacters:
    """Test handling of special characters and encodings."""

    def test_unicode_in_statements(self):
        """Test validation with unicode characters."""
        layer = FundamentalsLayer()

        statements = [
            "测试中文",
            "Тест на русском",
            "اختبار بالعربية",
            "🔥 emoji test 🎉",
            "Ελληνικά",
        ]

        for stmt in statements:
            result = layer.validate(stmt)
            assert isinstance(result, dict)
            assert "valid" in result

    def test_special_math_symbols(self):
        """Test mathematical validator with special symbols."""
        validator = MathematicalValidator()

        statement = {
            "equation": "x² + 2x + 1 = 0",
            "steps": ["x² + 2x + 1 = 0"],
            "result": None
        }

        valid, confidence, details = validator.validate(statement)
        assert isinstance(valid, bool)

    def test_newlines_in_statements(self):
        """Test handling of multi-line statements."""
        layer = FundamentalsLayer()

        statement = "Line 1\nLine 2\nLine 3"
        result = layer.validate(statement)

        assert isinstance(result, dict)

    def test_tabs_and_whitespace(self):
        """Test handling of tabs and whitespace."""
        layer = FundamentalsLayer()

        statement = "test\t\twith\t\ttabs"
        result = layer.validate(statement)

        assert isinstance(result, dict)


@pytest.mark.edge_case
class TestConcurrencyAndState:
    """Test state management and potential concurrency issues."""

    def test_multiple_validations_preserve_history(self):
        """Test that validation history is preserved correctly."""
        validator = LogicalValidator()

        for i in range(10):
            chain = ReasoningChain(
                steps=[ReasoningStep(step_id=0, statement=f"Statement {i}")]
            )
            validator.validate(chain)

        history = validator.get_validation_history()
        assert len(history) == 10

    def test_statistics_accuracy_under_load(self):
        """Test statistics remain accurate under multiple validations."""
        layer = FundamentalsLayer()

        # Perform many validations
        for i in range(50):
            layer.validate(f"test {i}")

        stats = layer.get_statistics()
        assert stats["total_validations"] == 50
        assert stats["cache_hits"] + stats["cache_misses"] == 50

    def test_cache_consistency(self):
        """Test that cache remains consistent."""
        layer = FundamentalsLayer({"cache_size": 10})

        # Add items to cache
        for i in range(20):
            layer.validate(f"test {i}")

        # Cache should not exceed size
        assert len(layer.validation_cache) <= layer.cache_size

        # Items should still be retrievable
        result = layer.validate("test 19")
        assert "valid" in result


@pytest.mark.edge_case
class TestMemoryAndPerformance:
    """Test memory usage and performance characteristics."""

    def test_large_validation_cache_cleanup(self):
        """Test that cache cleanup works correctly."""
        layer = FundamentalsLayer({"cache_size": 100})

        # Fill cache beyond capacity
        for i in range(200):
            layer.validate(f"statement {i}")

        # Cache should be at capacity
        assert len(layer.validation_cache) <= layer.cache_size

    def test_validation_time_tracking_accuracy(self):
        """Test that validation times are tracked accurately."""
        layer = FundamentalsLayer()

        layer.validate("test 1")
        layer.validate("test 2")

        stats = layer.get_statistics()
        assert len(stats["validation_times"]) == 2
        assert all(t > 0 for t in stats["validation_times"])

    def test_principle_statistics_accumulation(self):
        """Test that principle statistics accumulate correctly."""
        principles = LogicalPrinciples()

        for i in range(10):
            principles.validate_all(i)

        stats = principles.get_statistics()
        for principle_stats in stats.values():
            assert principle_stats["validation_count"] == 10
