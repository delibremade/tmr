"""
Unit tests for logical principles.

Tests all five core logical principles:
- Identity
- Non-Contradiction
- Excluded Middle
- Causality
- Conservation
"""

import pytest
from tmr.fundamentals.principles import (
    IdentityPrinciple,
    NonContradictionPrinciple,
    ExcludedMiddlePrinciple,
    CausalityPrinciple,
    ConservationPrinciple,
    LogicalPrinciples,
    PrincipleType,
    ValidationResult,
)


class TestIdentityPrinciple:
    """Test the Identity Principle (A = A)."""

    @pytest.fixture
    def principle(self):
        return IdentityPrinciple()

    @pytest.mark.unit
    def test_identity_simple_types(self, principle):
        """Test identity for simple types."""
        # Test integers
        result = principle.validate(42)
        assert result.valid is True
        assert result.confidence == 1.0
        assert result.principle == PrincipleType.IDENTITY

        # Test strings
        result = principle.validate("hello")
        assert result.valid is True
        assert result.confidence == 1.0

        # Test floats
        result = principle.validate(3.14)
        assert result.valid is True
        assert result.confidence == 1.0

        # Test booleans
        result = principle.validate(True)
        assert result.valid is True
        assert result.confidence == 1.0

    @pytest.mark.unit
    def test_identity_entity_reference(self, principle):
        """Test identity with entity-reference structure."""
        statement = {
            "entity": "Alice",
            "reference": "Alice"
        }
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 0.9

        # Test non-matching entities
        statement = {
            "entity": "Alice",
            "reference": "Bob"
        }
        result = principle.validate(statement)
        assert result.valid is False
        assert result.confidence == 0.1

    @pytest.mark.unit
    def test_identity_case_insensitive(self, principle):
        """Test semantic similarity with case differences."""
        statement = {
            "entity": "ALICE",
            "reference": "alice"
        }
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 0.9

    @pytest.mark.unit
    def test_identity_list_elements(self, principle):
        """Test identity for list elements."""
        result = principle.validate([1, 2, 3, 4])
        assert result.valid is True
        assert result.confidence == 0.95

        result = principle.validate(["a", "b", "c"])
        assert result.valid is True
        assert result.confidence == 0.95

    @pytest.mark.unit
    def test_identity_tuple_elements(self, principle):
        """Test identity for tuple elements."""
        result = principle.validate((1, 2, 3))
        assert result.valid is True
        assert result.confidence == 0.95

    @pytest.mark.unit
    def test_identity_dict_without_entity(self, principle):
        """Test identity for dict without entity/reference keys."""
        statement = {"key": "value"}
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 0.5

    @pytest.mark.unit
    def test_identity_statistics(self, principle):
        """Test statistics tracking."""
        principle.validate(42)
        principle.validate("test")
        principle.validate([1, 2, 3])

        assert principle.validation_count == 3
        assert principle.success_count == 3
        assert principle.success_rate == 1.0

        # Add a failed validation
        principle.validate({"entity": "A", "reference": "B"})
        assert principle.validation_count == 4
        assert principle.success_count == 3
        assert principle.success_rate == 0.75

    @pytest.mark.unit
    def test_identity_reset_stats(self, principle):
        """Test statistics reset."""
        principle.validate(42)
        principle.validate("test")
        principle.reset_stats()

        assert principle.validation_count == 0
        assert principle.success_count == 0
        assert principle.success_rate == 0.0


class TestNonContradictionPrinciple:
    """Test the Non-Contradiction Principle (¬(P ∧ ¬P))."""

    @pytest.fixture
    def principle(self):
        return NonContradictionPrinciple()

    @pytest.mark.unit
    def test_non_contradiction_simple_case(self, principle):
        """Test basic non-contradiction."""
        statement = {
            "claim_true": True,
            "negation_true": False
        }
        result = principle.validate(statement)
        assert result.valid is True
        # Default confidence for unknown structure is 0.4
        assert result.confidence == 0.4
        assert result.principle == PrincipleType.NON_CONTRADICTION

    @pytest.mark.unit
    def test_contradiction_detected(self, principle):
        """Test that contradictions are detected."""
        statement = {
            "claim": "P",
            "negation": "not P",
            "claim_true": True,
            "negation_true": True
        }
        result = principle.validate(statement)
        assert result.valid is False
        assert result.confidence == 0.0

    @pytest.mark.unit
    def test_propositions_no_contradiction(self, principle):
        """Test multiple propositions without contradiction."""
        statement = {
            "propositions": [
                {"subject": "sky", "predicate": "blue"},
                {"subject": "grass", "predicate": "green"},
            ]
        }
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 0.95

    @pytest.mark.unit
    def test_propositions_with_contradiction(self, principle):
        """Test multiple propositions with contradiction."""
        statement = {
            "propositions": [
                {"subject": "sky", "predicate": "blue"},
                {"subject": "sky", "predicate": "not blue"},
            ]
        }
        result = principle.validate(statement)
        assert result.valid is False
        assert result.confidence == 0.05

    @pytest.mark.unit
    def test_list_consistency(self, principle):
        """Test list consistency check."""
        statement = ["P", "Q", "R"]
        result = principle.validate(statement)
        # Currently placeholder returns True
        assert result.valid is True
        assert result.confidence == 0.9

    @pytest.mark.unit
    def test_non_contradiction_statistics(self, principle):
        """Test statistics tracking."""
        principle.validate({"claim_true": True, "negation_true": False})
        principle.validate({"claim_true": True, "negation_true": False})

        assert principle.validation_count == 2
        assert principle.success_count == 2
        assert principle.success_rate == 1.0


class TestExcludedMiddlePrinciple:
    """Test the Excluded Middle Principle (P ∨ ¬P)."""

    @pytest.fixture
    def principle(self):
        return ExcludedMiddlePrinciple()

    @pytest.mark.unit
    def test_excluded_middle_true_value(self, principle):
        """Test with truth value True."""
        statement = {
            "proposition": "The sky is blue",
            "truth_value": True
        }
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 1.0
        assert result.principle == PrincipleType.EXCLUDED_MIDDLE

    @pytest.mark.unit
    def test_excluded_middle_false_value(self, principle):
        """Test with truth value False."""
        statement = {
            "proposition": "The sky is green",
            "truth_value": False
        }
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 1.0

    @pytest.mark.unit
    def test_excluded_middle_string_values(self, principle):
        """Test with string truth values."""
        statement = {
            "proposition": "Test",
            "truth_value": "true"
        }
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 1.0

        statement["truth_value"] = "false"
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 1.0

    @pytest.mark.unit
    def test_excluded_middle_numeric_values(self, principle):
        """Test with numeric truth values (1 and 0)."""
        statement = {
            "proposition": "Test",
            "truth_value": 1
        }
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 1.0

        statement["truth_value"] = 0
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 1.0

    @pytest.mark.unit
    def test_excluded_middle_invalid_value(self, principle):
        """Test with invalid truth value."""
        statement = {
            "proposition": "Test",
            "truth_value": "maybe"
        }
        result = principle.validate(statement)
        assert result.valid is False
        assert result.confidence == 0.0

    @pytest.mark.unit
    def test_excluded_middle_missing_proposition(self, principle):
        """Test with missing proposition key."""
        statement = {"some_key": "value"}
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 0.4

    @pytest.mark.unit
    def test_excluded_middle_statistics(self, principle):
        """Test statistics tracking."""
        principle.validate({"proposition": "P", "truth_value": True})
        principle.validate({"proposition": "Q", "truth_value": False})
        principle.validate({"proposition": "R", "truth_value": "maybe"})

        assert principle.validation_count == 3
        assert principle.success_count == 2
        assert principle.success_rate == pytest.approx(2/3)


class TestCausalityPrinciple:
    """Test the Causality Principle."""

    @pytest.fixture
    def principle(self):
        return CausalityPrinciple()

    @pytest.mark.unit
    def test_causality_basic_cause_effect(self, principle):
        """Test basic cause-effect validation."""
        statement = {
            "cause": "rain",
            "effect": "wet ground"
        }
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 0.9
        assert result.principle == PrincipleType.CAUSALITY

    @pytest.mark.unit
    def test_causality_with_timestamps(self, principle):
        """Test temporal ordering with timestamps."""
        statement = {
            "cause": "event A",
            "effect": "event B",
            "timestamp_cause": 100,
            "timestamp_effect": 200
        }
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 0.9

    @pytest.mark.unit
    def test_causality_reversed_timestamps(self, principle):
        """Test that reversed timestamps are detected."""
        statement = {
            "cause": "event A",
            "effect": "event B",
            "timestamp_cause": 200,
            "timestamp_effect": 100
        }
        result = principle.validate(statement)
        assert result.valid is False
        assert result.confidence == 0.1

    @pytest.mark.unit
    def test_causality_event_sequence(self, principle):
        """Test event sequence validation."""
        statement = {
            "events": [
                {"name": "event1", "timestamp": 100},
                {"name": "event2", "timestamp": 200},
                {"name": "event3", "timestamp": 300},
            ]
        }
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 0.85

    @pytest.mark.unit
    def test_causality_unordered_events(self, principle):
        """Test unordered event sequence."""
        statement = {
            "events": [
                {"name": "event1", "timestamp": 100},
                {"name": "event2", "timestamp": 300},
                {"name": "event3", "timestamp": 200},
            ]
        }
        result = principle.validate(statement)
        assert result.valid is False
        assert result.confidence == 0.15

    @pytest.mark.unit
    def test_causality_empty_events(self, principle):
        """Test empty event sequence."""
        statement = {"events": []}
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 0.85

    @pytest.mark.unit
    def test_causality_missing_cause(self, principle):
        """Test validation with missing cause."""
        statement = {
            "cause": None,
            "effect": "something"
        }
        result = principle.validate(statement)
        assert result.valid is False
        assert result.confidence == 0.1

    @pytest.mark.unit
    def test_causality_statistics(self, principle):
        """Test statistics tracking."""
        principle.validate({"cause": "A", "effect": "B"})
        principle.validate({"cause": "C", "effect": "D"})

        assert principle.validation_count == 2
        assert principle.success_count == 2
        assert principle.success_rate == 1.0


class TestConservationPrinciple:
    """Test the Conservation Principle."""

    @pytest.fixture
    def principle(self):
        return ConservationPrinciple()

    @pytest.mark.unit
    def test_conservation_numeric_equality(self, principle):
        """Test conservation of numeric values."""
        statement = {
            "before": 100,
            "after": 100
        }
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 0.95
        assert result.principle == PrincipleType.CONSERVATION

    @pytest.mark.unit
    def test_conservation_numeric_inequality(self, principle):
        """Test detection of non-conservation."""
        statement = {
            "before": 100,
            "after": 50
        }
        result = principle.validate(statement)
        assert result.valid is False
        assert result.confidence == 0.05

    @pytest.mark.unit
    def test_conservation_numeric_tolerance(self, principle):
        """Test numeric tolerance for floating point."""
        statement = {
            "before": 1.0000001,
            "after": 1.0000002
        }
        result = principle.validate(statement)
        # Should be valid due to 1e-6 tolerance
        assert result.valid is True
        assert result.confidence == 0.95

    @pytest.mark.unit
    def test_conservation_list_count(self, principle):
        """Test conservation of list element count."""
        statement = {
            "before": [1, 2, 3, 4],
            "after": [5, 6, 7, 8],
            "conserved_quantity": "count"
        }
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 0.95

    @pytest.mark.unit
    def test_conservation_list_sum(self, principle):
        """Test conservation of list sum."""
        statement = {
            "before": [1, 2, 3, 4],
            "after": [2, 3, 2, 3],
            "conserved_quantity": "sum"
        }
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 0.95

    @pytest.mark.unit
    def test_conservation_list_sum_violation(self, principle):
        """Test detection of sum non-conservation."""
        statement = {
            "before": [1, 2, 3, 4],  # sum = 10
            "after": [1, 1, 1, 1],   # sum = 4
            "conserved_quantity": "sum"
        }
        result = principle.validate(statement)
        assert result.valid is False
        assert result.confidence == 0.05

    @pytest.mark.unit
    def test_conservation_dict_property(self, principle):
        """Test conservation of dictionary property."""
        statement = {
            "before": {"energy": 100, "mass": 10},
            "after": {"energy": 100, "mass": 8},
            "conserved_quantity": "energy"
        }
        result = principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 0.95

    @pytest.mark.unit
    def test_conservation_statistics(self, principle):
        """Test statistics tracking."""
        principle.validate({"before": 10, "after": 10})
        principle.validate({"before": 20, "after": 20})
        principle.validate({"before": 30, "after": 50})

        assert principle.validation_count == 3
        assert principle.success_count == 2
        assert principle.success_rate == pytest.approx(2/3)


class TestLogicalPrinciples:
    """Test the LogicalPrinciples collection."""

    @pytest.fixture
    def principles(self):
        return LogicalPrinciples()

    @pytest.mark.unit
    def test_all_principles_initialized(self, principles):
        """Test that all principles are initialized."""
        assert PrincipleType.IDENTITY in principles.principles
        assert PrincipleType.NON_CONTRADICTION in principles.principles
        assert PrincipleType.EXCLUDED_MIDDLE in principles.principles
        assert PrincipleType.CAUSALITY in principles.principles
        assert PrincipleType.CONSERVATION in principles.principles

    @pytest.mark.unit
    def test_validate_all(self, principles):
        """Test validating against all principles."""
        statement = 42
        results = principles.validate_all(statement)

        assert len(results) == 5
        assert all(isinstance(r, ValidationResult) for r in results.values())

    @pytest.mark.unit
    def test_validate_specific(self, principles):
        """Test validating against specific principles."""
        statement = 42
        principle_types = [PrincipleType.IDENTITY, PrincipleType.NON_CONTRADICTION]
        results = principles.validate_specific(statement, principle_types)

        assert len(results) == 2
        assert PrincipleType.IDENTITY in results
        assert PrincipleType.NON_CONTRADICTION in results

    @pytest.mark.unit
    def test_aggregate_validity_all_valid(self, principles):
        """Test aggregate validity when all principles pass."""
        statement = 42
        results = principles.validate_all(statement)
        is_valid, confidence = principles.get_aggregate_validity(results)

        assert is_valid is True
        assert confidence > 0.0

    @pytest.mark.unit
    def test_aggregate_validity_some_invalid(self, principles):
        """Test aggregate validity with some failures."""
        results = {
            PrincipleType.IDENTITY: ValidationResult(True, PrincipleType.IDENTITY, 1.0),
            PrincipleType.NON_CONTRADICTION: ValidationResult(False, PrincipleType.NON_CONTRADICTION, 0.0),
        }
        is_valid, confidence = principles.get_aggregate_validity(results)

        assert is_valid is False
        assert confidence == 0.5

    @pytest.mark.unit
    def test_aggregate_validity_empty(self, principles):
        """Test aggregate validity with empty results."""
        is_valid, confidence = principles.get_aggregate_validity({})
        assert is_valid is True
        assert confidence == 0.0

    @pytest.mark.unit
    def test_get_statistics(self, principles):
        """Test statistics retrieval."""
        # Perform some validations
        principles.validate_all(42)
        principles.validate_all("test")

        stats = principles.get_statistics()
        assert len(stats) == 5
        for principle_name, principle_stats in stats.items():
            assert "validation_count" in principle_stats
            assert "success_count" in principle_stats
            assert "success_rate" in principle_stats
            assert principle_stats["validation_count"] == 2

    @pytest.mark.unit
    def test_reset_all_stats(self, principles):
        """Test resetting all statistics."""
        # Perform some validations
        principles.validate_all(42)
        principles.validate_all("test")

        # Reset
        principles.reset_all_stats()

        # Verify all reset
        stats = principles.get_statistics()
        for principle_stats in stats.values():
            assert principle_stats["validation_count"] == 0
            assert principle_stats["success_count"] == 0
            assert principle_stats["success_rate"] == 0.0


@pytest.mark.edge_case
class TestPrincipleEdgeCases:
    """Test edge cases across all principles."""

    @pytest.mark.unit
    def test_none_input(self):
        """Test handling of None input."""
        principle = IdentityPrinciple()
        result = principle.validate(None)
        # Should handle gracefully
        assert isinstance(result, ValidationResult)

    @pytest.mark.unit
    def test_empty_dict(self):
        """Test handling of empty dictionary."""
        principles = [
            IdentityPrinciple(),
            NonContradictionPrinciple(),
            ExcludedMiddlePrinciple(),
            CausalityPrinciple(),
            ConservationPrinciple(),
        ]
        for principle in principles:
            result = principle.validate({})
            assert isinstance(result, ValidationResult)

    @pytest.mark.unit
    def test_empty_list(self):
        """Test handling of empty list."""
        principle = IdentityPrinciple()
        result = principle.validate([])
        assert result.valid is True

    @pytest.mark.unit
    def test_nested_structures(self):
        """Test handling of nested data structures."""
        principle = IdentityPrinciple()
        statement = {
            "entity": {"nested": {"value": "A"}},
            "reference": {"nested": {"value": "A"}}
        }
        result = principle.validate(statement)
        assert isinstance(result, ValidationResult)

    @pytest.mark.unit
    def test_unicode_strings(self):
        """Test handling of unicode strings."""
        principle = IdentityPrinciple()
        result = principle.validate("こんにちは")
        assert result.valid is True
        assert result.confidence == 1.0

    @pytest.mark.unit
    def test_very_large_numbers(self):
        """Test handling of very large numbers."""
        principle = ConservationPrinciple()
        statement = {
            "before": 10**100,
            "after": 10**100
        }
        result = principle.validate(statement)
        assert result.valid is True

    @pytest.mark.unit
    def test_negative_numbers(self):
        """Test handling of negative numbers."""
        principle = ConservationPrinciple()
        statement = {
            "before": -100,
            "after": -100
        }
        result = principle.validate(statement)
        assert result.valid is True
