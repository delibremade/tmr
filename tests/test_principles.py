"""
Comprehensive tests for logical principles in the Fundamentals Layer.
Tests 100% coverage of identity, non-contradiction, excluded middle, causality, and conservation.
"""

import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    """Tests for the Law of Identity (A = A)"""

    def setup_method(self):
        self.principle = IdentityPrinciple()

    def test_simple_identity_valid(self):
        """Test that simple values are identical to themselves"""
        result = self.principle.validate("test")
        assert result.valid is True
        assert result.confidence == 1.0

    def test_numeric_identity(self):
        """Test numeric identity"""
        for value in [42, 3.14, 0, -100]:
            result = self.principle.validate(value)
            assert result.valid is True
            assert result.confidence == 1.0

    def test_boolean_identity(self):
        """Test boolean identity"""
        for value in [True, False]:
            result = self.principle.validate(value)
            assert result.valid is True

    def test_entity_reference_identity(self):
        """Test entity-reference identity checking"""
        statement = {"entity": "Python", "reference": "Python"}
        result = self.principle.validate(statement)
        assert result.valid is True

    def test_semantic_similarity_case_insensitive(self):
        """Test semantic similarity with case differences"""
        statement = {"entity": "HELLO", "reference": "hello"}
        result = self.principle.validate(statement)
        assert result.valid is True

    def test_semantic_similarity_similar_strings(self):
        """Test semantic similarity for similar strings"""
        # Levenshtein distance should detect similarity
        sim = self.principle._semantic_similarity("kitten", "sitting")
        assert sim > 0.5  # Should be somewhat similar

    def test_semantic_similarity_exact_match(self):
        """Test exact string match gives perfect score"""
        sim = self.principle._semantic_similarity("identical", "identical")
        assert sim == 1.0

    def test_semantic_similarity_case_match(self):
        """Test case-insensitive match"""
        sim = self.principle._semantic_similarity("Test", "test")
        assert sim >= 0.98

    def test_jaccard_similarity(self):
        """Test Jaccard similarity for word tokens"""
        sim = self.principle._jaccard_similarity("the quick brown fox", "the brown fox")
        assert sim > 0.5  # Significant overlap

    def test_lcs_similarity(self):
        """Test longest common subsequence similarity"""
        sim = self.principle._lcs_similarity("abcdef", "acdxyz")
        assert sim > 0.3  # Should find 'acd' as common subsequence

    def test_levenshtein_similarity(self):
        """Test Levenshtein distance similarity"""
        # Identical strings
        sim = self.principle._levenshtein_similarity("test", "test")
        assert sim == 1.0

        # One character difference
        sim = self.principle._levenshtein_similarity("test", "best")
        assert sim > 0.7

    def test_list_identity(self):
        """Test identity for lists"""
        result = self.principle.validate([1, 2, 3, "test", True])
        assert result.valid is True

    def test_statistics_tracking(self):
        """Test that validation statistics are tracked"""
        initial_count = self.principle.validation_count
        self.principle.validate("test")
        assert self.principle.validation_count == initial_count + 1
        assert self.principle.success_count > 0


class TestNonContradictionPrinciple:
    """Tests for the Law of Non-Contradiction (¬(P ∧ ¬P))"""

    def setup_method(self):
        self.principle = NonContradictionPrinciple()

    def test_no_contradiction_simple(self):
        """Test simple non-contradictory statement"""
        statement = {"claim": "It is raining", "claim_true": True, "negation_true": False}
        result = self.principle.validate(statement)
        assert result.valid is True

    def test_contradiction_detected(self):
        """Test that contradictions are detected"""
        statement = {"claim": "It is raining", "claim_true": True, "negation_true": True}
        result = self.principle.validate(statement)
        assert result.valid is False

    def test_propositions_no_contradiction(self):
        """Test multiple propositions without contradiction"""
        propositions = [
            {"subject": "sky", "predicate": "blue"},
            {"subject": "grass", "predicate": "green"},
        ]
        statement = {"propositions": propositions}
        result = self.principle.validate(statement)
        assert result.valid is True

    def test_propositions_with_contradiction(self):
        """Test detection of contradictory propositions"""
        propositions = [
            {"subject": "sky", "predicate": "blue"},
            {"subject": "sky", "predicate": "not blue"},
        ]
        statement = {"propositions": propositions}
        result = self.principle.validate(statement)
        assert result.valid is False

    def test_list_consistency_valid(self):
        """Test consistent list of statements"""
        statements = ["The sky is blue", "The grass is green", "Water is wet"]
        result = self.principle.validate(statements)
        assert result.valid is True

    def test_list_consistency_contradiction(self):
        """Test contradictory statements in list"""
        statements = ["The sky is blue", "The sky is not blue"]
        result = self.principle.validate(statements)
        assert result.valid is False

    def test_text_contradiction_negation(self):
        """Test detection of text contradictions with negation"""
        assert self.principle._text_contradicts("The answer is yes", "The answer is no")
        assert self.principle._text_contradicts("It is true", "It is false")
        assert self.principle._text_contradicts("Always happens", "Never happens")

    def test_text_contradiction_antonyms(self):
        """Test detection of antonym-based contradictions"""
        assert self.principle._text_contradicts(
            "Temperature will increase", "Temperature will decrease"
        )
        assert self.principle._text_contradicts("Value is greater", "Value is less")

    def test_text_contradiction_verb_negation(self):
        """Test detection of verb negation contradictions"""
        assert self.principle._text_contradicts("He is happy", "He is not happy")
        assert self.principle._text_contradicts("She can swim", "She cannot swim")

    def test_no_false_positives(self):
        """Test that non-contradictory statements aren't flagged"""
        assert not self.principle._text_contradicts(
            "The car is red", "The house is blue"
        )
        assert not self.principle._text_contradicts(
            "I like apples", "I like oranges"
        )

    def test_statements_contradict_mixed_types(self):
        """Test contradiction detection across different statement types"""
        stmt1 = {"text": "It is true", "type": "string"}
        stmt2 = {"text": "It is false", "type": "string"}
        assert self.principle._statements_contradict(stmt1, stmt2)


class TestExcludedMiddlePrinciple:
    """Tests for the Law of Excluded Middle (P ∨ ¬P)"""

    def setup_method(self):
        self.principle = ExcludedMiddlePrinciple()

    def test_definite_true_value(self):
        """Test that definite true value is valid"""
        statement = {"proposition": "The sky is blue", "truth_value": True}
        result = self.principle.validate(statement)
        assert result.valid is True
        assert result.confidence == 1.0

    def test_definite_false_value(self):
        """Test that definite false value is valid"""
        statement = {"proposition": "2 + 2 = 5", "truth_value": False}
        result = self.principle.validate(statement)
        assert result.valid is True

    def test_string_true_value(self):
        """Test string representation of truth values"""
        statement = {"proposition": "Test", "truth_value": "true"}
        result = self.principle.validate(statement)
        assert result.valid is True

    def test_numeric_truth_values(self):
        """Test numeric truth values (1, 0)"""
        for value in [1, 0]:
            statement = {"proposition": "Test", "truth_value": value}
            result = self.principle.validate(statement)
            assert result.valid is True

    def test_undefined_truth_value(self):
        """Test that undefined truth value is invalid"""
        statement = {"proposition": "Test", "truth_value": None}
        result = self.principle.validate(statement)
        assert result.valid is False

    def test_invalid_truth_value(self):
        """Test that invalid truth value is rejected"""
        statement = {"proposition": "Test", "truth_value": "maybe"}
        result = self.principle.validate(statement)
        assert result.valid is False


class TestCausalityPrinciple:
    """Tests for the Principle of Causality"""

    def setup_method(self):
        self.principle = CausalityPrinciple()

    def test_valid_causal_relationship(self):
        """Test valid cause-effect relationship"""
        statement = {
            "cause": "Rain",
            "effect": "Wet ground",
            "timestamp_cause": 1.0,
            "timestamp_effect": 2.0,
        }
        result = self.principle.validate(statement)
        assert result.valid is True

    def test_invalid_temporal_order(self):
        """Test that effect cannot precede cause"""
        statement = {
            "cause": "Rain",
            "effect": "Wet ground",
            "timestamp_cause": 2.0,
            "timestamp_effect": 1.0,
        }
        result = self.principle.validate(statement)
        assert result.valid is False

    def test_simultaneous_events(self):
        """Test events at same timestamp"""
        statement = {
            "cause": "Lightning",
            "effect": "Thunder",
            "timestamp_cause": 1.0,
            "timestamp_effect": 1.0,
        }
        # Simultaneous is invalid (cause must strictly precede)
        result = self.principle.validate(statement)
        assert result.valid is False

    def test_event_sequence_ordered(self):
        """Test properly ordered event sequence"""
        events = [
            {"name": "Event A", "timestamp": 1.0},
            {"name": "Event B", "timestamp": 2.0},
            {"name": "Event C", "timestamp": 3.0},
        ]
        statement = {"events": events}
        result = self.principle.validate(statement)
        assert result.valid is True

    def test_event_sequence_unordered(self):
        """Test improperly ordered event sequence"""
        events = [
            {"name": "Event A", "timestamp": 3.0},
            {"name": "Event B", "timestamp": 1.0},
            {"name": "Event C", "timestamp": 2.0},
        ]
        statement = {"events": events}
        result = self.principle.validate(statement)
        assert result.valid is False

    def test_empty_events(self):
        """Test empty event sequence"""
        statement = {"events": []}
        result = self.principle.validate(statement)
        assert result.valid is True

    def test_causal_relationship_without_timestamps(self):
        """Test causal relationship without temporal information"""
        statement = {"cause": "Fire", "effect": "Smoke"}
        result = self.principle.validate(statement)
        assert result.valid is True  # Valid if both exist


class TestConservationPrinciple:
    """Tests for Conservation Principles"""

    def setup_method(self):
        self.principle = ConservationPrinciple()

    def test_numeric_conservation(self):
        """Test conservation of numeric values"""
        statement = {"before": 100, "after": 100, "conserved_quantity": "total"}
        result = self.principle.validate(statement)
        assert result.valid is True
        assert result.confidence >= 0.95

    def test_numeric_non_conservation(self):
        """Test detection of non-conservation"""
        statement = {"before": 100, "after": 50, "conserved_quantity": "total"}
        result = self.principle.validate(statement)
        assert result.valid is False

    def test_floating_point_tolerance(self):
        """Test that small floating point errors are tolerated"""
        statement = {"before": 1.0, "after": 1.0000000001, "conserved_quantity": "total"}
        result = self.principle.validate(statement)
        assert result.valid is True

    def test_list_count_conservation(self):
        """Test conservation of list count"""
        statement = {
            "before": [1, 2, 3, 4],
            "after": [5, 6, 7, 8],
            "conserved_quantity": "count",
        }
        result = self.principle.validate(statement)
        assert result.valid is True

    def test_list_sum_conservation(self):
        """Test conservation of list sum"""
        statement = {
            "before": [1, 2, 3, 4],
            "after": [10],
            "conserved_quantity": "sum",
        }
        result = self.principle.validate(statement)
        assert result.valid is True

    def test_list_sum_non_conservation(self):
        """Test detection of sum non-conservation"""
        statement = {
            "before": [1, 2, 3],
            "after": [10],
            "conserved_quantity": "sum",
        }
        result = self.principle.validate(statement)
        assert result.valid is False

    def test_dict_property_conservation(self):
        """Test conservation of dict properties"""
        statement = {
            "before": {"energy": 100, "other": 50},
            "after": {"energy": 100, "other": 75},
            "conserved_quantity": "energy",
        }
        result = self.principle.validate(statement)
        assert result.valid is True


class TestLogicalPrinciples:
    """Tests for the LogicalPrinciples collection"""

    def setup_method(self):
        self.principles = LogicalPrinciples()

    def test_all_principles_initialized(self):
        """Test that all 5 principles are initialized"""
        assert len(self.principles.principles) == 5
        assert PrincipleType.IDENTITY in self.principles.principles
        assert PrincipleType.NON_CONTRADICTION in self.principles.principles
        assert PrincipleType.EXCLUDED_MIDDLE in self.principles.principles
        assert PrincipleType.CAUSALITY in self.principles.principles
        assert PrincipleType.CONSERVATION in self.principles.principles

    def test_validate_all(self):
        """Test validation against all principles"""
        statement = "test"
        results = self.principles.validate_all(statement)
        assert len(results) == 5
        assert all(isinstance(r, ValidationResult) for r in results.values())

    def test_validate_specific(self):
        """Test validation against specific principles"""
        statement = "test"
        principle_types = [PrincipleType.IDENTITY, PrincipleType.NON_CONTRADICTION]
        results = self.principles.validate_specific(statement, principle_types)
        assert len(results) == 2
        assert PrincipleType.IDENTITY in results
        assert PrincipleType.NON_CONTRADICTION in results

    def test_aggregate_validity_all_valid(self):
        """Test aggregate validity when all principles pass"""
        statement = "test"
        results = self.principles.validate_all(statement)
        is_valid, confidence = self.principles.get_aggregate_validity(results)
        assert is_valid is True
        assert 0.0 <= confidence <= 1.0

    def test_aggregate_validity_some_invalid(self):
        """Test aggregate validity when some principles fail"""
        # Create a contradictory statement
        statement = {
            "claim": "Test",
            "claim_true": True,
            "negation_true": True,  # Contradiction
        }
        results = self.principles.validate_all(statement)
        is_valid, confidence = self.principles.get_aggregate_validity(results)
        # Should be invalid due to contradiction
        assert is_valid is False

    def test_statistics(self):
        """Test statistics collection"""
        self.principles.validate_all("test1")
        self.principles.validate_all("test2")

        stats = self.principles.get_statistics()
        assert len(stats) == 5

        for principle_stats in stats.values():
            assert "validation_count" in principle_stats
            assert "success_count" in principle_stats
            assert "success_rate" in principle_stats
            assert principle_stats["validation_count"] == 2

    def test_reset_statistics(self):
        """Test resetting statistics"""
        self.principles.validate_all("test")
        self.principles.reset_all_stats()

        stats = self.principles.get_statistics()
        for principle_stats in stats.values():
            assert principle_stats["validation_count"] == 0
            assert principle_stats["success_count"] == 0


class TestPrincipleIntegration:
    """Integration tests for principle interactions"""

    def test_conservation_with_identity(self):
        """Test that conservation respects identity"""
        principles = LogicalPrinciples()

        # A conserved value should maintain identity
        conservation_stmt = {"before": 42, "after": 42}
        identity_stmt = {"entity": 42, "reference": 42}

        cons_result = principles.principles[PrincipleType.CONSERVATION].validate(
            conservation_stmt
        )
        id_result = principles.principles[PrincipleType.IDENTITY].validate(identity_stmt)

        assert cons_result.valid is True
        assert id_result.valid is True

    def test_causality_with_non_contradiction(self):
        """Test that causal chains don't contradict"""
        principles = LogicalPrinciples()

        # Events in causal order shouldn't contradict
        causal_stmt = {
            "events": [
                {"name": "Cause", "timestamp": 1.0},
                {"name": "Effect", "timestamp": 2.0},
            ]
        }

        causal_result = principles.principles[PrincipleType.CAUSALITY].validate(
            causal_stmt
        )
        assert causal_result.valid is True


if __name__ == "__main__":
    # Run tests with pytest if available
    if pytest:
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        print("pytest not available - use run_tests.py to run tests")
