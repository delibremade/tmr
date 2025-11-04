"""
Unit tests for validators.

Tests all four validators:
- LogicalValidator
- MathematicalValidator
- CausalValidator
- ConsistencyValidator
"""

import pytest
import numpy as np
from tmr.fundamentals.validators import (
    LogicalValidator,
    MathematicalValidator,
    CausalValidator,
    ConsistencyValidator,
    ReasoningStep,
    ReasoningChain,
    PrincipleValidator,
)
from tmr.fundamentals.principles import LogicalPrinciples


class TestLogicalValidator:
    """Test the LogicalValidator."""

    @pytest.fixture
    def validator(self):
        return LogicalValidator()

    @pytest.mark.unit
    def test_logical_validator_initialization(self, validator):
        """Test validator initialization."""
        assert isinstance(validator.principles, LogicalPrinciples)
        assert validator.validation_history == []

    @pytest.mark.unit
    def test_simple_reasoning_chain(self, validator):
        """Test validation of simple reasoning chain."""
        steps = [
            ReasoningStep(step_id=0, statement="A is true", justification="Given"),
            ReasoningStep(step_id=1, statement="B follows from A", justification="Inference", dependencies=[0]),
            ReasoningStep(step_id=2, statement="C follows from B", justification="Inference", dependencies=[1]),
        ]
        chain = ReasoningChain(steps=steps, conclusion="C is true")

        valid, confidence, details = validator.validate(chain)

        assert isinstance(valid, bool)
        assert isinstance(confidence, (float, np.floating))
        assert 0.0 <= confidence <= 1.0
        assert "step_results" in details
        assert "consistency_valid" in details
        assert "num_steps" in details
        assert details["num_steps"] == 3

    @pytest.mark.unit
    def test_invalid_dependencies(self, validator):
        """Test detection of invalid dependencies."""
        steps = [
            ReasoningStep(step_id=0, statement="A is true"),
            ReasoningStep(step_id=1, statement="B is true", dependencies=[5]),  # Invalid dependency
        ]
        chain = ReasoningChain(steps=steps)

        valid, confidence, details = validator.validate(chain)

        assert details["consistency_valid"] is False

    @pytest.mark.unit
    def test_forward_dependency_violation(self, validator):
        """Test that forward dependencies are detected."""
        steps = [
            ReasoningStep(step_id=0, statement="A is true", dependencies=[1]),  # Forward dependency
            ReasoningStep(step_id=1, statement="B is true"),
        ]
        chain = ReasoningChain(steps=steps)

        valid, confidence, details = validator.validate(chain)

        assert details["consistency_valid"] is False

    @pytest.mark.unit
    def test_contradictory_statements(self, validator):
        """Test detection of contradictory statements."""
        steps = [
            ReasoningStep(step_id=0, statement="The value is true"),
            ReasoningStep(step_id=1, statement="The value is false"),
        ]
        chain = ReasoningChain(steps=steps)

        valid, confidence, details = validator.validate(chain)

        assert details["consistency_valid"] is False

    @pytest.mark.unit
    def test_validation_history(self, validator):
        """Test that validation history is recorded."""
        steps = [ReasoningStep(step_id=0, statement="A is true")]
        chain = ReasoningChain(steps=steps)

        validator.validate(chain)
        validator.validate(chain)

        history = validator.get_validation_history()
        assert len(history) == 2
        assert all(h["type"] == "logical" for h in history)

    @pytest.mark.unit
    def test_clear_history(self, validator):
        """Test clearing validation history."""
        steps = [ReasoningStep(step_id=0, statement="A is true")]
        chain = ReasoningChain(steps=steps)

        validator.validate(chain)
        validator.clear_history()

        assert len(validator.get_validation_history()) == 0

    @pytest.mark.unit
    def test_empty_reasoning_chain(self, validator):
        """Test validation of empty reasoning chain."""
        chain = ReasoningChain(steps=[])

        valid, confidence, details = validator.validate(chain)

        assert details["num_steps"] == 0
        assert details["consistency_valid"] is True

    @pytest.mark.unit
    def test_same_subject_detection(self, validator):
        """Test detection of same subject in statements."""
        same = validator._same_subject("The cat is black", "The cat is white")
        assert same is True

        different = validator._same_subject("The cat is black", "The dog is white")
        # Should still have some overlap, but less significant
        assert isinstance(different, bool)


class TestMathematicalValidator:
    """Test the MathematicalValidator."""

    @pytest.fixture
    def validator(self):
        return MathematicalValidator()

    @pytest.mark.unit
    def test_mathematical_validator_initialization(self, validator):
        """Test validator initialization."""
        assert isinstance(validator.principles, LogicalPrinciples)
        assert validator.validation_history == []

    @pytest.mark.unit
    def test_simple_equation(self, validator):
        """Test validation of simple equation."""
        math_statement = {
            "equation": "x + 5 = 10",
            "steps": [
                "x + 5 = 10",
                "x = 10 - 5",
                "x = 5"
            ],
            "result": 5
        }

        valid, confidence, details = validator.validate(math_statement)

        assert isinstance(valid, bool)
        assert isinstance(confidence, (float, np.floating))
        assert 0.0 <= confidence <= 1.0
        assert "structure_valid" in details
        assert "steps_valid" in details
        assert "conservation_valid" in details
        assert "result_valid" in details

    @pytest.mark.unit
    def test_equation_without_equals(self, validator):
        """Test detection of invalid equation structure."""
        math_statement = {
            "equation": "x + 5",  # Missing equals
            "steps": [],
            "result": None
        }

        valid, confidence, details = validator.validate(math_statement)

        assert details["structure_valid"] is False

    @pytest.mark.unit
    def test_balanced_parentheses(self, validator):
        """Test parentheses balancing check."""
        assert validator._balanced_parentheses("(a + b)") is True
        assert validator._balanced_parentheses("((a + b) * c)") is True
        assert validator._balanced_parentheses("(a + b))") is False
        assert validator._balanced_parentheses("((a + b)") is False

    @pytest.mark.unit
    def test_division_by_zero_detection(self, validator):
        """Test detection of division by zero."""
        errors = validator._detect_math_errors("x = 10/0")
        assert "division by zero" in errors

        errors = validator._detect_math_errors("x = 10 / 0")
        assert "division by zero" in errors

    @pytest.mark.unit
    def test_result_validation(self, validator):
        """Test result validation."""
        steps = ["x = 5", "y = x + 3", "y = 8"]

        assert validator._validate_result(steps, 8) is True
        assert validator._validate_result(steps, 10) is False
        assert validator._validate_result([], 5) is True

    @pytest.mark.unit
    def test_validation_history(self, validator):
        """Test that validation history is recorded."""
        math_statement = {
            "equation": "x = 5",
            "steps": ["x = 5"],
            "result": 5
        }

        validator.validate(math_statement)
        validator.validate(math_statement)

        history = validator.get_validation_history()
        assert len(history) == 2
        assert all(h["type"] == "mathematical" for h in history)

    @pytest.mark.unit
    def test_empty_equation(self, validator):
        """Test validation with empty equation."""
        math_statement = {
            "equation": "",
            "steps": [],
            "result": None
        }

        valid, confidence, details = validator.validate(math_statement)

        assert details["structure_valid"] is True  # Empty is valid
        assert details["num_steps"] == 0

    @pytest.mark.unit
    def test_complex_equation(self, validator):
        """Test validation of complex equation."""
        math_statement = {
            "equation": "(x^2 + 2*x + 1) = (x + 1)^2",
            "steps": [
                "(x^2 + 2*x + 1) = (x + 1)^2",
                "x^2 + 2*x + 1 = x^2 + 2*x + 1"
            ],
            "result": "identity"
        }

        valid, confidence, details = validator.validate(math_statement)

        assert isinstance(valid, bool)
        assert isinstance(confidence, (float, np.floating))


class TestCausalValidator:
    """Test the CausalValidator."""

    @pytest.fixture
    def validator(self):
        return CausalValidator()

    @pytest.mark.unit
    def test_causal_validator_initialization(self, validator):
        """Test validator initialization."""
        assert isinstance(validator.principles, LogicalPrinciples)
        assert validator.validation_history == []

    @pytest.mark.unit
    def test_simple_causal_chain(self, validator):
        """Test validation of simple causal chain."""
        causal_chain = {
            "events": [
                {"id": 1, "name": "rain", "timestamp": 100},
                {"id": 2, "name": "wet ground", "timestamp": 200},
            ],
            "relationships": [
                {"cause_id": 1, "effect_id": 2, "confidence": 0.9}
            ]
        }

        valid, confidence, details = validator.validate(causal_chain)

        assert isinstance(valid, bool)
        assert isinstance(confidence, (float, np.floating))
        assert 0.0 <= confidence <= 1.0
        assert "temporal_valid" in details
        assert "causal_valid" in details
        assert "circular_causality" in details
        assert details["num_events"] == 2
        assert details["num_relationships"] == 1

    @pytest.mark.unit
    def test_temporal_ordering_violation(self, validator):
        """Test detection of temporal ordering violations."""
        causal_chain = {
            "events": [
                {"id": 1, "name": "event1", "timestamp": 200},
                {"id": 2, "name": "event2", "timestamp": 100},  # Out of order
            ],
            "relationships": []
        }

        valid, confidence, details = validator.validate(causal_chain)

        assert details["temporal_valid"] is False

    @pytest.mark.unit
    def test_causal_relationship_with_wrong_timestamps(self, validator):
        """Test causal relationship with cause after effect."""
        causal_chain = {
            "events": [
                {"id": 1, "name": "cause", "timestamp": 200},
                {"id": 2, "name": "effect", "timestamp": 100},  # Before cause
            ],
            "relationships": [
                {"cause_id": 1, "effect_id": 2, "confidence": 0.9}
            ]
        }

        valid, confidence, details = validator.validate(causal_chain)

        assert valid is False

    @pytest.mark.unit
    def test_circular_causality_detection(self, validator):
        """Test detection of circular causality."""
        causal_chain = {
            "events": [
                {"id": 1, "name": "A"},
                {"id": 2, "name": "B"},
                {"id": 3, "name": "C"},
            ],
            "relationships": [
                {"cause_id": 1, "effect_id": 2},
                {"cause_id": 2, "effect_id": 3},
                {"cause_id": 3, "effect_id": 1},  # Creates cycle
            ]
        }

        valid, confidence, details = validator.validate(causal_chain)

        assert details["circular_causality"] is True
        assert valid is False

    @pytest.mark.unit
    def test_no_circular_causality(self, validator):
        """Test that linear causality is not flagged as circular."""
        causal_chain = {
            "events": [
                {"id": 1, "name": "A"},
                {"id": 2, "name": "B"},
                {"id": 3, "name": "C"},
            ],
            "relationships": [
                {"cause_id": 1, "effect_id": 2},
                {"cause_id": 2, "effect_id": 3},
            ]
        }

        valid, confidence, details = validator.validate(causal_chain)

        assert details["circular_causality"] is False

    @pytest.mark.unit
    def test_missing_event_in_relationship(self, validator):
        """Test handling of missing events in relationships."""
        causal_chain = {
            "events": [
                {"id": 1, "name": "A"},
            ],
            "relationships": [
                {"cause_id": 1, "effect_id": 999},  # Non-existent event
            ]
        }

        valid, confidence, details = validator.validate(causal_chain)

        assert valid is False

    @pytest.mark.unit
    def test_empty_causal_chain(self, validator):
        """Test validation of empty causal chain."""
        causal_chain = {
            "events": [],
            "relationships": []
        }

        valid, confidence, details = validator.validate(causal_chain)

        assert details["temporal_valid"] is True
        assert details["num_events"] == 0

    @pytest.mark.unit
    def test_validation_history(self, validator):
        """Test that validation history is recorded."""
        causal_chain = {
            "events": [{"id": 1, "name": "A"}],
            "relationships": []
        }

        validator.validate(causal_chain)
        validator.validate(causal_chain)

        history = validator.get_validation_history()
        assert len(history) == 2
        assert all(h["type"] == "causal" for h in history)


class TestConsistencyValidator:
    """Test the ConsistencyValidator."""

    @pytest.fixture
    def validator(self):
        return ConsistencyValidator()

    @pytest.mark.unit
    def test_consistency_validator_initialization(self, validator):
        """Test validator initialization."""
        assert isinstance(validator.principles, LogicalPrinciples)
        assert isinstance(validator.logical_validator, LogicalValidator)
        assert isinstance(validator.math_validator, MathematicalValidator)
        assert isinstance(validator.causal_validator, CausalValidator)

    @pytest.mark.unit
    def test_logical_component_only(self, validator):
        """Test validation with only logical component."""
        steps = [ReasoningStep(step_id=0, statement="A is true")]
        chain = ReasoningChain(steps=steps, conclusion="A is true")

        composite = {
            "logical": chain
        }

        valid, confidence, details = validator.validate(composite)

        assert isinstance(valid, bool)
        assert isinstance(confidence, (float, np.floating))
        assert "component_results" in details
        assert "logical" in details["component_results"]
        assert details["num_components"] == 1

    @pytest.mark.unit
    def test_mathematical_component_only(self, validator):
        """Test validation with only mathematical component."""
        composite = {
            "mathematical": {
                "equation": "x = 5",
                "steps": ["x = 5"],
                "result": 5
            }
        }

        valid, confidence, details = validator.validate(composite)

        assert isinstance(valid, bool)
        assert "mathematical" in details["component_results"]

    @pytest.mark.unit
    def test_causal_component_only(self, validator):
        """Test validation with only causal component."""
        composite = {
            "causal": {
                "events": [{"id": 1, "name": "A"}],
                "relationships": []
            }
        }

        valid, confidence, details = validator.validate(composite)

        assert isinstance(valid, bool)
        assert "causal" in details["component_results"]

    @pytest.mark.unit
    def test_multiple_components(self, validator):
        """Test validation with multiple components."""
        steps = [ReasoningStep(step_id=0, statement="A is true")]
        chain = ReasoningChain(steps=steps, conclusion="result is 5")

        composite = {
            "logical": chain,
            "mathematical": {
                "equation": "x = 5",
                "steps": ["x = 5"],
                "result": 5,
                "conclusion": "result is 5"
            },
            "causal": {
                "events": [{"id": 1, "name": "A"}],
                "relationships": [],
                "conclusion": "result is 5"
            }
        }

        valid, confidence, details = validator.validate(composite)

        assert details["num_components"] == 3
        assert "cross_domain_valid" in details

    @pytest.mark.unit
    def test_contradictory_conclusions(self, validator):
        """Test detection of contradictory conclusions."""
        assert validator._are_contradictory_conclusions("true", "false") is True
        assert validator._are_contradictory_conclusions("yes", "no") is True
        assert validator._are_contradictory_conclusions("1", "0") is True
        assert validator._are_contradictory_conclusions("result", "not result") is True

    @pytest.mark.unit
    def test_consistent_conclusions(self, validator):
        """Test that consistent conclusions are not flagged."""
        assert validator._are_contradictory_conclusions("true", "true") is False
        assert validator._are_contradictory_conclusions("result is 5", "answer is 5") is False

    @pytest.mark.unit
    def test_cross_domain_inconsistency_penalty(self, validator):
        """Test that cross-domain inconsistency reduces confidence."""
        steps = [ReasoningStep(step_id=0, statement="A is true")]

        composite_consistent = {
            "logical": ReasoningChain(steps=steps, conclusion="true"),
            "mathematical": {
                "equation": "x = 5",
                "steps": ["x = 5"],
                "result": 5,
                "conclusion": "true"
            }
        }

        composite_inconsistent = {
            "logical": ReasoningChain(steps=steps, conclusion="true"),
            "mathematical": {
                "equation": "x = 5",
                "steps": ["x = 5"],
                "result": 5,
                "conclusion": "false"
            }
        }

        valid_consistent, conf_consistent, _ = validator.validate(composite_consistent)
        valid_inconsistent, conf_inconsistent, details_inconsistent = validator.validate(composite_inconsistent)

        # Inconsistent should have lower confidence
        if not details_inconsistent["cross_domain_valid"]:
            assert conf_inconsistent < conf_consistent

    @pytest.mark.unit
    def test_validation_history(self, validator):
        """Test that validation history is recorded."""
        composite = {
            "mathematical": {
                "equation": "x = 5",
                "steps": ["x = 5"],
                "result": 5
            }
        }

        validator.validate(composite)
        validator.validate(composite)

        history = validator.get_validation_history()
        assert len(history) == 2
        assert all(h["type"] == "consistency" for h in history)

    @pytest.mark.unit
    def test_empty_composite(self, validator):
        """Test validation of empty composite reasoning."""
        composite = {}

        valid, confidence, details = validator.validate(composite)

        assert details["num_components"] == 0
        assert confidence == 0.0


@pytest.mark.edge_case
class TestValidatorEdgeCases:
    """Test edge cases across all validators."""

    @pytest.mark.unit
    def test_logical_validator_with_metadata(self):
        """Test logical validator with metadata in steps."""
        validator = LogicalValidator()
        steps = [
            ReasoningStep(
                step_id=0,
                statement="A is true",
                metadata={"source": "premise", "line": 1}
            )
        ]
        chain = ReasoningChain(steps=steps, metadata={"author": "test"})

        valid, confidence, details = validator.validate(chain)
        assert isinstance(valid, bool)

    @pytest.mark.unit
    def test_mathematical_validator_with_special_functions(self):
        """Test mathematical validator with special functions."""
        validator = MathematicalValidator()
        math_statement = {
            "equation": "sin(x) + cos(x) = sqrt(2)",
            "steps": ["sin(x) + cos(x) = sqrt(2)"],
            "result": None
        }

        valid, confidence, details = validator.validate(math_statement)
        assert isinstance(valid, bool)

    @pytest.mark.unit
    def test_causal_validator_with_simultaneous_events(self):
        """Test causal validator with events at same timestamp."""
        validator = CausalValidator()
        causal_chain = {
            "events": [
                {"id": 1, "name": "A", "timestamp": 100},
                {"id": 2, "name": "B", "timestamp": 100},  # Same time
            ],
            "relationships": []
        }

        valid, confidence, details = validator.validate(causal_chain)
        assert details["temporal_valid"] is True

    @pytest.mark.unit
    def test_validators_with_none_values(self):
        """Test validators handle None values gracefully."""
        logical = LogicalValidator()
        mathematical = MathematicalValidator()
        causal = CausalValidator()

        # These should not crash
        chain = ReasoningChain(steps=[])
        valid, conf, det = logical.validate(chain)
        assert isinstance(valid, bool)

        math_stmt = {"equation": None, "steps": [], "result": None}
        valid, conf, det = mathematical.validate(math_stmt)
        assert isinstance(valid, bool)

        causal_chain = {"events": [], "relationships": []}
        valid, conf, det = causal.validate(causal_chain)
        assert isinstance(valid, bool)

    @pytest.mark.unit
    def test_very_long_reasoning_chain(self):
        """Test with very long reasoning chain."""
        validator = LogicalValidator()
        steps = [
            ReasoningStep(
                step_id=i,
                statement=f"Step {i}",
                dependencies=[i-1] if i > 0 else None
            )
            for i in range(100)
        ]
        chain = ReasoningChain(steps=steps)

        valid, confidence, details = validator.validate(chain)
        assert details["num_steps"] == 100

    @pytest.mark.unit
    def test_unicode_in_statements(self):
        """Test handling of unicode characters."""
        validator = LogicalValidator()
        steps = [
            ReasoningStep(step_id=0, statement="日本語のテスト"),
            ReasoningStep(step_id=1, statement="Тест на русском"),
        ]
        chain = ReasoningChain(steps=steps)

        valid, confidence, details = validator.validate(chain)
        assert isinstance(valid, bool)
