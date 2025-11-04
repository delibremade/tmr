"""
Comprehensive tests for validators in the Fundamentals Layer.
Tests all validator types: Logical, Mathematical, Causal, and Consistency.
"""

import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tmr.fundamentals.validators import (
    LogicalValidator,
    MathematicalValidator,
    CausalValidator,
    ConsistencyValidator,
    ReasoningStep,
    ReasoningChain,
)
from tmr.fundamentals.principles import LogicalPrinciples


class TestLogicalValidator:
    """Tests for LogicalValidator"""

    def setup_method(self):
        self.validator = LogicalValidator()

    def test_valid_reasoning_chain(self):
        """Test validation of a valid reasoning chain"""
        steps = [
            ReasoningStep(
                step_id=1, statement="All humans are mortal", justification="Premise"
            ),
            ReasoningStep(
                step_id=2,
                statement="Socrates is human",
                justification="Premise",
                dependencies=[1],
            ),
            ReasoningStep(
                step_id=3,
                statement="Socrates is mortal",
                justification="Modus ponens",
                dependencies=[1, 2],
            ),
        ]

        chain = ReasoningChain(
            steps=steps, conclusion="Socrates is mortal", domain="logic"
        )

        valid, confidence, details = self.validator.validate(chain)
        assert valid is True
        assert confidence > 0.0
        assert details["num_steps"] == 3
        assert details["consistency_valid"] is True

    def test_chain_with_invalid_dependency(self):
        """Test detection of invalid dependencies"""
        steps = [
            ReasoningStep(step_id=1, statement="Step 1"),
            ReasoningStep(
                step_id=2, statement="Step 2", dependencies=[5]
            ),  # Invalid dependency
        ]

        chain = ReasoningChain(steps=steps)
        valid, confidence, details = self.validator.validate(chain)
        assert details["consistency_valid"] is False

    def test_chain_with_forward_dependency(self):
        """Test detection of forward-looking dependencies"""
        steps = [
            ReasoningStep(
                step_id=1, statement="Step 1", dependencies=[2]
            ),  # Depends on future step
            ReasoningStep(step_id=2, statement="Step 2"),
        ]

        chain = ReasoningChain(steps=steps)
        valid, confidence, details = self.validator.validate(chain)
        assert details["consistency_valid"] is False

    def test_contradictory_statements_detection(self):
        """Test detection of contradictory statements in chain"""
        steps = [
            ReasoningStep(step_id=1, statement="The answer is yes"),
            ReasoningStep(step_id=2, statement="The answer is no"),
        ]

        chain = ReasoningChain(steps=steps)
        valid, confidence, details = self.validator.validate(chain)
        assert details["consistency_valid"] is False

    def test_same_subject_detection(self):
        """Test detection of statements about same subject"""
        assert self.validator._same_subject(
            "The sky is blue", "The sky is beautiful"
        )
        assert not self.validator._same_subject(
            "The sky is blue", "The car is red"
        )

    def test_empty_chain(self):
        """Test validation of empty chain"""
        chain = ReasoningChain(steps=[])
        valid, confidence, details = self.validator.validate(chain)
        assert valid is True  # Empty chain is vacuously valid


class TestMathematicalValidator:
    """Tests for MathematicalValidator"""

    def setup_method(self):
        self.validator = MathematicalValidator()

    def test_valid_equation_structure(self):
        """Test validation of proper equation structure"""
        assert self.validator._validate_equation_structure("x + 5 = 10")
        assert self.validator._validate_equation_structure("(a + b) * c = d")

    def test_invalid_equation_no_equals(self):
        """Test rejection of expression without equals sign"""
        assert not self.validator._validate_equation_structure("x + 5")

    def test_balanced_parentheses_valid(self):
        """Test balanced parentheses detection"""
        assert self.validator._balanced_parentheses("((a + b) * c)")
        assert self.validator._balanced_parentheses("(x + (y - z))")

    def test_balanced_parentheses_invalid(self):
        """Test unbalanced parentheses detection"""
        assert not self.validator._balanced_parentheses("((a + b)")
        assert not self.validator._balanced_parentheses("a + b)")

    def test_division_by_zero_detection(self):
        """Test detection of division by zero"""
        errors = self.validator._detect_math_errors("x / 0")
        assert "division by zero" in errors

        errors = self.validator._detect_math_errors("x / 5")
        assert "division by zero" not in errors

    def test_valid_numeric_equation(self):
        """Test validation of numeric equations"""
        math_stmt = {"equation": "5 + 5 = 10", "steps": ["5 + 5 = 10"], "result": 10}
        valid, confidence, details = self.validator.validate(math_stmt)
        assert valid is True
        assert details["structure_valid"] is True

    def test_invalid_numeric_equation(self):
        """Test rejection of incorrect equations"""
        math_stmt = {
            "equation": "5 + 5 = 11",  # Wrong!
            "steps": ["5 + 5 = 11"],
            "result": 11,
        }
        valid, confidence, details = self.validator.validate(math_stmt)
        # Should detect imbalanced equation
        assert valid is False or confidence < 0.5

    def test_safe_eval_simple_arithmetic(self):
        """Test safe evaluation of arithmetic"""
        assert self.validator._safe_eval("5 + 5") == 10
        assert self.validator._safe_eval("10 - 3") == 7
        assert self.validator._safe_eval("4 * 3") == 12
        assert self.validator._safe_eval("15 / 3") == 5

    def test_safe_eval_with_parentheses(self):
        """Test safe evaluation with parentheses"""
        assert self.validator._safe_eval("(5 + 3) * 2") == 16
        assert self.validator._safe_eval("10 / (2 + 3)") == 2

    def test_safe_eval_returns_none_for_variables(self):
        """Test that expressions with variables return None"""
        assert self.validator._safe_eval("x + 5") is None
        assert self.validator._safe_eval("2 * x") is None

    def test_safe_eval_power_operator(self):
        """Test power operator conversion"""
        result = self.validator._safe_eval("2 ^ 3")
        assert result == 8  # ^ converted to **

    def test_algebraic_operation_detection(self):
        """Test detection of algebraic operations"""
        assert self.validator._contains_algebraic_operation("x + 5 = 10")
        assert self.validator._contains_algebraic_operation("sin(x) = 0")
        assert not self.validator._contains_algebraic_operation("5 + 5 = 10")

    def test_algebraic_operation_validation(self):
        """Test validation of algebraic operations"""
        assert self.validator._validate_algebraic_operation("x + 5")
        assert self.validator._validate_algebraic_operation("(x + y) * z")
        assert not self.validator._validate_algebraic_operation("x +/ 5")  # Invalid

    def test_transformation_valid_numeric(self):
        """Test validation of numeric transformations"""
        assert self.validator._transformation_valid("5 + 5 = 10", "10 = 10")
        assert self.validator._transformation_valid("2 * 3 = 6", "6 = 6")

    def test_transformation_structure_change(self):
        """Test rejection of structure changes"""
        # Changing from equation to expression
        assert not self.validator._transformation_valid("x = 5", "x + 5")

    def test_math_step_validation_balanced(self):
        """Test validation of balanced equation step"""
        valid, conf = self.validator._validate_math_step("10 = 10", 0, ["10 = 10"])
        assert valid is True
        assert conf > 0.9

    def test_math_step_validation_unbalanced(self):
        """Test rejection of unbalanced equation"""
        valid, conf = self.validator._validate_math_step("10 = 11", 0, ["10 = 11"])
        assert valid is False

    def test_tokenize_math(self):
        """Test mathematical expression tokenization"""
        tokens = self.validator._tokenize_math("x + 5 = 10")
        assert "x" in tokens
        assert "+" in tokens
        assert "5" in tokens
        assert "=" in tokens
        assert "10" in tokens

    def test_conservation_validation(self):
        """Test mathematical conservation validation"""
        steps = ["x + 5 = 10", "x = 5"]
        # Conservation should be maintained through transformation
        assert self.validator._validate_mathematical_conservation(steps)


class TestCausalValidator:
    """Tests for CausalValidator"""

    def setup_method(self):
        self.validator = CausalValidator()

    def test_valid_temporal_ordering(self):
        """Test validation of properly ordered events"""
        events = [
            {"id": 1, "name": "Event A", "timestamp": 1.0},
            {"id": 2, "name": "Event B", "timestamp": 2.0},
            {"id": 3, "name": "Event C", "timestamp": 3.0},
        ]
        assert self.validator._validate_temporal_ordering(events) is True

    def test_invalid_temporal_ordering(self):
        """Test detection of misordered events"""
        events = [
            {"id": 1, "name": "Event A", "timestamp": 3.0},
            {"id": 2, "name": "Event B", "timestamp": 1.0},
        ]
        assert self.validator._validate_temporal_ordering(events) is False

    def test_valid_causal_relationship(self):
        """Test validation of proper causal relationship"""
        events = [
            {"id": 1, "timestamp": 1.0},
            {"id": 2, "timestamp": 2.0},
        ]
        relationship = {"cause_id": 1, "effect_id": 2, "confidence": 0.9}

        valid, conf = self.validator._validate_causal_relationship(relationship, events)
        assert valid is True
        assert conf == 0.9

    def test_invalid_causal_temporal_order(self):
        """Test rejection of backwards causality"""
        events = [
            {"id": 1, "timestamp": 2.0},
            {"id": 2, "timestamp": 1.0},
        ]
        relationship = {"cause_id": 1, "effect_id": 2}

        valid, conf = self.validator._validate_causal_relationship(relationship, events)
        assert valid is False

    def test_circular_causality_detection(self):
        """Test detection of circular causal chains"""
        # A causes B, B causes C, C causes A (circular!)
        relationships = [
            {"cause_id": "A", "effect_id": "B"},
            {"cause_id": "B", "effect_id": "C"},
            {"cause_id": "C", "effect_id": "A"},
        ]
        assert self.validator._detect_circular_causality(relationships) is True

    def test_no_circular_causality(self):
        """Test that linear causality is not flagged as circular"""
        relationships = [
            {"cause_id": "A", "effect_id": "B"},
            {"cause_id": "B", "effect_id": "C"},
            {"cause_id": "C", "effect_id": "D"},
        ]
        assert self.validator._detect_circular_causality(relationships) is False

    def test_complex_circular_detection(self):
        """Test circular detection in complex graph"""
        # A -> B -> C
        #      ^    |
        #      |----+  (C causes B, creating cycle)
        relationships = [
            {"cause_id": "A", "effect_id": "B"},
            {"cause_id": "B", "effect_id": "C"},
            {"cause_id": "C", "effect_id": "B"},
        ]
        assert self.validator._detect_circular_causality(relationships) is True

    def test_full_causal_chain_validation(self):
        """Test full causal chain validation"""
        causal_chain = {
            "events": [
                {"id": 1, "name": "Rain", "timestamp": 1.0},
                {"id": 2, "name": "Wet ground", "timestamp": 2.0},
                {"id": 3, "name": "Slippery", "timestamp": 3.0},
            ],
            "relationships": [
                {"cause_id": 1, "effect_id": 2, "confidence": 0.95},
                {"cause_id": 2, "effect_id": 3, "confidence": 0.85},
            ],
        }

        valid, confidence, details = self.validator.validate(causal_chain)
        assert valid is True
        assert details["temporal_valid"] is True
        assert details["causal_valid"] is True
        assert details["circular_causality"] is False

    def test_causal_chain_with_circularity(self):
        """Test rejection of causal chain with circular relationships"""
        causal_chain = {
            "events": [
                {"id": 1, "timestamp": 1.0},
                {"id": 2, "timestamp": 2.0},
            ],
            "relationships": [
                {"cause_id": 1, "effect_id": 2},
                {"cause_id": 2, "effect_id": 1},  # Circular!
            ],
        }

        valid, confidence, details = self.validator.validate(causal_chain)
        assert valid is False
        assert details["circular_causality"] is True


class TestConsistencyValidator:
    """Tests for ConsistencyValidator"""

    def setup_method(self):
        self.validator = ConsistencyValidator()

    def test_logical_component_validation(self):
        """Test validation of logical component"""
        composite = {
            "logical": ReasoningChain(
                steps=[
                    ReasoningStep(step_id=1, statement="Premise 1"),
                    ReasoningStep(step_id=2, statement="Conclusion", dependencies=[1]),
                ]
            )
        }

        valid, confidence, details = self.validator.validate(composite)
        assert "logical" in details["component_results"]
        assert details["component_results"]["logical"]["valid"] is True

    def test_mathematical_component_validation(self):
        """Test validation of mathematical component"""
        composite = {
            "mathematical": {
                "equation": "2 + 2 = 4",
                "steps": ["2 + 2 = 4"],
                "result": 4,
            }
        }

        valid, confidence, details = self.validator.validate(composite)
        assert "mathematical" in details["component_results"]

    def test_causal_component_validation(self):
        """Test validation of causal component"""
        composite = {
            "causal": {
                "events": [{"id": 1, "timestamp": 1.0}, {"id": 2, "timestamp": 2.0}],
                "relationships": [{"cause_id": 1, "effect_id": 2}],
            }
        }

        valid, confidence, details = self.validator.validate(composite)
        assert "causal" in details["component_results"]

    def test_cross_domain_consistency(self):
        """Test cross-domain consistency validation"""
        composite = {
            "logical": ReasoningChain(
                steps=[ReasoningStep(step_id=1, statement="Test")],
                conclusion="Result A",
            ),
            "mathematical": {
                "equation": "x = 1",
                "steps": ["x = 1"],
                "conclusion": "Result A",  # Same conclusion
            },
        }

        valid, confidence, details = self.validator.validate(composite)
        assert details["cross_domain_valid"] is True

    def test_contradictory_conclusions_detection(self):
        """Test detection of contradictory conclusions"""
        assert self.validator._are_contradictory_conclusions("true", "false")
        assert self.validator._are_contradictory_conclusions("yes", "no")
        assert self.validator._are_contradictory_conclusions(1, 0)

    def test_non_contradictory_conclusions(self):
        """Test that similar conclusions aren't flagged"""
        assert not self.validator._are_contradictory_conclusions("result A", "result A")
        assert not self.validator._are_contradictory_conclusions("positive", "good")

    def test_multi_domain_validation(self):
        """Test validation across all three domains"""
        composite = {
            "logical": ReasoningChain(
                steps=[ReasoningStep(step_id=1, statement="Test")]
            ),
            "mathematical": {"equation": "x = 1", "steps": ["x = 1"], "result": 1},
            "causal": {
                "events": [{"id": 1, "timestamp": 1.0}],
                "relationships": [],
            },
        }

        valid, confidence, details = self.validator.validate(composite)
        assert len(details["component_results"]) == 3
        assert "logical" in details["component_results"]
        assert "mathematical" in details["component_results"]
        assert "causal" in details["component_results"]


class TestValidatorIntegration:
    """Integration tests for validator interactions"""

    def test_validator_history_tracking(self):
        """Test that validators track validation history"""
        validator = LogicalValidator()
        chain = ReasoningChain(steps=[ReasoningStep(step_id=1, statement="Test")])

        validator.validate(chain)
        validator.validate(chain)

        history = validator.get_validation_history()
        assert len(history) == 2
        assert all(h["type"] == "logical" for h in history)

    def test_validator_history_clear(self):
        """Test clearing validation history"""
        validator = LogicalValidator()
        chain = ReasoningChain(steps=[ReasoningStep(step_id=1, statement="Test")])

        validator.validate(chain)
        validator.clear_history()

        assert len(validator.get_validation_history()) == 0

    def test_principles_shared_across_validators(self):
        """Test that validators can share principle instances"""
        principles = LogicalPrinciples()

        validator1 = LogicalValidator(principles)
        validator2 = MathematicalValidator(principles)

        # Both should use the same principle instances
        assert validator1.principles is validator2.principles


if __name__ == "__main__":
    # Run tests with pytest if available
    if pytest:
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        print("pytest not available - use run_tests.py to run tests")
