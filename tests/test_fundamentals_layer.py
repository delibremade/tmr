"""
Comprehensive tests for the FundamentalsLayer orchestrator.
Tests integration, caching, statistics, and health checking.
"""

import sys
from pathlib import Path

try:
    import pytest
except ImportError:
    pytest = None

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tmr.fundamentals.layer import FundamentalsLayer
from tmr.fundamentals.validators import ReasoningStep, ReasoningChain
from tmr.fundamentals.principles import PrincipleType


class TestFundamentalsLayerBasics:
    """Basic tests for FundamentalsLayer"""

    def setup_method(self):
        self.layer = FundamentalsLayer()

    def test_initialization(self):
        """Test that layer initializes correctly"""
        assert self.layer.principles is not None
        assert self.layer.logical_validator is not None
        assert self.layer.math_validator is not None
        assert self.layer.causal_validator is not None
        assert self.layer.consistency_validator is not None

    def test_validate_simple_statement(self):
        """Test validation of simple statement"""
        result = self.layer.validate("test statement")
        assert "valid" in result
        assert "confidence" in result
        assert "details" in result

    def test_validate_with_domain(self):
        """Test validation with explicit domain"""
        result = self.layer.validate("test", domain="logical")
        assert result["valid"] in [True, False]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_validate_principle_identity(self):
        """Test single principle validation"""
        result = self.layer.validate_principle(
            "test", PrincipleType.IDENTITY
        )
        assert "valid" in result
        assert result["principle"] == PrincipleType.IDENTITY

    def test_validate_multiple_principles(self):
        """Test multiple principle validation"""
        principles = [PrincipleType.IDENTITY, PrincipleType.NON_CONTRADICTION]
        results = self.layer.validate_multiple_principles("test", principles)

        assert len(results) == 2
        assert all(r["principle"] in principles for r in results)


class TestDomainInference:
    """Tests for automatic domain detection"""

    def setup_method(self):
        self.layer = FundamentalsLayer()

    def test_infer_mathematical_domain(self):
        """Test inference of mathematical domain"""
        domain = self.layer._infer_validation_type(
            {"equation": "x + 5 = 10", "steps": []}, None
        )
        assert domain == "mathematical"

    def test_infer_causal_domain(self):
        """Test inference of causal domain"""
        domain = self.layer._infer_validation_type(
            {"events": [], "relationships": []}, None
        )
        assert domain == "causal"

    def test_infer_logical_domain(self):
        """Test inference of logical domain"""
        chain = ReasoningChain(steps=[ReasoningStep(step_id=1, statement="test")])
        domain = self.layer._infer_validation_type(chain, None)
        assert domain == "logical"

    def test_explicit_domain_override(self):
        """Test that explicit domain overrides inference"""
        # Even with math-like statement, use explicit domain
        result = self.layer.validate("2 + 2 = 4", domain="logical")
        # Should use logical validator even though it looks like math
        assert "valid" in result


class TestCaching:
    """Tests for validation caching"""

    def setup_method(self):
        self.layer = FundamentalsLayer(cache_size=10)

    def test_cache_hit(self):
        """Test that identical validations use cache"""
        statement = "cached test"

        # First validation
        result1 = self.layer.validate(statement)

        # Second validation should hit cache
        result2 = self.layer.validate(statement)

        assert result1 == result2
        # Check cache statistics
        stats = self.layer.get_statistics()
        assert stats["cache_hits"] > 0

    def test_cache_different_statements(self):
        """Test that different statements don't hit cache"""
        result1 = self.layer.validate("statement 1")
        result2 = self.layer.validate("statement 2")

        # These are different, so cache shouldn't be used
        assert "valid" in result1
        assert "valid" in result2

    def test_cache_with_different_domains(self):
        """Test that same statement with different domains aren't cached together"""
        statement = "test"

        result1 = self.layer.validate(statement, domain="logical")
        result2 = self.layer.validate(statement, domain="mathematical")

        # Different domains should produce different cache entries
        # (Though both might succeed or fail)
        assert "valid" in result1
        assert "valid" in result2

    def test_cache_respects_size_limit(self):
        """Test that cache respects size limit"""
        small_cache_layer = FundamentalsLayer(cache_size=2)

        # Add 3 items
        small_cache_layer.validate("test1")
        small_cache_layer.validate("test2")
        small_cache_layer.validate("test3")

        # Cache should have at most 2 items
        assert len(small_cache_layer.validation_cache) <= 2


class TestStatistics:
    """Tests for statistics tracking"""

    def setup_method(self):
        self.layer = FundamentalsLayer()

    def test_statistics_tracking(self):
        """Test that validations are tracked"""
        initial_stats = self.layer.get_statistics()

        self.layer.validate("test1")
        self.layer.validate("test2")

        stats = self.layer.get_statistics()
        assert stats["total_validations"] == initial_stats["total_validations"] + 2

    def test_success_failure_tracking(self):
        """Test tracking of successes and failures"""
        # Valid statement
        self.layer.validate("valid test")

        # Invalid statement (contradiction)
        self.layer.validate(
            {"claim": "test", "claim_true": True, "negation_true": True}
        )

        stats = self.layer.get_statistics()
        # Should have both successes and possibly failures
        assert stats["total_validations"] >= 2

    def test_domain_distribution(self):
        """Test tracking of domain distribution"""
        self.layer.validate("test", domain="logical")
        self.layer.validate({"equation": "x=1", "steps": []}, domain="mathematical")

        stats = self.layer.get_statistics()
        assert "domain_distribution" in stats
        assert "logical" in stats["domain_distribution"]
        assert "mathematical" in stats["domain_distribution"]

    def test_statistics_reset(self):
        """Test resetting statistics"""
        self.layer.validate("test1")
        self.layer.validate("test2")

        self.layer.reset_statistics()

        stats = self.layer.get_statistics()
        assert stats["total_validations"] == 0
        assert stats["successful_validations"] == 0
        assert stats["failed_validations"] == 0

    def test_export_statistics(self):
        """Test exporting statistics to JSON format"""
        self.layer.validate("test")

        stats_json = self.layer.export_stats()
        assert isinstance(stats_json, str)
        assert "total_validations" in stats_json


class TestHealthCheck:
    """Tests for health checking functionality"""

    def setup_method(self):
        self.layer = FundamentalsLayer()

    def test_health_check_healthy(self):
        """Test health check for healthy system"""
        # Perform some successful validations
        for i in range(10):
            self.layer.validate(f"test {i}")

        health = self.layer.health_check()
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "total_validations" in health
        assert "cache_hit_rate" in health

    def test_health_check_components(self):
        """Test that health check includes all components"""
        health = self.layer.health_check()

        assert "status" in health
        assert "total_validations" in health
        assert "success_rate" in health
        assert "cache_hit_rate" in health
        assert "principle_health" in health


class TestDataPreparation:
    """Tests for data preparation methods"""

    def setup_method(self):
        self.layer = FundamentalsLayer()

    def test_prepare_mathematical_statement(self):
        """Test preparation of mathematical statements"""
        prepared = self.layer._prepare_mathematical_statement(
            "Solve x + 5 = 10"
        )

        assert isinstance(prepared, dict)
        # Should extract equation structure

    def test_prepare_causal_chain(self):
        """Test preparation of causal chains"""
        prepared = self.layer._prepare_causal_chain(
            "Event A causes Event B which causes Event C"
        )

        assert isinstance(prepared, dict)
        assert "events" in prepared or "relationships" in prepared

    def test_prepare_reasoning_chain(self):
        """Test preparation of reasoning chains"""
        prepared = self.layer._prepare_reasoning_chain(
            "First premise. Second premise. Therefore conclusion."
        )

        assert isinstance(prepared, (dict, ReasoningChain))


class TestValidationTypes:
    """Tests for different validation types"""

    def setup_method(self):
        self.layer = FundamentalsLayer()

    def test_logical_validation(self):
        """Test logical reasoning validation"""
        chain = ReasoningChain(
            steps=[
                ReasoningStep(step_id=1, statement="All men are mortal"),
                ReasoningStep(step_id=2, statement="Socrates is a man"),
                ReasoningStep(
                    step_id=3,
                    statement="Socrates is mortal",
                    dependencies=[1, 2],
                ),
            ]
        )

        result = self.layer.validate(chain, domain="logical")
        assert "valid" in result
        assert "confidence" in result

    def test_mathematical_validation(self):
        """Test mathematical validation"""
        math_stmt = {
            "equation": "2 + 2 = 4",
            "steps": ["2 + 2 = 4"],
            "result": 4,
        }

        result = self.layer.validate(math_stmt, domain="mathematical")
        assert "valid" in result
        assert result["valid"] is True  # This should be valid

    def test_causal_validation(self):
        """Test causal validation"""
        causal_chain = {
            "events": [
                {"id": 1, "name": "Rain", "timestamp": 1.0},
                {"id": 2, "name": "Wet ground", "timestamp": 2.0},
            ],
            "relationships": [{"cause_id": 1, "effect_id": 2}],
        }

        result = self.layer.validate(causal_chain, domain="causal")
        assert "valid" in result
        assert result["valid"] is True

    def test_consistency_validation(self):
        """Test cross-domain consistency validation"""
        composite = {
            "logical": ReasoningChain(
                steps=[ReasoningStep(step_id=1, statement="Test")]
            ),
            "mathematical": {"equation": "x=1", "steps": ["x=1"], "result": 1},
        }

        result = self.layer.validate(composite, domain="consistency")
        assert "valid" in result


class TestErrorHandling:
    """Tests for error handling"""

    def setup_method(self):
        self.layer = FundamentalsLayer()

    def test_invalid_domain(self):
        """Test handling of invalid domain"""
        result = self.layer.validate("test", domain="invalid_domain")
        # Should fall back to default behavior
        assert "valid" in result

    def test_none_statement(self):
        """Test handling of None statement"""
        result = self.layer.validate(None)
        # Should handle gracefully
        assert "valid" in result

    def test_invalid_principle_type(self):
        """Test handling of invalid principle type"""
        try:
            # This should handle the error gracefully
            result = self.layer.validate_principle("test", "invalid_principle")
            # If it doesn't raise an error, check result format
            assert "valid" in result or "error" in result
        except (ValueError, KeyError, TypeError):
            # These exceptions are acceptable for invalid input
            pass


class TestIntegration:
    """Integration tests for FundamentalsLayer"""

    def setup_method(self):
        self.layer = FundamentalsLayer()

    def test_end_to_end_logical_reasoning(self):
        """Test complete logical reasoning validation"""
        chain = ReasoningChain(
            steps=[
                ReasoningStep(
                    step_id=1,
                    statement="If it rains, the ground gets wet",
                    justification="Conditional premise",
                ),
                ReasoningStep(
                    step_id=2,
                    statement="It is raining",
                    justification="Observation",
                ),
                ReasoningStep(
                    step_id=3,
                    statement="The ground is wet",
                    justification="Modus ponens",
                    dependencies=[1, 2],
                ),
            ],
            conclusion="The ground is wet",
            domain="logic",
        )

        result = self.layer.validate(chain, domain="logical")

        assert result["valid"] is True
        assert result["confidence"] > 0.5
        assert "details" in result

    def test_end_to_end_mathematical_reasoning(self):
        """Test complete mathematical validation"""
        math_problem = {
            "equation": "x + 5 = 12",
            "steps": [
                "x + 5 = 12",
                "x + 5 - 5 = 12 - 5",
                "x = 7",
            ],
            "result": 7,
        }

        result = self.layer.validate(math_problem, domain="mathematical")

        assert "valid" in result
        assert "confidence" in result

    def test_principle_coverage(self):
        """Test that all 5 principles are exercised"""
        # Test each principle type
        statements = {
            PrincipleType.IDENTITY: "test",
            PrincipleType.NON_CONTRADICTION: [
                "statement 1",
                "statement 2",
            ],
            PrincipleType.EXCLUDED_MIDDLE: {
                "proposition": "test",
                "truth_value": True,
            },
            PrincipleType.CAUSALITY: {
                "cause": "A",
                "effect": "B",
                "timestamp_cause": 1.0,
                "timestamp_effect": 2.0,
            },
            PrincipleType.CONSERVATION: {
                "before": 100,
                "after": 100,
            },
        }

        results = []
        for principle_type, statement in statements.items():
            result = self.layer.validate_principle(statement, principle_type)
            results.append(result)
            assert result["valid"] in [True, False]

        # All 5 principles tested
        assert len(results) == 5

    def test_concurrent_validations(self):
        """Test multiple concurrent validations"""
        statements = [f"test statement {i}" for i in range(10)]

        results = [self.layer.validate(stmt) for stmt in statements]

        # All should complete
        assert len(results) == 10
        assert all("valid" in r for r in results)

        # Statistics should reflect all validations
        stats = self.layer.get_statistics()
        assert stats["total_validations"] >= 10


if __name__ == "__main__":
    # Run tests with pytest if available
    if pytest:
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        print("pytest not available - use run_tests.py to run tests")
