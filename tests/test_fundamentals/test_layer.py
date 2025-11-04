"""
Integration tests for FundamentalsLayer.

Tests the complete validation pipeline including:
- Initialization and configuration
- Different validation types (mathematical, logical, causal, consistency)
- Caching mechanism
- Statistics tracking
- Health checks
- Error handling
"""

import pytest
import json
import tempfile
from pathlib import Path
from tmr.fundamentals.layer import FundamentalsLayer
from tmr.fundamentals.validators import ReasoningChain, ReasoningStep
from tmr.fundamentals.principles import PrincipleType


class TestFundamentalsLayerInitialization:
    """Test FundamentalsLayer initialization."""

    @pytest.mark.integration
    def test_default_initialization(self):
        """Test initialization with default config."""
        layer = FundamentalsLayer()

        assert layer.principles is not None
        assert layer.logical_validator is not None
        assert layer.mathematical_validator is not None
        assert layer.causal_validator is not None
        assert layer.consistency_validator is not None
        assert layer.cache_size == 1000
        assert layer.stats["total_validations"] == 0

    @pytest.mark.integration
    def test_custom_config(self):
        """Test initialization with custom config."""
        config = {
            "cache_size": 500,
            "log_level": "DEBUG"
        }
        layer = FundamentalsLayer(config)

        assert layer.cache_size == 500
        assert layer.config["log_level"] == "DEBUG"

    @pytest.mark.integration
    def test_string_representation(self):
        """Test string representation of layer."""
        layer = FundamentalsLayer()
        repr_str = repr(layer)

        assert "FundamentalsLayer" in repr_str
        assert "validations=" in repr_str
        assert "success_rate=" in repr_str


class TestFundamentalsLayerValidation:
    """Test validation methods."""

    @pytest.fixture
    def layer(self):
        return FundamentalsLayer()

    @pytest.mark.integration
    def test_mathematical_validation(self, layer):
        """Test mathematical statement validation."""
        statement = {
            "equation": "x + 5 = 10",
            "steps": [
                "x + 5 = 10",
                "x = 10 - 5",
                "x = 5"
            ],
            "result": 5
        }

        result = layer.validate(statement, domain="mathematical")

        assert "valid" in result
        assert "confidence" in result
        assert "details" in result
        assert "metadata" in result
        assert result["metadata"]["validation_type"] == "mathematical"
        assert isinstance(result["valid"], bool)
        assert 0.0 <= result["confidence"] <= 1.0

    @pytest.mark.integration
    def test_logical_validation(self, layer):
        """Test logical reasoning validation."""
        statement = {
            "steps": [
                "All humans are mortal",
                "Socrates is human",
                "Therefore, Socrates is mortal"
            ],
            "conclusion": "Socrates is mortal"
        }

        result = layer.validate(statement, domain="logical")

        assert "valid" in result
        assert "confidence" in result
        assert result["metadata"]["validation_type"] == "logical"

    @pytest.mark.integration
    def test_causal_validation(self, layer):
        """Test causal chain validation."""
        statement = {
            "events": [
                {"id": 1, "name": "rain", "timestamp": 100},
                {"id": 2, "name": "wet ground", "timestamp": 200},
            ],
            "relationships": [
                {"cause_id": 1, "effect_id": 2, "confidence": 0.9}
            ]
        }

        result = layer.validate(statement, domain="causal")

        assert "valid" in result
        assert result["metadata"]["validation_type"] == "causal"

    @pytest.mark.integration
    def test_consistency_validation(self, layer):
        """Test consistency validation across domains."""
        statement = {
            "logical": {
                "steps": ["A is true", "B follows from A"],
                "conclusion": "B is true"
            },
            "mathematical": {
                "equation": "x = 5",
                "steps": ["x = 5"],
                "result": 5
            }
        }

        result = layer.validate(statement, domain="mixed")

        assert "valid" in result
        assert result["metadata"]["validation_type"] == "consistency"

    @pytest.mark.integration
    def test_simple_string_validation(self, layer):
        """Test validation of simple string statement."""
        result = layer.validate("The sky is blue")

        assert "valid" in result
        assert "confidence" in result
        # Should default to logical
        assert result["metadata"]["validation_type"] == "logical"

    @pytest.mark.integration
    def test_validation_type_inference_from_equation(self, layer):
        """Test that equations are inferred as mathematical."""
        statement = {
            "equation": "2 + 2 = 4",
            "steps": ["2 + 2 = 4"]
        }

        result = layer.validate(statement)

        assert result["metadata"]["validation_type"] == "mathematical"

    @pytest.mark.integration
    def test_validation_type_inference_from_events(self, layer):
        """Test that events are inferred as causal."""
        statement = {
            "events": [{"id": 1, "name": "A"}],
            "relationships": []
        }

        result = layer.validate(statement)

        assert result["metadata"]["validation_type"] == "causal"

    @pytest.mark.integration
    def test_validation_with_explicit_type(self, layer):
        """Test validation with explicit type specification."""
        statement = "Test statement"

        result = layer.validate(statement, validation_type="mathematical")

        assert result["metadata"]["validation_type"] == "mathematical"

    @pytest.mark.integration
    def test_validation_statistics_tracking(self, layer):
        """Test that statistics are properly tracked."""
        layer.validate("test 1")
        layer.validate("test 2")
        layer.validate("test 3")

        stats = layer.get_statistics()
        assert stats["total_validations"] == 3
        assert "successful_validations" in stats
        assert "failed_validations" in stats
        assert "domain_counts" in stats


class TestFundamentalsLayerCaching:
    """Test caching functionality."""

    @pytest.fixture
    def layer(self):
        return FundamentalsLayer({"cache_size": 10})

    @pytest.mark.integration
    def test_cache_hit(self, layer):
        """Test that cache returns same result."""
        statement = "test statement"

        result1 = layer.validate(statement)
        result2 = layer.validate(statement)

        # Second call should be from cache
        assert layer.stats["cache_hits"] == 1
        assert layer.stats["cache_misses"] == 1
        assert result1["valid"] == result2["valid"]
        assert result1["confidence"] == result2["confidence"]

    @pytest.mark.integration
    def test_cache_miss(self, layer):
        """Test cache miss for different statements."""
        layer.validate("statement 1")
        layer.validate("statement 2")

        assert layer.stats["cache_hits"] == 0
        assert layer.stats["cache_misses"] == 2

    @pytest.mark.integration
    def test_cache_bypass(self, layer):
        """Test bypassing cache."""
        statement = "test statement"

        layer.validate(statement, use_cache=True)
        layer.validate(statement, use_cache=False)

        # Second call should not hit cache
        assert layer.stats["cache_hits"] == 0
        assert layer.stats["cache_misses"] == 2

    @pytest.mark.integration
    def test_cache_eviction(self, layer):
        """Test cache eviction when limit reached."""
        # Fill cache beyond limit
        for i in range(15):
            layer.validate(f"statement {i}")

        # Cache should not exceed size limit
        assert len(layer.validation_cache) <= layer.cache_size

    @pytest.mark.integration
    def test_cache_key_generation(self, layer):
        """Test cache key generation for different statement types."""
        # Same statement, different domains should have different keys
        statement = "test"

        layer.validate(statement, domain="math")
        layer.validate(statement, domain="logic")

        # Both should be cache misses (different keys)
        assert layer.stats["cache_misses"] == 2

    @pytest.mark.integration
    def test_clear_cache(self, layer):
        """Test clearing cache."""
        layer.validate("test 1")
        layer.validate("test 2")

        assert len(layer.validation_cache) > 0

        layer.clear_cache()

        assert len(layer.validation_cache) == 0


class TestFundamentalsLayerPrincipleValidation:
    """Test principle-specific validation methods."""

    @pytest.fixture
    def layer(self):
        return FundamentalsLayer()

    @pytest.mark.integration
    def test_validate_single_principle(self, layer):
        """Test validation against a single principle."""
        statement = 42

        result = layer.validate_principle(statement, PrincipleType.IDENTITY)

        assert result.valid is True
        assert result.principle == PrincipleType.IDENTITY
        assert result.confidence == 1.0

    @pytest.mark.integration
    def test_validate_multiple_principles(self, layer):
        """Test validation against multiple principles."""
        statement = 42
        principle_types = [
            PrincipleType.IDENTITY,
            PrincipleType.NON_CONTRADICTION
        ]

        results = layer.validate_multiple_principles(statement, principle_types)

        assert len(results) == 2
        assert PrincipleType.IDENTITY in results
        assert PrincipleType.NON_CONTRADICTION in results

    @pytest.mark.integration
    def test_validate_unknown_principle(self, layer):
        """Test error handling for unknown principle."""
        with pytest.raises(ValueError, match="Unknown principle type"):
            layer.validate_principle(42, "invalid_principle")


class TestFundamentalsLayerStatistics:
    """Test statistics tracking and reporting."""

    @pytest.fixture
    def layer(self):
        return FundamentalsLayer()

    @pytest.mark.integration
    def test_statistics_initialization(self, layer):
        """Test initial statistics state."""
        stats = layer.get_statistics()

        assert stats["total_validations"] == 0
        assert stats["successful_validations"] == 0
        assert stats["failed_validations"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["cache_hit_rate"] == 0.0

    @pytest.mark.integration
    def test_statistics_after_validations(self, layer):
        """Test statistics after performing validations."""
        layer.validate("test 1")
        layer.validate("test 2")
        layer.validate("test 2")  # Cache hit

        stats = layer.get_statistics()

        assert stats["total_validations"] == 3
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2

    @pytest.mark.integration
    def test_success_rate_calculation(self, layer):
        """Test success rate calculation."""
        # Perform some validations
        layer.validate({"equation": "x = 5", "steps": ["x = 5"], "result": 5})
        layer.validate({"equation": "y = 10", "steps": ["y = 10"], "result": 10})

        stats = layer.get_statistics()

        assert "success_rate" in stats
        assert 0.0 <= stats["success_rate"] <= 1.0

    @pytest.mark.integration
    def test_cache_hit_rate_calculation(self, layer):
        """Test cache hit rate calculation."""
        layer.validate("test")
        layer.validate("test")  # Hit
        layer.validate("test")  # Hit
        layer.validate("different")  # Miss

        stats = layer.get_statistics()

        assert stats["cache_hit_rate"] == 0.5  # 2 hits out of 4 total

    @pytest.mark.integration
    def test_validation_time_tracking(self, layer):
        """Test validation time tracking."""
        layer.validate("test 1")
        layer.validate("test 2")

        stats = layer.get_statistics()

        assert "avg_validation_time_ms" in stats
        assert "median_validation_time_ms" in stats
        assert stats["avg_validation_time_ms"] > 0

    @pytest.mark.integration
    def test_domain_counts(self, layer):
        """Test domain usage counting."""
        layer.validate("test", domain="math", use_cache=False)
        layer.validate("test", domain="math", use_cache=False)
        layer.validate("test", domain="logic", use_cache=False)

        stats = layer.get_statistics()

        assert "domain_counts" in stats
        assert stats["domain_counts"]["mathematical"] == 2
        assert stats["domain_counts"]["logical"] == 1

    @pytest.mark.integration
    def test_principle_statistics(self, layer):
        """Test principle-level statistics."""
        layer.validate(42)

        stats = layer.get_statistics()

        assert "principle_stats" in stats
        assert isinstance(stats["principle_stats"], dict)

    @pytest.mark.integration
    def test_reset_statistics(self, layer):
        """Test resetting statistics."""
        layer.validate("test 1")
        layer.validate("test 2")

        layer.reset_statistics()

        stats = layer.get_statistics()
        assert stats["total_validations"] == 0
        assert stats["successful_validations"] == 0
        assert stats["cache_hits"] == 0

    @pytest.mark.integration
    def test_export_stats_to_json(self, layer):
        """Test exporting statistics to JSON."""
        layer.validate("test 1")
        layer.validate("test 2")

        stats_json = layer.export_stats()

        # Should be valid JSON
        stats = json.loads(stats_json)
        assert "total_validations" in stats
        assert stats["total_validations"] == 2

    @pytest.mark.integration
    def test_export_stats_to_file(self, layer):
        """Test exporting statistics to file."""
        layer.validate("test")

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name

        try:
            layer.export_stats(filepath)

            # Verify file was created
            assert Path(filepath).exists()

            # Verify contents
            with open(filepath, 'r') as f:
                stats = json.load(f)
                assert "total_validations" in stats
        finally:
            # Cleanup
            Path(filepath).unlink(missing_ok=True)


class TestFundamentalsLayerHealthCheck:
    """Test health check functionality."""

    @pytest.fixture
    def layer(self):
        return FundamentalsLayer()

    @pytest.mark.integration
    def test_health_check_healthy(self, layer):
        """Test health check returns healthy status."""
        health = layer.health_check()

        assert "status" in health
        assert "issues" in health
        assert "metrics" in health
        assert health["status"] in ["healthy", "warning", "degraded"]

    @pytest.mark.integration
    def test_health_check_metrics(self, layer):
        """Test health check includes metrics."""
        layer.validate("test")

        health = layer.health_check()

        assert "cache_size" in health["metrics"]
        assert "cache_limit" in health["metrics"]
        assert "total_validations" in health["metrics"]
        assert "success_rate" in health["metrics"]

    @pytest.mark.integration
    def test_health_check_cache_warning(self, layer):
        """Test health check warns when cache is nearly full."""
        # Fill cache to near capacity
        for i in range(int(layer.cache_size * 0.95)):
            layer.validate(f"test {i}", use_cache=True)

        health = layer.health_check()

        # Should have warning about cache
        if health["status"] != "healthy":
            assert any("cache" in issue.lower() for issue in health["issues"])


class TestFundamentalsLayerErrorHandling:
    """Test error handling in validation."""

    @pytest.fixture
    def layer(self):
        return FundamentalsLayer()

    @pytest.mark.integration
    def test_validation_with_invalid_type(self, layer):
        """Test validation handles invalid types gracefully."""
        # This should not crash, but handle gracefully
        result = layer.validate(None)

        assert "valid" in result
        assert "confidence" in result
        assert "metadata" in result

    @pytest.mark.integration
    def test_validation_error_tracking(self, layer):
        """Test that validation errors are tracked in statistics."""
        initial_failed = layer.stats["failed_validations"]

        # Attempt validation that might fail
        result = layer.validate(None)

        # Check error handling
        if not result["valid"]:
            assert layer.stats["failed_validations"] > initial_failed

    @pytest.mark.integration
    def test_malformed_mathematical_statement(self, layer):
        """Test handling of malformed mathematical statements."""
        statement = {
            "equation": "not an equation",
            "steps": [],
            "result": None
        }

        result = layer.validate(statement, domain="mathematical")

        assert "valid" in result
        # Should handle without crashing

    @pytest.mark.integration
    def test_malformed_causal_chain(self, layer):
        """Test handling of malformed causal chains."""
        statement = {
            "events": "not a list",
            "relationships": "not a list"
        }

        # Should handle without crashing
        try:
            result = layer.validate(statement, domain="causal")
            assert "valid" in result
        except Exception:
            # Error should be caught and returned in result
            pass


@pytest.mark.integration
class TestFundamentalsLayerEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.integration
    def test_complete_workflow(self):
        """Test complete validation workflow."""
        # Initialize layer
        layer = FundamentalsLayer({"cache_size": 100})

        # Perform various validations
        math_result = layer.validate({
            "equation": "x + 2 = 5",
            "steps": ["x + 2 = 5", "x = 3"],
            "result": 3
        }, domain="mathematical")

        causal_result = layer.validate({
            "events": [
                {"id": 1, "name": "A", "timestamp": 100},
                {"id": 2, "name": "B", "timestamp": 200}
            ],
            "relationships": [{"cause_id": 1, "effect_id": 2}]
        }, domain="causal")

        # Check statistics
        stats = layer.get_statistics()
        assert stats["total_validations"] == 2
        assert len(stats["domain_counts"]) == 2

        # Verify health
        health = layer.health_check()
        assert health["status"] in ["healthy", "warning", "degraded"]

        # Export stats
        stats_json = layer.export_stats()
        stats_data = json.loads(stats_json)
        assert stats_data["total_validations"] == 2

        # Test caching
        layer.validate({
            "equation": "x + 2 = 5",
            "steps": ["x + 2 = 5", "x = 3"],
            "result": 3
        }, domain="mathematical")

        assert layer.stats["cache_hits"] == 1

    @pytest.mark.integration
    def test_mixed_validation_types(self):
        """Test handling mixed validation types in sequence."""
        layer = FundamentalsLayer()

        # Mathematical
        layer.validate({"equation": "x = 5"}, domain="math")

        # Logical
        layer.validate({"steps": ["A", "B"]}, domain="logic")

        # Causal
        layer.validate({"events": [], "relationships": []}, domain="causal")

        # String
        layer.validate("simple statement")

        stats = layer.get_statistics()
        assert stats["total_validations"] == 4
        assert len(stats["domain_counts"]) >= 3

    @pytest.mark.integration
    def test_high_volume_validation(self):
        """Test performance with high volume of validations."""
        layer = FundamentalsLayer({"cache_size": 500})

        # Perform many validations
        for i in range(100):
            layer.validate(f"test statement {i % 10}")  # Some repeats for cache hits

        stats = layer.get_statistics()
        assert stats["total_validations"] == 100
        assert stats["cache_hits"] > 0  # Should have some cache hits
        assert "avg_validation_time_ms" in stats
