"""
Tests for Nuance Layer
"""

import unittest
import tempfile
from pathlib import Path

from tmr.nuance import (
    NuanceLayer,
    DomainType,
    Pattern,
    PatternComplexity
)


class TestNuanceLayer(unittest.TestCase):
    """Test NuanceLayer class."""

    def setUp(self):
        """Set up test layer."""
        # Use temporary directory for storage
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "test_patterns.json"

        self.layer = NuanceLayer(config={
            "storage_path": str(self.storage_path),
            "load_core_patterns": True
        })

    def test_initialization(self):
        """Test layer initialization."""
        self.assertIsNotNone(self.layer.classifier)
        self.assertIsNotNone(self.layer.library)
        self.assertIsNotNone(self.layer.extractors)
        self.assertGreater(len(self.layer.library), 0)  # Core patterns loaded

    def test_extract_patterns_math(self):
        """Test extracting mathematical patterns."""
        content = {
            "equation": "2x + 5 = 15",
            "steps": ["2x = 10", "x = 5"],
            "solution": "x = 5"
        }

        patterns = self.layer.extract_patterns(content, domain=DomainType.MATH)
        self.assertIsInstance(patterns, list)
        self.assertGreater(self.layer.stats["total_extractions"], 0)

    def test_extract_patterns_code(self):
        """Test extracting code patterns."""
        content = """
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        """

        patterns = self.layer.extract_patterns(content, domain=DomainType.CODE)
        self.assertIsInstance(patterns, list)

    def test_extract_patterns_logic(self):
        """Test extracting logical patterns."""
        content = {
            "premises": [
                "If it rains, the ground is wet",
                "It is raining"
            ],
            "conclusion": "The ground is wet"
        }

        patterns = self.layer.extract_patterns(content, domain=DomainType.LOGIC)
        self.assertIsInstance(patterns, list)

    def test_extract_patterns_auto_classify(self):
        """Test pattern extraction with automatic domain classification."""
        content = "Calculate 10 + 5 and solve x = 15"

        patterns = self.layer.extract_patterns(content)
        self.assertIsInstance(patterns, list)
        self.assertGreater(self.layer.stats["total_classifications"], 0)

    def test_retrieve_patterns_math(self):
        """Test retrieving mathematical patterns."""
        # First add a pattern
        pattern = Pattern(
            pattern_id="",
            domain=DomainType.MATH,
            name="test_arithmetic",
            description="Test arithmetic pattern",
            structure={"operation": "addition", "operands": 2},
            complexity=PatternComplexity.SIMPLE
        )
        self.layer.add_pattern(pattern)

        # Retrieve with query
        query = {
            "domain": DomainType.MATH.value,
            "structure": {"operation": "addition"}
        }

        matches = self.layer.retrieve_patterns(query, domain=DomainType.MATH)
        self.assertIsInstance(matches, list)
        self.assertGreater(self.layer.stats["total_retrievals"], 0)

    def test_classify_content(self):
        """Test content classification."""
        content = "Implement a function to calculate factorial"
        domain, confidence, scores = self.layer.classify_content(content)

        self.assertIsInstance(domain, DomainType)
        self.assertIsInstance(confidence, float)
        self.assertIsInstance(scores, dict)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

    def test_add_pattern(self):
        """Test manually adding a pattern."""
        initial_count = len(self.layer.library)

        pattern = Pattern(
            pattern_id="manual_test",
            domain=DomainType.MATH,
            name="manual_pattern",
            description="Manually added test pattern",
            structure={"test": "value"}
        )

        success = self.layer.add_pattern(pattern)
        self.assertTrue(success)
        self.assertEqual(len(self.layer.library), initial_count + 1)

        # Try adding same pattern again
        success = self.layer.add_pattern(pattern)
        self.assertFalse(success)

    def test_get_pattern(self):
        """Test getting a pattern by ID."""
        # Add a pattern
        pattern = Pattern(
            pattern_id="get_test",
            domain=DomainType.CODE,
            name="get_test_pattern",
            description="Test",
            structure={}
        )
        self.layer.add_pattern(pattern)

        # Retrieve it
        retrieved = self.layer.get_pattern("get_test")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "get_test_pattern")

    def test_get_patterns_by_domain(self):
        """Test getting patterns by domain."""
        patterns = self.layer.get_patterns_by_domain(DomainType.MATH)
        self.assertIsInstance(patterns, list)

        # All patterns should be math domain
        for pattern in patterns:
            self.assertEqual(pattern.domain, DomainType.MATH)

    def test_get_patterns_by_complexity(self):
        """Test getting patterns by complexity."""
        patterns = self.layer.get_patterns_by_complexity(PatternComplexity.SIMPLE)
        self.assertIsInstance(patterns, list)

        # All patterns should be simple complexity
        for pattern in patterns:
            self.assertEqual(pattern.complexity, PatternComplexity.SIMPLE)

    def test_suggest_patterns(self):
        """Test pattern suggestions."""
        content = "Solve the equation 2x + 5 = 15"
        suggestions = self.layer.suggest_patterns(content, max_suggestions=3)

        self.assertIsInstance(suggestions, list)
        self.assertLessEqual(len(suggestions), 3)

        # Each suggestion should be a tuple of (pattern, confidence)
        for pattern, confidence in suggestions:
            self.assertIsInstance(pattern, Pattern)
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    def test_apply_pattern(self):
        """Test applying a pattern."""
        # Add a pattern
        pattern = Pattern(
            pattern_id="apply_test",
            domain=DomainType.MATH,
            name="apply_test_pattern",
            description="Test",
            structure={},
            transformation_steps=[
                {"action": "step1", "description": "First step"}
            ]
        )
        self.layer.add_pattern(pattern)

        # Apply it
        context = {"value": 10}
        result = self.layer.apply_pattern("apply_test", context)

        self.assertIsInstance(result, dict)
        self.assertIn("pattern_id", result)

    def test_save_and_load_library(self):
        """Test saving and loading pattern library."""
        # Add some patterns
        for i in range(3):
            pattern = Pattern(
                pattern_id=f"save_test_{i}",
                domain=DomainType.MATH,
                name=f"pattern_{i}",
                description=f"Test pattern {i}",
                structure={"index": i}
            )
            self.layer.add_pattern(pattern)

        initial_count = len(self.layer.library)

        # Save
        self.layer.save_library()
        self.assertTrue(self.storage_path.exists())

        # Create new layer and load
        new_layer = NuanceLayer(config={
            "storage_path": str(self.storage_path),
            "load_core_patterns": False
        })
        new_layer.load_library()

        # Should have same number of patterns
        self.assertEqual(len(new_layer.library), initial_count)

    def test_get_statistics(self):
        """Test getting layer statistics."""
        # Perform some operations
        self.layer.classify_content("test content")
        self.layer.extract_patterns("solve x = 5", domain=DomainType.MATH)

        stats = self.layer.get_statistics()

        self.assertIsInstance(stats, dict)
        self.assertIn("total_extractions", stats)
        self.assertIn("total_retrievals", stats)
        self.assertIn("total_classifications", stats)
        self.assertIn("library", stats)
        self.assertIn("supported_domains", stats)

    def test_reset_statistics(self):
        """Test resetting statistics."""
        # Perform some operations
        self.layer.classify_content("test")
        self.layer.extract_patterns("test", domain=DomainType.MATH)

        self.assertGreater(self.layer.stats["total_extractions"], 0)

        # Reset
        self.layer.reset_statistics()

        self.assertEqual(self.layer.stats["total_extractions"], 0)
        self.assertEqual(self.layer.stats["total_retrievals"], 0)
        self.assertEqual(self.layer.stats["total_classifications"], 0)

    def test_health_check(self):
        """Test health check."""
        health = self.layer.health_check()

        self.assertIsInstance(health, dict)
        self.assertIn("status", health)
        self.assertIn("issues", health)
        self.assertIn("metrics", health)

        # Should be healthy initially
        self.assertIn(health["status"], ["healthy", "warning", "degraded"])

    def test_core_patterns_loaded(self):
        """Test that core patterns are loaded."""
        # Check math patterns
        math_patterns = self.layer.get_patterns_by_domain(DomainType.MATH)
        self.assertGreater(len(math_patterns), 0)

        # Check code patterns
        code_patterns = self.layer.get_patterns_by_domain(DomainType.CODE)
        self.assertGreater(len(code_patterns), 0)

        # Check logic patterns
        logic_patterns = self.layer.get_patterns_by_domain(DomainType.LOGIC)
        self.assertGreater(len(logic_patterns), 0)

    def test_export_import_patterns(self):
        """Test exporting and importing patterns."""
        export_path = Path(self.temp_dir) / "export.json"

        # Export math patterns
        self.layer.export_patterns(export_path, domain=DomainType.MATH)
        self.assertTrue(export_path.exists())

        # Create new layer and import
        new_layer = NuanceLayer(config={"load_core_patterns": False})
        initial_count = len(new_layer.library)

        imported = new_layer.import_patterns(export_path)
        self.assertGreater(imported, 0)
        self.assertGreater(len(new_layer.library), initial_count)

    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.layer)
        self.assertIsInstance(repr_str, str)
        self.assertIn("NuanceLayer", repr_str)


if __name__ == '__main__':
    unittest.main()
