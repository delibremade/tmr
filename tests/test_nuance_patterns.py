"""
Tests for Pattern Data Structures
"""

import unittest
from datetime import datetime

from tmr.nuance.patterns import (
    Pattern,
    PatternMetadata,
    PatternMatch,
    DomainType,
    PatternComplexity
)


class TestPatternMetadata(unittest.TestCase):
    """Test PatternMetadata class."""

    def test_initialization(self):
        """Test metadata initialization."""
        metadata = PatternMetadata()
        self.assertEqual(metadata.usage_count, 0)
        self.assertEqual(metadata.success_rate, 0.0)
        self.assertIsInstance(metadata.created_at, datetime)

    def test_update_usage_success(self):
        """Test updating usage with success."""
        metadata = PatternMetadata()
        metadata.update_usage(success=True, confidence=0.8)

        self.assertEqual(metadata.usage_count, 1)
        self.assertEqual(metadata.success_rate, 1.0)
        self.assertEqual(len(metadata.confidence_scores), 1)
        self.assertEqual(metadata.confidence_scores[0], 0.8)

    def test_update_usage_failure(self):
        """Test updating usage with failure."""
        metadata = PatternMetadata()
        metadata.update_usage(success=False, confidence=0.3)

        self.assertEqual(metadata.usage_count, 1)
        self.assertEqual(metadata.success_rate, 0.0)

    def test_update_usage_multiple(self):
        """Test multiple usage updates."""
        metadata = PatternMetadata()
        metadata.update_usage(success=True, confidence=0.9)
        metadata.update_usage(success=True, confidence=0.8)
        metadata.update_usage(success=False, confidence=0.4)

        self.assertEqual(metadata.usage_count, 3)
        self.assertAlmostEqual(metadata.success_rate, 2/3, places=2)

    def test_get_avg_confidence(self):
        """Test average confidence calculation."""
        metadata = PatternMetadata()
        metadata.update_usage(success=True, confidence=0.8)
        metadata.update_usage(success=True, confidence=0.6)
        metadata.update_usage(success=True, confidence=0.9)

        avg = metadata.get_avg_confidence()
        self.assertAlmostEqual(avg, 0.7667, places=3)


class TestPattern(unittest.TestCase):
    """Test Pattern class."""

    def setUp(self):
        """Set up test patterns."""
        self.simple_pattern = Pattern(
            pattern_id="test_001",
            domain=DomainType.MATH,
            name="simple_arithmetic",
            description="Basic arithmetic pattern",
            structure={"operation": "addition"},
            complexity=PatternComplexity.SIMPLE
        )

    def test_initialization(self):
        """Test pattern initialization."""
        self.assertEqual(self.simple_pattern.domain, DomainType.MATH)
        self.assertEqual(self.simple_pattern.name, "simple_arithmetic")
        self.assertEqual(self.simple_pattern.complexity, PatternComplexity.SIMPLE)

    def test_auto_id_generation(self):
        """Test automatic ID generation."""
        pattern = Pattern(
            pattern_id="",
            domain=DomainType.CODE,
            name="test_pattern",
            description="Test",
            structure={"test": "value"}
        )
        self.assertNotEqual(pattern.pattern_id, "")
        self.assertIsInstance(pattern.pattern_id, str)

    def test_matches_exact(self):
        """Test pattern matching with exact match."""
        query = {
            "domain": DomainType.MATH.value,
            "structure": {"operation": "addition"}
        }
        matches, similarity = self.simple_pattern.matches(query, threshold=0.5)
        self.assertTrue(matches)
        self.assertGreater(similarity, 0.5)

    def test_matches_partial(self):
        """Test pattern matching with partial match."""
        query = {
            "domain": DomainType.MATH.value,
            "structure": {"operation": "addition", "extra": "field"}
        }
        matches, similarity = self.simple_pattern.matches(query, threshold=0.3)
        self.assertTrue(matches)

    def test_no_match(self):
        """Test pattern matching with no match."""
        query = {
            "domain": DomainType.CODE.value,
            "structure": {"different": "structure"}
        }
        matches, similarity = self.simple_pattern.matches(query, threshold=0.7)
        self.assertFalse(matches)

    def test_add_example(self):
        """Test adding examples."""
        example = {"input": "2 + 2", "output": "4"}
        self.simple_pattern.add_example(example)
        self.assertEqual(len(self.simple_pattern.examples), 1)
        self.assertEqual(self.simple_pattern.examples[0], example)

    def test_add_variation(self):
        """Test adding variations."""
        self.simple_pattern.add_variation("subtraction")
        self.assertEqual(len(self.simple_pattern.variations), 1)
        self.assertIn("subtraction", self.simple_pattern.variations)

        # Adding same variation should not duplicate
        self.simple_pattern.add_variation("subtraction")
        self.assertEqual(len(self.simple_pattern.variations), 1)

    def test_to_dict(self):
        """Test converting pattern to dictionary."""
        pattern_dict = self.simple_pattern.to_dict()
        self.assertIsInstance(pattern_dict, dict)
        self.assertEqual(pattern_dict["name"], "simple_arithmetic")
        self.assertEqual(pattern_dict["domain"], DomainType.MATH.value)
        self.assertIn("metadata", pattern_dict)

    def test_from_dict(self):
        """Test creating pattern from dictionary."""
        pattern_dict = self.simple_pattern.to_dict()
        restored = Pattern.from_dict(pattern_dict)

        self.assertEqual(restored.name, self.simple_pattern.name)
        self.assertEqual(restored.domain, self.simple_pattern.domain)
        self.assertEqual(restored.complexity, self.simple_pattern.complexity)

    def test_apply_pattern(self):
        """Test applying pattern to context."""
        context = {"value1": 5, "value2": 3}
        result = self.simple_pattern.apply(context)

        self.assertIsInstance(result, dict)
        self.assertIn("pattern_id", result)
        self.assertIn("confidence", result)


class TestPatternMatch(unittest.TestCase):
    """Test PatternMatch class."""

    def test_initialization(self):
        """Test pattern match initialization."""
        pattern = Pattern(
            pattern_id="test",
            domain=DomainType.LOGIC,
            name="test",
            description="Test pattern",
            structure={}
        )
        match = PatternMatch(
            pattern=pattern,
            similarity_score=0.8,
            confidence=0.9
        )

        self.assertEqual(match.similarity_score, 0.8)
        self.assertEqual(match.confidence, 0.9)
        self.assertEqual(match.pattern, pattern)

    def test_sorting(self):
        """Test sorting pattern matches."""
        pattern1 = Pattern("p1", DomainType.MATH, "p1", "P1", {})
        pattern2 = Pattern("p2", DomainType.MATH, "p2", "P2", {})
        pattern3 = Pattern("p3", DomainType.MATH, "p3", "P3", {})

        matches = [
            PatternMatch(pattern1, 0.5, 0.6),
            PatternMatch(pattern2, 0.9, 0.8),
            PatternMatch(pattern3, 0.7, 0.7)
        ]

        sorted_matches = sorted(matches, reverse=True)
        self.assertEqual(sorted_matches[0].similarity_score, 0.9)
        self.assertEqual(sorted_matches[1].similarity_score, 0.7)
        self.assertEqual(sorted_matches[2].similarity_score, 0.5)


if __name__ == '__main__':
    unittest.main()
