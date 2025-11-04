"""Tests for response parsing."""

import pytest
from tmr.integrations.response_parsing import (
    ReasoningStep,
    ParsedResponse,
    ResponseParser,
    CostTracker,
)


def test_reasoning_step():
    """Test reasoning step creation."""
    step = ReasoningStep(
        step_number=1,
        description="First step",
        confidence=0.9,
        dependencies=[0]
    )

    assert step.step_number == 1
    assert step.description == "First step"
    assert step.confidence == 0.9
    assert step.dependencies == [0]

    # Test to_dict
    step_dict = step.to_dict()
    assert step_dict["step_number"] == 1
    assert step_dict["description"] == "First step"


def test_parsed_response():
    """Test parsed response creation."""
    response = ParsedResponse(
        content="Test content",
        conclusion="Test conclusion",
        confidence=0.85
    )

    assert response.content == "Test content"
    assert response.conclusion == "Test conclusion"
    assert response.confidence == 0.85

    # Test to_dict
    response_dict = response.to_dict()
    assert response_dict["content"] == "Test content"
    assert response_dict["conclusion"] == "Test conclusion"


def test_response_parser_extract_steps():
    """Test extracting reasoning steps from content."""
    parser = ResponseParser()

    content = """
    Step 1: First, we identify the problem.
    Step 2: Then, we analyze the data.
    Step 3: Finally, we draw conclusions.
    """

    steps = parser._extract_reasoning_steps(content)

    assert len(steps) == 3
    assert steps[0].step_number == 1
    assert "identify" in steps[0].description.lower()
    assert steps[1].step_number == 2
    assert "analyze" in steps[1].description.lower()


def test_response_parser_extract_steps_numbered_list():
    """Test extracting steps from numbered list."""
    parser = ResponseParser()

    content = """
    1. First item
    2. Second item
    3. Third item
    """

    steps = parser._extract_reasoning_steps(content)

    assert len(steps) == 3
    assert steps[0].step_number == 1
    assert "First" in steps[0].description


def test_response_parser_extract_conclusion():
    """Test extracting conclusion from content."""
    parser = ResponseParser()

    content = """
    We analyzed the data carefully.

    Conclusion: The hypothesis is supported by the evidence.
    """

    conclusion = parser._extract_conclusion(content)

    assert conclusion is not None
    assert "hypothesis" in conclusion.lower()
    assert "supported" in conclusion.lower()


def test_response_parser_extract_confidence():
    """Test extracting confidence from content."""
    parser = ResponseParser()

    # Test percentage format
    content1 = "I am 85% confident in this result."
    confidence1 = parser._extract_confidence(content1)
    assert confidence1 == pytest.approx(0.85, rel=1e-6)

    # Test decimal format
    content2 = "Confidence: 0.9"
    confidence2 = parser._extract_confidence(content2)
    assert confidence2 == pytest.approx(0.9, rel=1e-6)

    # Test no confidence
    content3 = "No confidence mentioned."
    confidence3 = parser._extract_confidence(content3)
    assert confidence3 is None


def test_response_parser_extract_assumptions():
    """Test extracting assumptions from content."""
    parser = ResponseParser()

    content = """
    Assumptions:
    - The data is accurate
    - The sample is representative
    - External factors are minimal
    """

    assumptions = parser._extract_assumptions(content)

    assert len(assumptions) == 3
    assert any("accurate" in a.lower() for a in assumptions)
    assert any("representative" in a.lower() for a in assumptions)


def test_cost_tracker():
    """Test cost tracking."""
    tracker = CostTracker()

    # Add some usage
    tracker.add_usage("gpt-4", 1000, 500, 0.045)
    tracker.add_usage("gpt-4", 2000, 1000, 0.090)
    tracker.add_usage("gpt-3.5-turbo", 1000, 500, 0.001)

    stats = tracker.get_statistics()

    assert stats["total_requests"] == 3
    assert stats["total_input_tokens"] == 4000
    assert stats["total_output_tokens"] == 2000
    assert stats["total_tokens"] == 6000
    assert stats["total_cost_usd"] == pytest.approx(0.136, rel=1e-3)
    assert "gpt-4" in stats["costs_by_model"]
    assert "gpt-3.5-turbo" in stats["costs_by_model"]

    # Reset
    tracker.reset()
    stats = tracker.get_statistics()
    assert stats["total_requests"] == 0
    assert stats["total_cost_usd"] == 0.0
