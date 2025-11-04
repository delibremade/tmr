"""Tests for prompt templates."""

import pytest
from tmr.integrations.prompt_templates import (
    PromptType,
    PromptTemplate,
    PromptBuilder,
    get_template,
    create_messages,
)


def test_prompt_template_basic():
    """Test basic prompt template."""
    template = PromptTemplate(
        system_prompt="You are a helpful assistant.",
        user_template="Please answer: {question}"
    )

    messages = template.format(question="What is 2+2?")

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "user"
    assert "What is 2+2?" in messages[1]["content"]


def test_prompt_template_with_examples():
    """Test prompt template with few-shot examples."""
    template = PromptTemplate(
        system_prompt="You are a math tutor.",
        user_template="Solve: {problem}",
        few_shot_examples=[
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ]
    )

    messages = template.format(problem="What is 2+2?")

    assert len(messages) == 4  # system + 2 examples + user
    assert messages[1]["content"] == "What is 1+1?"
    assert messages[2]["content"] == "2"


def test_get_template():
    """Test getting predefined templates."""
    # Get logical template
    logical_template = get_template(PromptType.LOGICAL)
    assert logical_template is not None
    assert "logical" in logical_template.system_prompt.lower()

    # Get mathematical template
    math_template = get_template(PromptType.MATHEMATICAL)
    assert math_template is not None
    assert "mathematical" in math_template.system_prompt.lower()


def test_create_messages():
    """Test creating messages from template."""
    messages = create_messages(
        PromptType.MATHEMATICAL,
        problem="Solve for x: 2x + 5 = 13"
    )

    assert len(messages) >= 2
    assert messages[0]["role"] == "system"
    assert any("2x + 5 = 13" in msg["content"] for msg in messages)


def test_prompt_builder():
    """Test prompt builder."""
    builder = PromptBuilder(PromptType.GENERAL)
    builder.add_context("This is important context")
    builder.add_constraint("Be concise")

    messages = builder.build(question="What is AI?")

    assert len(messages) >= 2
    user_message = messages[-1]["content"]
    assert "What is AI?" in user_message
    assert "important context" in user_message
    assert "Be concise" in user_message


def test_prompt_builder_chaining():
    """Test prompt builder method chaining."""
    builder = PromptBuilder(PromptType.LOGICAL)
    messages = (
        builder
        .add_context("Context 1")
        .add_context("Context 2")
        .add_constraint("Constraint 1")
        .build(problem="Test problem")
    )

    assert len(messages) >= 2
    user_message = messages[-1]["content"]
    assert "Context 1" in user_message
    assert "Context 2" in user_message
    assert "Constraint 1" in user_message


def test_prompt_builder_verification():
    """Test TMR verification prompt builder."""
    messages = PromptBuilder.for_verification(
        reasoning="Step 1: A. Step 2: B. Conclusion: C",
        context="Mathematical proof"
    )

    assert len(messages) >= 2
    assert any("verification" in msg["content"].lower() for msg in messages)
    assert any("Step 1" in msg["content"] for msg in messages)


def test_prompt_builder_structured_output():
    """Test structured output prompt builder."""
    messages = PromptBuilder.for_structured_output(
        question="What is the weather like?"
    )

    assert len(messages) >= 2
    system_message = messages[0]["content"]
    assert "JSON" in system_message or "json" in system_message
