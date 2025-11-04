"""Advanced usage examples for TMR OpenAI integration."""

import os
import asyncio
from tmr.integrations import (
    TMROpenAI,
    PromptBuilder,
    PromptType,
    OpenAIConfig,
)


# Example 1: Custom prompt builder
def custom_prompt_builder():
    """Building custom prompts with constraints."""
    print("=== Example 1: Custom Prompt Builder ===")

    client = TMROpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Build a custom prompt with context and constraints
    builder = PromptBuilder(PromptType.LOGICAL)
    builder.add_context("This is a philosophical debate about free will")
    builder.add_constraint("Provide exactly 3 arguments")
    builder.add_constraint("Each argument should be 2-3 sentences")

    messages = builder.build(
        problem="Does free will exist?"
    )

    response = client.chat_completion(messages)
    print(f"Response:\n{response.content}")
    print()


# Example 2: Batch processing with rate limiting
async def batch_processing():
    """Process multiple requests with automatic rate limiting."""
    print("=== Example 2: Batch Processing with Rate Limiting ===")

    client = TMROpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        enable_rate_limiting=True
    )

    questions = [
        "What is the speed of light?",
        "What is the boiling point of water?",
        "What is the atomic number of carbon?",
        "What is Newton's first law?",
        "What is the chemical formula for water?",
    ]

    # Process all questions concurrently (rate limiter handles throttling)
    tasks = [
        client.chat_completion_async([{"role": "user", "content": q}])
        for q in questions
    ]

    responses = await asyncio.gather(*tasks)

    for question, response in zip(questions, responses):
        print(f"Q: {question}")
        print(f"A: {response.content[:100]}...")
        print()

    # Show rate limiting stats
    stats = client.get_usage_statistics()
    print(f"Rate limit hits: {stats['rate_limiting']['statistics']['rate_limit_hits']}")
    print(f"Total wait time: {stats['rate_limiting']['statistics']['total_wait_time_seconds']}s")
    print()


# Example 3: Multi-step reasoning with verification
def multi_step_reasoning():
    """Complex reasoning with step-by-step verification."""
    print("=== Example 3: Multi-step Reasoning ===")

    client = TMROpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {
            "role": "system",
            "content": "You are a mathematical reasoning expert. Always show your work step by step."
        },
        {
            "role": "user",
            "content": """Solve this problem step by step:

A train travels 120 km in 2 hours. If it maintains the same speed,
how far will it travel in 5 hours? Show all steps."""
        }
    ]

    response = client.chat_completion(messages, verify=True)

    print("Reasoning Steps:")
    for step in response.reasoning_steps:
        print(f"{step.step_number}. {step.description}")

    print(f"\nConclusion: {response.conclusion}")

    if "tmr_verification" in response.metadata:
        verification = response.metadata["tmr_verification"]
        print(f"\nTMR Verification:")
        print(f"  Valid: {verification['valid']}")
        print(f"  Confidence: {verification['confidence']}")

    print()


# Example 4: Structured output parsing
def structured_output():
    """Request and parse structured JSON output."""
    print("=== Example 4: Structured Output ===")

    client = TMROpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = PromptBuilder.for_structured_output(
        "Analyze the sentiment of this review: 'This product exceeded my expectations!'"
    )

    response = client.chat_completion(messages)

    print(f"Raw response:\n{response.content}\n")

    # Try to parse as structured output
    try:
        response_json = client.parser.parse_json_response(
            # Note: This would need the actual OpenAI response object
            # This is just for demonstration
            response
        )
        print(f"Structured data available: {bool(response_json.metadata.get('raw_json'))}")
    except Exception as e:
        print(f"Could not parse as JSON: {e}")

    print()


# Example 5: Cost optimization
def cost_optimization():
    """Optimize costs by selecting appropriate models."""
    print("=== Example 5: Cost Optimization ===")

    from tmr.integrations import calculate_cost

    # Simple task - use cheaper model
    simple_config = OpenAIConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",
        max_tokens=100
    )

    # Complex task - use more capable model
    complex_config = OpenAIConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4",
        max_tokens=500
    )

    # Estimate costs
    simple_cost = calculate_cost("gpt-3.5-turbo", 100, 100)
    complex_cost = calculate_cost("gpt-4", 500, 500)

    print(f"Simple task estimated cost: ${simple_cost:.4f}")
    print(f"Complex task estimated cost: ${complex_cost:.4f}")
    print(f"Cost difference: {complex_cost / simple_cost:.1f}x")
    print()


# Example 6: Causal reasoning template
def causal_reasoning():
    """Using causal reasoning template."""
    print("=== Example 6: Causal Reasoning ===")

    client = TMROpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.create_with_template(
        prompt_type=PromptType.CAUSAL,
        scenario="""A company's sales dropped by 30% after they raised prices by 20%
        and a competitor launched a similar product at a lower price point.""",
        verify=True
    )

    print(f"Analysis:\n{response.content}")

    if response.assumptions:
        print(f"\nAssumptions made:")
        for assumption in response.assumptions:
            print(f"  - {assumption}")

    if response.uncertainties:
        print(f"\nUncertainties:")
        for uncertainty in response.uncertainties:
            print(f"  - {uncertainty}")

    print()


# Example 7: TMR verification prompt
def tmr_verification():
    """Using TMR-specific verification."""
    print("=== Example 7: TMR Verification ===")

    client = TMROpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # First, get a reasoning response
    reasoning_response = client.chat_completion([
        {"role": "user", "content": "Explain why the sky is blue."}
    ])

    # Then verify it with TMR-specific prompt
    verification_messages = PromptBuilder.for_verification(
        reasoning=reasoning_response.content,
        context="Scientific explanation"
    )

    verification = client.chat_completion(verification_messages)

    print(f"Original reasoning:\n{reasoning_response.content}\n")
    print(f"TMR Verification:\n{verification.content}")
    print()


if __name__ == "__main__":
    print("TMR OpenAI Advanced Examples\n")

    # Run examples (uncomment and provide API key):
    # custom_prompt_builder()
    # asyncio.run(batch_processing())
    # multi_step_reasoning()
    # structured_output()
    # cost_optimization()
    # causal_reasoning()
    # tmr_verification()

    print("\nNote: Set OPENAI_API_KEY environment variable to run these examples.")
