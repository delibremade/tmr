"""Basic usage examples for TMR OpenAI integration."""

import os
from tmr.integrations import TMROpenAI, OpenAIConfig, PromptType

# Example 1: Basic completion
def basic_completion():
    """Simple completion without TMR verification."""
    print("=== Example 1: Basic Completion ===")

    client = TMROpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {"role": "user", "content": "What is 2 + 2?"}
    ]

    response = client.chat_completion(messages)

    print(f"Response: {response.content}")
    print(f"Confidence: {response.confidence}")
    print(f"Token Usage: {response.token_usage}")
    print()


# Example 2: Completion with TMR verification
def completion_with_verification():
    """Completion with TMR verification enabled."""
    print("=== Example 2: Completion with TMR Verification ===")

    client = TMROpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {"role": "user", "content": "Solve for x: 2x + 5 = 13. Show your steps."}
    ]

    response = client.chat_completion(messages, verify=True)

    print(f"Response: {response.content}")
    print(f"Reasoning Steps: {len(response.reasoning_steps)}")
    for step in response.reasoning_steps:
        print(f"  Step {step.step_number}: {step.description}")
    print(f"Conclusion: {response.conclusion}")
    print(f"TMR Verification: {response.metadata.get('tmr_verification', 'N/A')}")
    print()


# Example 3: Using prompt templates
def using_templates():
    """Using predefined prompt templates."""
    print("=== Example 3: Using Prompt Templates ===")

    client = TMROpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Mathematical reasoning template
    response = client.create_with_template(
        prompt_type=PromptType.MATHEMATICAL,
        problem="Find the derivative of f(x) = 3x^2 + 2x + 1",
        verify=True
    )

    print(f"Response: {response.content}")
    print(f"Steps: {len(response.reasoning_steps)}")
    print()


# Example 4: Custom configuration
def custom_configuration():
    """Using custom configuration."""
    print("=== Example 4: Custom Configuration ===")

    config = OpenAIConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=500,
        rpm=100,  # Custom rate limit
        tpm=10000,
    )

    client = TMROpenAI(config=config)

    messages = [
        {"role": "user", "content": "Explain quantum entanglement briefly."}
    ]

    response = client.chat_completion(messages)

    print(f"Model: {response.model}")
    print(f"Response: {response.content}")
    print()


# Example 5: Usage statistics
def usage_statistics():
    """Tracking usage statistics."""
    print("=== Example 5: Usage Statistics ===")

    client = TMROpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        enable_cost_tracking=True
    )

    # Make several requests
    for i in range(3):
        messages = [{"role": "user", "content": f"Count to {i + 1}"}]
        client.chat_completion(messages)

    # Get statistics
    stats = client.get_usage_statistics()

    print(f"Total Requests: {stats['costs']['total_requests']}")
    print(f"Total Tokens: {stats['costs']['total_tokens']}")
    print(f"Total Cost: ${stats['costs']['total_cost_usd']}")
    print(f"Average Cost per Request: ${stats['costs']['average_cost_per_request']}")
    print()


# Example 6: Async usage
async def async_completion():
    """Async completion example."""
    print("=== Example 6: Async Completion ===")

    client = TMROpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]

    response = await client.chat_completion_async(messages)

    print(f"Response: {response.content}")
    print()


# Example 7: Error handling
def error_handling():
    """Demonstrating error handling."""
    print("=== Example 7: Error Handling ===")

    from tmr.integrations import AuthenticationError

    try:
        # Try with invalid API key
        client = TMROpenAI(api_key="invalid-key")
        messages = [{"role": "user", "content": "Hello"}]
        client.chat_completion(messages)
    except AuthenticationError as e:
        print(f"Caught authentication error: {e}")
    print()


if __name__ == "__main__":
    # Run examples (comment out those requiring API key)
    print("TMR OpenAI Integration Examples\n")

    # Uncomment and run with valid API key:
    # basic_completion()
    # completion_with_verification()
    # using_templates()
    # custom_configuration()
    # usage_statistics()
    # error_handling()

    # For async example:
    # import asyncio
    # asyncio.run(async_completion())

    print("\nNote: Set OPENAI_API_KEY environment variable to run these examples.")
