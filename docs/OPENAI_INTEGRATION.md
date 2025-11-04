# OpenAI API Integration for TMR Framework

Complete guide to using the OpenAI API integration with the Trinity Meta-Reasoning Framework.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Features](#features)
6. [Usage Examples](#usage-examples)
7. [API Reference](#api-reference)
8. [Advanced Usage](#advanced-usage)
9. [Best Practices](#best-practices)

## Overview

The TMR OpenAI integration provides a powerful wrapper around OpenAI's API with additional features:

- **Automatic Rate Limiting**: Token-based (TPM) and request-based (RPM) rate limiting
- **Error Handling**: Robust error handling with automatic retries and exponential backoff
- **Response Parsing**: Structured parsing of LLM outputs with reasoning chain extraction
- **Cost Tracking**: Automatic tracking of API usage and costs
- **Prompt Templates**: Pre-built templates for different reasoning tasks
- **TMR Verification**: Optional verification of reasoning using TMR's fundamental principles

## Installation

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install openai>=1.12.0 numpy>=1.21.0 python-dotenv>=0.19.0
```

## Quick Start

```python
from tmr.integrations import TMROpenAI

# Initialize client
client = TMROpenAI(api_key="your-api-key-here")

# Make a simple completion
response = client.chat_completion(
    messages=[{"role": "user", "content": "What is 2+2?"}]
)

print(response.content)
```

### With Environment Variables

Create a `.env` file:

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4
```

Then use:

```python
from tmr.integrations import TMROpenAI, OpenAIConfig

# Load from environment
config = OpenAIConfig.from_env()
client = TMROpenAI(config=config)
```

## Configuration

### OpenAI Configuration

```python
from tmr.integrations import OpenAIConfig

config = OpenAIConfig(
    api_key="your-key",
    org_id="your-org-id",  # Optional

    # Model settings
    model="gpt-4",
    temperature=0.7,
    max_tokens=2000,
    top_p=1.0,

    # Rate limiting
    rpm=3500,  # Requests per minute
    tpm=90000,  # Tokens per minute
    rate_limit_buffer=0.9,  # Use 90% of limits

    # Request settings
    timeout=30,
    max_retries=3,

    # Features
    enable_streaming=False,
    enable_cost_tracking=True,
)
```

### TMR Configuration

```python
from tmr.core import TMRConfig, VerificationDepth

tmr_config = TMRConfig(
    verification_depth=VerificationDepth.ADAPTIVE,
    confidence_threshold=0.7,
    max_retries=3,
    cache_enabled=True,
)
```

## Features

### 1. Rate Limiting

Automatic token and request rate limiting:

```python
from tmr.integrations import TMROpenAI

client = TMROpenAI(
    api_key="your-key",
    enable_rate_limiting=True
)

# Rate limiter automatically throttles requests
for i in range(100):
    response = client.chat_completion(
        messages=[{"role": "user", "content": f"Question {i}"}]
    )
```

### 2. Error Handling

Built-in retry logic with exponential backoff:

```python
from tmr.integrations import TMROpenAI, RateLimitError

client = TMROpenAI(api_key="your-key")

try:
    response = client.chat_completion(messages)
except RateLimitError as e:
    print(f"Rate limit hit: {e}")
    print(f"Retry after: {e.retry_after} seconds")
```

### 3. Response Parsing

Automatic extraction of reasoning steps, conclusions, and confidence:

```python
response = client.chat_completion(
    messages=[{
        "role": "user",
        "content": "Solve: 2x + 5 = 13. Show steps."
    }]
)

# Access parsed components
print(f"Content: {response.content}")
print(f"Steps: {len(response.reasoning_steps)}")
for step in response.reasoning_steps:
    print(f"  {step.step_number}. {step.description}")
print(f"Conclusion: {response.conclusion}")
print(f"Confidence: {response.confidence}")
```

### 4. Cost Tracking

Track API usage and costs:

```python
client = TMROpenAI(
    api_key="your-key",
    enable_cost_tracking=True
)

# Make requests
for i in range(10):
    client.chat_completion(messages)

# Get statistics
stats = client.get_usage_statistics()
print(f"Total requests: {stats['costs']['total_requests']}")
print(f"Total tokens: {stats['costs']['total_tokens']}")
print(f"Total cost: ${stats['costs']['total_cost_usd']}")
```

### 5. Prompt Templates

Use pre-built templates for different reasoning tasks:

```python
from tmr.integrations import PromptType

# Mathematical reasoning
response = client.create_with_template(
    prompt_type=PromptType.MATHEMATICAL,
    problem="Find the derivative of f(x) = 3x^2 + 2x + 1"
)

# Logical reasoning
response = client.create_with_template(
    prompt_type=PromptType.LOGICAL,
    problem="If all humans are mortal, and Socrates is human..."
)

# Causal reasoning
response = client.create_with_template(
    prompt_type=PromptType.CAUSAL,
    scenario="Company sales dropped after price increase..."
)
```

### 6. TMR Verification

Apply TMR's fundamental principles to verify reasoning:

```python
response = client.chat_completion(
    messages=[{
        "role": "user",
        "content": "Prove that 1+1=2"
    }],
    verify=True  # Enable TMR verification
)

# Check verification results
if "tmr_verification" in response.metadata:
    verification = response.metadata["tmr_verification"]
    print(f"Valid: {verification['valid']}")
    print(f"Confidence: {verification['confidence']}")
    print(f"Details: {verification['details']}")
```

## Usage Examples

### Example 1: Basic Q&A

```python
from tmr.integrations import quick_completion

answer = quick_completion(
    "What is the capital of France?",
    api_key="your-key"
)
print(answer)
```

### Example 2: Multi-step Problem Solving

```python
from tmr.integrations import TMROpenAI, PromptType

client = TMROpenAI(api_key="your-key")

response = client.create_with_template(
    prompt_type=PromptType.MATHEMATICAL,
    problem="""
    A train travels 120 km in 2 hours.
    How far will it travel in 5 hours at the same speed?
    """,
    verify=True
)

for step in response.reasoning_steps:
    print(f"{step.step_number}. {step.description}")
```

### Example 3: Custom Prompt Building

```python
from tmr.integrations import PromptBuilder, PromptType

builder = PromptBuilder(PromptType.LOGICAL)
builder.add_context("This is a philosophical debate")
builder.add_constraint("Provide exactly 3 arguments")
builder.add_constraint("Each argument should be 2-3 sentences")

messages = builder.build(problem="Does free will exist?")

response = client.chat_completion(messages)
```

### Example 4: Async Batch Processing

```python
import asyncio
from tmr.integrations import TMROpenAI

async def process_batch():
    client = TMROpenAI(api_key="your-key")

    questions = [
        "What is the speed of light?",
        "What is E=mc^2?",
        "Who discovered gravity?",
    ]

    tasks = [
        client.chat_completion_async([
            {"role": "user", "content": q}
        ])
        for q in questions
    ]

    responses = await asyncio.gather(*tasks)
    return responses

responses = asyncio.run(process_batch())
```

### Example 5: Cost Optimization

```python
from tmr.integrations import OpenAIConfig, calculate_cost

# Simple tasks: use cheaper model
simple_config = OpenAIConfig(
    api_key="your-key",
    model="gpt-3.5-turbo",
    max_tokens=100
)

# Complex tasks: use more capable model
complex_config = OpenAIConfig(
    api_key="your-key",
    model="gpt-4",
    max_tokens=500
)

# Estimate costs
simple_cost = calculate_cost("gpt-3.5-turbo", 100, 100)
complex_cost = calculate_cost("gpt-4", 500, 500)

print(f"Simple: ${simple_cost:.4f}")
print(f"Complex: ${complex_cost:.4f}")
```

## API Reference

### TMROpenAI

Main client class for OpenAI integration.

```python
class TMROpenAI:
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[OpenAIConfig] = None,
        enable_rate_limiting: bool = True,
        enable_cost_tracking: bool = True,
    )

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        verify: bool = False,
        parse_response: bool = True,
        **kwargs,
    ) -> Union[Any, ParsedResponse]

    async def chat_completion_async(...)

    def create_with_template(
        self,
        prompt_type: PromptType,
        verify: bool = False,
        **template_kwargs,
    ) -> ParsedResponse

    def get_usage_statistics(self) -> Dict[str, Any]
    def reset_statistics(self)
```

### ParsedResponse

Response object with structured information.

```python
@dataclass
class ParsedResponse:
    content: str
    reasoning_steps: List[ReasoningStep]
    conclusion: Optional[str]
    confidence: Optional[float]
    assumptions: List[str]
    uncertainties: List[str]
    token_usage: Optional[Dict[str, int]]
    model: Optional[str]
    metadata: Dict[str, Any]
```

### PromptType

Enum for prompt template types.

```python
class PromptType(Enum):
    LOGICAL = "logical"
    MATHEMATICAL = "mathematical"
    CAUSAL = "causal"
    GENERAL = "general"
    VERIFICATION = "verification"
    CHAIN_OF_THOUGHT = "chain_of_thought"
```

## Advanced Usage

### Custom Rate Limiting

```python
from tmr.integrations import RateLimitConfig

rate_config = RateLimitConfig(
    requests_per_minute=100,
    tokens_per_minute=10000,
    max_retries=5,
    retry_delay=2.0,
    exponential_backoff=True,
)
```

### Custom Error Handling

```python
from tmr.integrations import with_retry, ErrorContext

@with_retry(max_retries=5)
def my_api_call():
    # Your API call logic
    pass

# Or use context manager
with ErrorContext("my operation") as ctx:
    # Your code
    pass
```

### Token Estimation

```python
from tmr.integrations import TokenEstimator

text = "Hello, world!"
estimated_tokens = TokenEstimator.estimate_tokens(text, model="gpt-4")

messages = [{"role": "user", "content": "Hello"}]
estimated_tokens = TokenEstimator.estimate_message_tokens(messages, model="gpt-4")
```

## Best Practices

### 1. Use Environment Variables

Store API keys in environment variables, not in code:

```python
# .env
OPENAI_API_KEY=sk-...

# Code
config = OpenAIConfig.from_env()
```

### 2. Enable Rate Limiting

Always enable rate limiting for production:

```python
client = TMROpenAI(
    api_key="your-key",
    enable_rate_limiting=True  # Prevents 429 errors
)
```

### 3. Use Appropriate Models

Choose models based on task complexity:

- Simple tasks: `gpt-3.5-turbo` (cost-effective)
- Complex reasoning: `gpt-4` (more capable)
- Long context: `gpt-4-turbo` (128k context)

### 4. Track Costs

Monitor API usage to control costs:

```python
client = TMROpenAI(enable_cost_tracking=True)

# Periodically check
stats = client.get_usage_statistics()
if stats['costs']['total_cost_usd'] > 10.0:
    print("Warning: Cost threshold exceeded!")
```

### 5. Handle Errors Gracefully

Always wrap API calls in try-except:

```python
from tmr.integrations import TMRAPIError

try:
    response = client.chat_completion(messages)
except TMRAPIError as e:
    logger.error(f"API error: {e}")
    # Implement fallback logic
```

### 6. Use TMR Verification Selectively

TMR verification adds overhead - use for critical reasoning:

```python
# Critical reasoning tasks
response = client.chat_completion(messages, verify=True)

# Simple Q&A
response = client.chat_completion(messages, verify=False)
```

### 7. Optimize Token Usage

Minimize tokens to reduce costs:

```python
config = OpenAIConfig(
    api_key="your-key",
    max_tokens=200,  # Limit response length
    temperature=0.3,  # More deterministic (fewer retries)
)
```

## Troubleshooting

### Issue: Authentication Error

```python
AuthenticationError: Invalid API key
```

**Solution**: Check that OPENAI_API_KEY is set correctly.

### Issue: Rate Limit Exceeded

```python
RateLimitError: Rate limit exceeded
```

**Solution**: Enable rate limiting or reduce request frequency:

```python
client = TMROpenAI(enable_rate_limiting=True)
```

### Issue: Context Length Exceeded

```python
ContextLengthExceededError: Input exceeds model's context length
```

**Solution**: Reduce input length or use a model with larger context:

```python
config = OpenAIConfig(model="gpt-4-turbo")  # 128k context
```

## Support

For issues and questions:
- GitHub Issues: [repository]/issues
- Documentation: [repository]/docs
- Examples: [repository]/examples
