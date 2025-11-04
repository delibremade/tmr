"""Prompt templates for OpenAI integration with TMR framework."""

from typing import Dict, List, Optional, Any
from enum import Enum


class PromptType(Enum):
    """Types of prompts for different reasoning tasks."""

    LOGICAL = "logical"
    MATHEMATICAL = "mathematical"
    CAUSAL = "causal"
    GENERAL = "general"
    VERIFICATION = "verification"
    CHAIN_OF_THOUGHT = "chain_of_thought"


class PromptTemplate:
    """Base class for prompt templates."""

    def __init__(
        self,
        system_prompt: str,
        user_template: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ):
        """Initialize prompt template.

        Args:
            system_prompt: System-level instructions
            user_template: Template for user messages (with placeholders)
            few_shot_examples: List of example interactions
        """
        self.system_prompt = system_prompt
        self.user_template = user_template
        self.few_shot_examples = few_shot_examples or []

    def format(self, **kwargs) -> List[Dict[str, str]]:
        """Format the template with provided variables.

        Args:
            **kwargs: Variables to fill in the template

        Returns:
            List of message dicts for OpenAI API
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add few-shot examples
        for example in self.few_shot_examples:
            messages.append(example)

        # Add user message
        user_content = self.user_template.format(**kwargs)
        messages.append({"role": "user", "content": user_content})

        return messages


# System prompts for different reasoning types
SYSTEM_PROMPTS = {
    PromptType.LOGICAL: """You are an expert in logical reasoning and formal logic.
Your task is to provide clear, step-by-step logical reasoning that follows these principles:
1. Identity: Each entity maintains its identity (A = A)
2. Non-contradiction: No statement can be both true and false simultaneously
3. Excluded middle: Every statement is either true or false

Always structure your reasoning as explicit steps with clear dependencies.""",
    PromptType.MATHEMATICAL: """You are an expert in mathematical reasoning and problem-solving.
Your task is to provide clear, step-by-step mathematical reasoning that:
1. Shows all intermediate steps
2. Maintains conservation of equality
3. Uses valid mathematical operations
4. Verifies results

Always show your work and explain each transformation.""",
    PromptType.CAUSAL: """You are an expert in causal reasoning and analysis.
Your task is to identify and explain causal relationships that:
1. Respect temporal ordering (causes precede effects)
2. Show clear causal mechanisms
3. Avoid circular causality
4. Consider alternative explanations

Always establish clear causal chains with evidence.""",
    PromptType.GENERAL: """You are an AI assistant specialized in clear, logical reasoning.
Provide well-structured responses that:
1. Break down complex problems into steps
2. Show your reasoning process
3. Acknowledge uncertainties
4. Support claims with evidence

Be precise, accurate, and transparent about your reasoning.""",
    PromptType.VERIFICATION: """You are a reasoning verification expert.
Your task is to analyze reasoning chains and identify:
1. Logical errors or fallacies
2. Mathematical mistakes
3. Causal inconsistencies
4. Missing steps or assumptions

Provide a detailed analysis with specific issues highlighted.""",
    PromptType.CHAIN_OF_THOUGHT: """You are an expert problem solver.
Use chain-of-thought reasoning to solve problems:
1. Break down the problem into components
2. Reason through each component step-by-step
3. Show intermediate conclusions
4. Synthesize the final answer

Think step-by-step and show all your reasoning.""",
}


# Template definitions
TEMPLATES = {
    PromptType.LOGICAL: PromptTemplate(
        system_prompt=SYSTEM_PROMPTS[PromptType.LOGICAL],
        user_template="""Analyze the following logical reasoning problem:

Problem: {problem}

Provide a step-by-step logical analysis with clear reasoning chains.""",
        few_shot_examples=[
            {
                "role": "user",
                "content": "If all humans are mortal, and Socrates is human, what can we conclude?",
            },
            {
                "role": "assistant",
                "content": """Step 1: Premise 1 - All humans are mortal (Universal statement)
Step 2: Premise 2 - Socrates is human (Particular statement)
Step 3: Apply universal instantiation - If all humans are mortal, then Socrates (being human) is mortal
Step 4: Conclusion - Socrates is mortal (Valid by modus ponens)

This is a valid deductive argument following the form of categorical syllogism.""",
            },
        ],
    ),
    PromptType.MATHEMATICAL: PromptTemplate(
        system_prompt=SYSTEM_PROMPTS[PromptType.MATHEMATICAL],
        user_template="""Solve the following mathematical problem:

Problem: {problem}

Show all steps and verify your answer.""",
        few_shot_examples=[
            {
                "role": "user",
                "content": "Solve for x: 2x + 5 = 13",
            },
            {
                "role": "assistant",
                "content": """Step 1: Start with equation: 2x + 5 = 13
Step 2: Subtract 5 from both sides: 2x + 5 - 5 = 13 - 5
Step 3: Simplify: 2x = 8
Step 4: Divide both sides by 2: 2x/2 = 8/2
Step 5: Simplify: x = 4

Verification: 2(4) + 5 = 8 + 5 = 13 ✓

Answer: x = 4""",
            },
        ],
    ),
    PromptType.CAUSAL: PromptTemplate(
        system_prompt=SYSTEM_PROMPTS[PromptType.CAUSAL],
        user_template="""Analyze the causal relationships in the following scenario:

Scenario: {scenario}

Identify causes, effects, and the causal mechanisms involved.""",
        few_shot_examples=[
            {
                "role": "user",
                "content": "Why did the plant die?",
            },
            {
                "role": "assistant",
                "content": """Causal Analysis:

Event: Plant died (Effect)

Potential Causes (with temporal ordering):
1. T0: Plant was placed in location with insufficient light
2. T1: Lack of photosynthesis due to insufficient light
3. T2: Depletion of energy reserves
4. T3: Cellular death and plant death

Causal Mechanism:
- Light deficiency → Reduced photosynthesis → Energy depletion → Cell death

Alternative causes to consider:
- Lack of water (dehydration)
- Disease or pests
- Extreme temperatures
- Nutrient deficiency

Evidence needed:
- Light exposure levels
- Watering history
- Plant appearance (color, texture)
- Environmental conditions""",
            },
        ],
    ),
    PromptType.GENERAL: PromptTemplate(
        system_prompt=SYSTEM_PROMPTS[PromptType.GENERAL],
        user_template="""{question}""",
    ),
    PromptType.VERIFICATION: PromptTemplate(
        system_prompt=SYSTEM_PROMPTS[PromptType.VERIFICATION],
        user_template="""Verify the following reasoning chain:

{reasoning_chain}

Identify any logical errors, inconsistencies, or missing steps.""",
    ),
    PromptType.CHAIN_OF_THOUGHT: PromptTemplate(
        system_prompt=SYSTEM_PROMPTS[PromptType.CHAIN_OF_THOUGHT],
        user_template="""{problem}

Let's solve this step by step:""",
    ),
}


# Structured output templates
STRUCTURED_OUTPUT_TEMPLATE = PromptTemplate(
    system_prompt="""You are an AI assistant that provides structured reasoning outputs.

Your response must follow this exact JSON format:
{
  "reasoning_steps": [
    {"step": 1, "description": "...", "confidence": 0.95},
    {"step": 2, "description": "...", "confidence": 0.90}
  ],
  "conclusion": "...",
  "overall_confidence": 0.92,
  "assumptions": ["..."],
  "uncertainties": ["..."]
}

Be precise and always return valid JSON.""",
    user_template="""{question}

Provide your reasoning in the structured JSON format.""",
)


# TMR-specific verification template
TMR_VERIFICATION_TEMPLATE = PromptTemplate(
    system_prompt="""You are a Trinity Meta-Reasoning verification assistant.

Your task is to analyze reasoning and identify potential issues related to:

1. Identity Principle: Does each entity maintain consistent identity?
2. Non-Contradiction Principle: Are there any contradictory statements?
3. Excluded Middle Principle: Are truth values properly binary?
4. Causality Principle: Is temporal ordering respected?
5. Conservation Principle: Are properties properly conserved?

Provide detailed verification feedback.""",
    user_template="""Verify the following reasoning against TMR principles:

Reasoning:
{reasoning}

Context:
{context}

Analyze adherence to each of the five TMR fundamental principles.""",
)


class PromptBuilder:
    """Builder class for constructing complex prompts."""

    def __init__(self, prompt_type: PromptType = PromptType.GENERAL):
        """Initialize prompt builder.

        Args:
            prompt_type: Type of prompt to build
        """
        self.prompt_type = prompt_type
        self.template = TEMPLATES.get(prompt_type, TEMPLATES[PromptType.GENERAL])
        self.additional_context: List[str] = []
        self.constraints: List[str] = []

    def add_context(self, context: str) -> "PromptBuilder":
        """Add additional context to the prompt.

        Args:
            context: Context to add

        Returns:
            Self for chaining
        """
        self.additional_context.append(context)
        return self

    def add_constraint(self, constraint: str) -> "PromptBuilder":
        """Add a constraint to the prompt.

        Args:
            constraint: Constraint to add

        Returns:
            Self for chaining
        """
        self.constraints.append(constraint)
        return self

    def build(self, **kwargs) -> List[Dict[str, str]]:
        """Build the final prompt.

        Args:
            **kwargs: Variables for template formatting

        Returns:
            List of message dicts for OpenAI API
        """
        # Get base messages
        messages = self.template.format(**kwargs)

        # Add additional context if present
        if self.additional_context:
            context_text = "\n\nAdditional Context:\n" + "\n".join(
                f"- {ctx}" for ctx in self.additional_context
            )
            messages[-1]["content"] += context_text

        # Add constraints if present
        if self.constraints:
            constraints_text = "\n\nConstraints:\n" + "\n".join(
                f"- {constraint}" for constraint in self.constraints
            )
            messages[-1]["content"] += constraints_text

        return messages

    @staticmethod
    def for_verification(reasoning: str, context: str = "") -> List[Dict[str, str]]:
        """Build a TMR verification prompt.

        Args:
            reasoning: Reasoning to verify
            context: Additional context

        Returns:
            List of message dicts for OpenAI API
        """
        return TMR_VERIFICATION_TEMPLATE.format(
            reasoning=reasoning, context=context or "No additional context"
        )

    @staticmethod
    def for_structured_output(question: str) -> List[Dict[str, str]]:
        """Build a prompt for structured output.

        Args:
            question: Question to answer

        Returns:
            List of message dicts for OpenAI API
        """
        return STRUCTURED_OUTPUT_TEMPLATE.format(question=question)


def get_template(prompt_type: PromptType) -> PromptTemplate:
    """Get a prompt template by type.

    Args:
        prompt_type: Type of prompt template

    Returns:
        PromptTemplate instance

    Raises:
        ValueError: If prompt type is not found
    """
    template = TEMPLATES.get(prompt_type)
    if not template:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return template


def create_messages(
    prompt_type: PromptType, **kwargs
) -> List[Dict[str, str]]:
    """Create formatted messages for a prompt type.

    Args:
        prompt_type: Type of prompt
        **kwargs: Variables for template formatting

    Returns:
        List of message dicts for OpenAI API
    """
    template = get_template(prompt_type)
    return template.format(**kwargs)
