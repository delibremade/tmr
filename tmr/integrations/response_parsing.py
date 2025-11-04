"""Response parsing utilities for OpenAI API responses."""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ReasoningStep:
    """A single reasoning step extracted from LLM output.

    Attributes:
        step_number: Step number in sequence
        description: Description of the reasoning step
        confidence: Confidence score (0.0-1.0)
        dependencies: List of step numbers this depends on
    """

    step_number: int
    description: str
    confidence: float = 1.0
    dependencies: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_number": self.step_number,
            "description": self.description,
            "confidence": self.confidence,
            "dependencies": self.dependencies,
        }


@dataclass
class ParsedResponse:
    """Parsed LLM response with structured information.

    Attributes:
        content: Raw text content
        reasoning_steps: Extracted reasoning steps
        conclusion: Final conclusion
        confidence: Overall confidence score
        assumptions: List of assumptions made
        uncertainties: List of identified uncertainties
        token_usage: Token usage information
        model: Model used
        finish_reason: Reason for completion
        created_at: Timestamp of creation
        metadata: Additional metadata
    """

    content: str
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    conclusion: Optional[str] = None
    confidence: Optional[float] = None
    assumptions: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    token_usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "reasoning_steps": [step.to_dict() for step in self.reasoning_steps],
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "assumptions": self.assumptions,
            "uncertainties": self.uncertainties,
            "token_usage": self.token_usage,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata,
        }


class ResponseParser:
    """Parser for OpenAI API responses."""

    # Patterns for extracting structured information
    STEP_PATTERNS = [
        r"Step\s+(\d+):\s*(.+?)(?=Step\s+\d+:|$)",  # Step 1: ...
        r"(\d+)\.\s+(.+?)(?=\d+\.|$)",  # 1. ...
        r"(?:^|\n)[-•]\s*(.+?)(?=\n[-•]|$)",  # - ... or • ...
    ]

    CONCLUSION_PATTERNS = [
        r"(?:Conclusion|Therefore|Thus|Hence|Final Answer|Answer):\s*(.+?)(?=\n\n|$)",
        r"(?:In conclusion|To conclude),\s*(.+?)(?=\n\n|$)",
    ]

    CONFIDENCE_PATTERNS = [
        r"(?:confidence|certainty):\s*(\d+(?:\.\d+)?)\s*(?:%|/100)?",
        r"(?:I am|We are)\s+(\d+)%\s+(?:confident|certain)",
    ]

    def __init__(self):
        """Initialize response parser."""
        self.step_pattern = re.compile(
            "|".join(self.STEP_PATTERNS), re.MULTILINE | re.DOTALL
        )
        self.conclusion_pattern = re.compile(
            "|".join(self.CONCLUSION_PATTERNS), re.IGNORECASE | re.DOTALL
        )
        self.confidence_pattern = re.compile(
            "|".join(self.CONFIDENCE_PATTERNS), re.IGNORECASE
        )

    def parse(self, response: Any) -> ParsedResponse:
        """Parse OpenAI API response.

        Args:
            response: OpenAI API response object

        Returns:
            ParsedResponse with structured information
        """
        # Extract basic information
        content = self._extract_content(response)
        token_usage = self._extract_token_usage(response)
        model = getattr(response, "model", None)
        finish_reason = self._extract_finish_reason(response)

        # Parse content for structured information
        reasoning_steps = self._extract_reasoning_steps(content)
        conclusion = self._extract_conclusion(content)
        confidence = self._extract_confidence(content)
        assumptions = self._extract_assumptions(content)
        uncertainties = self._extract_uncertainties(content)

        return ParsedResponse(
            content=content,
            reasoning_steps=reasoning_steps,
            conclusion=conclusion,
            confidence=confidence,
            assumptions=assumptions,
            uncertainties=uncertainties,
            token_usage=token_usage,
            model=model,
            finish_reason=finish_reason,
            created_at=datetime.now(),
        )

    def parse_json_response(self, response: Any) -> ParsedResponse:
        """Parse structured JSON response.

        Args:
            response: OpenAI API response with JSON content

        Returns:
            ParsedResponse with structured information

        Raises:
            ValueError: If JSON parsing fails
        """
        content = self._extract_content(response)

        try:
            # Try to extract JSON from content
            json_data = self._extract_json(content)

            # Parse structured fields
            reasoning_steps = []
            if "reasoning_steps" in json_data:
                for step_data in json_data["reasoning_steps"]:
                    reasoning_steps.append(
                        ReasoningStep(
                            step_number=step_data.get("step", 0),
                            description=step_data.get("description", ""),
                            confidence=step_data.get("confidence", 1.0),
                            dependencies=step_data.get("dependencies", []),
                        )
                    )

            return ParsedResponse(
                content=content,
                reasoning_steps=reasoning_steps,
                conclusion=json_data.get("conclusion"),
                confidence=json_data.get("overall_confidence"),
                assumptions=json_data.get("assumptions", []),
                uncertainties=json_data.get("uncertainties", []),
                token_usage=self._extract_token_usage(response),
                model=getattr(response, "model", None),
                finish_reason=self._extract_finish_reason(response),
                created_at=datetime.now(),
                metadata={"raw_json": json_data},
            )

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback to regular parsing
            return self.parse(response)

    def _extract_content(self, response: Any) -> str:
        """Extract content from response."""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                return choice.message.content or ""
            elif hasattr(choice, "text"):
                return choice.text or ""
        return str(response)

    def _extract_token_usage(self, response: Any) -> Optional[Dict[str, int]]:
        """Extract token usage information."""
        if hasattr(response, "usage"):
            usage = response.usage
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
        return None

    def _extract_finish_reason(self, response: Any) -> Optional[str]:
        """Extract finish reason."""
        if hasattr(response, "choices") and response.choices:
            return getattr(response.choices[0], "finish_reason", None)
        return None

    def _extract_reasoning_steps(self, content: str) -> List[ReasoningStep]:
        """Extract reasoning steps from content."""
        steps = []

        # Try numbered steps (Step 1:, 1., etc.)
        step_matches = re.finditer(
            r"(?:Step\s+)?(\d+)[.:]?\s+(.+?)(?=(?:Step\s+)?\d+[.:]|\n\n|$)",
            content,
            re.IGNORECASE | re.DOTALL,
        )

        for match in step_matches:
            step_num = int(match.group(1))
            description = match.group(2).strip()
            if description:
                steps.append(
                    ReasoningStep(
                        step_number=step_num,
                        description=description,
                        confidence=1.0,
                    )
                )

        # If no numbered steps, try bullet points
        if not steps:
            bullet_matches = re.finditer(
                r"^[\s]*[-•*]\s+(.+?)$", content, re.MULTILINE
            )
            for i, match in enumerate(bullet_matches, 1):
                description = match.group(1).strip()
                if description:
                    steps.append(
                        ReasoningStep(
                            step_number=i, description=description, confidence=1.0
                        )
                    )

        return steps

    def _extract_conclusion(self, content: str) -> Optional[str]:
        """Extract conclusion from content."""
        match = self.conclusion_pattern.search(content)
        if match:
            # Get the last non-empty group
            for group in reversed(match.groups()):
                if group:
                    return group.strip()
        return None

    def _extract_confidence(self, content: str) -> Optional[float]:
        """Extract confidence score from content."""
        match = self.confidence_pattern.search(content)
        if match:
            for group in match.groups():
                if group:
                    try:
                        value = float(group)
                        # Normalize to 0-1 if it's a percentage
                        if value > 1.0:
                            value /= 100.0
                        return max(0.0, min(1.0, value))
                    except ValueError:
                        continue
        return None

    def _extract_assumptions(self, content: str) -> List[str]:
        """Extract assumptions from content."""
        assumptions = []

        # Look for "Assumptions:" section
        match = re.search(
            r"Assumptions?:\s*(.+?)(?=\n\n|\n[A-Z][a-z]+:|$)",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            assumptions_text = match.group(1)
            # Extract bullet points or numbered items
            items = re.findall(r"[-•*\d.]\s+(.+?)(?=\n[-•*\d.]|\n\n|$)", assumptions_text)
            assumptions.extend(item.strip() for item in items if item.strip())

        return assumptions

    def _extract_uncertainties(self, content: str) -> List[str]:
        """Extract uncertainties from content."""
        uncertainties = []

        # Look for "Uncertainties:" or "Limitations:" section
        match = re.search(
            r"(?:Uncertainties?|Limitations?):\s*(.+?)(?=\n\n|\n[A-Z][a-z]+:|$)",
            content,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            uncertainties_text = match.group(1)
            # Extract bullet points or numbered items
            items = re.findall(
                r"[-•*\d.]\s+(.+?)(?=\n[-•*\d.]|\n\n|$)", uncertainties_text
            )
            uncertainties.extend(item.strip() for item in items if item.strip())

        # Also look for phrases indicating uncertainty
        uncertainty_phrases = [
            r"(?:I'm |I am )?not (?:entirely |completely )?(?:sure|certain)",
            r"(?:It's |It is )?unclear",
            r"(?:might|may|could) be",
            r"(?:possibly|perhaps|maybe)",
        ]

        for phrase in uncertainty_phrases:
            matches = re.finditer(
                f"({phrase}.+?[.!?])", content, re.IGNORECASE
            )
            for match in matches:
                uncertainties.append(match.group(1).strip())

        return list(set(uncertainties))  # Remove duplicates

    def _extract_json(self, content: str) -> Dict[str, Any]:
        """Extract JSON from content.

        Args:
            content: Content that may contain JSON

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If no valid JSON found
        """
        # Try to find JSON block
        json_match = re.search(r"```json\s*(\{.+?\})\s*```", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Try to find raw JSON
        json_match = re.search(r"(\{.+\})", content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Try parsing the whole content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            raise ValueError("No valid JSON found in content")


class CostTracker:
    """Track API usage costs."""

    def __init__(self):
        """Initialize cost tracker."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.total_cost = 0.0
        self.costs_by_model: Dict[str, float] = {}

    def add_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
    ):
        """Add usage information.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost: Cost in USD
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_requests += 1
        self.total_cost += cost

        if model not in self.costs_by_model:
            self.costs_by_model[model] = 0.0
        self.costs_by_model[model] += cost

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "average_cost_per_request": (
                round(self.total_cost / self.total_requests, 4)
                if self.total_requests > 0
                else 0.0
            ),
            "costs_by_model": {
                model: round(cost, 4)
                for model, cost in self.costs_by_model.items()
            },
        }

    def reset(self):
        """Reset all statistics."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_requests = 0
        self.total_cost = 0.0
        self.costs_by_model.clear()
