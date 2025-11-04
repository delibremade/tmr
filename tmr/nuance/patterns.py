"""
Pattern Data Structures and Types

Defines the core pattern structures used in the nuance layer.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
import hashlib
import json


class DomainType(Enum):
    """Enumeration of supported domain types."""
    MATH = "mathematical"
    CODE = "code"
    LOGIC = "logical"
    UNKNOWN = "unknown"


class PatternComplexity(Enum):
    """Pattern complexity levels."""
    SIMPLE = "simple"
    INTERMEDIATE = "intermediate"
    COMPLEX = "complex"
    ADVANCED = "advanced"


@dataclass
class PatternMetadata:
    """Metadata associated with a pattern."""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0
    confidence_scores: List[float] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    source: Optional[str] = None

    def update_usage(self, success: bool, confidence: float):
        """Update usage statistics."""
        self.usage_count += 1
        self.confidence_scores.append(confidence)

        # Recalculate success rate
        if success:
            self.success_rate = (self.success_rate * (self.usage_count - 1) + 1) / self.usage_count
        else:
            self.success_rate = (self.success_rate * (self.usage_count - 1)) / self.usage_count

        self.updated_at = datetime.now()

    def get_avg_confidence(self) -> float:
        """Get average confidence score."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)


@dataclass
class Pattern:
    """
    Core pattern structure.

    Represents a reusable pattern extracted from reasoning instances.
    """
    pattern_id: str
    domain: DomainType
    name: str
    description: str
    structure: Dict[str, Any]

    # Pattern characteristics
    complexity: PatternComplexity = PatternComplexity.SIMPLE
    prerequisites: List[str] = field(default_factory=list)
    applicable_contexts: List[str] = field(default_factory=list)

    # Examples and variations
    examples: List[Dict[str, Any]] = field(default_factory=list)
    variations: List[str] = field(default_factory=list)

    # Validation and application
    validation_rules: List[Dict[str, Any]] = field(default_factory=list)
    transformation_steps: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    metadata: PatternMetadata = field(default_factory=PatternMetadata)

    def __post_init__(self):
        """Post-initialization validation."""
        if not self.pattern_id:
            self.pattern_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique pattern ID."""
        content = f"{self.domain.value}:{self.name}:{json.dumps(self.structure, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def matches(self, query: Dict[str, Any], threshold: float = 0.7) -> Tuple[bool, float]:
        """
        Check if this pattern matches a given query.

        Args:
            query: Query structure to match against
            threshold: Minimum similarity threshold

        Returns:
            Tuple of (matches, similarity_score)
        """
        similarity = self._calculate_similarity(query)
        return similarity >= threshold, similarity

    def _calculate_similarity(self, query: Dict[str, Any]) -> float:
        """Calculate similarity between pattern and query."""
        # Simple structural similarity for now
        # Can be enhanced with more sophisticated matching

        if not query:
            return 0.0

        # Check domain match
        domain_score = 0.3 if query.get("domain") == self.domain.value else 0.0

        # Check structural similarity
        structure_score = self._structure_similarity(self.structure, query.get("structure", {}))

        # Check context match
        context_score = 0.0
        if "context" in query and self.applicable_contexts:
            query_context = query["context"]
            if any(ctx in query_context for ctx in self.applicable_contexts):
                context_score = 0.2

        return domain_score + (structure_score * 0.5) + context_score

    def _structure_similarity(self, struct1: Dict, struct2: Dict) -> float:
        """Calculate structural similarity between two structures."""
        if not struct1 or not struct2:
            return 0.0

        # Count matching keys
        keys1 = set(struct1.keys())
        keys2 = set(struct2.keys())

        if not keys1 or not keys2:
            return 0.0

        intersection = keys1 & keys2
        union = keys1 | keys2

        return len(intersection) / len(union) if union else 0.0

    def apply(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply this pattern to a given context.

        Args:
            context: The context to apply the pattern to

        Returns:
            Result of pattern application
        """
        result = {
            "pattern_id": self.pattern_id,
            "pattern_name": self.name,
            "applied": False,
            "output": None,
            "confidence": 0.0
        }

        # Check prerequisites
        if not self._check_prerequisites(context):
            result["error"] = "Prerequisites not met"
            return result

        # Apply transformation steps
        try:
            output = context.copy()
            for step in self.transformation_steps:
                output = self._apply_transformation_step(output, step)

            result["applied"] = True
            result["output"] = output
            result["confidence"] = self._calculate_application_confidence(context, output)

        except Exception as e:
            result["error"] = str(e)

        return result

    def _check_prerequisites(self, context: Dict[str, Any]) -> bool:
        """Check if prerequisites are met."""
        if not self.prerequisites:
            return True

        for prereq in self.prerequisites:
            if prereq not in context:
                return False

        return True

    def _apply_transformation_step(self, data: Dict[str, Any],
                                   step: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a single transformation step."""
        # Placeholder for transformation logic
        # In a real implementation, this would execute the transformation
        return data

    def _calculate_application_confidence(self, input_data: Dict[str, Any],
                                         output_data: Dict[str, Any]) -> float:
        """Calculate confidence in the pattern application."""
        # Use metadata and validation rules to calculate confidence
        base_confidence = self.metadata.get_avg_confidence()

        # Adjust based on success rate
        adjusted_confidence = base_confidence * (0.5 + 0.5 * self.metadata.success_rate)

        return min(1.0, adjusted_confidence)

    def add_example(self, example: Dict[str, Any]):
        """Add an example usage of this pattern."""
        self.examples.append(example)
        self.metadata.updated_at = datetime.now()

    def add_variation(self, variation: str):
        """Add a variation of this pattern."""
        if variation not in self.variations:
            self.variations.append(variation)
            self.metadata.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary representation."""
        return {
            "pattern_id": self.pattern_id,
            "domain": self.domain.value,
            "name": self.name,
            "description": self.description,
            "structure": self.structure,
            "complexity": self.complexity.value,
            "prerequisites": self.prerequisites,
            "applicable_contexts": self.applicable_contexts,
            "examples": self.examples,
            "variations": self.variations,
            "validation_rules": self.validation_rules,
            "transformation_steps": self.transformation_steps,
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
                "usage_count": self.metadata.usage_count,
                "success_rate": self.metadata.success_rate,
                "avg_confidence": self.metadata.get_avg_confidence(),
                "tags": list(self.metadata.tags),
                "source": self.metadata.source
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pattern':
        """Create a Pattern from dictionary representation."""
        metadata_dict = data.get("metadata", {})
        metadata = PatternMetadata(
            created_at=datetime.fromisoformat(metadata_dict.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(metadata_dict.get("updated_at", datetime.now().isoformat())),
            usage_count=metadata_dict.get("usage_count", 0),
            success_rate=metadata_dict.get("success_rate", 0.0),
            tags=set(metadata_dict.get("tags", [])),
            source=metadata_dict.get("source")
        )

        return cls(
            pattern_id=data["pattern_id"],
            domain=DomainType(data["domain"]),
            name=data["name"],
            description=data["description"],
            structure=data["structure"],
            complexity=PatternComplexity(data.get("complexity", "simple")),
            prerequisites=data.get("prerequisites", []),
            applicable_contexts=data.get("applicable_contexts", []),
            examples=data.get("examples", []),
            variations=data.get("variations", []),
            validation_rules=data.get("validation_rules", []),
            transformation_steps=data.get("transformation_steps", []),
            metadata=metadata
        )

    def __repr__(self) -> str:
        """String representation of the pattern."""
        return (f"Pattern(id={self.pattern_id}, domain={self.domain.value}, "
                f"name={self.name}, complexity={self.complexity.value})")


@dataclass
class PatternMatch:
    """Represents a match between a pattern and a query."""
    pattern: Pattern
    similarity_score: float
    confidence: float
    matched_features: List[str] = field(default_factory=list)
    suggested_adaptations: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: 'PatternMatch') -> bool:
        """Allow sorting by similarity score."""
        return self.similarity_score < other.similarity_score
