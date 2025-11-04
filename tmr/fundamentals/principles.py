"""
Core Logical Principles

Implements the immutable logical principles that form the foundation of verified reasoning.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class PrincipleType(Enum):
    """Types of logical principles."""
    IDENTITY = "identity"
    NON_CONTRADICTION = "non_contradiction"
    EXCLUDED_MIDDLE = "excluded_middle"
    CAUSALITY = "causality"
    CONSERVATION = "conservation"


@dataclass
class ValidationResult:
    """Result of principle validation."""
    valid: bool
    principle: PrincipleType
    confidence: float
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class LogicalPrinciple(ABC):
    """Abstract base class for logical principles."""
    
    def __init__(self, name: str):
        self.name = name
        self.validation_count = 0
        self.success_count = 0
    
    @abstractmethod
    def validate(self, statement: Any, context: Optional[Dict] = None) -> ValidationResult:
        """
        Validate a statement against this principle.
        
        Args:
            statement: The statement to validate
            context: Optional context information
            
        Returns:
            ValidationResult with validation outcome
        """
        pass
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of validations."""
        if self.validation_count == 0:
            return 0.0
        return self.success_count / self.validation_count
    
    def reset_stats(self):
        """Reset validation statistics."""
        self.validation_count = 0
        self.success_count = 0


class IdentityPrinciple(LogicalPrinciple):
    """
    Law of Identity: A = A
    A thing is identical to itself.
    """
    
    def __init__(self):
        super().__init__("Identity")
    
    def validate(self, statement: Any, context: Optional[Dict] = None) -> ValidationResult:
        """Validate that entities maintain their identity."""
        self.validation_count += 1
        
        try:
            # For simple types, direct comparison
            if isinstance(statement, (str, int, float, bool)):
                valid = statement == statement
                confidence = 1.0 if valid else 0.0
            
            # For dictionaries, check key-value consistency
            elif isinstance(statement, dict):
                if "entity" in statement and "reference" in statement:
                    valid = self._check_entity_identity(
                        statement["entity"], 
                        statement["reference"]
                    )
                    confidence = 0.9 if valid else 0.1
                else:
                    valid = True  # Assume valid if structure unknown
                    confidence = 0.5
            
            # For lists/tuples, check element identity
            elif isinstance(statement, (list, tuple)):
                valid = all(x == x for x in statement)
                confidence = 0.95 if valid else 0.05
            
            else:
                # Unknown type, can't validate
                valid = True
                confidence = 0.3
            
            if valid:
                self.success_count += 1
            
            return ValidationResult(
                valid=valid,
                principle=PrincipleType.IDENTITY,
                confidence=confidence,
                reason="Identity principle validation"
            )
            
        except Exception as e:
            logger.error(f"Identity validation error: {e}")
            return ValidationResult(
                valid=False,
                principle=PrincipleType.IDENTITY,
                confidence=0.0,
                reason=f"Validation error: {str(e)}"
            )
    
    def _check_entity_identity(self, entity: Any, reference: Any) -> bool:
        """Check if entity and reference refer to the same thing."""
        if entity == reference:
            return True
        
        # Check semantic similarity for strings
        if isinstance(entity, str) and isinstance(reference, str):
            return self._semantic_similarity(entity, reference) > 0.9
        
        return False
    
    def _semantic_similarity(self, s1: str, s2: str) -> float:
        """Simple semantic similarity check."""
        # Placeholder - in production would use embeddings
        if s1.lower() == s2.lower():
            return 1.0
        
        # Check if one contains the other
        if s1.lower() in s2.lower() or s2.lower() in s1.lower():
            return 0.7
        
        return 0.0


class NonContradictionPrinciple(LogicalPrinciple):
    """
    Law of Non-Contradiction: ¬(P ∧ ¬P)
    A statement cannot be both true and false at the same time.
    """
    
    def __init__(self):
        super().__init__("Non-Contradiction")
    
    def validate(self, statement: Any, context: Optional[Dict] = None) -> ValidationResult:
        """Validate that statements don't contradict themselves."""
        self.validation_count += 1
        
        try:
            if isinstance(statement, dict):
                if "propositions" in statement:
                    valid = self._check_propositions(statement["propositions"])
                    confidence = 0.95 if valid else 0.05
                elif "claim" in statement and "negation" in statement:
                    # Direct contradiction check
                    valid = not (statement.get("claim_true", False) and 
                               statement.get("negation_true", False))
                    confidence = 1.0 if valid else 0.0
                else:
                    valid = True
                    confidence = 0.4
            
            elif isinstance(statement, list):
                # Check for contradictions in list of statements
                valid = self._check_list_consistency(statement)
                confidence = 0.9 if valid else 0.1
            
            else:
                valid = True
                confidence = 0.3
            
            if valid:
                self.success_count += 1
            
            return ValidationResult(
                valid=valid,
                principle=PrincipleType.NON_CONTRADICTION,
                confidence=confidence,
                reason="Non-contradiction validation"
            )
            
        except Exception as e:
            logger.error(f"Non-contradiction validation error: {e}")
            return ValidationResult(
                valid=False,
                principle=PrincipleType.NON_CONTRADICTION,
                confidence=0.0,
                reason=f"Validation error: {str(e)}"
            )
    
    def _check_propositions(self, propositions: List[Dict]) -> bool:
        """Check if propositions contain contradictions."""
        for i, prop1 in enumerate(propositions):
            for prop2 in propositions[i+1:]:
                if self._are_contradictory(prop1, prop2):
                    return False
        return True
    
    def _are_contradictory(self, prop1: Dict, prop2: Dict) -> bool:
        """Check if two propositions are contradictory."""
        # Simple check - in production would use NLI model
        if prop1.get("subject") == prop2.get("subject"):
            if prop1.get("predicate") == f"not {prop2.get('predicate')}":
                return True
            if prop2.get("predicate") == f"not {prop1.get('predicate')}":
                return True
        return False
    
    def _check_list_consistency(self, statements: List) -> bool:
        """Check consistency in a list of statements."""
        # Placeholder implementation
        return True


class ExcludedMiddlePrinciple(LogicalPrinciple):
    """
    Law of Excluded Middle: P ∨ ¬P
    Every proposition is either true or false.
    """
    
    def __init__(self):
        super().__init__("Excluded Middle")
    
    def validate(self, statement: Any, context: Optional[Dict] = None) -> ValidationResult:
        """Validate that propositions have definite truth values."""
        self.validation_count += 1
        
        try:
            if isinstance(statement, dict):
                if "proposition" in statement:
                    # Check if proposition has definite truth value
                    truth_value = statement.get("truth_value")
                    valid = truth_value in [True, False, "true", "false", 1, 0]
                    confidence = 1.0 if valid else 0.0
                else:
                    valid = True
                    confidence = 0.4
            else:
                valid = True
                confidence = 0.3
            
            if valid:
                self.success_count += 1
            
            return ValidationResult(
                valid=valid,
                principle=PrincipleType.EXCLUDED_MIDDLE,
                confidence=confidence,
                reason="Excluded middle validation"
            )
            
        except Exception as e:
            logger.error(f"Excluded middle validation error: {e}")
            return ValidationResult(
                valid=False,
                principle=PrincipleType.EXCLUDED_MIDDLE,
                confidence=0.0,
                reason=f"Validation error: {str(e)}"
            )


class CausalityPrinciple(LogicalPrinciple):
    """
    Principle of Causality: Every effect has a cause.
    Temporal ordering and causal relationships must be maintained.
    """
    
    def __init__(self):
        super().__init__("Causality")
    
    def validate(self, statement: Any, context: Optional[Dict] = None) -> ValidationResult:
        """Validate causal relationships and temporal ordering."""
        self.validation_count += 1
        
        try:
            if isinstance(statement, dict):
                if "cause" in statement and "effect" in statement:
                    valid = self._validate_causal_relationship(
                        statement["cause"],
                        statement["effect"],
                        statement.get("timestamp_cause"),
                        statement.get("timestamp_effect")
                    )
                    confidence = 0.9 if valid else 0.1
                elif "events" in statement:
                    valid = self._validate_event_sequence(statement["events"])
                    confidence = 0.85 if valid else 0.15
                else:
                    valid = True
                    confidence = 0.4
            else:
                valid = True
                confidence = 0.3
            
            if valid:
                self.success_count += 1
            
            return ValidationResult(
                valid=valid,
                principle=PrincipleType.CAUSALITY,
                confidence=confidence,
                reason="Causality validation"
            )
            
        except Exception as e:
            logger.error(f"Causality validation error: {e}")
            return ValidationResult(
                valid=False,
                principle=PrincipleType.CAUSALITY,
                confidence=0.0,
                reason=f"Validation error: {str(e)}"
            )
    
    def _validate_causal_relationship(self, cause: Any, effect: Any, 
                                     t_cause: Optional[float] = None, 
                                     t_effect: Optional[float] = None) -> bool:
        """Validate that cause precedes effect."""
        # Check temporal ordering if timestamps provided
        if t_cause is not None and t_effect is not None:
            if t_cause > t_effect:
                return False  # Cause must precede effect
        
        # Basic validation that cause and effect are defined
        return cause is not None and effect is not None
    
    def _validate_event_sequence(self, events: List[Dict]) -> bool:
        """Validate that event sequence maintains causal ordering."""
        if not events:
            return True
        
        # Check if events are properly ordered
        for i in range(len(events) - 1):
            if "timestamp" in events[i] and "timestamp" in events[i+1]:
                if events[i]["timestamp"] > events[i+1]["timestamp"]:
                    return False
        
        return True


class ConservationPrinciple(LogicalPrinciple):
    """
    Conservation Principles: Certain properties remain constant.
    (e.g., conservation of mass, energy, information)
    """
    
    def __init__(self):
        super().__init__("Conservation")
    
    def validate(self, statement: Any, context: Optional[Dict] = None) -> ValidationResult:
        """Validate conservation laws."""
        self.validation_count += 1
        
        try:
            if isinstance(statement, dict):
                if "before" in statement and "after" in statement:
                    valid = self._check_conservation(
                        statement["before"],
                        statement["after"],
                        statement.get("conserved_quantity", "total")
                    )
                    confidence = 0.95 if valid else 0.05
                else:
                    valid = True
                    confidence = 0.4
            else:
                valid = True
                confidence = 0.3
            
            if valid:
                self.success_count += 1
            
            return ValidationResult(
                valid=valid,
                principle=PrincipleType.CONSERVATION,
                confidence=confidence,
                reason="Conservation validation"
            )
            
        except Exception as e:
            logger.error(f"Conservation validation error: {e}")
            return ValidationResult(
                valid=False,
                principle=PrincipleType.CONSERVATION,
                confidence=0.0,
                reason=f"Validation error: {str(e)}"
            )
    
    def _check_conservation(self, before: Any, after: Any, quantity: str) -> bool:
        """Check if a quantity is conserved."""
        try:
            # For numerical values
            if isinstance(before, (int, float)) and isinstance(after, (int, float)):
                return abs(before - after) < 1e-6  # Allow small numerical errors
            
            # For lists/collections
            if isinstance(before, list) and isinstance(after, list):
                if quantity == "count":
                    return len(before) == len(after)
                elif quantity == "sum":
                    return abs(sum(before) - sum(after)) < 1e-6
            
            # For dictionaries
            if isinstance(before, dict) and isinstance(after, dict):
                if quantity in before and quantity in after:
                    return before[quantity] == after[quantity]
            
            return True  # Default to valid if can't determine
            
        except:
            return False


class LogicalPrinciples:
    """Collection of all logical principles."""
    
    def __init__(self):
        self.principles = {
            PrincipleType.IDENTITY: IdentityPrinciple(),
            PrincipleType.NON_CONTRADICTION: NonContradictionPrinciple(),
            PrincipleType.EXCLUDED_MIDDLE: ExcludedMiddlePrinciple(),
            PrincipleType.CAUSALITY: CausalityPrinciple(),
            PrincipleType.CONSERVATION: ConservationPrinciple(),
        }
    
    def validate_all(self, statement: Any, context: Optional[Dict] = None) -> Dict[PrincipleType, ValidationResult]:
        """Validate statement against all principles."""
        results = {}
        for principle_type, principle in self.principles.items():
            results[principle_type] = principle.validate(statement, context)
        return results
    
    def validate_specific(self, statement: Any, principle_types: List[PrincipleType], 
                         context: Optional[Dict] = None) -> Dict[PrincipleType, ValidationResult]:
        """Validate statement against specific principles."""
        results = {}
        for principle_type in principle_types:
            if principle_type in self.principles:
                results[principle_type] = self.principles[principle_type].validate(statement, context)
        return results
    
    def get_aggregate_validity(self, results: Dict[PrincipleType, ValidationResult]) -> Tuple[bool, float]:
        """
        Calculate aggregate validity and confidence.
        
        Returns:
            Tuple of (is_valid, confidence)
        """
        if not results:
            return True, 0.0
        
        valid_count = sum(1 for r in results.values() if r.valid)
        total_count = len(results)
        
        # All principles must be valid
        is_valid = valid_count == total_count
        
        # Weighted confidence
        confidence = sum(r.confidence for r in results.values()) / total_count
        
        return is_valid, confidence
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get validation statistics for all principles."""
        stats = {}
        for principle_type, principle in self.principles.items():
            stats[principle_type.value] = {
                "validation_count": principle.validation_count,
                "success_count": principle.success_count,
                "success_rate": principle.success_rate
            }
        return stats
    
    def reset_all_stats(self):
        """Reset statistics for all principles."""
        for principle in self.principles.values():
            principle.reset_stats()