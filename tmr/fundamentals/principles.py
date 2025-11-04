"""
Core Logical Principles

Implements the immutable logical principles that form the foundation of verified reasoning.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
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
        """
        Calculate semantic similarity between two strings.
        Returns a score between 0.0 and 1.0.
        """
        # Exact match
        if s1 == s2:
            return 1.0

        # Case-insensitive exact match
        s1_lower = s1.lower().strip()
        s2_lower = s2.lower().strip()

        if s1_lower == s2_lower:
            return 0.98

        # Calculate multiple similarity metrics
        scores = []

        # 1. Levenshtein-based similarity
        lev_sim = self._levenshtein_similarity(s1_lower, s2_lower)
        scores.append(lev_sim)

        # 2. Token-based Jaccard similarity
        jaccard_sim = self._jaccard_similarity(s1_lower, s2_lower)
        scores.append(jaccard_sim)

        # 3. Longest common subsequence similarity
        lcs_sim = self._lcs_similarity(s1_lower, s2_lower)
        scores.append(lcs_sim)

        # 4. Containment check
        if s1_lower in s2_lower or s2_lower in s1_lower:
            containment_score = min(len(s1_lower), len(s2_lower)) / max(len(s1_lower), len(s2_lower))
            scores.append(containment_score)

        # Return weighted average, with emphasis on highest score
        if not scores:
            return 0.0

        # Take max and average for final score
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)

        # Weighted combination favoring the best metric
        return 0.6 * max_score + 0.4 * avg_score

    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity based on Levenshtein distance."""
        if not s1 or not s2:
            return 0.0

        # Create distance matrix
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        # Convert distance to similarity
        max_len = max(m, n)
        if max_len == 0:
            return 1.0

        distance = dp[m][n]
        similarity = 1.0 - (distance / max_len)
        return max(0.0, similarity)

    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """Calculate Jaccard similarity based on word tokens."""
        # Tokenize
        tokens1 = set(s1.split())
        tokens2 = set(s2.split())

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        # Calculate Jaccard index
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union) if union else 0.0

    def _lcs_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity based on longest common subsequence."""
        if not s1 or not s2:
            return 0.0

        m, n = len(s1), len(s2)

        # Create LCS matrix
        lcs = [[0] * (n + 1) for _ in range(m + 1)]

        # Fill matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    lcs[i][j] = lcs[i-1][j-1] + 1
                else:
                    lcs[i][j] = max(lcs[i-1][j], lcs[i][j-1])

        # Convert to similarity
        lcs_length = lcs[m][n]
        max_len = max(m, n)

        return lcs_length / max_len if max_len > 0 else 0.0


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
        if not statements or len(statements) < 2:
            return True

        # Convert all statements to comparable format
        normalized_statements = []
        for stmt in statements:
            if isinstance(stmt, str):
                normalized_statements.append({"text": stmt, "type": "string"})
            elif isinstance(stmt, dict):
                normalized_statements.append({"text": str(stmt), "dict": stmt, "type": "dict"})
            elif isinstance(stmt, (int, float, bool)):
                normalized_statements.append({"text": str(stmt), "value": stmt, "type": "value"})
            else:
                normalized_statements.append({"text": str(stmt), "type": "other"})

        # Check pairwise for contradictions
        for i, stmt1 in enumerate(normalized_statements):
            for stmt2 in normalized_statements[i+1:]:
                if self._statements_contradict(stmt1, stmt2):
                    logger.debug(f"Contradiction found between: {stmt1['text']} and {stmt2['text']}")
                    return False

        return True

    def _statements_contradict(self, stmt1: Dict, stmt2: Dict) -> bool:
        """Check if two normalized statements contradict each other."""
        # Handle boolean values
        if stmt1["type"] == "value" and stmt2["type"] == "value":
            if isinstance(stmt1.get("value"), bool) and isinstance(stmt2.get("value"), bool):
                # Same statement with different truth values
                if stmt1["text"] == stmt2["text"]:
                    return stmt1["value"] != stmt2["value"]

        # Handle string statements
        if stmt1["type"] == "string" and stmt2["type"] == "string":
            return self._text_contradicts(stmt1["text"], stmt2["text"])

        # Handle dict propositions
        if stmt1["type"] == "dict" and stmt2["type"] == "dict":
            return self._are_contradictory(stmt1["dict"], stmt2["dict"])

        # Handle mixed types
        if stmt1["type"] in ["string", "dict"] and stmt2["type"] in ["string", "dict"]:
            # Extract text from both and compare
            text1 = stmt1["text"]
            text2 = stmt2["text"]
            return self._text_contradicts(text1, text2)

        return False

    def _text_contradicts(self, text1: str, text2: str) -> bool:
        """Check if two text statements contradict each other."""
        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        # Direct negation patterns
        negation_patterns = [
            ("is", "is not"),
            ("are", "are not"),
            ("was", "was not"),
            ("were", "were not"),
            ("will", "will not"),
            ("can", "cannot"),
            ("should", "should not"),
            ("has", "has not"),
            ("have", "have not"),
            ("does", "does not"),
            ("do", "do not"),
        ]

        for positive, negative in negation_patterns:
            # Check if one has positive and other has negative
            if positive in t1 and negative in t2:
                # Extract parts before and after the verb
                parts1 = t1.split(positive)
                parts2 = t2.split(negative)
                if len(parts1) == 2 and len(parts2) == 2:
                    if parts1[0].strip() == parts2[0].strip() and parts1[1].strip() == parts2[1].strip():
                        return True

            if negative in t1 and positive in t2:
                parts1 = t1.split(negative)
                parts2 = t2.split(positive)
                if len(parts1) == 2 and len(parts2) == 2:
                    if parts1[0].strip() == parts2[0].strip() and parts1[1].strip() == parts2[1].strip():
                        return True

        # Check for "not" insertion/removal
        if "not" in t1 and "not" not in t2:
            t1_without_not = t1.replace(" not ", " ").replace("not ", "").strip()
            if t1_without_not == t2:
                return True

        if "not" in t2 and "not" not in t1:
            t2_without_not = t2.replace(" not ", " ").replace("not ", "").strip()
            if t2_without_not == t1:
                return True

        # Check for antonym pairs
        antonym_pairs = [
            ("true", "false"),
            ("yes", "no"),
            ("always", "never"),
            ("all", "none"),
            ("everything", "nothing"),
            ("everyone", "no one"),
            ("increase", "decrease"),
            ("rise", "fall"),
            ("grow", "shrink"),
            ("expand", "contract"),
            ("greater", "less"),
            ("more", "fewer"),
            ("above", "below"),
            ("before", "after"),
            ("present", "absent"),
            ("exist", "not exist"),
            ("possible", "impossible"),
            ("valid", "invalid"),
            ("correct", "incorrect"),
        ]

        # Extract words
        words1 = set(t1.split())
        words2 = set(t2.split())

        for word1, word2 in antonym_pairs:
            if word1 in words1 and word2 in words2:
                # Check if rest of statement is similar
                remaining1 = words1 - {word1}
                remaining2 = words2 - {word2}
                overlap = remaining1.intersection(remaining2)
                if len(overlap) >= min(len(remaining1), len(remaining2)) * 0.6:
                    return True

            # Check reverse
            if word2 in words1 and word1 in words2:
                remaining1 = words1 - {word2}
                remaining2 = words2 - {word1}
                overlap = remaining1.intersection(remaining2)
                if len(overlap) >= min(len(remaining1), len(remaining2)) * 0.6:
                    return True

        return False


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