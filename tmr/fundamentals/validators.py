"""
Validators for the Fundamentals Layer

These validators apply logical principles to different types of reasoning.
"""

from typing import Any, Dict, List, Optional, Tuple
import re
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from .principles import LogicalPrinciples, PrincipleType, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """Represents a single step in a reasoning chain."""
    step_id: int
    statement: str
    justification: Optional[str] = None
    dependencies: Optional[List[int]] = None
    step_type: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class ReasoningChain:
    """Represents a complete reasoning chain."""
    steps: List[ReasoningStep]
    conclusion: Optional[str] = None
    domain: Optional[str] = None
    metadata: Optional[Dict] = None


class PrincipleValidator(ABC):
    """Abstract base class for principle-based validators."""
    
    def __init__(self, principles: Optional[LogicalPrinciples] = None):
        self.principles = principles or LogicalPrinciples()
        self.validation_history = []
    
    @abstractmethod
    def validate(self, input_data: Any) -> Tuple[bool, float, Dict]:
        """
        Validate input data.
        
        Returns:
            Tuple of (is_valid, confidence, details)
        """
        pass
    
    def get_validation_history(self) -> List[Dict]:
        """Get history of validations."""
        return self.validation_history
    
    def clear_history(self):
        """Clear validation history."""
        self.validation_history = []


class LogicalValidator(PrincipleValidator):
    """Validator for logical reasoning."""
    
    def validate(self, reasoning_chain: ReasoningChain) -> Tuple[bool, float, Dict]:
        """Validate a logical reasoning chain."""
        results = []
        
        for step in reasoning_chain.steps:
            # Check each step against logical principles
            step_results = self.principles.validate_all(
                self._convert_step_to_statement(step)
            )
            results.append(step_results)
        
        # Check consistency across steps
        consistency_valid = self._check_chain_consistency(reasoning_chain)
        
        # Aggregate results
        all_valid = all(
            all(r.valid for r in step_results.values())
            for step_results in results
        ) and consistency_valid
        
        # Calculate confidence
        if results:
            confidences = []
            for step_results in results:
                _, step_confidence = self.principles.get_aggregate_validity(step_results)
                confidences.append(step_confidence)
            avg_confidence = np.mean(confidences)
        else:
            avg_confidence = 0.0
        
        # Adjust confidence based on consistency
        if not consistency_valid:
            avg_confidence *= 0.5
        
        details = {
            "step_results": results,
            "consistency_valid": consistency_valid,
            "num_steps": len(reasoning_chain.steps)
        }
        
        self.validation_history.append({
            "type": "logical",
            "valid": all_valid,
            "confidence": avg_confidence,
            "details": details
        })
        
        return all_valid, avg_confidence, details
    
    def _convert_step_to_statement(self, step: ReasoningStep) -> Dict:
        """Convert reasoning step to statement for validation."""
        return {
            "proposition": step.statement,
            "justification": step.justification,
            "dependencies": step.dependencies,
            "metadata": step.metadata
        }
    
    def _check_chain_consistency(self, chain: ReasoningChain) -> bool:
        """Check consistency across the entire reasoning chain."""
        # Check that dependencies are valid
        step_ids = {step.step_id for step in chain.steps}
        
        for step in chain.steps:
            if step.dependencies:
                for dep_id in step.dependencies:
                    if dep_id not in step_ids or dep_id >= step.step_id:
                        return False  # Invalid dependency
        
        # Check for contradictions between steps
        statements = [step.statement for step in chain.steps]
        for i, stmt1 in enumerate(statements):
            for stmt2 in statements[i+1:]:
                if self._are_contradictory_statements(stmt1, stmt2):
                    return False
        
        return True
    
    def _are_contradictory_statements(self, stmt1: str, stmt2: str) -> bool:
        """Check if two statements are contradictory."""
        # Simple heuristic - in production would use NLI
        stmt1_lower = stmt1.lower()
        stmt2_lower = stmt2.lower()
        
        # Check for explicit negation
        if ("not" in stmt1_lower and stmt1_lower.replace("not ", "") == stmt2_lower):
            return True
        if ("not" in stmt2_lower and stmt2_lower.replace("not ", "") == stmt1_lower):
            return True
        
        # Check for opposite assertions
        opposites = [
            ("true", "false"),
            ("yes", "no"),
            ("always", "never"),
            ("all", "none"),
            ("increases", "decreases"),
            ("greater", "less")
        ]
        
        for word1, word2 in opposites:
            if word1 in stmt1_lower and word2 in stmt2_lower:
                # Check if they're about the same subject
                if self._same_subject(stmt1, stmt2):
                    return True
            if word2 in stmt1_lower and word1 in stmt2_lower:
                if self._same_subject(stmt1, stmt2):
                    return True
        
        return False
    
    def _same_subject(self, stmt1: str, stmt2: str) -> bool:
        """Check if two statements are about the same subject."""
        # Extract potential subjects (nouns)
        # Simple heuristic - in production would use NLP
        words1 = set(stmt1.lower().split())
        words2 = set(stmt2.lower().split())
        
        # Check for significant overlap
        overlap = words1.intersection(words2)
        if len(overlap) > len(words1) * 0.3:
            return True
        
        return False


class MathematicalValidator(PrincipleValidator):
    """Validator for mathematical reasoning."""
    
    def validate(self, math_statement: Dict) -> Tuple[bool, float, Dict]:
        """Validate mathematical reasoning."""
        
        # Extract components
        equation = math_statement.get("equation", "")
        steps = math_statement.get("steps", [])
        result = math_statement.get("result")
        
        # Validate equation structure
        structure_valid = self._validate_equation_structure(equation)
        
        # Validate calculation steps
        steps_valid = True
        step_confidences = []
        
        for i, step in enumerate(steps):
            step_valid, step_conf = self._validate_math_step(step, i, steps)
            steps_valid = steps_valid and step_valid
            step_confidences.append(step_conf)
        
        # Validate conservation (e.g., equality preservation)
        conservation_valid = self._validate_mathematical_conservation(steps)
        
        # Check result
        result_valid = self._validate_result(steps, result) if result else True
        
        # Aggregate validation
        all_valid = structure_valid and steps_valid and conservation_valid and result_valid
        
        # Calculate confidence
        base_confidence = 0.25 * (
            (1.0 if structure_valid else 0.0) +
            (1.0 if steps_valid else 0.0) +
            (1.0 if conservation_valid else 0.0) +
            (1.0 if result_valid else 0.0)
        )
        
        if step_confidences:
            step_conf_avg = np.mean(step_confidences)
            confidence = 0.7 * base_confidence + 0.3 * step_conf_avg
        else:
            confidence = base_confidence
        
        details = {
            "structure_valid": structure_valid,
            "steps_valid": steps_valid,
            "conservation_valid": conservation_valid,
            "result_valid": result_valid,
            "num_steps": len(steps)
        }
        
        self.validation_history.append({
            "type": "mathematical",
            "valid": all_valid,
            "confidence": confidence,
            "details": details
        })
        
        return all_valid, confidence, details
    
    def _validate_equation_structure(self, equation: str) -> bool:
        """Validate basic equation structure."""
        if not equation:
            return True  # No equation to validate
        
        # Check for basic mathematical syntax
        # Must have an equals sign
        if "=" not in equation:
            return False
        
        # Check balanced parentheses
        if not self._balanced_parentheses(equation):
            return False
        
        # Check for valid operators
        valid_ops = ["+", "-", "*", "/", "^", "=", "(", ")", "sqrt", "sin", "cos", "tan", "log", "ln"]
        # Simple check - in production would use proper parser
        
        return True
    
    def _balanced_parentheses(self, text: str) -> bool:
        """Check if parentheses are balanced."""
        count = 0
        for char in text:
            if char == "(":
                count += 1
            elif char == ")":
                count -= 1
                if count < 0:
                    return False
        return count == 0
    
    def _validate_math_step(self, step: str, index: int, all_steps: List[str]) -> Tuple[bool, float]:
        """Validate a single mathematical step."""
        # Check if step maintains equality
        if "=" in step:
            parts = step.split("=")
            if len(parts) == 2:
                # In production, would evaluate both sides
                # For now, basic validation
                return True, 0.8
        
        # Check for common errors
        errors = self._detect_math_errors(step)
        if errors:
            return False, 0.2
        
        return True, 0.7
    
    def _detect_math_errors(self, step: str) -> List[str]:
        """Detect common mathematical errors."""
        errors = []
        
        # Division by zero
        if "/0" in step or "/ 0" in step:
            errors.append("division by zero")
        
        # Sign errors (simplified check)
        if "--" in step and "- -" not in step:
            # Might be intentional, but flag for review
            pass
        
        return errors
    
    def _validate_mathematical_conservation(self, steps: List[str]) -> bool:
        """Validate that mathematical operations preserve equality."""
        # Check that each transformation preserves the equation
        # Simplified - in production would use symbolic math
        
        for i in range(len(steps) - 1):
            if not self._transformation_valid(steps[i], steps[i+1]):
                return False
        
        return True
    
    def _transformation_valid(self, step1: str, step2: str) -> bool:
        """Check if transformation from step1 to step2 is valid."""
        # Placeholder - in production would use symbolic math
        return True
    
    def _validate_result(self, steps: List[str], result: Any) -> bool:
        """Validate that the result follows from the steps."""
        if not steps:
            return True
        
        # Check if result appears in final step
        final_step = steps[-1]
        if str(result) in final_step:
            return True
        
        # More sophisticated checking would be done in production
        return False


class CausalValidator(PrincipleValidator):
    """Validator for causal reasoning."""
    
    def validate(self, causal_chain: Dict) -> Tuple[bool, float, Dict]:
        """Validate causal reasoning."""
        
        events = causal_chain.get("events", [])
        relationships = causal_chain.get("relationships", [])
        
        # Validate temporal ordering
        temporal_valid = self._validate_temporal_ordering(events)
        
        # Validate causal relationships
        causal_valid = True
        relationship_confidences = []
        
        for rel in relationships:
            rel_valid, rel_conf = self._validate_causal_relationship(rel, events)
            causal_valid = causal_valid and rel_valid
            relationship_confidences.append(rel_conf)
        
        # Check for circular causality
        circular = self._detect_circular_causality(relationships)
        
        # Aggregate validation
        all_valid = temporal_valid and causal_valid and not circular
        
        # Calculate confidence
        base_confidence = (0.4 if temporal_valid else 0.0) + \
                         (0.4 if causal_valid else 0.0) + \
                         (0.2 if not circular else 0.0)
        
        if relationship_confidences:
            rel_conf_avg = np.mean(relationship_confidences)
            confidence = 0.7 * base_confidence + 0.3 * rel_conf_avg
        else:
            confidence = base_confidence
        
        details = {
            "temporal_valid": temporal_valid,
            "causal_valid": causal_valid,
            "circular_causality": circular,
            "num_events": len(events),
            "num_relationships": len(relationships)
        }
        
        self.validation_history.append({
            "type": "causal",
            "valid": all_valid,
            "confidence": confidence,
            "details": details
        })
        
        return all_valid, confidence, details
    
    def _validate_temporal_ordering(self, events: List[Dict]) -> bool:
        """Validate that events are temporally ordered."""
        if not events:
            return True
        
        for i in range(len(events) - 1):
            if "timestamp" in events[i] and "timestamp" in events[i+1]:
                if events[i]["timestamp"] > events[i+1]["timestamp"]:
                    return False
        
        return True
    
    def _validate_causal_relationship(self, relationship: Dict, events: List[Dict]) -> Tuple[bool, float]:
        """Validate a single causal relationship."""
        cause_id = relationship.get("cause_id")
        effect_id = relationship.get("effect_id")
        
        # Find events
        cause_event = next((e for e in events if e.get("id") == cause_id), None)
        effect_event = next((e for e in events if e.get("id") == effect_id), None)
        
        if not cause_event or not effect_event:
            return False, 0.0
        
        # Check temporal constraint
        if "timestamp" in cause_event and "timestamp" in effect_event:
            if cause_event["timestamp"] >= effect_event["timestamp"]:
                return False, 0.0
        
        # Check for plausible causation (simplified)
        confidence = relationship.get("confidence", 0.5)
        
        return True, confidence
    
    def _detect_circular_causality(self, relationships: List[Dict]) -> bool:
        """Detect circular causal relationships."""
        # Build adjacency list
        graph = {}
        for rel in relationships:
            cause = rel.get("cause_id")
            effect = rel.get("effect_id")
            if cause and effect:
                if cause not in graph:
                    graph[cause] = []
                graph[cause].append(effect)
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True
        
        return False


class ConsistencyValidator(PrincipleValidator):
    """Validator for overall consistency across different types of reasoning."""
    
    def __init__(self, principles: Optional[LogicalPrinciples] = None):
        super().__init__(principles)
        self.logical_validator = LogicalValidator(principles)
        self.math_validator = MathematicalValidator(principles)
        self.causal_validator = CausalValidator(principles)
    
    def validate(self, composite_reasoning: Dict) -> Tuple[bool, float, Dict]:
        """Validate consistency across multiple reasoning types."""
        
        results = {}
        confidences = []
        
        # Validate each component if present
        if "logical" in composite_reasoning:
            valid, conf, details = self.logical_validator.validate(
                composite_reasoning["logical"]
            )
            results["logical"] = {"valid": valid, "confidence": conf, "details": details}
            confidences.append(conf)
        
        if "mathematical" in composite_reasoning:
            valid, conf, details = self.math_validator.validate(
                composite_reasoning["mathematical"]
            )
            results["mathematical"] = {"valid": valid, "confidence": conf, "details": details}
            confidences.append(conf)
        
        if "causal" in composite_reasoning:
            valid, conf, details = self.causal_validator.validate(
                composite_reasoning["causal"]
            )
            results["causal"] = {"valid": valid, "confidence": conf, "details": details}
            confidences.append(conf)
        
        # Check cross-domain consistency
        cross_valid = self._validate_cross_domain_consistency(composite_reasoning)
        
        # Aggregate results
        all_valid = all(r["valid"] for r in results.values()) and cross_valid
        
        # Calculate overall confidence
        if confidences:
            avg_confidence = np.mean(confidences)
            if not cross_valid:
                avg_confidence *= 0.8  # Penalty for cross-domain inconsistency
        else:
            avg_confidence = 0.0
        
        details = {
            "component_results": results,
            "cross_domain_valid": cross_valid,
            "num_components": len(results)
        }
        
        self.validation_history.append({
            "type": "consistency",
            "valid": all_valid,
            "confidence": avg_confidence,
            "details": details
        })
        
        return all_valid, avg_confidence, details
    
    def _validate_cross_domain_consistency(self, composite: Dict) -> bool:
        """Validate consistency across different reasoning domains."""
        # Check that conclusions align across domains
        conclusions = []
        
        for domain in ["logical", "mathematical", "causal"]:
            if domain in composite:
                conclusion = composite[domain].get("conclusion")
                if conclusion:
                    conclusions.append(conclusion)
        
        # Check for contradictions in conclusions
        for i, conc1 in enumerate(conclusions):
            for conc2 in conclusions[i+1:]:
                if self._are_contradictory_conclusions(conc1, conc2):
                    return False
        
        return True
    
    def _are_contradictory_conclusions(self, conc1: Any, conc2: Any) -> bool:
        """Check if two conclusions are contradictory."""
        # Convert to strings for comparison
        str1 = str(conc1).lower()
        str2 = str(conc2).lower()
        
        # Simple contradiction detection
        if str1 == f"not {str2}" or str2 == f"not {str1}":
            return True
        
        # Check for opposite values
        if str1 in ["true", "yes", "1"] and str2 in ["false", "no", "0"]:
            return True
        if str2 in ["true", "yes", "1"] and str1 in ["false", "no", "0"]:
            return True
        
        return False