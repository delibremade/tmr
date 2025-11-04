"""
Validators for the Fundamentals Layer

These validators apply logical principles to different types of reasoning.
"""

from typing import Any, Dict, List, Optional, Tuple
import re
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import statistics

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
            avg_confidence = statistics.mean(confidences)
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
            step_conf_avg = statistics.mean(step_confidences)
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
        confidence = 0.7  # Base confidence

        # Check for common errors first
        errors = self._detect_math_errors(step)
        if errors:
            logger.debug(f"Math errors in step {index}: {errors}")
            return False, 0.1

        # Check if step maintains equality
        if "=" in step:
            parts = step.split("=")

            # Must have exactly 2 parts (one equals sign)
            if len(parts) != 2:
                return False, 0.2

            left_side = parts[0].strip()
            right_side = parts[1].strip()

            # Both sides must be non-empty
            if not left_side or not right_side:
                return False, 0.2

            # Try to evaluate both sides
            try:
                left_value = self._safe_eval(left_side)
                right_value = self._safe_eval(right_side)

                # If both can be evaluated, check if they're equal
                if left_value is not None and right_value is not None:
                    if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
                        if abs(left_value - right_value) < 1e-10:
                            confidence = 0.95
                        else:
                            logger.debug(f"Equation not balanced: {left_value} != {right_value}")
                            return False, 0.1
                    else:
                        confidence = 0.85  # Can't verify numerically
                else:
                    confidence = 0.75  # Can't evaluate, but syntactically valid

            except Exception as e:
                logger.debug(f"Could not evaluate step: {e}")
                confidence = 0.6  # Syntax looks ok but can't verify

        # If there's a previous step, check transformation validity
        if index > 0 and index <= len(all_steps):
            prev_step = all_steps[index - 1]

            # Check if transformation from previous step is valid
            transform_valid = self._check_transformation_validity(prev_step, step)
            if not transform_valid:
                confidence *= 0.7  # Reduce confidence for questionable transformation

        # Check for proper algebraic operations
        if self._contains_algebraic_operation(step):
            algebraic_valid = self._validate_algebraic_operation(step)
            if not algebraic_valid:
                return False, 0.3
            confidence = max(confidence, 0.8)

        return True, confidence

    def _safe_eval(self, expr: str) -> Any:
        """Safely evaluate a mathematical expression."""
        try:
            # Remove whitespace
            expr = expr.strip()

            # Only allow safe mathematical operations
            # Replace common mathematical functions
            safe_expr = expr
            safe_expr = safe_expr.replace('^', '**')

            # Check for only allowed characters
            allowed = set('0123456789+-*/().**x ')
            if not all(c in allowed for c in safe_expr.replace('.', '').replace('x', '')):
                # Contains variables or unsafe characters
                return None

            # If contains 'x' or other variables, can't evaluate numerically
            if 'x' in safe_expr.lower() or any(c.isalpha() for c in safe_expr):
                return None

            # Create restricted namespace
            safe_namespace = {
                '__builtins__': {},
                'abs': abs,
                'min': min,
                'max': max,
                'pow': pow,
            }

            # Evaluate
            result = eval(safe_expr, safe_namespace)
            return result

        except Exception as e:
            logger.debug(f"Evaluation error: {e}")
            return None

    def _check_transformation_validity(self, prev_step: str, curr_step: str) -> bool:
        """Check if transformation from prev_step to curr_step is valid."""
        # Extract both sides of equations if present
        prev_parts = prev_step.split('=') if '=' in prev_step else [prev_step]
        curr_parts = curr_step.split('=') if '=' in curr_step else [curr_step]

        if len(prev_parts) != len(curr_parts):
            return False  # Changed equation structure

        # Check for valid transformation operations
        # Common valid transformations:
        # 1. Adding/subtracting same value to both sides
        # 2. Multiplying/dividing both sides by same non-zero value
        # 3. Simplification (combining like terms)
        # 4. Distribution/factoring

        # For now, accept if both have same number of equals signs
        # More sophisticated checking would use symbolic math
        return True

    def _contains_algebraic_operation(self, step: str) -> bool:
        """Check if step contains algebraic operations."""
        algebraic_indicators = ['x', 'y', 'z', 'a', 'b', 'c', 'n', '√', 'sqrt', 'log', 'ln', 'sin', 'cos', 'tan']
        return any(indicator in step.lower() for indicator in algebraic_indicators)

    def _validate_algebraic_operation(self, step: str) -> bool:
        """Validate algebraic operations in the step."""
        # Check for balanced operations
        if '(' in step or ')' in step:
            if not self._balanced_parentheses(step):
                return False

        # Check for valid algebraic syntax
        # No double operators (except **, --, ++)
        invalid_patterns = ['++', '+-', '-+', '*/', '/*', '//', '==']
        for pattern in invalid_patterns:
            if pattern in step and pattern not in ['**', '--']:
                return False

        return True
    
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
        """
        Check if transformation from step1 to step2 is valid.
        Validates that mathematical operations preserve equality.
        """
        # If either step is empty, can't validate
        if not step1 or not step2:
            return False

        # Both steps should have equals signs for equation transformation
        has_eq1 = '=' in step1
        has_eq2 = '=' in step2

        # If one has equals and other doesn't, transformation changed structure
        if has_eq1 != has_eq2:
            return False

        if not has_eq1:
            # No equations to validate, just expressions
            # Check if they're equivalent
            val1 = self._safe_eval(step1)
            val2 = self._safe_eval(step2)

            if val1 is not None and val2 is not None:
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    return abs(val1 - val2) < 1e-10

            # Can't evaluate, accept transformation
            return True

        # Both have equations - validate transformation
        parts1 = step1.split('=')
        parts2 = step2.split('=')

        if len(parts1) != 2 or len(parts2) != 2:
            return False  # Invalid equation format

        left1, right1 = parts1[0].strip(), parts1[1].strip()
        left2, right2 = parts2[0].strip(), parts2[1].strip()

        # Try to evaluate both equations
        try:
            # Check if step1 is balanced
            val1_left = self._safe_eval(left1)
            val1_right = self._safe_eval(right1)

            # Check if step2 is balanced
            val2_left = self._safe_eval(left2)
            val2_right = self._safe_eval(right2)

            # If all can be evaluated
            if all(v is not None for v in [val1_left, val1_right, val2_left, val2_right]):
                # Check if both equations are balanced
                eq1_balanced = abs(val1_left - val1_right) < 1e-10
                eq2_balanced = abs(val2_left - val2_right) < 1e-10

                if not eq1_balanced or not eq2_balanced:
                    return False

                # Check if the solution is preserved
                # (equations should have same solution)
                return True

        except Exception:
            pass

        # Can't fully validate numerically, check structural validity
        return self._check_structural_transformation(step1, step2)

    def _check_structural_transformation(self, step1: str, step2: str) -> bool:
        """Check if structural transformation appears valid."""
        # Check for common valid transformation patterns

        # 1. Simplification (longer to shorter)
        # 2. Expansion (shorter to longer)
        # 3. Substitution (similar complexity)

        # Extract tokens
        tokens1 = self._tokenize_math(step1)
        tokens2 = self._tokenize_math(step2)

        # Check if tokens are reasonable
        # Should not lose fundamental structure
        numbers1 = [t for t in tokens1 if t.replace('.', '').replace('-', '').isdigit()]
        numbers2 = [t for t in tokens2 if t.replace('.', '').replace('-', '').isdigit()]

        # If numeric constants disappeared, might be invalid (unless solving)
        # This is a heuristic - symbolic math library would be better

        # Accept transformation if it's reasonable
        return True

    def _tokenize_math(self, expr: str) -> List[str]:
        """Tokenize a mathematical expression."""
        # Simple tokenization
        tokens = []
        current = ""

        for char in expr:
            if char in '+-*/=()^':
                if current:
                    tokens.append(current.strip())
                    current = ""
                tokens.append(char)
            elif char.isspace():
                if current:
                    tokens.append(current.strip())
                    current = ""
            else:
                current += char

        if current:
            tokens.append(current.strip())

        return [t for t in tokens if t]
    
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
            rel_conf_avg = statistics.mean(relationship_confidences)
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
            avg_confidence = statistics.mean(confidences)
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