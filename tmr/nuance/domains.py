"""
Domain Classification Logic

Classifies reasoning problems into specific domains (math, code, logic).
"""

from typing import Any, Dict, List, Optional, Tuple
import re
from .patterns import DomainType


class DomainClassifier:
    """
    Classifies reasoning problems into specific domains.

    Supports: Mathematical, Code, and Logical domains.
    """

    def __init__(self):
        """Initialize the domain classifier with domain-specific patterns."""
        self.domain_patterns = {
            DomainType.MATH: self._init_math_patterns(),
            DomainType.CODE: self._init_code_patterns(),
            DomainType.LOGIC: self._init_logic_patterns()
        }

        # Keywords for quick domain identification
        self.domain_keywords = {
            DomainType.MATH: {
                'calculate', 'equation', 'solve', 'formula', 'derivative',
                'integral', 'theorem', 'proof', 'number', 'sum', 'product',
                'division', 'multiply', 'subtract', 'add', 'algebra',
                'geometry', 'calculus', 'statistics', 'probability',
                'matrix', 'vector', 'function', 'expression'
            },
            DomainType.CODE: {
                'function', 'class', 'method', 'variable', 'algorithm',
                'data structure', 'implementation', 'program', 'code',
                'syntax', 'compile', 'execute', 'debug', 'refactor',
                'optimize', 'loop', 'conditional', 'recursion', 'array',
                'list', 'dictionary', 'object', 'module', 'import',
                'return', 'parameter', 'argument'
            },
            DomainType.LOGIC: {
                'if', 'then', 'therefore', 'implies', 'because', 'since',
                'given', 'assume', 'conclude', 'deduce', 'infer', 'premise',
                'conclusion', 'argument', 'valid', 'sound', 'fallacy',
                'contradiction', 'consistent', 'necessary', 'sufficient',
                'all', 'some', 'none', 'exists', 'forall'
            }
        }

    def _init_math_patterns(self) -> Dict[str, Any]:
        """Initialize mathematical domain patterns."""
        return {
            "equation_patterns": [
                r'\d+\s*[+\-*/=]\s*\d+',  # Simple arithmetic
                r'[a-z]\s*=\s*[^=]+',  # Variable assignment
                r'[a-z]\^?\d*\s*[+\-*/]\s*[a-z]\^?\d*',  # Algebraic expressions
                r'∫|∑|∏|lim|sin|cos|tan|log|ln|exp',  # Mathematical functions
                r'\d+/\d+',  # Fractions
                r'\d+\.\d+',  # Decimals
            ],
            "structure_keys": [
                "equation", "formula", "calculation", "expression",
                "theorem", "proof", "result", "solution"
            ],
            "operators": ['+', '-', '*', '/', '=', '^', '√', '∫', '∑', '∏'],
            "complexity_indicators": {
                "simple": ["add", "subtract", "multiply", "divide"],
                "intermediate": ["equation", "solve", "formula"],
                "complex": ["derivative", "integral", "theorem", "proof"],
                "advanced": ["multivariate", "differential", "topology"]
            }
        }

    def _init_code_patterns(self) -> Dict[str, Any]:
        """Initialize code domain patterns."""
        return {
            "code_patterns": [
                r'def\s+\w+\s*\(',  # Python function definition
                r'class\s+\w+',  # Class definition
                r'function\s+\w+\s*\(',  # JavaScript function
                r'for\s+\w+\s+in\s+',  # For loop
                r'if\s+.*:',  # Conditional
                r'return\s+',  # Return statement
                r'\w+\s*=\s*\[.*\]',  # List/array assignment
                r'\w+\s*=\s*\{.*\}',  # Dict/object assignment
            ],
            "structure_keys": [
                "function", "class", "method", "algorithm", "implementation",
                "code", "program", "module", "library"
            ],
            "language_indicators": {
                "python": ["def", "class", "import", "self", "None"],
                "javascript": ["const", "let", "var", "function", "=>"],
                "java": ["public", "private", "static", "void"],
                "cpp": ["#include", "std::", "int main"]
            },
            "complexity_indicators": {
                "simple": ["variable", "assignment", "print"],
                "intermediate": ["function", "loop", "conditional"],
                "complex": ["class", "recursion", "data structure"],
                "advanced": ["design pattern", "optimization", "concurrency"]
            }
        }

    def _init_logic_patterns(self) -> Dict[str, Any]:
        """Initialize logical domain patterns."""
        return {
            "logic_patterns": [
                r'if\s+.+\s+then\s+.+',  # Conditional logic
                r'all\s+\w+\s+are\s+',  # Universal quantification
                r'some\s+\w+\s+are\s+',  # Existential quantification
                r'therefore\s+',  # Conclusion marker
                r'because\s+',  # Causation
                r'implies\s+',  # Implication
            ],
            "structure_keys": [
                "premise", "conclusion", "argument", "inference",
                "deduction", "induction", "reasoning", "logic"
            ],
            "logical_connectives": [
                "and", "or", "not", "if", "then", "implies", "iff",
                "∧", "∨", "¬", "→", "↔", "∀", "∃"
            ],
            "complexity_indicators": {
                "simple": ["and", "or", "not"],
                "intermediate": ["if-then", "implies", "because"],
                "complex": ["all", "some", "exists", "forall"],
                "advanced": ["modal logic", "temporal logic", "higher-order"]
            }
        }

    def classify(self, content: Any) -> DomainType:
        """
        Classify content into a domain.

        Args:
            content: The content to classify (string, dict, or list)

        Returns:
            DomainType classification
        """
        # Convert to string representation for analysis
        content_str = self._prepare_content(content)

        # Calculate scores for each domain
        scores = {
            DomainType.MATH: self._score_math(content, content_str),
            DomainType.CODE: self._score_code(content, content_str),
            DomainType.LOGIC: self._score_logic(content, content_str)
        }

        # Return domain with highest score
        max_domain = max(scores, key=scores.get)

        # If all scores are very low, return UNKNOWN
        if scores[max_domain] < 0.2:
            return DomainType.UNKNOWN

        return max_domain

    def classify_with_confidence(self, content: Any) -> Tuple[DomainType, float, Dict[DomainType, float]]:
        """
        Classify content with confidence scores.

        Args:
            content: The content to classify

        Returns:
            Tuple of (domain, confidence, all_scores)
        """
        content_str = self._prepare_content(content)

        scores = {
            DomainType.MATH: self._score_math(content, content_str),
            DomainType.CODE: self._score_code(content, content_str),
            DomainType.LOGIC: self._score_logic(content, content_str)
        }

        # Find max score
        max_domain = max(scores, key=scores.get)
        max_score = scores[max_domain]

        # Calculate confidence based on score separation
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            # Confidence is higher when there's a clear winner
            separation = sorted_scores[0] - sorted_scores[1]
            confidence = min(1.0, max_score * (1.0 + separation))
        else:
            confidence = max_score

        # Check for unknown domain
        if max_score < 0.2:
            return DomainType.UNKNOWN, 0.0, scores

        return max_domain, confidence, scores

    def _prepare_content(self, content: Any) -> str:
        """Prepare content for analysis."""
        if isinstance(content, str):
            return content.lower()
        elif isinstance(content, dict):
            # Extract relevant text fields
            text_parts = []
            for key, value in content.items():
                text_parts.append(str(key))
                if isinstance(value, (str, int, float)):
                    text_parts.append(str(value))
            return ' '.join(text_parts).lower()
        elif isinstance(content, list):
            return ' '.join(str(item) for item in content).lower()
        else:
            return str(content).lower()

    def _score_math(self, content: Any, content_str: str) -> float:
        """Calculate mathematical domain score."""
        score = 0.0

        # Check keywords
        keyword_score = self._keyword_score(content_str, DomainType.MATH)
        score += keyword_score * 0.4

        # Check patterns
        pattern_score = self._pattern_score(content_str, self.domain_patterns[DomainType.MATH]["equation_patterns"])
        score += pattern_score * 0.3

        # Check structure (if dict)
        if isinstance(content, dict):
            structure_score = self._structure_score(content, self.domain_patterns[DomainType.MATH]["structure_keys"])
            score += structure_score * 0.3

        return min(1.0, score)

    def _score_code(self, content: Any, content_str: str) -> float:
        """Calculate code domain score."""
        score = 0.0

        # Check keywords
        keyword_score = self._keyword_score(content_str, DomainType.CODE)
        score += keyword_score * 0.4

        # Check patterns
        pattern_score = self._pattern_score(content_str, self.domain_patterns[DomainType.CODE]["code_patterns"])
        score += pattern_score * 0.3

        # Check structure
        if isinstance(content, dict):
            structure_score = self._structure_score(content, self.domain_patterns[DomainType.CODE]["structure_keys"])
            score += structure_score * 0.3

        return min(1.0, score)

    def _score_logic(self, content: Any, content_str: str) -> float:
        """Calculate logical domain score."""
        score = 0.0

        # Check keywords
        keyword_score = self._keyword_score(content_str, DomainType.LOGIC)
        score += keyword_score * 0.4

        # Check patterns
        pattern_score = self._pattern_score(content_str, self.domain_patterns[DomainType.LOGIC]["logic_patterns"])
        score += pattern_score * 0.3

        # Check structure
        if isinstance(content, dict):
            structure_score = self._structure_score(content, self.domain_patterns[DomainType.LOGIC]["structure_keys"])
            score += structure_score * 0.3

        return min(1.0, score)

    def _keyword_score(self, text: str, domain: DomainType) -> float:
        """Calculate keyword matching score."""
        keywords = self.domain_keywords[domain]
        matches = sum(1 for keyword in keywords if keyword in text)
        return min(1.0, matches / max(5, len(keywords) * 0.1))

    def _pattern_score(self, text: str, patterns: List[str]) -> float:
        """Calculate regex pattern matching score."""
        matches = sum(1 for pattern in patterns if re.search(pattern, text))
        return min(1.0, matches / max(1, len(patterns) * 0.3))

    def _structure_score(self, content: Dict, structure_keys: List[str]) -> float:
        """Calculate structural matching score."""
        if not isinstance(content, dict):
            return 0.0

        matches = sum(1 for key in structure_keys if key in content)
        return min(1.0, matches / max(1, len(structure_keys) * 0.2))

    def get_domain_characteristics(self, domain: DomainType) -> Dict[str, Any]:
        """
        Get characteristics of a specific domain.

        Args:
            domain: The domain type

        Returns:
            Dictionary of domain characteristics
        """
        if domain == DomainType.UNKNOWN:
            return {"name": "Unknown", "description": "Unclassified domain"}

        return {
            "name": domain.value,
            "keywords": list(self.domain_keywords[domain]),
            "patterns": self.domain_patterns[domain],
            "complexity_levels": list(self.domain_patterns[domain].get("complexity_indicators", {}).keys())
        }

    def suggest_domain(self, content: Any) -> Dict[str, Any]:
        """
        Suggest domain with detailed explanation.

        Args:
            content: Content to classify

        Returns:
            Dictionary with domain suggestion and reasoning
        """
        domain, confidence, scores = self.classify_with_confidence(content)

        return {
            "suggested_domain": domain.value,
            "confidence": confidence,
            "all_scores": {d.value: s for d, s in scores.items()},
            "reasoning": self._explain_classification(content, domain, scores)
        }

    def _explain_classification(self, content: Any, domain: DomainType,
                               scores: Dict[DomainType, float]) -> List[str]:
        """Generate explanation for classification."""
        explanations = []

        content_str = self._prepare_content(content)

        # Explain winning domain
        if domain != DomainType.UNKNOWN:
            keywords = [kw for kw in self.domain_keywords[domain] if kw in content_str]
            if keywords:
                explanations.append(f"Contains {domain.value} keywords: {', '.join(keywords[:3])}")

            patterns = self.domain_patterns[domain]
            for pattern_list_key in [k for k in patterns.keys() if 'patterns' in k.lower()]:
                if any(re.search(p, content_str) for p in patterns[pattern_list_key]):
                    explanations.append(f"Matches {domain.value} patterns")
                    break

        # Compare scores
        sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_domains) > 1:
            diff = sorted_domains[0][1] - sorted_domains[1][1]
            if diff > 0.3:
                explanations.append(f"Clear separation from other domains (margin: {diff:.2f})")
            elif diff < 0.1:
                explanations.append(f"Close call with {sorted_domains[1][0].value} domain")

        return explanations
