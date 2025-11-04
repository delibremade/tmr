"""
Pattern Extractors

Domain-specific pattern extraction logic for math, code, and logic domains.
"""

from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import re
from datetime import datetime

from .patterns import Pattern, DomainType, PatternComplexity, PatternMetadata


class PatternExtractor(ABC):
    """Base class for pattern extractors."""

    def __init__(self, domain: DomainType):
        """Initialize extractor for specific domain."""
        self.domain = domain
        self.extraction_count = 0

    @abstractmethod
    def extract(self, content: Any, context: Optional[Dict] = None) -> List[Pattern]:
        """
        Extract patterns from content.

        Args:
            content: Content to extract patterns from
            context: Optional context information

        Returns:
            List of extracted patterns
        """
        pass

    @abstractmethod
    def identify_structure(self, content: Any) -> Dict[str, Any]:
        """Identify the structure of the content."""
        pass

    def _create_pattern(self, name: str, description: str, structure: Dict,
                       complexity: PatternComplexity = PatternComplexity.SIMPLE,
                       **kwargs) -> Pattern:
        """Helper to create a pattern."""
        metadata = PatternMetadata(
            source=f"{self.domain.value}_extractor",
            tags={self.domain.value, complexity.value}
        )

        return Pattern(
            pattern_id="",  # Will be auto-generated
            domain=self.domain,
            name=name,
            description=description,
            structure=structure,
            complexity=complexity,
            metadata=metadata,
            **kwargs
        )


class MathPatternExtractor(PatternExtractor):
    """Extract mathematical patterns."""

    def __init__(self):
        """Initialize mathematical pattern extractor."""
        super().__init__(DomainType.MATH)

        self.operation_patterns = {
            "arithmetic": r'(\d+)\s*([+\-*/])\s*(\d+)',
            "equation": r'([a-z])\s*=\s*([^=]+)',
            "function": r'f\(([^)]+)\)\s*=\s*([^=]+)',
            "derivative": r'd/d([a-z])\s*\[?([^\]]+)\]?',
            "integral": r'∫\s*([^d]+)\s*d([a-z])',
        }

    def extract(self, content: Any, context: Optional[Dict] = None) -> List[Pattern]:
        """Extract mathematical patterns."""
        self.extraction_count += 1
        patterns = []

        content_str = str(content)

        # Extract arithmetic patterns
        arithmetic_patterns = self._extract_arithmetic(content_str)
        patterns.extend(arithmetic_patterns)

        # Extract equation patterns
        equation_patterns = self._extract_equations(content, content_str)
        patterns.extend(equation_patterns)

        # Extract function patterns
        function_patterns = self._extract_functions(content_str)
        patterns.extend(function_patterns)

        # Extract problem-solving patterns
        if isinstance(content, dict):
            problem_patterns = self._extract_problem_patterns(content)
            patterns.extend(problem_patterns)

        return patterns

    def identify_structure(self, content: Any) -> Dict[str, Any]:
        """Identify mathematical structure."""
        structure = {
            "type": "mathematical",
            "components": []
        }

        content_str = str(content)

        # Identify operations
        for op_type, pattern in self.operation_patterns.items():
            if re.search(pattern, content_str):
                structure["components"].append(op_type)

        # Check for equations
        if isinstance(content, dict):
            if "equation" in content or "formula" in content:
                structure["has_equation"] = True
            if "steps" in content:
                structure["has_solution_steps"] = True
                structure["step_count"] = len(content.get("steps", []))

        return structure

    def _extract_arithmetic(self, content_str: str) -> List[Pattern]:
        """Extract arithmetic operation patterns."""
        patterns = []

        matches = re.finditer(self.operation_patterns["arithmetic"], content_str)
        for match in matches:
            operand1, operator, operand2 = match.groups()

            pattern = self._create_pattern(
                name=f"arithmetic_{operator}",
                description=f"Arithmetic operation: {operator}",
                structure={
                    "operation": operator,
                    "operand_type": "numeric",
                    "format": "binary_operation"
                },
                complexity=PatternComplexity.SIMPLE,
                applicable_contexts=["arithmetic", "calculation"],
                examples=[{
                    "expression": f"{operand1} {operator} {operand2}",
                    "structure": "operand1 operator operand2"
                }]
            )
            patterns.append(pattern)

        return patterns

    def _extract_equations(self, content: Any, content_str: str) -> List[Pattern]:
        """Extract equation solving patterns."""
        patterns = []

        # Look for equation structures
        if isinstance(content, dict) and "equation" in content:
            equation = content["equation"]
            steps = content.get("steps", [])

            complexity = PatternComplexity.SIMPLE
            if len(steps) > 3:
                complexity = PatternComplexity.INTERMEDIATE
            if any(keyword in content_str.lower() for keyword in ["quadratic", "polynomial", "system"]):
                complexity = PatternComplexity.COMPLEX

            pattern = self._create_pattern(
                name="equation_solving",
                description="Pattern for solving equations",
                structure={
                    "equation": equation,
                    "has_steps": len(steps) > 0,
                    "step_count": len(steps)
                },
                complexity=complexity,
                applicable_contexts=["algebra", "equation_solving"],
                transformation_steps=[
                    {"action": "isolate_variable", "description": "Isolate variable on one side"},
                    {"action": "simplify", "description": "Simplify expression"},
                    {"action": "solve", "description": "Solve for variable"}
                ]
            )
            patterns.append(pattern)

        return patterns

    def _extract_functions(self, content_str: str) -> List[Pattern]:
        """Extract function patterns."""
        patterns = []

        matches = re.finditer(self.operation_patterns["function"], content_str)
        for match in matches:
            var, expression = match.groups()

            pattern = self._create_pattern(
                name="function_definition",
                description="Mathematical function definition pattern",
                structure={
                    "variable": var,
                    "expression_type": "function",
                    "has_domain": False
                },
                complexity=PatternComplexity.INTERMEDIATE,
                applicable_contexts=["functions", "calculus"]
            )
            patterns.append(pattern)

        return patterns

    def _extract_problem_patterns(self, content: Dict) -> List[Pattern]:
        """Extract problem-solving patterns from structured content."""
        patterns = []

        # Word problem pattern
        if "problem" in content and "solution" in content:
            pattern = self._create_pattern(
                name="word_problem_solving",
                description="Pattern for solving mathematical word problems",
                structure={
                    "has_problem_statement": True,
                    "has_solution": True,
                    "components": list(content.keys())
                },
                complexity=PatternComplexity.INTERMEDIATE,
                applicable_contexts=["word_problems", "applications"],
                transformation_steps=[
                    {"action": "parse_problem", "description": "Extract relevant information"},
                    {"action": "formulate_equation", "description": "Convert to mathematical form"},
                    {"action": "solve", "description": "Solve the equation"},
                    {"action": "interpret", "description": "Interpret result in context"}
                ]
            )
            patterns.append(pattern)

        return patterns


class CodePatternExtractor(PatternExtractor):
    """Extract code patterns."""

    def __init__(self):
        """Initialize code pattern extractor."""
        super().__init__(DomainType.CODE)

        self.code_patterns = {
            "function_def": r'def\s+(\w+)\s*\(([^)]*)\)',
            "class_def": r'class\s+(\w+)',
            "loop_for": r'for\s+(\w+)\s+in\s+([^:]+):',
            "loop_while": r'while\s+([^:]+):',
            "conditional": r'if\s+([^:]+):',
            "list_comp": r'\[([^\]]+)\s+for\s+([^\]]+)\]',
        }

    def extract(self, content: Any, context: Optional[Dict] = None) -> List[Pattern]:
        """Extract code patterns."""
        self.extraction_count += 1
        patterns = []

        content_str = str(content)

        # Extract function patterns
        function_patterns = self._extract_functions(content_str)
        patterns.extend(function_patterns)

        # Extract loop patterns
        loop_patterns = self._extract_loops(content_str)
        patterns.extend(loop_patterns)

        # Extract algorithm patterns
        if isinstance(content, dict):
            algo_patterns = self._extract_algorithms(content)
            patterns.extend(algo_patterns)

        return patterns

    def identify_structure(self, content: Any) -> Dict[str, Any]:
        """Identify code structure."""
        structure = {
            "type": "code",
            "components": []
        }

        content_str = str(content)

        # Identify code constructs
        for construct_type, pattern in self.code_patterns.items():
            if re.search(pattern, content_str):
                structure["components"].append(construct_type)

        # Check for algorithmic patterns
        if isinstance(content, dict):
            if "algorithm" in content:
                structure["has_algorithm"] = True
            if "complexity" in content:
                structure["has_complexity_analysis"] = True

        return structure

    def _extract_functions(self, content_str: str) -> List[Pattern]:
        """Extract function definition patterns."""
        patterns = []

        matches = re.finditer(self.code_patterns["function_def"], content_str)
        for match in matches:
            func_name, params = match.groups()

            param_count = len([p.strip() for p in params.split(',') if p.strip()])

            pattern = self._create_pattern(
                name="function_definition",
                description=f"Function definition pattern: {func_name}",
                structure={
                    "construct": "function",
                    "name": func_name,
                    "parameter_count": param_count,
                    "has_parameters": param_count > 0
                },
                complexity=PatternComplexity.SIMPLE if param_count <= 2 else PatternComplexity.INTERMEDIATE,
                applicable_contexts=["functions", "procedures", "methods"]
            )
            patterns.append(pattern)

        return patterns

    def _extract_loops(self, content_str: str) -> List[Pattern]:
        """Extract loop patterns."""
        patterns = []

        # For loops
        for_matches = re.finditer(self.code_patterns["loop_for"], content_str)
        for match in for_matches:
            pattern = self._create_pattern(
                name="for_loop_iteration",
                description="For loop iteration pattern",
                structure={
                    "construct": "for_loop",
                    "iteration_type": "collection"
                },
                complexity=PatternComplexity.SIMPLE,
                applicable_contexts=["iteration", "loops", "collections"]
            )
            patterns.append(pattern)

        # While loops
        while_matches = re.finditer(self.code_patterns["loop_while"], content_str)
        for match in while_matches:
            pattern = self._create_pattern(
                name="while_loop_iteration",
                description="While loop iteration pattern",
                structure={
                    "construct": "while_loop",
                    "iteration_type": "conditional"
                },
                complexity=PatternComplexity.SIMPLE,
                applicable_contexts=["iteration", "loops", "conditions"]
            )
            patterns.append(pattern)

        return patterns

    def _extract_algorithms(self, content: Dict) -> List[Pattern]:
        """Extract algorithmic patterns."""
        patterns = []

        if "algorithm" in content:
            algo_name = content.get("name", "unnamed_algorithm")
            steps = content.get("steps", [])
            complexity = content.get("complexity", "unknown")

            pattern_complexity = PatternComplexity.INTERMEDIATE
            if "O(n^2)" in complexity or "quadratic" in complexity:
                pattern_complexity = PatternComplexity.COMPLEX
            elif "O(2^n)" in complexity or "exponential" in complexity:
                pattern_complexity = PatternComplexity.ADVANCED

            pattern = self._create_pattern(
                name=f"algorithm_{algo_name}",
                description=f"Algorithm pattern: {algo_name}",
                structure={
                    "type": "algorithm",
                    "name": algo_name,
                    "step_count": len(steps),
                    "complexity": complexity
                },
                complexity=pattern_complexity,
                applicable_contexts=["algorithms", "optimization", "problem_solving"],
                transformation_steps=[
                    {"action": "initialize", "description": "Initialize data structures"},
                    {"action": "process", "description": "Execute algorithm steps"},
                    {"action": "return_result", "description": "Return computed result"}
                ]
            )
            patterns.append(pattern)

        return patterns


class LogicPatternExtractor(PatternExtractor):
    """Extract logical reasoning patterns."""

    def __init__(self):
        """Initialize logic pattern extractor."""
        super().__init__(DomainType.LOGIC)

        self.logic_patterns = {
            "implication": r'if\s+(.+?)\s+then\s+(.+)',
            "universal": r'all\s+(\w+)\s+are\s+(.+)',
            "existential": r'some\s+(\w+)\s+are\s+(.+)',
            "negation": r'not\s+(.+)',
            "conjunction": r'(.+?)\s+and\s+(.+)',
            "disjunction": r'(.+?)\s+or\s+(.+)',
        }

    def extract(self, content: Any, context: Optional[Dict] = None) -> List[Pattern]:
        """Extract logical patterns."""
        self.extraction_count += 1
        patterns = []

        content_str = str(content).lower()

        # Extract logical connective patterns
        connective_patterns = self._extract_connectives(content_str)
        patterns.extend(connective_patterns)

        # Extract argument patterns
        if isinstance(content, dict):
            argument_patterns = self._extract_arguments(content)
            patterns.extend(argument_patterns)

        # Extract inference patterns
        inference_patterns = self._extract_inferences(content, content_str)
        patterns.extend(inference_patterns)

        return patterns

    def identify_structure(self, content: Any) -> Dict[str, Any]:
        """Identify logical structure."""
        structure = {
            "type": "logical",
            "components": []
        }

        content_str = str(content).lower()

        # Identify logical constructs
        for construct_type, pattern in self.logic_patterns.items():
            if re.search(pattern, content_str, re.IGNORECASE):
                structure["components"].append(construct_type)

        # Check for argument structure
        if isinstance(content, dict):
            if "premises" in content or "premise" in content:
                structure["has_premises"] = True
            if "conclusion" in content:
                structure["has_conclusion"] = True
            if "premises" in content and "conclusion" in content:
                structure["is_argument"] = True

        return structure

    def _extract_connectives(self, content_str: str) -> List[Pattern]:
        """Extract logical connective patterns."""
        patterns = []

        # Implication pattern
        impl_matches = re.finditer(self.logic_patterns["implication"], content_str, re.IGNORECASE)
        for match in impl_matches:
            pattern = self._create_pattern(
                name="implication",
                description="Logical implication (if-then) pattern",
                structure={
                    "connective": "implication",
                    "form": "if P then Q",
                    "components": ["antecedent", "consequent"]
                },
                complexity=PatternComplexity.SIMPLE,
                applicable_contexts=["conditional_reasoning", "implications"]
            )
            patterns.append(pattern)

        # Conjunction pattern
        conj_matches = re.finditer(self.logic_patterns["conjunction"], content_str, re.IGNORECASE)
        for match in conj_matches:
            pattern = self._create_pattern(
                name="conjunction",
                description="Logical conjunction (and) pattern",
                structure={
                    "connective": "conjunction",
                    "form": "P and Q",
                    "components": ["conjunct1", "conjunct2"]
                },
                complexity=PatternComplexity.SIMPLE,
                applicable_contexts=["compound_statements", "logical_operators"]
            )
            patterns.append(pattern)

        return patterns

    def _extract_arguments(self, content: Dict) -> List[Pattern]:
        """Extract argument patterns."""
        patterns = []

        if "premises" in content and "conclusion" in content:
            premises = content["premises"]
            premise_count = len(premises) if isinstance(premises, list) else 1

            complexity = PatternComplexity.SIMPLE
            if premise_count > 2:
                complexity = PatternComplexity.INTERMEDIATE
            if premise_count > 4:
                complexity = PatternComplexity.COMPLEX

            pattern = self._create_pattern(
                name="deductive_argument",
                description="Deductive argument pattern",
                structure={
                    "type": "argument",
                    "form": "deductive",
                    "premise_count": premise_count,
                    "has_conclusion": True
                },
                complexity=complexity,
                applicable_contexts=["arguments", "reasoning", "inference"],
                validation_rules=[
                    {"rule": "premises_valid", "description": "All premises must be valid"},
                    {"rule": "conclusion_follows", "description": "Conclusion must follow from premises"}
                ],
                transformation_steps=[
                    {"action": "validate_premises", "description": "Check premise validity"},
                    {"action": "check_inference", "description": "Verify logical connection"},
                    {"action": "validate_conclusion", "description": "Confirm conclusion validity"}
                ]
            )
            patterns.append(pattern)

        return patterns

    def _extract_inferences(self, content: Any, content_str: str) -> List[Pattern]:
        """Extract inference patterns."""
        patterns = []

        # Modus ponens pattern
        if re.search(r'if\s+.*\s+then.*therefore', content_str, re.IGNORECASE):
            pattern = self._create_pattern(
                name="modus_ponens",
                description="Modus ponens inference pattern",
                structure={
                    "type": "inference_rule",
                    "name": "modus_ponens",
                    "form": "If P then Q; P; Therefore Q"
                },
                complexity=PatternComplexity.INTERMEDIATE,
                applicable_contexts=["inference", "deduction", "logic"],
                validation_rules=[
                    {"rule": "has_conditional", "description": "Must have if-then statement"},
                    {"rule": "affirms_antecedent", "description": "Must affirm the antecedent"}
                ]
            )
            patterns.append(pattern)

        return patterns


class PatternExtractorFactory:
    """Factory for creating pattern extractors."""

    _extractors = {
        DomainType.MATH: MathPatternExtractor,
        DomainType.CODE: CodePatternExtractor,
        DomainType.LOGIC: LogicPatternExtractor
    }

    @classmethod
    def get_extractor(cls, domain: DomainType) -> PatternExtractor:
        """
        Get extractor for specific domain.

        Args:
            domain: Domain type

        Returns:
            PatternExtractor instance

        Raises:
            ValueError: If domain is not supported
        """
        if domain not in cls._extractors:
            raise ValueError(f"No extractor available for domain: {domain}")

        return cls._extractors[domain]()

    @classmethod
    def get_all_extractors(cls) -> Dict[DomainType, PatternExtractor]:
        """Get all available extractors."""
        return {domain: extractor() for domain, extractor in cls._extractors.items()}
