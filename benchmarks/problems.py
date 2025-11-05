"""
Benchmark Problem Definitions

This module defines benchmark problems across MATH, CODE, and LOGIC domains
with varying complexity levels to test TMR's verification capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Complexity levels for benchmark problems"""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"


class ProblemDomain(Enum):
    """Problem domains matching TMR's target domains"""
    MATH = "math"
    CODE = "code"
    LOGIC = "logic"
    MIXED = "mixed"


@dataclass
class BenchmarkProblem:
    """
    A single benchmark problem for TMR validation.

    Attributes:
        id: Unique identifier for the problem
        domain: Problem domain (MATH, CODE, LOGIC, MIXED)
        complexity: Complexity level
        title: Human-readable title
        description: Problem description
        input_statement: Statement to be verified
        expected_valid: Whether statement should be valid
        expected_confidence: Expected confidence score (0.0-1.0)
        ground_truth: Ground truth explanation
        metadata: Additional metadata
        validator: Optional custom validation function
    """
    id: str
    domain: ProblemDomain
    complexity: ComplexityLevel
    title: str
    description: str
    input_statement: str
    expected_valid: bool
    expected_confidence: float
    ground_truth: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    validator: Optional[Callable[[Any], bool]] = None

    def validate_result(self, result: Any) -> bool:
        """
        Validate the result against expected outcome.

        Args:
            result: Verification result from TMR

        Returns:
            True if result matches expected outcome
        """
        if self.validator:
            return self.validator(result)

        # Default validation: check valid flag
        if isinstance(result, dict):
            return result.get("valid", False) == self.expected_valid
        return False


@dataclass
class ProblemSet:
    """
    A collection of related benchmark problems.

    Attributes:
        name: Problem set name
        domain: Primary domain
        description: Set description
        problems: List of problems in set
    """
    name: str
    domain: ProblemDomain
    description: str
    problems: List[BenchmarkProblem] = field(default_factory=list)

    def add_problem(self, problem: BenchmarkProblem) -> None:
        """Add a problem to the set"""
        self.problems.append(problem)

    def get_problems_by_complexity(self, complexity: ComplexityLevel) -> List[BenchmarkProblem]:
        """Get problems filtered by complexity level"""
        return [p for p in self.problems if p.complexity == complexity]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the problem set"""
        return {
            "total_problems": len(self.problems),
            "by_complexity": {
                level.value: len(self.get_problems_by_complexity(level))
                for level in ComplexityLevel
            },
            "expected_valid_rate": sum(1 for p in self.problems if p.expected_valid) / len(self.problems) if self.problems else 0,
            "avg_expected_confidence": sum(p.expected_confidence for p in self.problems) / len(self.problems) if self.problems else 0,
        }


# ============================================================================
# MATHEMATICAL DOMAIN PROBLEMS
# ============================================================================

def create_math_problems() -> ProblemSet:
    """Create mathematical domain benchmark problems"""
    problem_set = ProblemSet(
        name="Mathematical Reasoning",
        domain=ProblemDomain.MATH,
        description="Mathematical problems covering arithmetic, algebra, calculus, and proofs"
    )

    # TRIVIAL: Basic arithmetic
    problem_set.add_problem(BenchmarkProblem(
        id="MATH-001",
        domain=ProblemDomain.MATH,
        complexity=ComplexityLevel.TRIVIAL,
        title="Simple Addition",
        description="Basic arithmetic verification",
        input_statement="2 + 2 = 4",
        expected_valid=True,
        expected_confidence=0.95,
        ground_truth="Basic identity principle: 2 + 2 equals 4 by definition of addition",
        metadata={"category": "arithmetic", "operations": ["addition"]}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="MATH-002",
        domain=ProblemDomain.MATH,
        complexity=ComplexityLevel.TRIVIAL,
        title="Contradictory Arithmetic",
        description="Invalid arithmetic statement",
        input_statement="5 + 3 = 9",
        expected_valid=False,
        expected_confidence=0.90,
        ground_truth="Violates non-contradiction: 5 + 3 = 8, not 9",
        metadata={"category": "arithmetic", "operations": ["addition"]}
    ))

    # SIMPLE: Algebraic identities
    problem_set.add_problem(BenchmarkProblem(
        id="MATH-003",
        domain=ProblemDomain.MATH,
        complexity=ComplexityLevel.SIMPLE,
        title="Algebraic Identity",
        description="Basic algebraic equation",
        input_statement="If x = 5, then 2x = 10",
        expected_valid=True,
        expected_confidence=0.90,
        ground_truth="Substitution and multiplication: 2 * 5 = 10",
        metadata={"category": "algebra", "operations": ["substitution", "multiplication"]}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="MATH-004",
        domain=ProblemDomain.MATH,
        complexity=ComplexityLevel.SIMPLE,
        title="Quadratic Formula Application",
        description="Solve quadratic equation",
        input_statement="The equation x² - 5x + 6 = 0 has solutions x = 2 and x = 3",
        expected_valid=True,
        expected_confidence=0.88,
        ground_truth="Factorization: (x-2)(x-3) = 0, so x = 2 or x = 3",
        metadata={"category": "algebra", "operations": ["quadratic", "factorization"]}
    ))

    # MODERATE: Calculus and functions
    problem_set.add_problem(BenchmarkProblem(
        id="MATH-005",
        domain=ProblemDomain.MATH,
        complexity=ComplexityLevel.MODERATE,
        title="Derivative Calculation",
        description="Basic calculus derivative",
        input_statement="The derivative of f(x) = x² is f'(x) = 2x",
        expected_valid=True,
        expected_confidence=0.85,
        ground_truth="Power rule: d/dx(x^n) = n*x^(n-1), so d/dx(x²) = 2x",
        metadata={"category": "calculus", "operations": ["derivative", "power_rule"]}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="MATH-006",
        domain=ProblemDomain.MATH,
        complexity=ComplexityLevel.MODERATE,
        title="Integration",
        description="Definite integral evaluation",
        input_statement="The integral of f(x) = 2x from 0 to 1 equals 1",
        expected_valid=True,
        expected_confidence=0.83,
        ground_truth="∫₀¹ 2x dx = [x²]₀¹ = 1² - 0² = 1",
        metadata={"category": "calculus", "operations": ["integration", "definite_integral"]}
    ))

    # COMPLEX: Multi-step reasoning
    problem_set.add_problem(BenchmarkProblem(
        id="MATH-007",
        domain=ProblemDomain.MATH,
        complexity=ComplexityLevel.COMPLEX,
        title="Chain Rule Application",
        description="Composite function derivative",
        input_statement="The derivative of f(x) = (2x + 1)³ is f'(x) = 6(2x + 1)²",
        expected_valid=True,
        expected_confidence=0.80,
        ground_truth="Chain rule: d/dx[g(h(x))] = g'(h(x)) * h'(x). Here g(u)=u³, h(x)=2x+1, so 3u² * 2 = 6(2x+1)²",
        metadata={"category": "calculus", "operations": ["chain_rule", "composition"]}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="MATH-008",
        domain=ProblemDomain.MATH,
        complexity=ComplexityLevel.COMPLEX,
        title="Exponential Growth",
        description="Exponential function reasoning",
        input_statement="If a population doubles every 3 hours and starts at 100, after 9 hours it will be 800",
        expected_valid=True,
        expected_confidence=0.78,
        ground_truth="Exponential growth: 100 * 2^(9/3) = 100 * 2³ = 100 * 8 = 800",
        metadata={"category": "exponential", "operations": ["exponentiation", "modeling"]}
    ))

    # ADVANCED: Proof-based reasoning
    problem_set.add_problem(BenchmarkProblem(
        id="MATH-009",
        domain=ProblemDomain.MATH,
        complexity=ComplexityLevel.ADVANCED,
        title="Proof by Induction Structure",
        description="Mathematical induction reasoning",
        input_statement="To prove that 1 + 2 + ... + n = n(n+1)/2 for all n ≥ 1, we show: (1) Base case: n=1 gives 1 = 1(2)/2 = 1. (2) Inductive step: If true for k, then 1+...+k+(k+1) = k(k+1)/2 + (k+1) = (k+1)(k+2)/2.",
        expected_valid=True,
        expected_confidence=0.75,
        ground_truth="Valid proof by induction: base case verified, inductive step correctly derives P(k+1) from P(k)",
        metadata={"category": "proof", "operations": ["induction", "series"]}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="MATH-010",
        domain=ProblemDomain.MATH,
        complexity=ComplexityLevel.ADVANCED,
        title="Limit Calculation",
        description="Advanced limit evaluation",
        input_statement="The limit as x approaches 0 of sin(x)/x equals 1",
        expected_valid=True,
        expected_confidence=0.72,
        ground_truth="Standard calculus limit: lim(x→0) sin(x)/x = 1 (L'Hôpital's rule or Taylor series)",
        metadata={"category": "calculus", "operations": ["limits", "trigonometry"]}
    ))

    return problem_set


# ============================================================================
# CODE DOMAIN PROBLEMS
# ============================================================================

def create_code_problems() -> ProblemSet:
    """Create code domain benchmark problems"""
    problem_set = ProblemSet(
        name="Code Reasoning",
        domain=ProblemDomain.CODE,
        description="Code problems covering functions, algorithms, and data structures"
    )

    # TRIVIAL: Variable assignment
    problem_set.add_problem(BenchmarkProblem(
        id="CODE-001",
        domain=ProblemDomain.CODE,
        complexity=ComplexityLevel.TRIVIAL,
        title="Variable Assignment",
        description="Simple variable assignment",
        input_statement="After executing 'x = 5', the variable x holds the value 5",
        expected_valid=True,
        expected_confidence=0.95,
        ground_truth="Direct assignment: x = 5 assigns value 5 to variable x",
        metadata={"language": "python", "category": "variables"}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="CODE-002",
        domain=ProblemDomain.CODE,
        complexity=ComplexityLevel.TRIVIAL,
        title="Invalid Type Operation",
        description="Type error in operation",
        input_statement="In Python, the expression '5' + 3 returns '53'",
        expected_valid=False,
        expected_confidence=0.90,
        ground_truth="Type error: cannot add string and integer without conversion in Python",
        metadata={"language": "python", "category": "types", "error": "TypeError"}
    ))

    # SIMPLE: Function behavior
    problem_set.add_problem(BenchmarkProblem(
        id="CODE-003",
        domain=ProblemDomain.CODE,
        complexity=ComplexityLevel.SIMPLE,
        title="Function Return Value",
        description="Simple function evaluation",
        input_statement="def double(x): return x * 2\nCalling double(5) returns 10",
        expected_valid=True,
        expected_confidence=0.92,
        ground_truth="Function evaluation: double(5) = 5 * 2 = 10",
        metadata={"language": "python", "category": "functions"}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="CODE-004",
        domain=ProblemDomain.CODE,
        complexity=ComplexityLevel.SIMPLE,
        title="List Indexing",
        description="Array/list access",
        input_statement="Given arr = [1, 2, 3, 4], arr[2] equals 3",
        expected_valid=True,
        expected_confidence=0.90,
        ground_truth="Zero-based indexing: arr[2] accesses third element, which is 3",
        metadata={"language": "python", "category": "data_structures", "type": "list"}
    ))

    # MODERATE: Loops and iteration
    problem_set.add_problem(BenchmarkProblem(
        id="CODE-005",
        domain=ProblemDomain.CODE,
        complexity=ComplexityLevel.MODERATE,
        title="Loop Iteration Count",
        description="For loop analysis",
        input_statement="The loop 'for i in range(5): print(i)' prints 5 numbers (0 through 4)",
        expected_valid=True,
        expected_confidence=0.88,
        ground_truth="range(5) generates [0, 1, 2, 3, 4], which is 5 numbers",
        metadata={"language": "python", "category": "loops", "type": "for_loop"}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="CODE-006",
        domain=ProblemDomain.CODE,
        complexity=ComplexityLevel.MODERATE,
        title="Recursive Function",
        description="Simple recursion",
        input_statement="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)\nfactorial(4) returns 24",
        expected_valid=True,
        expected_confidence=0.85,
        ground_truth="factorial(4) = 4 * 3 * 2 * 1 = 24",
        metadata={"language": "python", "category": "recursion"}
    ))

    # COMPLEX: Algorithm analysis
    problem_set.add_problem(BenchmarkProblem(
        id="CODE-007",
        domain=ProblemDomain.CODE,
        complexity=ComplexityLevel.COMPLEX,
        title="Binary Search Correctness",
        description="Algorithm verification",
        input_statement="Binary search on a sorted array of size n has O(log n) time complexity because it halves the search space each iteration",
        expected_valid=True,
        expected_confidence=0.82,
        ground_truth="Each iteration reduces search space by half: n → n/2 → n/4 → ... → 1, taking log₂(n) steps",
        metadata={"category": "algorithms", "type": "search", "complexity": "O(log n)"}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="CODE-008",
        domain=ProblemDomain.CODE,
        complexity=ComplexityLevel.COMPLEX,
        title="Sorting Algorithm Stability",
        description="Algorithm property verification",
        input_statement="Merge sort is a stable sorting algorithm because equal elements maintain their relative order",
        expected_valid=True,
        expected_confidence=0.80,
        ground_truth="Merge sort preserves order during merge operation when elements are equal",
        metadata={"category": "algorithms", "type": "sorting", "property": "stability"}
    ))

    # ADVANCED: Design patterns and optimization
    problem_set.add_problem(BenchmarkProblem(
        id="CODE-009",
        domain=ProblemDomain.CODE,
        complexity=ComplexityLevel.ADVANCED,
        title="Dynamic Programming Optimization",
        description="Complexity optimization analysis",
        input_statement="Computing Fibonacci numbers with memoization reduces time complexity from O(2^n) to O(n) by caching intermediate results",
        expected_valid=True,
        expected_confidence=0.78,
        ground_truth="Memoization eliminates redundant calculations: each fib(k) computed once, total n values",
        metadata={"category": "optimization", "technique": "dynamic_programming"}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="CODE-010",
        domain=ProblemDomain.CODE,
        complexity=ComplexityLevel.ADVANCED,
        title="Concurrent Data Structure Safety",
        description="Thread safety analysis",
        input_statement="A thread-safe queue using locks ensures that enqueue and dequeue operations are atomic, preventing race conditions",
        expected_valid=True,
        expected_confidence=0.75,
        ground_truth="Locks provide mutual exclusion, ensuring only one thread modifies queue at a time",
        metadata={"category": "concurrency", "property": "thread_safety"}
    ))

    return problem_set


# ============================================================================
# LOGICAL DOMAIN PROBLEMS
# ============================================================================

def create_logic_problems() -> ProblemSet:
    """Create logical domain benchmark problems"""
    problem_set = ProblemSet(
        name="Logical Reasoning",
        domain=ProblemDomain.LOGIC,
        description="Logical problems covering deduction, induction, and formal reasoning"
    )

    # TRIVIAL: Basic logical operators
    problem_set.add_problem(BenchmarkProblem(
        id="LOGIC-001",
        domain=ProblemDomain.LOGIC,
        complexity=ComplexityLevel.TRIVIAL,
        title="AND Operation",
        description="Basic conjunction",
        input_statement="If P is true and Q is true, then (P AND Q) is true",
        expected_valid=True,
        expected_confidence=0.95,
        ground_truth="Definition of conjunction: (P ∧ Q) is true iff both P and Q are true",
        metadata={"category": "propositional", "operator": "AND"}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="LOGIC-002",
        domain=ProblemDomain.LOGIC,
        complexity=ComplexityLevel.TRIVIAL,
        title="NOT Operation",
        description="Basic negation",
        input_statement="If P is false, then NOT P is true",
        expected_valid=True,
        expected_confidence=0.95,
        ground_truth="Definition of negation: ¬P is true when P is false",
        metadata={"category": "propositional", "operator": "NOT"}
    ))

    # SIMPLE: Basic inference rules
    problem_set.add_problem(BenchmarkProblem(
        id="LOGIC-003",
        domain=ProblemDomain.LOGIC,
        complexity=ComplexityLevel.SIMPLE,
        title="Modus Ponens",
        description="Basic deductive inference",
        input_statement="Given: If P then Q. Given: P is true. Conclusion: Q is true.",
        expected_valid=True,
        expected_confidence=0.92,
        ground_truth="Modus ponens: (P → Q) ∧ P ⊢ Q is a valid inference rule",
        metadata={"category": "inference", "rule": "modus_ponens"}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="LOGIC-004",
        domain=ProblemDomain.LOGIC,
        complexity=ComplexityLevel.SIMPLE,
        title="Modus Tollens",
        description="Contrapositive reasoning",
        input_statement="Given: If P then Q. Given: Q is false. Conclusion: P is false.",
        expected_valid=True,
        expected_confidence=0.90,
        ground_truth="Modus tollens: (P → Q) ∧ ¬Q ⊢ ¬P is a valid inference rule",
        metadata={"category": "inference", "rule": "modus_tollens"}
    ))

    # MODERATE: Compound reasoning
    problem_set.add_problem(BenchmarkProblem(
        id="LOGIC-005",
        domain=ProblemDomain.LOGIC,
        complexity=ComplexityLevel.MODERATE,
        title="Disjunctive Syllogism",
        description="Elimination reasoning",
        input_statement="Given: P OR Q. Given: NOT P. Conclusion: Q must be true.",
        expected_valid=True,
        expected_confidence=0.88,
        ground_truth="Disjunctive syllogism: (P ∨ Q) ∧ ¬P ⊢ Q",
        metadata={"category": "inference", "rule": "disjunctive_syllogism"}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="LOGIC-006",
        domain=ProblemDomain.LOGIC,
        complexity=ComplexityLevel.MODERATE,
        title="Hypothetical Syllogism",
        description="Transitive implication",
        input_statement="Given: If P then Q. Given: If Q then R. Conclusion: If P then R.",
        expected_valid=True,
        expected_confidence=0.87,
        ground_truth="Hypothetical syllogism: (P → Q) ∧ (Q → R) ⊢ (P → R)",
        metadata={"category": "inference", "rule": "hypothetical_syllogism"}
    ))

    # COMPLEX: Multi-step reasoning
    problem_set.add_problem(BenchmarkProblem(
        id="LOGIC-007",
        domain=ProblemDomain.LOGIC,
        complexity=ComplexityLevel.COMPLEX,
        title="Proof by Contradiction",
        description="Indirect proof",
        input_statement="To prove P, assume NOT P. If this leads to a contradiction, then P must be true.",
        expected_valid=True,
        expected_confidence=0.85,
        ground_truth="Proof by contradiction (reductio ad absurdum): (¬P → ⊥) ⊢ P",
        metadata={"category": "proof", "technique": "contradiction"}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="LOGIC-008",
        domain=ProblemDomain.LOGIC,
        complexity=ComplexityLevel.COMPLEX,
        title="Universal Instantiation",
        description="Quantifier reasoning",
        input_statement="Given: For all x, P(x) is true. Given: a is a specific instance. Conclusion: P(a) is true.",
        expected_valid=True,
        expected_confidence=0.83,
        ground_truth="Universal instantiation: ∀x P(x) ⊢ P(a) for any a in domain",
        metadata={"category": "quantifiers", "rule": "universal_instantiation"}
    ))

    # ADVANCED: Complex logical reasoning
    problem_set.add_problem(BenchmarkProblem(
        id="LOGIC-009",
        domain=ProblemDomain.LOGIC,
        complexity=ComplexityLevel.ADVANCED,
        title="De Morgan's Law",
        description="Logical equivalence",
        input_statement="NOT (P AND Q) is logically equivalent to (NOT P) OR (NOT Q)",
        expected_valid=True,
        expected_confidence=0.82,
        ground_truth="De Morgan's law: ¬(P ∧ Q) ≡ (¬P ∨ ¬Q), proven by truth table",
        metadata={"category": "equivalence", "law": "de_morgan"}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="LOGIC-010",
        domain=ProblemDomain.LOGIC,
        complexity=ComplexityLevel.ADVANCED,
        title="Existential Generalization",
        description="Quantifier introduction",
        input_statement="Given: P(a) is true for specific a. Conclusion: There exists an x such that P(x) is true.",
        expected_valid=True,
        expected_confidence=0.80,
        ground_truth="Existential generalization: P(a) ⊢ ∃x P(x)",
        metadata={"category": "quantifiers", "rule": "existential_generalization"}
    ))

    return problem_set


# ============================================================================
# MIXED DOMAIN PROBLEMS
# ============================================================================

def create_mixed_problems() -> ProblemSet:
    """Create mixed domain benchmark problems"""
    problem_set = ProblemSet(
        name="Mixed Domain Reasoning",
        domain=ProblemDomain.MIXED,
        description="Problems combining multiple domains"
    )

    problem_set.add_problem(BenchmarkProblem(
        id="MIXED-001",
        domain=ProblemDomain.MIXED,
        complexity=ComplexityLevel.MODERATE,
        title="Mathematical Code Reasoning",
        description="Code implementing mathematical concept",
        input_statement="A function that computes n! (factorial) must multiply all integers from 1 to n, so factorial(5) = 5 * 4 * 3 * 2 * 1 = 120",
        expected_valid=True,
        expected_confidence=0.87,
        ground_truth="Combines mathematical definition of factorial with code execution trace",
        metadata={"domains": ["math", "code"]}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="MIXED-002",
        domain=ProblemDomain.MIXED,
        complexity=ComplexityLevel.COMPLEX,
        title="Logical Code Reasoning",
        description="Code implementing logical rules",
        input_statement="A boolean expression (A AND B) OR (A AND C) is equivalent to A AND (B OR C) by distributive law, so both evaluate to the same result for all inputs",
        expected_valid=True,
        expected_confidence=0.84,
        ground_truth="Combines logical equivalence with code evaluation",
        metadata={"domains": ["logic", "code"]}
    ))

    problem_set.add_problem(BenchmarkProblem(
        id="MIXED-003",
        domain=ProblemDomain.MIXED,
        complexity=ComplexityLevel.ADVANCED,
        title="Mathematical Proof in Code",
        description="Code correctness proof",
        input_statement="A binary search function is correct because: (1) Base case: empty array returns not found. (2) Recursive case: if mid element matches, return found; otherwise search appropriate half. This maintains the invariant that if target exists, it's in the current range.",
        expected_valid=True,
        expected_confidence=0.80,
        ground_truth="Combines mathematical proof by induction with code algorithm verification",
        metadata={"domains": ["math", "code", "logic"]}
    ))

    return problem_set


# ============================================================================
# PROBLEM SET AGGREGATION
# ============================================================================

def get_all_problem_sets() -> List[ProblemSet]:
    """
    Get all benchmark problem sets.

    Returns:
        List of all problem sets across all domains
    """
    return [
        create_math_problems(),
        create_code_problems(),
        create_logic_problems(),
        create_mixed_problems(),
    ]


def get_all_problems() -> List[BenchmarkProblem]:
    """
    Get all benchmark problems as a flat list.

    Returns:
        List of all problems across all domains
    """
    all_problems = []
    for problem_set in get_all_problem_sets():
        all_problems.extend(problem_set.problems)
    return all_problems


def get_problems_by_domain(domain: ProblemDomain) -> List[BenchmarkProblem]:
    """
    Get problems filtered by domain.

    Args:
        domain: Target domain

    Returns:
        List of problems in specified domain
    """
    return [p for p in get_all_problems() if p.domain == domain]


def get_problems_by_complexity(complexity: ComplexityLevel) -> List[BenchmarkProblem]:
    """
    Get problems filtered by complexity.

    Args:
        complexity: Target complexity level

    Returns:
        List of problems at specified complexity
    """
    return [p for p in get_all_problems() if p.complexity == complexity]


def get_benchmark_statistics() -> Dict[str, Any]:
    """
    Get overall benchmark statistics.

    Returns:
        Dictionary with benchmark statistics
    """
    all_problems = get_all_problems()

    return {
        "total_problems": len(all_problems),
        "by_domain": {
            domain.value: len(get_problems_by_domain(domain))
            for domain in ProblemDomain
        },
        "by_complexity": {
            level.value: len(get_problems_by_complexity(level))
            for level in ComplexityLevel
        },
        "problem_sets": len(get_all_problem_sets()),
    }
