"""
Nuance Layer Implementation

Layer 2: Nuance - Domain-Specific Pattern Recognition

This layer extracts, stores, and retrieves domain-specific patterns
for mathematical, code, and logical reasoning.
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime

from .patterns import Pattern, DomainType, PatternMatch, PatternComplexity
from .domains import DomainClassifier
from .extractors import PatternExtractorFactory
from .storage import PatternLibrary

logger = logging.getLogger(__name__)


class NuanceLayer:
    """
    Layer 2: Nuance - Domain-Specific Pattern Recognition

    Manages pattern extraction, storage, and retrieval across domains.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Nuance Layer.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

        # Initialize domain classifier
        self.classifier = DomainClassifier()

        # Initialize pattern library
        storage_path = self.config.get("storage_path")
        if storage_path:
            storage_path = Path(storage_path)
        self.library = PatternLibrary(storage_path)

        # Initialize pattern extractors
        self.extractors = PatternExtractorFactory.get_all_extractors()

        # Statistics
        self.stats = {
            "total_extractions": 0,
            "total_retrievals": 0,
            "total_classifications": 0,
            "extractions_by_domain": {},
            "retrievals_by_domain": {},
            "extraction_times": [],
            "retrieval_times": []
        }

        # Initialize with core patterns if configured
        if self.config.get("load_core_patterns", True):
            self._load_core_patterns()

        logger.info("NuanceLayer initialized")

    def extract_patterns(self, content: Any,
                        domain: Optional[DomainType] = None,
                        context: Optional[Dict] = None) -> List[Pattern]:
        """
        Extract patterns from content.

        Args:
            content: Content to extract patterns from
            domain: Optional domain (will be auto-detected if not provided)
            context: Optional context information

        Returns:
            List of extracted patterns
        """
        start_time = datetime.now()
        self.stats["total_extractions"] += 1

        # Classify domain if not provided
        if not domain:
            domain = self.classifier.classify(content)
            logger.info(f"Auto-classified content as: {domain.value}")

        self.stats["total_classifications"] += 1

        # Update statistics
        domain_key = domain.value
        self.stats["extractions_by_domain"][domain_key] = \
            self.stats["extractions_by_domain"].get(domain_key, 0) + 1

        # Extract patterns using appropriate extractor
        if domain == DomainType.UNKNOWN:
            logger.warning("Cannot extract patterns from unknown domain")
            return []

        try:
            extractor = self.extractors[domain]
            patterns = extractor.extract(content, context)

            # Add patterns to library
            added = self.library.add_patterns(patterns)

            # Record extraction time
            extraction_time = (datetime.now() - start_time).total_seconds() * 1000
            self.stats["extraction_times"].append(extraction_time)

            logger.info(f"Extracted {len(patterns)} patterns ({added} new) "
                       f"from {domain.value} domain in {extraction_time:.2f}ms")

            return patterns

        except Exception as e:
            logger.error(f"Error extracting patterns: {e}", exc_info=True)
            return []

    def retrieve_patterns(self, query: Dict[str, Any],
                         domain: Optional[DomainType] = None,
                         min_similarity: float = 0.5,
                         max_results: int = 10) -> List[PatternMatch]:
        """
        Retrieve patterns matching a query.

        Args:
            query: Query structure
            domain: Optional domain filter
            min_similarity: Minimum similarity threshold
            max_results: Maximum number of results

        Returns:
            List of pattern matches
        """
        start_time = datetime.now()
        self.stats["total_retrievals"] += 1

        # Auto-detect domain from query if not provided
        if not domain and "content" in query:
            domain = self.classifier.classify(query["content"])
            query["domain"] = domain.value

        # Update statistics
        if domain:
            domain_key = domain.value
            self.stats["retrievals_by_domain"][domain_key] = \
                self.stats["retrievals_by_domain"].get(domain_key, 0) + 1

        # Search library
        matches = self.library.search(
            query=query,
            domain=domain,
            min_similarity=min_similarity,
            max_results=max_results
        )

        # Record retrieval time
        retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
        self.stats["retrieval_times"].append(retrieval_time)

        logger.info(f"Retrieved {len(matches)} pattern matches "
                   f"in {retrieval_time:.2f}ms")

        return matches

    def classify_content(self, content: Any) -> Tuple[DomainType, float, Dict]:
        """
        Classify content into a domain.

        Args:
            content: Content to classify

        Returns:
            Tuple of (domain, confidence, scores)
        """
        self.stats["total_classifications"] += 1
        domain, confidence, scores = self.classifier.classify_with_confidence(content)

        logger.info(f"Classified content as {domain.value} "
                   f"with confidence {confidence:.2f}")

        return domain, confidence, scores

    def apply_pattern(self, pattern_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a pattern to a context.

        Args:
            pattern_id: Pattern ID to apply
            context: Context to apply the pattern to

        Returns:
            Result of pattern application
        """
        pattern = self.library.get_pattern(pattern_id)
        if not pattern:
            return {
                "success": False,
                "error": f"Pattern {pattern_id} not found"
            }

        result = pattern.apply(context)

        # Update pattern usage statistics
        if result.get("applied"):
            self.library.update_pattern_usage(
                pattern_id,
                success=True,
                confidence=result.get("confidence", 0.0)
            )

        return result

    def add_pattern(self, pattern: Pattern) -> bool:
        """
        Manually add a pattern to the library.

        Args:
            pattern: Pattern to add

        Returns:
            True if added successfully
        """
        return self.library.add_pattern(pattern)

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """
        Get a pattern by ID.

        Args:
            pattern_id: Pattern ID

        Returns:
            Pattern if found, None otherwise
        """
        return self.library.get_pattern(pattern_id)

    def get_patterns_by_domain(self, domain: DomainType) -> List[Pattern]:
        """
        Get all patterns for a domain.

        Args:
            domain: Domain type

        Returns:
            List of patterns
        """
        return self.library.find_by_domain(domain)

    def get_patterns_by_complexity(self, complexity: PatternComplexity) -> List[Pattern]:
        """
        Get patterns by complexity level.

        Args:
            complexity: Complexity level

        Returns:
            List of patterns
        """
        return self.library.find_by_complexity(complexity)

    def suggest_patterns(self, content: Any,
                        max_suggestions: int = 5) -> List[Tuple[Pattern, float]]:
        """
        Suggest patterns for given content.

        Args:
            content: Content to analyze
            max_suggestions: Maximum number of suggestions

        Returns:
            List of (pattern, confidence) tuples
        """
        # Classify content
        domain, confidence, _ = self.classify_content(content)

        if domain == DomainType.UNKNOWN:
            return []

        # Create query from content
        query = {
            "content": content,
            "domain": domain.value,
            "structure": self._analyze_structure(content, domain)
        }

        # Retrieve matches
        matches = self.retrieve_patterns(
            query=query,
            domain=domain,
            max_results=max_suggestions
        )

        # Return patterns with confidence
        suggestions = [(match.pattern, match.confidence) for match in matches]

        return suggestions

    def _analyze_structure(self, content: Any, domain: DomainType) -> Dict[str, Any]:
        """Analyze content structure using domain extractor."""
        if domain in self.extractors:
            return self.extractors[domain].identify_structure(content)
        return {}

    def _load_core_patterns(self):
        """Load core patterns for each domain."""
        logger.info("Loading core patterns...")

        # Math core patterns
        math_patterns = [
            Pattern(
                pattern_id="",
                domain=DomainType.MATH,
                name="linear_equation",
                description="Solve linear equations of form ax + b = c",
                structure={
                    "form": "ax + b = c",
                    "operations": ["isolate", "solve"]
                },
                complexity=PatternComplexity.SIMPLE,
                applicable_contexts=["algebra", "equations"],
                transformation_steps=[
                    {"action": "subtract_b", "description": "Subtract b from both sides"},
                    {"action": "divide_a", "description": "Divide both sides by a"}
                ]
            ),
            Pattern(
                pattern_id="",
                domain=DomainType.MATH,
                name="quadratic_formula",
                description="Solve quadratic equations using the quadratic formula",
                structure={
                    "form": "ax^2 + bx + c = 0",
                    "formula": "x = (-b ± √(b^2 - 4ac)) / 2a"
                },
                complexity=PatternComplexity.INTERMEDIATE,
                applicable_contexts=["algebra", "quadratic", "equations"]
            )
        ]

        # Code core patterns
        code_patterns = [
            Pattern(
                pattern_id="",
                domain=DomainType.CODE,
                name="iteration_pattern",
                description="Iterate over a collection",
                structure={
                    "construct": "for_loop",
                    "elements": ["collection", "iterator", "body"]
                },
                complexity=PatternComplexity.SIMPLE,
                applicable_contexts=["iteration", "loops", "collections"]
            ),
            Pattern(
                pattern_id="",
                domain=DomainType.CODE,
                name="recursive_pattern",
                description="Recursive function pattern",
                structure={
                    "construct": "recursion",
                    "elements": ["base_case", "recursive_case"]
                },
                complexity=PatternComplexity.INTERMEDIATE,
                applicable_contexts=["recursion", "algorithms"]
            )
        ]

        # Logic core patterns
        logic_patterns = [
            Pattern(
                pattern_id="",
                domain=DomainType.LOGIC,
                name="modus_ponens",
                description="If P then Q; P; Therefore Q",
                structure={
                    "type": "inference_rule",
                    "premises": ["P → Q", "P"],
                    "conclusion": "Q"
                },
                complexity=PatternComplexity.SIMPLE,
                applicable_contexts=["deduction", "inference"]
            ),
            Pattern(
                pattern_id="",
                domain=DomainType.LOGIC,
                name="modus_tollens",
                description="If P then Q; Not Q; Therefore not P",
                structure={
                    "type": "inference_rule",
                    "premises": ["P → Q", "¬Q"],
                    "conclusion": "¬P"
                },
                complexity=PatternComplexity.SIMPLE,
                applicable_contexts=["deduction", "inference"]
            )
        ]

        # Add all core patterns
        all_core_patterns = math_patterns + code_patterns + logic_patterns
        added = self.library.add_patterns(all_core_patterns)

        logger.info(f"Loaded {added} core patterns")

    def save_library(self, filepath: Optional[Path] = None):
        """
        Save pattern library to disk.

        Args:
            filepath: Optional custom filepath
        """
        self.library.save(filepath)

    def load_library(self, filepath: Optional[Path] = None):
        """
        Load pattern library from disk.

        Args:
            filepath: Optional custom filepath
        """
        self.library.load(filepath)

    def export_patterns(self, filepath: Path, domain: Optional[DomainType] = None):
        """
        Export patterns to file.

        Args:
            filepath: Export file path
            domain: Optional domain filter
        """
        self.library.export_patterns(filepath, domain)

    def import_patterns(self, filepath: Path) -> int:
        """
        Import patterns from file.

        Args:
            filepath: Import file path

        Returns:
            Number of patterns imported
        """
        return self.library.import_patterns(filepath)

    def get_statistics(self) -> Dict[str, Any]:
        """Get layer statistics."""
        stats = self.stats.copy()

        # Add library statistics
        stats["library"] = self.library.get_statistics()

        # Calculate averages
        if self.stats["extraction_times"]:
            import statistics
            stats["avg_extraction_time_ms"] = statistics.mean(self.stats["extraction_times"])
        if self.stats["retrieval_times"]:
            import statistics
            stats["avg_retrieval_time_ms"] = statistics.mean(self.stats["retrieval_times"])

        # Add domain information
        stats["supported_domains"] = [d.value for d in self.extractors.keys()]

        return stats

    def reset_statistics(self):
        """Reset layer statistics."""
        self.stats = {
            "total_extractions": 0,
            "total_retrievals": 0,
            "total_classifications": 0,
            "extractions_by_domain": {},
            "retrievals_by_domain": {},
            "extraction_times": [],
            "retrieval_times": []
        }
        logger.info("Statistics reset")

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the layer."""
        health = {
            "status": "healthy",
            "issues": [],
            "metrics": {}
        }

        # Check library
        health["metrics"]["library_size"] = len(self.library)
        if len(self.library) == 0:
            health["issues"].append("Pattern library is empty")
            if health["status"] == "healthy":
                health["status"] = "warning"

        # Check extractors
        health["metrics"]["extractors"] = len(self.extractors)
        if len(self.extractors) < 3:
            health["issues"].append("Not all domain extractors initialized")
            health["status"] = "degraded"

        # Check classifier
        try:
            test_content = "test content"
            self.classifier.classify(test_content)
            health["metrics"]["classifier"] = "operational"
        except Exception as e:
            health["issues"].append(f"Classifier error: {e}")
            health["status"] = "degraded"

        # Add usage metrics
        health["metrics"].update({
            "total_extractions": self.stats["total_extractions"],
            "total_retrievals": self.stats["total_retrievals"],
            "total_classifications": self.stats["total_classifications"]
        })

        return health

    def __repr__(self) -> str:
        """String representation of the layer."""
        return (f"NuanceLayer(patterns={len(self.library)}, "
                f"domains={len(self.extractors)}, "
                f"extractions={self.stats['total_extractions']}, "
                f"retrievals={self.stats['total_retrievals']})")
