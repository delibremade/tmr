"""
Pattern Storage and Retrieval System

Manages persistent storage, indexing, and retrieval of patterns.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import logging

from .patterns import Pattern, DomainType, PatternComplexity, PatternMatch

logger = logging.getLogger(__name__)


class PatternLibrary:
    """
    Central repository for storing and retrieving patterns.

    Provides indexing, search, and retrieval capabilities.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize pattern library.

        Args:
            storage_path: Optional path for persistent storage
        """
        self.storage_path = storage_path
        self.patterns: Dict[str, Pattern] = {}

        # Indexes for efficient retrieval
        self.domain_index: Dict[DomainType, Set[str]] = defaultdict(set)
        self.complexity_index: Dict[PatternComplexity, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.context_index: Dict[str, Set[str]] = defaultdict(set)

        # Statistics
        self.stats = {
            "total_patterns": 0,
            "patterns_by_domain": defaultdict(int),
            "patterns_by_complexity": defaultdict(int),
            "total_retrievals": 0,
            "total_matches": 0
        }

        # Load patterns if storage path exists
        if self.storage_path and self.storage_path.exists():
            self.load()

    def add_pattern(self, pattern: Pattern) -> bool:
        """
        Add a pattern to the library.

        Args:
            pattern: Pattern to add

        Returns:
            True if added successfully, False if already exists
        """
        if pattern.pattern_id in self.patterns:
            logger.warning(f"Pattern {pattern.pattern_id} already exists")
            return False

        # Store pattern
        self.patterns[pattern.pattern_id] = pattern

        # Update indexes
        self._update_indexes(pattern)

        # Update statistics
        self.stats["total_patterns"] += 1
        self.stats["patterns_by_domain"][pattern.domain.value] += 1
        self.stats["patterns_by_complexity"][pattern.complexity.value] += 1

        logger.info(f"Added pattern: {pattern.pattern_id} ({pattern.name})")
        return True

    def add_patterns(self, patterns: List[Pattern]) -> int:
        """
        Add multiple patterns to the library.

        Args:
            patterns: List of patterns to add

        Returns:
            Number of patterns successfully added
        """
        added = 0
        for pattern in patterns:
            if self.add_pattern(pattern):
                added += 1
        return added

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """
        Retrieve a pattern by ID.

        Args:
            pattern_id: Pattern ID

        Returns:
            Pattern if found, None otherwise
        """
        return self.patterns.get(pattern_id)

    def remove_pattern(self, pattern_id: str) -> bool:
        """
        Remove a pattern from the library.

        Args:
            pattern_id: Pattern ID to remove

        Returns:
            True if removed, False if not found
        """
        if pattern_id not in self.patterns:
            return False

        pattern = self.patterns[pattern_id]

        # Remove from indexes
        self._remove_from_indexes(pattern)

        # Remove pattern
        del self.patterns[pattern_id]

        # Update statistics
        self.stats["total_patterns"] -= 1
        self.stats["patterns_by_domain"][pattern.domain.value] -= 1
        self.stats["patterns_by_complexity"][pattern.complexity.value] -= 1

        logger.info(f"Removed pattern: {pattern_id}")
        return True

    def search(self, query: Dict[str, Any],
              domain: Optional[DomainType] = None,
              min_similarity: float = 0.5,
              max_results: int = 10) -> List[PatternMatch]:
        """
        Search for patterns matching a query.

        Args:
            query: Query structure
            domain: Optional domain filter
            min_similarity: Minimum similarity threshold
            max_results: Maximum number of results

        Returns:
            List of pattern matches, sorted by similarity
        """
        self.stats["total_retrievals"] += 1

        # Get candidate patterns
        candidates = self._get_candidates(query, domain)

        # Score each candidate
        matches = []
        for pattern in candidates:
            is_match, similarity = pattern.matches(query, min_similarity)
            if is_match:
                confidence = self._calculate_match_confidence(pattern, similarity)
                match = PatternMatch(
                    pattern=pattern,
                    similarity_score=similarity,
                    confidence=confidence,
                    matched_features=self._identify_matched_features(pattern, query)
                )
                matches.append(match)

        # Sort by similarity (descending)
        matches.sort(reverse=True)

        self.stats["total_matches"] += len(matches)

        # Return top results
        return matches[:max_results]

    def find_by_domain(self, domain: DomainType) -> List[Pattern]:
        """
        Find all patterns for a specific domain.

        Args:
            domain: Domain type

        Returns:
            List of patterns in the domain
        """
        pattern_ids = self.domain_index.get(domain, set())
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]

    def find_by_complexity(self, complexity: PatternComplexity) -> List[Pattern]:
        """
        Find all patterns of a specific complexity.

        Args:
            complexity: Complexity level

        Returns:
            List of patterns with that complexity
        """
        pattern_ids = self.complexity_index.get(complexity, set())
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]

    def find_by_tag(self, tag: str) -> List[Pattern]:
        """
        Find patterns by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of patterns with the tag
        """
        pattern_ids = self.tag_index.get(tag, set())
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]

    def find_by_context(self, context: str) -> List[Pattern]:
        """
        Find patterns applicable to a context.

        Args:
            context: Context string

        Returns:
            List of applicable patterns
        """
        pattern_ids = self.context_index.get(context, set())
        return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]

    def _get_candidates(self, query: Dict[str, Any],
                       domain: Optional[DomainType] = None) -> List[Pattern]:
        """Get candidate patterns for matching."""
        if domain:
            # Filter by domain
            pattern_ids = self.domain_index.get(domain, set())
            return [self.patterns[pid] for pid in pattern_ids if pid in self.patterns]
        else:
            # Consider all patterns
            return list(self.patterns.values())

    def _calculate_match_confidence(self, pattern: Pattern, similarity: float) -> float:
        """Calculate confidence in a pattern match."""
        # Base confidence from similarity
        base_confidence = similarity

        # Adjust based on pattern's historical performance
        success_rate = pattern.metadata.success_rate
        usage_count = pattern.metadata.usage_count

        # Boost confidence for patterns with good track record
        if usage_count > 0:
            performance_boost = success_rate * 0.2
            # Reduce boost for patterns with little usage data
            if usage_count < 5:
                performance_boost *= (usage_count / 5)
        else:
            performance_boost = 0.0

        confidence = min(1.0, base_confidence + performance_boost)
        return confidence

    def _identify_matched_features(self, pattern: Pattern,
                                   query: Dict[str, Any]) -> List[str]:
        """Identify which features of the pattern matched the query."""
        matched = []

        # Check domain match
        if query.get("domain") == pattern.domain.value:
            matched.append("domain")

        # Check structural matches
        if "structure" in query:
            query_keys = set(query["structure"].keys())
            pattern_keys = set(pattern.structure.keys())
            if query_keys & pattern_keys:
                matched.append("structure")

        # Check context match
        if "context" in query and pattern.applicable_contexts:
            query_context = query["context"]
            if any(ctx in query_context for ctx in pattern.applicable_contexts):
                matched.append("context")

        return matched

    def _update_indexes(self, pattern: Pattern):
        """Update indexes when adding a pattern."""
        # Domain index
        self.domain_index[pattern.domain].add(pattern.pattern_id)

        # Complexity index
        self.complexity_index[pattern.complexity].add(pattern.pattern_id)

        # Tag index
        for tag in pattern.metadata.tags:
            self.tag_index[tag].add(pattern.pattern_id)

        # Context index
        for context in pattern.applicable_contexts:
            self.context_index[context].add(pattern.pattern_id)

    def _remove_from_indexes(self, pattern: Pattern):
        """Remove pattern from indexes."""
        # Domain index
        self.domain_index[pattern.domain].discard(pattern.pattern_id)

        # Complexity index
        self.complexity_index[pattern.complexity].discard(pattern.pattern_id)

        # Tag index
        for tag in pattern.metadata.tags:
            self.tag_index[tag].discard(pattern.pattern_id)

        # Context index
        for context in pattern.applicable_contexts:
            self.context_index[context].discard(pattern.pattern_id)

    def update_pattern_usage(self, pattern_id: str, success: bool, confidence: float):
        """
        Update pattern usage statistics.

        Args:
            pattern_id: Pattern ID
            success: Whether the pattern application was successful
            confidence: Confidence in the application
        """
        if pattern_id in self.patterns:
            self.patterns[pattern_id].metadata.update_usage(success, confidence)

    def save(self, filepath: Optional[Path] = None):
        """
        Save library to disk.

        Args:
            filepath: Optional custom filepath
        """
        save_path = filepath or self.storage_path
        if not save_path:
            logger.warning("No storage path specified, cannot save")
            return

        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize patterns
        data = {
            "patterns": [pattern.to_dict() for pattern in self.patterns.values()],
            "stats": {k: dict(v) if isinstance(v, defaultdict) else v
                     for k, v in self.stats.items()},
            "saved_at": datetime.now().isoformat()
        }

        # Write to file
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved {len(self.patterns)} patterns to {save_path}")

    def load(self, filepath: Optional[Path] = None):
        """
        Load library from disk.

        Args:
            filepath: Optional custom filepath
        """
        load_path = filepath or self.storage_path
        if not load_path or not load_path.exists():
            logger.warning("No file to load from")
            return

        # Read from file
        with open(load_path, 'r') as f:
            data = json.load(f)

        # Clear existing data
        self.patterns = {}
        self.domain_index = defaultdict(set)
        self.complexity_index = defaultdict(set)
        self.tag_index = defaultdict(set)
        self.context_index = defaultdict(set)

        # Load patterns
        for pattern_dict in data.get("patterns", []):
            pattern = Pattern.from_dict(pattern_dict)
            self.patterns[pattern.pattern_id] = pattern
            self._update_indexes(pattern)

        # Restore stats
        if "stats" in data:
            self.stats = data["stats"]
            # Convert back to defaultdict
            for key in ["patterns_by_domain", "patterns_by_complexity"]:
                if key in self.stats:
                    self.stats[key] = defaultdict(int, self.stats[key])

        logger.info(f"Loaded {len(self.patterns)} patterns from {load_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        stats = self.stats.copy()

        # Add additional metrics
        stats["domains"] = {
            domain.value: len(pattern_ids)
            for domain, pattern_ids in self.domain_index.items()
        }
        stats["complexities"] = {
            complexity.value: len(pattern_ids)
            for complexity, pattern_ids in self.complexity_index.items()
        }
        stats["total_tags"] = len(self.tag_index)
        stats["total_contexts"] = len(self.context_index)

        # Calculate retrieval stats
        if stats["total_retrievals"] > 0:
            stats["avg_matches_per_retrieval"] = \
                stats["total_matches"] / stats["total_retrievals"]
        else:
            stats["avg_matches_per_retrieval"] = 0.0

        return stats

    def export_patterns(self, filepath: Path, domain: Optional[DomainType] = None):
        """
        Export patterns to a JSON file.

        Args:
            filepath: Export file path
            domain: Optional domain filter
        """
        if domain:
            patterns = self.find_by_domain(domain)
        else:
            patterns = list(self.patterns.values())

        export_data = {
            "patterns": [p.to_dict() for p in patterns],
            "count": len(patterns),
            "exported_at": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported {len(patterns)} patterns to {filepath}")

    def import_patterns(self, filepath: Path) -> int:
        """
        Import patterns from a JSON file.

        Args:
            filepath: Import file path

        Returns:
            Number of patterns imported
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        patterns = [Pattern.from_dict(p) for p in data.get("patterns", [])]
        return self.add_patterns(patterns)

    def clear(self):
        """Clear all patterns from the library."""
        self.patterns = {}
        self.domain_index = defaultdict(set)
        self.complexity_index = defaultdict(set)
        self.tag_index = defaultdict(set)
        self.context_index = defaultdict(set)
        self.stats = {
            "total_patterns": 0,
            "patterns_by_domain": defaultdict(int),
            "patterns_by_complexity": defaultdict(int),
            "total_retrievals": 0,
            "total_matches": 0
        }
        logger.info("Cleared all patterns from library")

    def __len__(self) -> int:
        """Return number of patterns in library."""
        return len(self.patterns)

    def __repr__(self) -> str:
        """String representation of the library."""
        return (f"PatternLibrary(patterns={len(self.patterns)}, "
                f"domains={len(self.domain_index)}, "
                f"retrievals={self.stats['total_retrievals']})")
