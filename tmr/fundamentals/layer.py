"""
Fundamentals Layer Implementation

The main class that orchestrates all fundamental principles and validators.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import json
from datetime import datetime
from pathlib import Path

from .principles import LogicalPrinciples, PrincipleType, ValidationResult
from .validators import (
    LogicalValidator,
    MathematicalValidator,
    CausalValidator,
    ConsistencyValidator,
    ReasoningChain,
    ReasoningStep
)

logger = logging.getLogger(__name__)


class FundamentalsLayer:
    """
    Layer 1: Fundamentals - Immutable Logical Principles
    
    This layer implements the core logical principles that remain constant
    across all domains and reasoning types.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Fundamentals Layer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Initialize principles
        self.principles = LogicalPrinciples()
        
        # Initialize validators
        self.logical_validator = LogicalValidator(self.principles)
        self.mathematical_validator = MathematicalValidator(self.principles)
        self.causal_validator = CausalValidator(self.principles)
        self.consistency_validator = ConsistencyValidator(self.principles)
        
        # Validation cache for performance
        self.validation_cache = {}
        self.cache_size = self.config.get("cache_size", 1000)
        
        # Statistics tracking
        self.stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "validation_times": [],
            "domain_counts": {}
        }
        
        # Initialize logging
        self.setup_logging()
        
        logger.info("FundamentalsLayer initialized")
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def validate(self, 
                 statement: Union[str, Dict, List],
                 domain: Optional[str] = None,
                 validation_type: Optional[str] = None,
                 use_cache: bool = True) -> Dict[str, Any]:
        """
        Main validation method for the Fundamentals Layer.
        
        Args:
            statement: The statement or reasoning to validate
            domain: Optional domain specification (e.g., "mathematical", "logical")
            validation_type: Specific validation type to use
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary containing validation results
        """
        start_time = datetime.now()
        self.stats["total_validations"] += 1
        
        # Check cache
        cache_key = self._generate_cache_key(statement, domain, validation_type)
        if use_cache and cache_key in self.validation_cache:
            self.stats["cache_hits"] += 1
            logger.debug(f"Cache hit for key: {cache_key[:50]}...")
            return self.validation_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        # Determine validation type
        if validation_type:
            val_type = validation_type
        else:
            val_type = self._infer_validation_type(statement, domain)
        
        # Update domain statistics
        self.stats["domain_counts"][val_type] = \
            self.stats["domain_counts"].get(val_type, 0) + 1
        
        # Perform validation
        try:
            result = self._perform_validation(statement, val_type)
            
            # Add metadata
            result["metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "validation_type": val_type,
                "domain": domain,
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000
            }
            
            # Update statistics
            if result["valid"]:
                self.stats["successful_validations"] += 1
            else:
                self.stats["failed_validations"] += 1
            
            self.stats["validation_times"].append(result["metadata"]["processing_time_ms"])
            
            # Cache result
            if use_cache:
                self._cache_result(cache_key, result)
            
            logger.info(f"Validation completed: valid={result['valid']}, "
                       f"confidence={result['confidence']:.2f}, type={val_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            self.stats["failed_validations"] += 1
            
            return {
                "valid": False,
                "confidence": 0.0,
                "error": str(e),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "validation_type": val_type,
                    "domain": domain,
                    "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                    "error": True
                }
            }
    
    def _generate_cache_key(self, statement: Any, domain: Optional[str], 
                           validation_type: Optional[str]) -> str:
        """Generate a cache key for the validation request."""
        # Convert statement to string representation
        if isinstance(statement, dict):
            stmt_str = json.dumps(statement, sort_keys=True)
        elif isinstance(statement, list):
            stmt_str = json.dumps(statement, sort_keys=True)
        else:
            stmt_str = str(statement)
        
        # Combine with domain and type
        key_parts = [stmt_str, domain or "none", validation_type or "auto"]
        return "|".join(key_parts)
    
    def _infer_validation_type(self, statement: Any, domain: Optional[str]) -> str:
        """Infer the validation type from the statement and domain."""
        if domain:
            if domain.lower() in ["math", "mathematical", "numeric"]:
                return "mathematical"
            elif domain.lower() in ["logic", "logical", "boolean"]:
                return "logical"
            elif domain.lower() in ["causal", "temporal", "sequence"]:
                return "causal"
            elif domain.lower() in ["mixed", "composite", "multi"]:
                return "consistency"
        
        # Infer from statement structure
        if isinstance(statement, dict):
            if "equation" in statement or "calculation" in statement:
                return "mathematical"
            elif "events" in statement or "cause" in statement:
                return "causal"
            elif "steps" in statement or "chain" in statement:
                return "logical"
            elif any(k in statement for k in ["logical", "mathematical", "causal"]):
                return "consistency"
        
        # Default to logical validation
        return "logical"
    
    def _perform_validation(self, statement: Any, validation_type: str) -> Dict[str, Any]:
        """Perform the actual validation based on type."""
        
        if validation_type == "mathematical":
            # Convert statement to expected format if needed
            math_statement = self._prepare_mathematical_statement(statement)
            valid, confidence, details = self.mathematical_validator.validate(math_statement)
            
        elif validation_type == "causal":
            causal_chain = self._prepare_causal_chain(statement)
            valid, confidence, details = self.causal_validator.validate(causal_chain)
            
        elif validation_type == "logical":
            reasoning_chain = self._prepare_reasoning_chain(statement)
            valid, confidence, details = self.logical_validator.validate(reasoning_chain)
            
        elif validation_type == "consistency":
            composite = self._prepare_composite_reasoning(statement)
            valid, confidence, details = self.consistency_validator.validate(composite)
            
        else:
            # Fallback to principle validation
            results = self.principles.validate_all(statement)
            valid, confidence = self.principles.get_aggregate_validity(results)
            details = {"principle_results": results}
        
        return {
            "valid": valid,
            "confidence": confidence,
            "details": details
        }
    
    def _prepare_mathematical_statement(self, statement: Any) -> Dict:
        """Prepare statement for mathematical validation."""
        if isinstance(statement, dict):
            return statement
        
        # Try to parse as mathematical expression
        return {
            "equation": str(statement),
            "steps": [],
            "result": None
        }
    
    def _prepare_causal_chain(self, statement: Any) -> Dict:
        """Prepare statement for causal validation."""
        if isinstance(statement, dict):
            return statement
        
        # Convert to causal chain format
        return {
            "events": [],
            "relationships": []
        }
    
    def _prepare_reasoning_chain(self, statement: Any) -> ReasoningChain:
        """Prepare statement for logical validation."""
        if isinstance(statement, ReasoningChain):
            return statement
        
        if isinstance(statement, dict):
            # Try to extract steps
            steps = statement.get("steps", [])
            reasoning_steps = []
            for i, step in enumerate(steps):
                if isinstance(step, ReasoningStep):
                    reasoning_steps.append(step)
                else:
                    reasoning_steps.append(ReasoningStep(
                        step_id=i,
                        statement=str(step),
                        justification=None,
                        dependencies=None
                    ))
            
            return ReasoningChain(
                steps=reasoning_steps,
                conclusion=statement.get("conclusion"),
                domain=statement.get("domain")
            )
        
        # Convert simple statement to single-step chain
        return ReasoningChain(
            steps=[ReasoningStep(
                step_id=0,
                statement=str(statement),
                justification=None,
                dependencies=None
            )],
            conclusion=None,
            domain=None
        )
    
    def _prepare_composite_reasoning(self, statement: Any) -> Dict:
        """Prepare statement for consistency validation."""
        if isinstance(statement, dict):
            # Check if already in composite format
            if any(k in statement for k in ["logical", "mathematical", "causal"]):
                return statement
        
        # Default composite structure
        return {
            "logical": self._prepare_reasoning_chain(statement),
            "mathematical": self._prepare_mathematical_statement(statement),
            "causal": self._prepare_causal_chain(statement)
        }
    
    def _cache_result(self, key: str, result: Dict):
        """Cache validation result."""
        # Implement LRU cache behavior
        if len(self.validation_cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self.validation_cache))
            del self.validation_cache[oldest_key]
        
        self.validation_cache[key] = result
    
    def validate_principle(self, statement: Any, principle_type: PrincipleType) -> ValidationResult:
        """
        Validate a statement against a specific principle.
        
        Args:
            statement: The statement to validate
            principle_type: The specific principle to apply
            
        Returns:
            ValidationResult from the principle
        """
        if principle_type in self.principles.principles:
            return self.principles.principles[principle_type].validate(statement)
        else:
            raise ValueError(f"Unknown principle type: {principle_type}")
    
    def validate_multiple_principles(self, 
                                   statement: Any, 
                                   principle_types: List[PrincipleType]) -> Dict[PrincipleType, ValidationResult]:
        """
        Validate a statement against multiple specific principles.
        
        Args:
            statement: The statement to validate
            principle_types: List of principles to apply
            
        Returns:
            Dictionary mapping principle types to results
        """
        return self.principles.validate_specific(statement, principle_types)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = self.stats.copy()
        
        # Add principle statistics
        stats["principle_stats"] = self.principles.get_statistics()
        
        # Calculate success rate
        if stats["total_validations"] > 0:
            stats["success_rate"] = stats["successful_validations"] / stats["total_validations"]
            stats["cache_hit_rate"] = stats["cache_hits"] / stats["total_validations"]
        else:
            stats["success_rate"] = 0.0
            stats["cache_hit_rate"] = 0.0
        
        # Calculate average validation time
        if stats["validation_times"]:
            import numpy as np
            stats["avg_validation_time_ms"] = np.mean(stats["validation_times"])
            stats["median_validation_time_ms"] = np.median(stats["validation_times"])
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics."""
        self.stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "validation_times": [],
            "domain_counts": {}
        }
        self.principles.reset_all_stats()
        logger.info("Statistics reset")
    
    def clear_cache(self):
        """Clear the validation cache."""
        self.validation_cache = {}
        logger.info("Cache cleared")
    
    def export_stats(self, filepath: Optional[str] = None) -> str:
        """
        Export statistics to JSON file.
        
        Args:
            filepath: Optional path to save stats
            
        Returns:
            JSON string of statistics
        """
        stats = self.get_statistics()
        stats_json = json.dumps(stats, indent=2, default=str)
        
        if filepath:
            Path(filepath).write_text(stats_json)
            logger.info(f"Statistics exported to {filepath}")
        
        return stats_json
    
    def __repr__(self) -> str:
        """String representation of the layer."""
        return (f"FundamentalsLayer(validations={self.stats['total_validations']}, "
                f"success_rate={self.stats.get('success_rate', 0):.2%}, "
                f"cache_size={len(self.validation_cache)})")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the layer."""
        health = {
            "status": "healthy",
            "issues": [],
            "metrics": {}
        }
        
        # Check validators
        validators = [
            self.logical_validator,
            self.mathematical_validator,
            self.causal_validator,
            self.consistency_validator
        ]
        
        for validator in validators:
            try:
                # Simple validation to test each validator
                test_input = {"test": "health_check"}
                validator.validate(test_input)
            except Exception as e:
                health["issues"].append(f"Validator error: {e}")
                health["status"] = "degraded"
        
        # Check cache
        health["metrics"]["cache_size"] = len(self.validation_cache)
        health["metrics"]["cache_limit"] = self.cache_size
        
        if len(self.validation_cache) >= self.cache_size * 0.9:
            health["issues"].append("Cache nearly full")
            if health["status"] == "healthy":
                health["status"] = "warning"
        
        # Add basic metrics
        health["metrics"].update({
            "total_validations": self.stats["total_validations"],
            "success_rate": self.stats.get("success_rate", 0.0),
            "cache_hit_rate": self.stats.get("cache_hit_rate", 0.0)
        })
        
        return health