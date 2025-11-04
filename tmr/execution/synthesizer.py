"""
Execution Layer Synthesizer

This module implements the execution layer synthesis with context-aware
verification selection and output formatting.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

from .depth_selector import DepthSelector, VerificationDepth, DepthProfile
from ..fundamentals.layer import FundamentalsLayer
from ..fundamentals.validators import ReasoningChain, ReasoningStep

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Output formatting options."""
    MINIMAL = "minimal"          # Just valid/invalid and confidence
    STANDARD = "standard"        # Include details and summary
    DETAILED = "detailed"        # Full validation information
    JSON = "json"                # Machine-readable JSON
    HUMAN_READABLE = "human"     # Formatted for human consumption


@dataclass
class VerificationResult:
    """Comprehensive verification result."""
    valid: bool
    confidence: float
    depth_used: VerificationDepth
    domain: str
    validators_used: List[str]
    principles_checked: List[str]
    details: Dict[str, Any]
    timestamp: str
    processing_time_ms: float
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisContext:
    """Context for synthesis operations."""
    input_data: Any
    domain: Optional[str] = None
    required_confidence: Optional[float] = None
    time_budget_ms: Optional[float] = None
    verification_depth: Optional[VerificationDepth] = None
    output_format: OutputFormat = OutputFormat.STANDARD
    custom_validators: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExecutionSynthesizer:
    """
    Execution Layer Synthesizer

    Orchestrates the verification process by:
    - Selecting appropriate verification depth
    - Choosing relevant validators based on context
    - Integrating fundamentals layer validation
    - Formatting output appropriately
    """

    def __init__(self,
                 fundamentals_layer: Optional[FundamentalsLayer] = None,
                 depth_selector: Optional[DepthSelector] = None,
                 config: Optional[Dict] = None):
        """
        Initialize the execution synthesizer.

        Args:
            fundamentals_layer: Optional FundamentalsLayer instance
            depth_selector: Optional DepthSelector instance
            config: Optional configuration dictionary
        """
        self.config = config or {}

        # Initialize layers
        self.fundamentals = fundamentals_layer or FundamentalsLayer(self.config.get("fundamentals"))
        self.depth_selector = depth_selector or DepthSelector(self.config.get("depth_selector"))

        # Synthesis statistics
        self.stats = {
            "total_syntheses": 0,
            "successful_verifications": 0,
            "failed_verifications": 0,
            "by_depth": {},
            "by_domain": {},
            "avg_processing_time_ms": 0.0
        }

        logger.info("ExecutionSynthesizer initialized")

    def synthesize(self, context: Union[SynthesisContext, Dict, Any]) -> VerificationResult:
        """
        Main synthesis method - orchestrates verification and formatting.

        Args:
            context: Synthesis context (SynthesisContext, dict, or raw input)

        Returns:
            VerificationResult with comprehensive verification information
        """
        start_time = datetime.now()
        self.stats["total_syntheses"] += 1

        # Normalize context
        if not isinstance(context, SynthesisContext):
            if isinstance(context, dict) and "input_data" in context:
                context = SynthesisContext(**context)
            else:
                context = SynthesisContext(input_data=context)

        try:
            # Step 1: Select verification depth
            depth_profile = self._select_verification_depth(context)

            # Step 2: Select validators based on context
            selected_validators = self._select_validators(context, depth_profile)

            # Step 3: Perform verification
            validation_results = self._perform_verification(
                context.input_data,
                selected_validators,
                depth_profile
            )

            # Step 4: Synthesize results
            verification_result = self._synthesize_results(
                validation_results,
                depth_profile,
                context,
                start_time
            )

            # Step 5: Format output
            formatted_result = self._format_output(verification_result, context.output_format)

            # Update statistics
            self._update_statistics(verification_result)

            # Record depth performance
            processing_time = verification_result.processing_time_ms
            self.depth_selector.record_performance(
                depth_profile.depth,
                verification_result.valid,
                processing_time
            )

            logger.info(f"Synthesis completed: valid={verification_result.valid}, "
                       f"confidence={verification_result.confidence:.2f}, "
                       f"depth={depth_profile.depth.value}")

            return formatted_result

        except Exception as e:
            logger.error(f"Synthesis error: {e}", exc_info=True)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            # Return error result
            return VerificationResult(
                valid=False,
                confidence=0.0,
                depth_used=VerificationDepth.MINIMAL,
                domain="unknown",
                validators_used=[],
                principles_checked=[],
                details={"error": str(e)},
                timestamp=datetime.now().isoformat(),
                processing_time_ms=processing_time,
                warnings=[f"Synthesis failed: {str(e)}"],
                suggestions=["Check input format and try again"]
            )

    def _select_verification_depth(self, context: SynthesisContext) -> DepthProfile:
        """Select appropriate verification depth based on context."""
        return self.depth_selector.select_depth(
            input_data=context.input_data,
            domain=context.domain,
            required_confidence=context.required_confidence,
            time_budget_ms=context.time_budget_ms,
            user_preference=context.verification_depth
        )

    def _select_validators(self, context: SynthesisContext,
                          depth_profile: DepthProfile) -> List[str]:
        """
        Select appropriate validators based on context and depth.

        Context-aware selection considers:
        - Domain type
        - Input structure
        - Depth profile requirements
        - Custom validator requests
        """
        # Start with depth profile validators
        validators = depth_profile.validators.copy()

        # Add custom validators if specified
        if context.custom_validators:
            validators.extend(context.custom_validators)
            validators = list(set(validators))  # Remove duplicates

        # Context-aware adjustments
        input_data = context.input_data

        # Check if we need mathematical validator
        if self._needs_mathematical_validator(input_data):
            if "mathematical" not in validators:
                validators.append("mathematical")

        # Check if we need causal validator
        if self._needs_causal_validator(input_data):
            if "causal" not in validators:
                validators.append("causal")

        # Check if we need consistency validator (for multi-domain)
        if self._needs_consistency_validator(input_data):
            if "consistency" not in validators:
                validators.append("consistency")

        logger.debug(f"Selected validators: {validators}")
        return validators

    def _needs_mathematical_validator(self, input_data: Any) -> bool:
        """Determine if mathematical validator is needed."""
        if isinstance(input_data, dict):
            math_keys = ["equation", "calculation", "result", "mathematical"]
            if any(key in input_data for key in math_keys):
                return True

        if isinstance(input_data, str):
            math_indicators = ['=', '+', '-', '*', '/', 'equation', 'calculate']
            if any(ind in input_data for ind in math_indicators):
                return True

        return False

    def _needs_causal_validator(self, input_data: Any) -> bool:
        """Determine if causal validator is needed."""
        if isinstance(input_data, dict):
            causal_keys = ["cause", "effect", "events", "causal", "temporal"]
            if any(key in input_data for key in causal_keys):
                return True

        if isinstance(input_data, str):
            causal_indicators = ['because', 'therefore', 'cause', 'effect', 'leads to']
            if any(ind in input_data.lower() for ind in causal_indicators):
                return True

        return False

    def _needs_consistency_validator(self, input_data: Any) -> bool:
        """Determine if consistency validator is needed."""
        if isinstance(input_data, dict):
            # Check for multiple reasoning domains
            domain_keys = ["logical", "mathematical", "causal"]
            present_domains = sum(1 for key in domain_keys if key in input_data)
            return present_domains > 1

        return False

    def _perform_verification(self,
                            input_data: Any,
                            validators: List[str],
                            depth_profile: DepthProfile) -> Dict[str, Any]:
        """
        Perform verification using selected validators.

        Returns comprehensive validation results.
        """
        results = {
            "validators": {},
            "principles": {},
            "overall": {}
        }

        # Perform validation using fundamentals layer
        fundamental_result = self.fundamentals.validate(
            statement=input_data,
            domain=None,  # Let it infer
            use_cache=True
        )

        results["fundamental_validation"] = fundamental_result

        # Validate against specific principles if needed
        if depth_profile.principles:
            principle_results = {}
            from ..fundamentals.principles import PrincipleType

            principle_map = {
                "identity": PrincipleType.IDENTITY,
                "non_contradiction": PrincipleType.NON_CONTRADICTION,
                "excluded_middle": PrincipleType.EXCLUDED_MIDDLE,
                "causality": PrincipleType.CAUSALITY,
                "conservation": PrincipleType.CONSERVATION
            }

            for principle_name in depth_profile.principles:
                if principle_name in principle_map:
                    principle_type = principle_map[principle_name]
                    try:
                        result = self.fundamentals.validate_principle(
                            input_data,
                            principle_type
                        )
                        principle_results[principle_name] = {
                            "valid": result.valid,
                            "confidence": result.confidence,
                            "reason": result.reason
                        }
                    except Exception as e:
                        logger.warning(f"Principle validation error for {principle_name}: {e}")

            results["principles"] = principle_results

        return results

    def _synthesize_results(self,
                          validation_results: Dict,
                          depth_profile: DepthProfile,
                          context: SynthesisContext,
                          start_time: datetime) -> VerificationResult:
        """Synthesize verification results into a unified result."""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        # Extract fundamental validation results
        fundamental = validation_results.get("fundamental_validation", {})
        fundamental_valid = fundamental.get("valid", False)
        fundamental_confidence = fundamental.get("confidence", 0.0)

        # Extract principle validation results
        principle_results = validation_results.get("principles", {})
        principle_valid_count = sum(1 for p in principle_results.values() if p.get("valid", False))
        principle_total = len(principle_results)

        # Calculate overall validity
        overall_valid = fundamental_valid
        if principle_total > 0:
            overall_valid = overall_valid and (principle_valid_count == principle_total)

        # Calculate overall confidence
        confidences = [fundamental_confidence]
        if principle_results:
            principle_confidences = [p.get("confidence", 0.0) for p in principle_results.values()]
            confidences.extend(principle_confidences)

        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Generate warnings
        warnings = []
        if not fundamental_valid:
            warnings.append("Fundamental validation failed")
        if principle_total > 0 and principle_valid_count < principle_total:
            failed_principles = [
                name for name, result in principle_results.items()
                if not result.get("valid", False)
            ]
            warnings.append(f"Failed principles: {', '.join(failed_principles)}")
        if overall_confidence < depth_profile.min_confidence:
            warnings.append(f"Confidence {overall_confidence:.2f} below threshold "
                          f"{depth_profile.min_confidence:.2f}")

        # Generate suggestions
        suggestions = []
        if not overall_valid:
            suggestions.append("Review reasoning steps for logical errors")
        if overall_confidence < 0.7:
            suggestions.append("Consider increasing verification depth")
        if warnings:
            suggestions.append("Address warnings before proceeding")

        # Determine domain
        domain = context.domain
        if not domain:
            domain = fundamental.get("metadata", {}).get("validation_type", "unknown")

        # Build details
        details = {
            "fundamental_validation": fundamental,
            "principle_validations": principle_results,
            "depth_profile": {
                "depth": depth_profile.depth.value,
                "validators": depth_profile.validators,
                "principles": depth_profile.principles,
                "min_confidence": depth_profile.min_confidence
            },
            "statistics": {
                "principles_checked": principle_total,
                "principles_passed": principle_valid_count
            }
        }

        return VerificationResult(
            valid=overall_valid,
            confidence=overall_confidence,
            depth_used=depth_profile.depth,
            domain=domain,
            validators_used=depth_profile.validators,
            principles_checked=depth_profile.principles,
            details=details,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time,
            warnings=warnings,
            suggestions=suggestions,
            metadata=context.metadata
        )

    def _format_output(self,
                      result: VerificationResult,
                      format_type: OutputFormat) -> VerificationResult:
        """
        Format output according to requested format.

        Different formats provide different levels of detail.
        """
        if format_type == OutputFormat.MINIMAL:
            # Keep only essential fields
            result.details = {}
            result.warnings = []
            result.suggestions = []

        elif format_type == OutputFormat.STANDARD:
            # Standard format (default) - keep as is
            pass

        elif format_type == OutputFormat.DETAILED:
            # Already includes all details
            pass

        elif format_type == OutputFormat.JSON:
            # Ensure all data is JSON-serializable
            # (already handled by dataclass)
            pass

        elif format_type == OutputFormat.HUMAN_READABLE:
            # Add formatted summary to metadata
            result.metadata["formatted_summary"] = self._create_human_readable_summary(result)

        return result

    def _create_human_readable_summary(self, result: VerificationResult) -> str:
        """Create a human-readable summary of verification results."""
        lines = []
        lines.append("=" * 60)
        lines.append("VERIFICATION SUMMARY")
        lines.append("=" * 60)
        lines.append("")

        # Status
        status = "✓ PASSED" if result.valid else "✗ FAILED"
        lines.append(f"Status: {status}")
        lines.append(f"Confidence: {result.confidence:.1%}")
        lines.append(f"Domain: {result.domain}")
        lines.append(f"Depth: {result.depth_used.value}")
        lines.append(f"Processing Time: {result.processing_time_ms:.2f}ms")
        lines.append("")

        # Validators used
        lines.append(f"Validators: {', '.join(result.validators_used)}")
        lines.append(f"Principles: {', '.join(result.principles_checked)}")
        lines.append("")

        # Warnings
        if result.warnings:
            lines.append("Warnings:")
            for warning in result.warnings:
                lines.append(f"  - {warning}")
            lines.append("")

        # Suggestions
        if result.suggestions:
            lines.append("Suggestions:")
            for suggestion in result.suggestions:
                lines.append(f"  - {suggestion}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def _update_statistics(self, result: VerificationResult):
        """Update synthesis statistics."""
        if result.valid:
            self.stats["successful_verifications"] += 1
        else:
            self.stats["failed_verifications"] += 1

        # Update by depth
        depth_key = result.depth_used.value
        if depth_key not in self.stats["by_depth"]:
            self.stats["by_depth"][depth_key] = {"count": 0, "successes": 0}
        self.stats["by_depth"][depth_key]["count"] += 1
        if result.valid:
            self.stats["by_depth"][depth_key]["successes"] += 1

        # Update by domain
        if result.domain not in self.stats["by_domain"]:
            self.stats["by_domain"][result.domain] = {"count": 0, "successes": 0}
        self.stats["by_domain"][result.domain]["count"] += 1
        if result.valid:
            self.stats["by_domain"][result.domain]["successes"] += 1

        # Update average processing time
        alpha = 0.1
        self.stats["avg_processing_time_ms"] = (
            alpha * result.processing_time_ms +
            (1 - alpha) * self.stats["avg_processing_time_ms"]
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get synthesis statistics."""
        stats = self.stats.copy()

        # Add success rate
        total = stats["total_syntheses"]
        if total > 0:
            stats["success_rate"] = stats["successful_verifications"] / total
        else:
            stats["success_rate"] = 0.0

        # Add depth selector stats
        stats["depth_performance"] = self.depth_selector.get_performance_stats()

        # Add fundamentals layer stats
        stats["fundamentals_stats"] = self.fundamentals.get_statistics()

        return stats

    def reset_statistics(self):
        """Reset all statistics."""
        self.stats = {
            "total_syntheses": 0,
            "successful_verifications": 0,
            "failed_verifications": 0,
            "by_depth": {},
            "by_domain": {},
            "avg_processing_time_ms": 0.0
        }
        self.fundamentals.reset_statistics()
        logger.info("Statistics reset")

    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the synthesis system."""
        health = {
            "status": "healthy",
            "issues": [],
            "metrics": {}
        }

        # Check fundamentals layer
        try:
            fundamental_health = self.fundamentals.health_check()
            if fundamental_health["status"] != "healthy":
                health["issues"].append("Fundamentals layer not healthy")
                health["status"] = "degraded"
        except Exception as e:
            health["issues"].append(f"Fundamentals layer error: {e}")
            health["status"] = "degraded"

        # Check depth selector
        try:
            depth_stats = self.depth_selector.get_performance_stats()
            health["metrics"]["depth_stats"] = depth_stats
        except Exception as e:
            health["issues"].append(f"Depth selector error: {e}")
            health["status"] = "degraded"

        # Add synthesis metrics
        stats = self.get_statistics()
        health["metrics"].update({
            "total_syntheses": stats["total_syntheses"],
            "success_rate": stats["success_rate"],
            "avg_processing_time_ms": stats["avg_processing_time_ms"]
        })

        return health

    def __repr__(self) -> str:
        """String representation of the synthesizer."""
        return (f"ExecutionSynthesizer(syntheses={self.stats['total_syntheses']}, "
                f"success_rate={self.stats.get('success_rate', 0):.2%})")
