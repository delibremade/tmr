"""
Simple test runner for TMR tests without pytest.
"""

import sys
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import test modules
try:
    from tests import test_principles, test_validators, test_fundamentals_layer
except ImportError as e:
    print(f"Error importing tests: {e}")
    sys.exit(1)


def run_test_class(test_class, class_name):
    """Run all test methods in a test class."""
    print(f"\n{'='*70}")
    print(f"Running {class_name}")
    print('='*70)

    instance = test_class()
    test_methods = [m for m in dir(instance) if m.startswith('test_')]

    passed = 0
    failed = 0
    errors = []

    for method_name in test_methods:
        try:
            # Run setup if it exists
            if hasattr(instance, 'setup_method'):
                instance.setup_method()

            # Run test
            method = getattr(instance, method_name)
            method()

            print(f"  ✓ {method_name}")
            passed += 1

        except AssertionError as e:
            print(f"  ✗ {method_name} - FAILED")
            failed += 1
            errors.append((class_name, method_name, str(e)))

        except Exception as e:
            print(f"  ✗ {method_name} - ERROR")
            failed += 1
            errors.append((class_name, method_name, traceback.format_exc()))

    return passed, failed, errors


def main():
    """Run all tests."""
    print("Trinity Meta-Reasoning Framework - Test Suite")
    print("=" * 70)

    all_passed = 0
    all_failed = 0
    all_errors = []

    # Test classes from test_principles
    principle_tests = [
        (test_principles.TestIdentityPrinciple, "TestIdentityPrinciple"),
        (test_principles.TestNonContradictionPrinciple, "TestNonContradictionPrinciple"),
        (test_principles.TestExcludedMiddlePrinciple, "TestExcludedMiddlePrinciple"),
        (test_principles.TestCausalityPrinciple, "TestCausalityPrinciple"),
        (test_principles.TestConservationPrinciple, "TestConservationPrinciple"),
        (test_principles.TestLogicalPrinciples, "TestLogicalPrinciples"),
        (test_principles.TestPrincipleIntegration, "TestPrincipleIntegration"),
    ]

    # Test classes from test_validators
    validator_tests = [
        (test_validators.TestLogicalValidator, "TestLogicalValidator"),
        (test_validators.TestMathematicalValidator, "TestMathematicalValidator"),
        (test_validators.TestCausalValidator, "TestCausalValidator"),
        (test_validators.TestConsistencyValidator, "TestConsistencyValidator"),
        (test_validators.TestValidatorIntegration, "TestValidatorIntegration"),
    ]

    # Test classes from test_fundamentals_layer
    layer_tests = [
        (test_fundamentals_layer.TestFundamentalsLayerBasics, "TestFundamentalsLayerBasics"),
        (test_fundamentals_layer.TestDomainInference, "TestDomainInference"),
        (test_fundamentals_layer.TestCaching, "TestCaching"),
        (test_fundamentals_layer.TestStatistics, "TestStatistics"),
        (test_fundamentals_layer.TestHealthCheck, "TestHealthCheck"),
        (test_fundamentals_layer.TestDataPreparation, "TestDataPreparation"),
        (test_fundamentals_layer.TestValidationTypes, "TestValidationTypes"),
        (test_fundamentals_layer.TestErrorHandling, "TestErrorHandling"),
        (test_fundamentals_layer.TestIntegration, "TestIntegration"),
    ]

    all_tests = principle_tests + validator_tests + layer_tests

    for test_class, name in all_tests:
        passed, failed, errors = run_test_class(test_class, name)
        all_passed += passed
        all_failed += failed
        all_errors.extend(errors)

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Passed: {all_passed}")
    print(f"Total Failed: {all_failed}")
    print(f"Total Tests:  {all_passed + all_failed}")

    if all_failed > 0:
        print(f"\n{'='*70}")
        print("FAILED TESTS DETAILS")
        print('='*70)
        for class_name, method_name, error in all_errors:
            print(f"\n{class_name}.{method_name}:")
            print(f"  {error}")

    print("\n" + "=" * 70)
    if all_failed == 0:
        print("✓ ALL TESTS PASSED!")
    else:
        print(f"✗ {all_failed} TESTS FAILED")
    print("=" * 70)

    return 0 if all_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
