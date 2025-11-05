#!/usr/bin/env python3
"""
TMR Benchmark Suite - CLI Runner

This script provides a command-line interface for running TMR benchmarks.

Usage:
    python run_benchmarks.py                    # Run all benchmarks
    python run_benchmarks.py --domain math      # Run math domain only
    python run_benchmarks.py --complexity simple # Run simple problems only
    python run_benchmarks.py --no-baselines     # Skip baseline generation
    python run_benchmarks.py --output-dir ./results # Custom output directory
"""

import argparse
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.runner import BenchmarkRunner, BenchmarkConfig
from benchmarks.problems import get_benchmark_statistics


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='TMR Benchmark Validation Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run all benchmarks:
    python run_benchmarks.py

  Run specific domain:
    python run_benchmarks.py --domain math

  Run specific complexity:
    python run_benchmarks.py --complexity simple

  Run without baselines:
    python run_benchmarks.py --no-baselines

  Custom output directory:
    python run_benchmarks.py --output-dir ./my_results

  Verbose output:
    python run_benchmarks.py --verbose
"""
    )

    # Domain selection
    parser.add_argument(
        '--domain',
        type=str,
        action='append',
        choices=['math', 'code', 'logic', 'mixed'],
        help='Run specific domain(s) only (can be specified multiple times)'
    )

    # Complexity selection
    parser.add_argument(
        '--complexity',
        type=str,
        action='append',
        choices=['trivial', 'simple', 'moderate', 'complex', 'advanced'],
        help='Run specific complexity level(s) only (can be specified multiple times)'
    )

    # Baseline configuration
    parser.add_argument(
        '--no-baselines',
        action='store_true',
        help='Skip baseline generation'
    )

    parser.add_argument(
        '--baseline',
        type=str,
        action='append',
        choices=['fundamentals_only', 'with_nuance', 'full_tmr', 'minimal_depth', 'standard_depth', 'exhaustive_depth'],
        help='Generate specific baseline(s) only (can be specified multiple times)'
    )

    # Performance configuration
    parser.add_argument(
        '--max-time',
        type=float,
        default=10000.0,
        help='Maximum time per problem in milliseconds (default: 10000)'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching'
    )

    parser.add_argument(
        '--depth',
        type=str,
        default='STANDARD',
        choices=['MINIMAL', 'QUICK', 'STANDARD', 'THOROUGH', 'EXHAUSTIVE'],
        help='Verification depth (default: STANDARD)'
    )

    # Output configuration
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./benchmark_results',
        help='Output directory for results (default: ./benchmark_results)'
    )

    parser.add_argument(
        '--format',
        type=str,
        action='append',
        choices=['text', 'json', 'html', 'markdown', 'csv'],
        help='Output format(s) (default: all formats)'
    )

    # Logging
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Suppress all output except errors'
    )

    # Statistics
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show benchmark statistics and exit'
    )

    args = parser.parse_args()

    # Setup logging
    if not args.quiet:
        setup_logging(args.verbose)

    # Show statistics if requested
    if args.stats:
        stats = get_benchmark_statistics()
        print("\n" + "=" * 60)
        print("TMR BENCHMARK STATISTICS")
        print("=" * 60)
        print(f"\nTotal Problems: {stats['total_problems']}")
        print(f"Problem Sets: {stats['problem_sets']}")
        print("\nBy Domain:")
        for domain, count in stats['by_domain'].items():
            print(f"  {domain}: {count}")
        print("\nBy Complexity:")
        for complexity, count in stats['by_complexity'].items():
            print(f"  {complexity}: {count}")
        print("\n" + "=" * 60 + "\n")
        return 0

    # Create benchmark configuration
    config = BenchmarkConfig(
        run_all_domains=(args.domain is None),
        domains=args.domain or [],
        complexities=args.complexity or [],
        max_time_per_problem_ms=args.max_time,
        use_caching=not args.no_cache,
        verification_depth=args.depth,
        generate_baselines=not args.no_baselines,
        baseline_types=args.baseline or [],
        enable_logging=not args.quiet,
        output_dir=args.output_dir,
    )

    # Create runner
    runner = BenchmarkRunner(config=config)

    # Print configuration
    if not args.quiet:
        print("\n" + "=" * 60)
        print("TMR BENCHMARK VALIDATION SUITE")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Domains: {config.domains if config.domains else 'all'}")
        print(f"  Complexities: {config.complexities if config.complexities else 'all'}")
        print(f"  Verification Depth: {config.verification_depth}")
        print(f"  Caching: {'enabled' if config.use_caching else 'disabled'}")
        print(f"  Baselines: {'enabled' if config.generate_baselines else 'disabled'}")
        if config.baseline_types:
            print(f"  Baseline Types: {', '.join(config.baseline_types)}")
        print(f"  Output Directory: {config.output_dir}")
        print("=" * 60 + "\n")

    # Run benchmarks
    try:
        if config.generate_baselines:
            results = runner.run_with_baselines()
        else:
            results = runner.run_benchmarks()

        # Generate reports in requested formats
        output_formats = args.format or ['text', 'json', 'html']

        for fmt in output_formats:
            try:
                filepath = runner.save_report(results, output_format=fmt)
                if not args.quiet:
                    print(f"Report saved: {filepath}")
            except Exception as e:
                logging.error(f"Error generating {fmt} report: {e}")

        # Print summary
        if not args.quiet:
            print("\n")
            print(runner.generate_report(results, output_format='text'))

        return 0

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Error running benchmarks: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
