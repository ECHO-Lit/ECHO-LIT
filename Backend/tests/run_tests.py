"""
Test Configuration and Runner Script
Master Test Plan Implementation - Comprehensive Test Execution
"""

import pytest
import shlex
import subprocess
import sys
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = Path(__file__).resolve().parent / "test_reports"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
from tests import _report  # noqa: E402

# Every tests/test_*.py belongs to exactly one category (guarded by TD-01), so
# `all`, `critical` and `report` cannot silently skip a module. Assignments
# follow each module's Test Case Matrix in tests/plans/.
TEST_CATEGORIES = {
    "data_integrity": {
        "section": "3.1.1",
        "description": "Data and Database Integrity Testing (Section 3.1.1)",
        "files": [
            "test_data_integrity.py",
            "test_data_integrity_artifacts.py",
            "test_data_integrity_store.py",
            "test_results_cache.py",
        ],
        "priority": "critical",
    },
    "function_testing": {
        "section": "3.1.2",
        "description": "Function Testing (Section 3.1.2)",
        "files": [
            "test_function_testing.py",
            "test_async_jobs.py",
            "test_queue.py",
            "test_jobs_api_contract.py",
            "test_custom_models_api.py",
            "test_custom_dataset_manifest.py",
            "test_dataset_labels_service.py",
            "test_dataset_management_api.py",
            "test_dataset_serving_api.py",
            "test_clustering.py",
            "test_probing_service.py",
            "test_layer_probe_job.py",
            "test_layer_probe_schema.py",
            "test_jacobian_lens.py",
            "test_faithfulness_job.py",
            "test_faithfulness_runner.py",
            "test_faithfulness_service.py",
            "test_fr7_acceptance.py",
            "test_fr7_api_contract.py",
            "test_fr7_dsp_invariants.py",
            "test_fr7_metrics.py",
            "test_fr7_orchestration.py",
            "test_fr10_api_contract.py",
            "test_fr10_cancellation_and_failures.py",
            "test_fr10_dataset_extensions.py",
            "test_fr10_grounding.py",
            "test_fr10_metrics.py",
            "test_fr10_orchestration.py",
            "test_fr10_partitioning.py",
            "test_whisper_transcript_consistency.py",
        ],
        "priority": "critical",
    },
    "performance_profiling": {
        "section": "3.1.4",
        "description": "Performance Profiling (Section 3.1.4)",
        "files": [
            "test_perf_control_plane.py",
            "test_perf_event_loop.py",
            "test_perf_worker_compute.py",
        ],
        "priority": "important",
    },
    "load_testing": {
        "section": "3.1.5",
        "description": "Load Testing (Section 3.1.5)",
        "files": [
            "test_load_workload_profiles.py",
            "test_load_background_workload.py",
            "test_load_worker_scaling.py",
            "test_load_capacity_growth.py",
        ],
        "priority": "important",
    },
    "security": {
        "section": "3.1.6",
        "description": "Security and Access Control Testing (Section 3.1.6)",
        "files": ["test_security.py", "test_session_cookie.py", "test_access_control.py"],
        "priority": "important",
    },
    "failover_recovery": {
        "section": "3.1.7",
        "description": "Failover and Recovery Testing (Section 3.1.7)",
        "files": [
            "test_failover_redis_outage.py",
            "test_failover_worker_recovery.py",
            "test_failover_stuck_jobs.py",
            "test_failover_corrupt_data.py",
            "test_failover_under_load.py",
            "test_worker_native_crash.py",
        ],
        "priority": "critical",
    },
    "configuration": {
        "section": "3.1.8",
        "description": "Configuration Testing (Section 3.1.8)",
        "files": [
            "test_config_settings.py",
            "test_config_deployment.py",
            "test_config_roles.py",
            "test_config_hardware.py",
            "test_config_storage.py",
            "test_device.py",
        ],
        "priority": "important",
    },
    "deliverables": {
        "section": "4",
        "description": "Test Deliverables Tooling (Section 4)",
        "files": ["test_deliverables_reporting.py"],
        "priority": "important",
    },
    "risks": {
        "section": "5",
        "description": "Risks, Dependencies, Assumptions and Constraints (Section 5)",
        "files": ["test_risk_register.py"],
        "priority": "important",
    },
}

# Performance budgets live with the cases that assert them, sourced from SRS
# PE-1..PE-3 -- see tests/plans/3.1.4-performance-profiling.md and
# tests/plans/3.1.5-load-testing.md.

# Test execution functions
def run_category_tests(category: str, verbose: bool = True):
    """Run tests for a specific category."""
    if category not in TEST_CATEGORIES:
        return False
    
    config = TEST_CATEGORIES[category]
    
    # Build pytest arguments
    pytest_args = []
    
    if verbose:
        pytest_args.extend(["-v", "-s"])
    
    # Add specific test files
    for test_file in config["files"]:
        test_path = Path(__file__).parent / test_file
        if test_path.exists():
            pytest_args.append(str(test_path))
        else:
            print(f"Warning: Test file not found: {test_file}")
    
    if not pytest_args or not any(Path(arg).exists() for arg in pytest_args if Path(arg).suffix == '.py'):
        print(f"No test files found for category: {category}")
        return False
    
    # Run tests
    result = pytest.main(pytest_args)
    return result == 0

def run_critical_tests():
    """Run only critical priority tests."""
    print("\n" + "="*80)
    print("RUNNING CRITICAL PRIORITY TESTS")
    print("="*80)
    
    critical_categories = [cat for cat, config in TEST_CATEGORIES.items() 
                          if config["priority"] == "critical"]
    
    results = {}
    for category in critical_categories:
        results[category] = run_category_tests(category)
    
    # Summary
    print(f"\n{'='*60}")
    print("CRITICAL TESTS SUMMARY")
    print(f"{'='*60}")
    
    for category, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{category}: {status}")
    
    return all(results.values())

def run_all_tests():
    """Run all test categories in order of priority."""
    print("\n" + "="*80) 
    print("RUNNING COMPLETE TEST SUITE")
    print("LIT for Voice - Master Test Plan Implementation")
    print("="*80)
    
    # Run critical tests first
    critical_categories = [cat for cat, config in TEST_CATEGORIES.items() 
                          if config["priority"] == "critical"]
    
    important_categories = [cat for cat, config in TEST_CATEGORIES.items()
                           if config["priority"] == "important"]
    
    all_results = {}
    
    # Critical tests
    print("\n🔴 CRITICAL PRIORITY TESTS")
    for category in critical_categories:
        all_results[category] = run_category_tests(category)
        
    # Important tests
    print("\n🟡 IMPORTANT PRIORITY TESTS") 
    for category in important_categories:
        all_results[category] = run_category_tests(category)
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL TEST EXECUTION SUMMARY")
    print(f"{'='*80}")
    
    critical_passed = all(all_results[cat] for cat in critical_categories)
    important_passed = all(all_results[cat] for cat in important_categories if cat in all_results)
    
    print(f"Critical Tests: {'PASS' if critical_passed else 'FAIL'}")
    print(f"Important Tests: {'PASS' if important_passed else 'FAIL'}")
    
    for category, passed in all_results.items():
        config = TEST_CATEGORIES[category]
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {category} ({config['priority']}): {status}")
    
    overall_success = critical_passed and important_passed
    print(f"\nOVERALL RESULT: {'✅ PASS' if overall_success else '❌ FAIL'}")
    
    return overall_success

def run_performance_benchmarks():
    """Run performance tests with detailed benchmarking."""
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK TESTING")
    print("="*80)
    
    # Every 3.1.4 / 3.1.5 case carries the `performance` marker; -s prints the
    # load reports.
    result = pytest.main(["-v", "-s", "-m", "performance", str(Path(__file__).parent)])
    
    print(f"\nPerformance Benchmarks: {'PASS' if result == 0 else 'FAIL'}")
    return result == 0

FRONTEND_COMMAND = "cd Frontend && npm run test:coverage"
STALE_ARTIFACTS = ("junit-backend.xml", "junit-performance.xml", "coverage.json")


def report_commands():
    """The two pytest runs behind `report`, as (label, argv) pairs.

    The `performance` cases assert absolute wall-clock budgets, which the
    coverage tracer would inflate, so they run separately without coverage.
    """
    out = "tests/test_reports"
    pytest_cmd = [sys.executable, "-m", "pytest", "tests", "-q"]
    return [
        ("Coverage run", pytest_cmd + [
            "-m", "not performance",
            "--cov",
            f"--cov-report=html:{out}/coverage-html",
            f"--cov-report=json:{out}/coverage.json",
            f"--junitxml={out}/junit-backend.xml",
        ]),
        ("Timing run", pytest_cmd + ["-m", "performance", f"--junitxml={out}/junit-performance.xml"]),
    ]


def _display_commands():
    shown = ["cd Backend && " + shlex.join(["python", *argv[1:]]) for _, argv in report_commands()]
    return shown + [FRONTEND_COMMAND]


def _child_env():
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    return env


def generate_test_report():
    """Run the suite with coverage and write the evaluation summary (Section 4)."""
    print("\n" + "="*80)
    print("GENERATING TEST EVALUATION SUMMARY")
    print("="*80)

    REPORT_DIR.mkdir(exist_ok=True)
    for name in STALE_ARTIFACTS:
        (REPORT_DIR / name).unlink(missing_ok=True)

    outcomes = []
    for label, argv in report_commands():
        print(f"\n--- {label}: {shlex.join(argv[1:])}")
        result = subprocess.run(argv, cwd=BACKEND_DIR, env=_child_env())
        outcomes.append((label, result.returncode))

    path, evaluation = _report.write_summary(TEST_CATEGORIES, _display_commands())

    print()
    for label, code in outcomes:
        print(f"{label}: {'PASS' if code == 0 else f'FAIL (exit {code})'}")
    print(f"Evaluation summary [{evaluation.verdict}]: {path}")
    return all(code == 0 for _, code in outcomes)


def render_summary():
    """Re-render the evaluation summary from existing artifacts, running nothing."""
    path, evaluation = _report.write_summary(TEST_CATEGORIES, _display_commands())
    print(f"Evaluation summary [{evaluation.verdict}]: {path}")
    return evaluation.verdict != "FAIL"

if __name__ == "__main__":
    """
    Test Runner Script
    
    Usage:
        python run_tests.py                    # Run all tests
        python run_tests.py critical          # Run critical tests only
        python run_tests.py data_integrity    # Run specific category
        python run_tests.py performance       # Run performance tests
        python run_tests.py report            # Coverage + evaluation summary
        python run_tests.py summary           # Re-render summary from artifacts
    """

    if len(sys.argv) < 2:
        # Run all tests by default
        success = run_all_tests()
        sys.exit(0 if success else 1)

    command = sys.argv[1].lower()

    if command == "critical":
        success = run_critical_tests()
    elif command == "performance":
        success = run_performance_benchmarks()
    elif command == "report":
        success = generate_test_report()
    elif command == "summary":
        success = render_summary()
    elif command in TEST_CATEGORIES:
        success = run_category_tests(command)
    elif command == "all":
        success = run_all_tests()
    else:
        print(f"Unknown command: {command}")
        print("\nAvailable commands:")
        print("  critical          - Run critical priority tests")
        print("  performance       - Run performance benchmarks")
        print("  report            - Run with coverage, write tests/test_reports/evaluation-summary.md")
        print("  summary           - Re-render the evaluation summary from existing artifacts")
        print("  all               - Run all tests")
        
        for category in TEST_CATEGORIES:
            config = TEST_CATEGORIES[category]
            print(f"  {category:<15} - {config['description']}")
        
        sys.exit(1)
    
    sys.exit(0 if success else 1)