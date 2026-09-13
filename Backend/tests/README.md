# LIT for Voice - Test Implementation Guide

## Overview

This test suite implements the comprehensive Master Test Plan for LIT for Voice application. The tests are designed to validate all critical functionality, performance requirements, and security measures as outlined in the test plan.

## Test Structure

### Test Categories

#### 🔴 **Critical Priority Tests**

1. **Data and Database Integrity Testing** (`test_data_integrity.py`)
   - Redis cache operations and consistency
   - Audio file storage integrity
   - Session management and isolation
   - Dataset metadata validation
   - **Coverage**: Section 3.1.1 of Test Plan
   - **Time**: ~10 minutes

2. **Function Testing** (`test_function_testing.py`)
   - ML model integration (Whisper, Wav2Vec2)
   - Audio processing pipeline
   - Attention mechanism extraction
   - API endpoint validation
   - **Coverage**: Section 3.1.2 of Test Plan
   - **Time**: ~30 minutes

#### 🟡 **Important Priority Tests**

3. **Performance and Load Testing** (`test_perf_*.py`, `test_load_*.py`)
   - Control-plane latency, event-loop blocking, worker compute budgets (SRS PE-1..PE-3)
   - Workload profiles, background load, worker scaling, capacity growth
   - **Coverage**: Sections 3.1.4 & 3.1.5 of Test Plan (`tests/plans/3.1.4-*.md`, `3.1.5-*.md`)

4. **Security Testing** (`test_security.py`)
   - Session security validation
   - File upload security checks
   - Input validation and injection prevention
   - Access control verification
   - **Coverage**: Section 3.1.6 of Test Plan
   - **Time**: ~20 minutes

## Running Tests

### Prerequisites

```bash
# Backend test dependencies (includes pytest-cov)
pip install -r requirements-dev.txt

# Frontend: Vitest, React Testing Library and @vitest/coverage-v8 are devDependencies
cd ../Frontend && npm install
```

### Test Execution Commands

```bash
# Run all tests
python tests/run_tests.py

# Run critical tests only
python tests/run_tests.py critical

# Run specific test category
python tests/run_tests.py data_integrity
python tests/run_tests.py function_testing
python tests/run_tests.py performance
python tests/run_tests.py security

# Run with pytest directly
pytest tests/ -v
pytest tests/test_data_integrity.py -v
pytest tests/test_perf_event_loop.py -v

# Run tests with specific markers
pytest -m "critical" -v
pytest -m "performance" -v
pytest -m "security" -v

# Coverage + test evaluation summary (see "Test Results and Reporting")
python tests/run_tests.py report
```

### Performance Benchmarks

Performance budgets are asserted by the `performance`-marked cases themselves, sourced from SRS PE-1..PE-3 — see `tests/plans/3.1.4-performance-profiling.md` and `tests/plans/3.1.5-load-testing.md`.

## Test Configuration

### Environment Setup

Tests automatically configure the following:

- **Fake Redis**: In-memory Redis for cache testing
- **Mock Audio Files**: Generated synthetic audio for testing
- **Temporary Directories**: Isolated file storage for each test
- **Mock ML Models**: Simulated model outputs for testing without GPU requirements

### Test Data

The test suite includes:

- **Synthetic Audio**: Generated WAV files with controlled properties
- **Mock Model Outputs**: Realistic attention and prediction data
- **Security Test Payloads**: Injection and malicious file samples
- **Performance Test Data**: Large datasets for load testing

## Test Results and Reporting

### Console Output

Tests provide detailed console output including:
- Performance timing measurements
- Memory usage statistics  
- Cache hit/miss ratios
- Security vulnerability detection
- Error handling validation

### Test Evaluation Summary and Coverage Reports (Test Plan Section 4)

```bash
cd Frontend && npm run test:coverage       # frontend tier (optional, run first)
cd Backend && python tests/run_tests.py report   # backend tier + summary of both
python tests/run_tests.py summary          # re-render the summary, running nothing
```

`report` runs the backend suite twice, as child processes with `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`:
coverage run (`-m "not performance"`, under coverage) and a timing run (`-m performance`, no coverage),
since the tracer would inflate the wall-clock budgets the performance cases assert. It exits non-zero
if either run fails, including the coverage floor. What it produces:

| Artifact | Contents |
|---|---|
| `tests/test_reports/evaluation-summary.md` | Verdict (PASS / FAIL / INCOMPLETE), environment and commands, per-run and per-test-plan-section totals for both tiers, every failure with its first message line, coverage against floors, the 10 lowest-covered backend modules |
| `tests/test_reports/coverage-html/index.html` | Backend line + branch coverage, per file |
| `tests/test_reports/junit-backend.xml`, `junit-performance.xml`, `coverage.json` | Machine-readable inputs to the summary |
| `Frontend/coverage/index.html`, `Frontend/coverage/coverage-summary.json` | Frontend v8 coverage |
| `Frontend/test-reports/junit.xml` | Frontend results |

All of it is gitignored. Form, content and frequency are specified in `tests/plans/4-deliverables.md`.

## Integration with CI/CD

There is no CI pipeline in this repository; the reports are produced locally. A pipeline would run:

```yaml
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - uses: actions/setup-node@v4
        with:
          node-version: '22'
      - name: Backend tests, coverage and evaluation summary
        run: |
          pip install -r Backend/requirements-dev.txt
          cd Backend && python tests/run_tests.py report
      - name: Frontend tests and coverage
        run: cd Frontend && npm ci && npm run test:coverage
      - name: Upload reports
        uses: actions/upload-artifact@v4
        with:
          name: test-reports
          path: |
            Backend/tests/test_reports/
            Frontend/coverage/
            Frontend/test-reports/
```

## Test Coverage

Coverage floors are enforced, not stated here:

| Tier | Metric | Enforced by |
|---|---|---|
| Backend | statements + branches over `app/` | `fail_under` in `Backend/.coveragerc` |
| Frontend | lines, statements, branches, functions over `src/` | `coverage.thresholds` in `Frontend/vite.config.ts` |

Each floor was set just below the measured baseline. It is raised when a test-plan section lands and
coverage rises, and never lowered without a recorded reason. The measured figures and floors are in
every evaluation summary (`tests/test_reports/evaluation-summary.md`).

Backend coverage is measured without the `performance`-marked cases. The frontend figure counts every
file under `src/`, including files no test imports. Functional coverage (which requirements and use
cases are exercised) is tracked per section in the Requirements Traceability tables of `tests/plans/`.

## Test Maintenance

### Adding New Tests

1. **Create test file** following naming convention `test_*.py`
2. **Add test category** to `run_tests.py` configuration
3. **Include appropriate markers** (`@pytest.mark.critical`, etc.)
4. **Update documentation** with new test coverage

### Mock Updates

When adding new models or features:

1. **Update mock outputs** in `conftest.py`
2. **Add new fixtures** for test data
3. **Update performance thresholds** if needed
4. **Add security test cases** for new endpoints

## Troubleshooting

### Common Issues

**Tests fail due to missing models:**
```bash
# Tests are designed to work without actual ML models
# They use mocks and skip tests when models unavailable
pytest tests/ --ignore-missing-models
```

**Memory issues during testing:**
```bash
# Run tests with memory limits
pytest tests/ -x --tb=short -q
```

**Redis connection issues:**
```bash
# Tests use fakeredis, no actual Redis needed
# If using real Redis, ensure it's running:
redis-server --port 6379
```

### Performance Test Variations

Performance results may vary based on:
- **Hardware**: CPU/GPU availability affects model inference times
- **System Load**: Background processes impact performance measurements
- **Network**: API response times affected by network latency
- **Storage**: File I/O performance varies by storage type

### Security Test Considerations

Security tests include:
- **Mock payloads**: No actual malicious code execution
- **Isolated environment**: Tests run in containers/virtual environments
- **Safe data**: No real sensitive information used in tests

## Contributing

When adding new tests:

1. **Follow test plan structure** - align with Master Test Plan sections
2. **Use appropriate priorities** - mark as critical/important/low
3. **Include performance metrics** - measure timing and resource usage
4. **Add security validations** - test input validation and access control
5. **Document new tests** - update this guide with new test coverage

## Contact and Support

For test-related questions:
- **Test Plan**: Refer to Master Test Plan document
- **Implementation**: Check individual test file documentation
- **Issues**: Report test failures with full console output
- **Performance**: Include system specifications for performance issues