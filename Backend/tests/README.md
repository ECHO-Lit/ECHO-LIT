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

2. **Function Testing** (`test_function_testing.py`)
   - ML model integration (Whisper, Wav2Vec2)
   - Audio processing pipeline
   - Attention mechanism extraction
   - API endpoint validation
   - **Coverage**: Section 3.1.2 of Test Plan

#### 🟡 **Important Priority Tests**

3. **Performance and Load Testing** (`test_perf_*.py`, `test_load_*.py`)
   - Control-plane latency, event-loop blocking, worker compute budgets (SRS PE-1..PE-3)
   - Workload profiles, background load, worker scaling, capacity growth
   - **Coverage**: Sections 3.1.4 & 3.1.5 of Test Plan (`tests/plans/3.1.4-*.md`, `3.1.5-*.md`)

4. **Security and Access Control Testing** (`test_access_control.py`, `test_security.py`, `test_session_cookie.py`)
   - Cross-session isolation of audio, custom datasets and jobs, with the owner's success in the same case
   - "Not yours" indistinguishable from "does not exist"
   - Session identity: a client-chosen `sid` is never adopted; cookie attributes
   - Deployment surface: legacy routes gated off, no admin routes, `/health` issues no cookie
   - **Coverage**: Section 3.1.6 of Test Plan (`tests/plans/3.1.6-security-testing.md`). Most of
     `test_security.py` passes whatever the app does; that plan records which of its cases are evidence

5. **Risk Register Guards** (`test_risk_register.py`)
   - Offline mode, corpus skip guards, timing-oracle markers, harness pins, README commands
   - **Coverage**: Section 5 of Test Plan (`tests/plans/5-risks-dependencies-assumptions-constraints.md`)

Every module belongs to exactly one category in `TEST_CATEGORIES` (`run_tests.py`), which maps it to its
test-plan section. That map, not this list, is authoritative, and TD-01 fails if a module is left out.

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
python tests/run_tests.py security    # 3.1.6
python tests/run_tests.py risks

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

Everything runs in-process: the FastAPI app is driven through httpx's ASGI transport, and no server,
container, Redis or Celery worker is started.

- **Offline by default**: `tests/conftest.py` sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
  before anything imports the Hugging Face libraries, so no run can download weights. An explicit value
  in your shell wins; see Troubleshooting for the one opt-out.
- **Fake Redis**: an autouse fixture replaces the three Redis clients (sessions DB0, jobs DB1, broker DB2)
  with fakeredis on one shared in-memory server.
- **Model stand-ins, not models**: inference is replaced at the adapter boundary by stand-ins with known
  outputs. The real models are exercised only by opt-in or cache-dependent cases (see Troubleshooting).
- **Isolated storage**: modules that write objects, sessions or uploads point them at `tmp_path` through
  per-module fixtures, so a run leaves the repository tree unchanged.

### Test Data

- **Synthetic audio**: WAV files generated in memory with controlled properties (`tests/_fixtures.py`).
- **Fetched corpora**: SAA, L2-ARCTIC, Common Voice and RAVDESS under `Backend/data/` (populated by the
  scripts in `scripts/`, see the root README). The directory is
  gitignored (the licences forbid redistribution), so a clean checkout has none of it and the cases that
  read it skip.
- **Security test payloads**: injection strings and malformed files, generated by the tests themselves.
- **Load profiles**: virtual users and a simulated worker (`tests/_load.py`), not large datasets.

## Test Results and Reporting

### Console Output

With `-s` (which `run_tests.py`'s category and `performance` commands pass), a few cases print what they
measured: per-endpoint latency statistics (`test_perf_control_plane.py`), load-run reports
(`test_load_workload_profiles.py`, `test_load_worker_scaling.py`) and fault-injection summaries
(`test_failover_under_load.py`). Every other case reports only through its assertions.

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

### Fixtures and Stand-ins

`tests/conftest.py` applies to every module, so it stays as it is: the fake Redis fixture, the offline
defaults and a few older fixtures. Isolation a new module needs goes in that module, not in the conftest:

1. **Per-module fixtures**: storage, dataset-root and broker isolation are autouse fixtures inside the
   module that needs them (see any `test_fr10_*.py` or `test_failover_*.py`).
2. **Shared helpers** are plain modules the tests import explicitly: `_fixtures.py` (synthetic audio,
   uploads), `_corpora.py` (corpus skip guards), `_perf.py` and `_load.py` (timing instruments),
   `_faults.py` (fault injection), `_config.py` (configuration readers).
3. **A case that reads a bundled corpus** carries `requires_corpora(...)` from `_corpora.py`.
4. **A module that submits jobs** patches `celery_app.send_task` (a `fake_broker` fixture); otherwise the
   submission goes to the broker in your real environment.

## Troubleshooting

**Cases skip with "Bundled corpus absent … (Backend/data/ is gitignored)":**
The corpora are not in the repository. Obtain them under their licences and place them in the layout
`app/services/dataset_service.py` registers: `Backend/data/common_voice_valid_dev/`,
`ravdess_subset/`, `L2_ARCTIC_dataset/` (with `l2_phone_error_annotations.csv` and `audio/`) and
`SAA_dataset/` (with `audio/`). To see what a clean checkout sees on a machine that has them:
```bash
python -m pytest tests/ -q -p tests._corpora
```

**A case needs the real models:**
Runs are offline by default. The real-model acceptance case runs only when asked, and needs the models in
the local Hugging Face cache (or permission to download them):
```bash
ECHO_MODEL_TESTS=1 HF_HUB_OFFLINE=0 pytest tests/test_fr7_acceptance.py
```
`HF_HUB_OFFLINE=0` is the opt-out; `TRANSFORMERS_OFFLINE=0` alone is not. The Whisper agreement case in
`test_whisper_transcript_consistency.py` runs the real `openai/whisper-base` when it is already cached,
and skips otherwise.

**Redis:**
No Redis is needed; fakeredis stands in for all three databases. A job-submitting module that does not
patch `send_task` will try to reach the broker at `localhost:6379/2`.

**A `performance` case fails:**
Those cases assert absolute or relative wall-clock budgets (SRS PE-1 to PE-3), so they need an idle
machine. Rerun the module on its own before suspecting a regression, or leave them out:
```bash
pytest tests/ -q -m "not performance"
```

**Where the tests run:**
In-process, in your virtualenv, on your machine. Nothing runs in a container.

The risks behind these entries, their mitigations and the cases that guard them are in the Section 5
register, `tests/plans/5-risks-dependencies-assumptions-constraints.md`.

## Contributing

When adding new tests:

1. **Follow test plan structure** - align with Master Test Plan sections
2. **Use appropriate priorities** - mark as critical/important
3. **Put timing assertions only in `performance` cases** - a wall-clock assertion anywhere else runs under
   the coverage tracer (RD-07 fails if one appears)
4. **Add security validations** - test input validation and access control
5. **Document new tests** - update this guide with new test coverage

## Contact and Support

For test-related questions:
- **Test Plan**: Refer to Master Test Plan document
- **Implementation**: Check individual test file documentation
- **Issues**: Report test failures with full console output
- **Performance**: Include system specifications for performance issues