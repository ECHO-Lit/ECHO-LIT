# 4 Deliverables

**Project:** ECHO (Explainable Computation for Hearing Outputs) — Learning Interpretability Tool for audio models
**Test Plan section:** 4 Deliverables (4.1 Test Evaluation Summaries, 4.2 Reporting on Test Coverage)
**Iteration:** Master
**Version:** 1.0
**Date:** 2026-09-12
**Branch:** `test/config-testing`
**Status:** Implemented, executed and remediated. One command, `python tests/run_tests.py report`, now produces the evaluation summary and measured coverage for both tiers, gated by ratchet floors. 24 new cases (TD-01 to TD-17) guard the tooling. Three test-run defects were found and fixed (TEST-03 to TEST-05), and five observations are recorded (OBS-53 to OBS-57). Nothing in the pre-existing suite regressed.

---

## Scope note

The RUP template asks Section 4 to list "the various artifacts that will be created by the test effort that are useful deliverables to the various stakeholders", and names two of them:

> **Test Evaluation Summaries** — "Provide a brief outline of both the form and content of the test evaluation summaries, and indicate how frequently they will be produced."
>
> **Reporting on Test Coverage** — "Provide a brief outline of both the form and content of the reports used to measure the extent of testing, and indicate how frequently they will be produced. Give an indication as to the method and tools used to record, measure, and report on the extent of testing."

Section 4 is not a test technique, so this document does not follow the 3.1.x technique headings. It specifies each deliverable (form, content, frequency, tools), records how the tooling behind them was built and verified, and records what that work found.

**Deliverables that already exist and are not repeated here.** Each `tests/plans/3.1.x-*.md` document is itself a per-section evaluation record: its Execution Results, Findings and Coverage Assessment sections are the section-level evaluation summary. This section adds the cross-section, per-run summary and the code-coverage measurement that no document produced.

**Decisions taken at planning time:**
- **Installs.** `pytest-cov` (backend) and `@vitest/coverage-v8@3.2.7` (frontend, pinned to match vitest 3.2.7) were approved. Nothing else is installed; in particular pytest-html is not, so the summary is rendered from built-in junit XML.
- **Coverage threshold.** Floors ratchet from the measured baseline (`floor(measured) − 1`). The README's unmeasured ">85% line coverage" claim is replaced, not enforced.
- **Harness.** In-process and local only, like every other section: there is no CI in the repository, and none is added.

---

## The state this section found

`currentTest.md` recorded Section 4 as "Partial — reporting tooling claimed, coverage measurement absent". Planning found it was worse than partial:

- **`run_tests.py report` could not run.** It passed `--html` and `--self-contained-html`, which belong to pytest-html. That plugin is in no requirements file and not installed, so pytest stopped with a usage error, and the script printed "Test report generated" anyway (TEST-03).
- **The category map covered 15 of 51 test modules.** `TEST_CATEGORIES` drives `all`, `critical` and `report`, and it had never been updated past 3.1.7. It left out the whole 3.1.8 suite, fr7, fr10, the 3.1.1 store and artifact modules and more: 695 of the 995 backend cases (TEST-04).
- **No coverage tooling existed in either tier.** No pytest-cov or coverage in the virtualenv, no `.coveragerc`, no Vitest coverage provider and no `coverage` block in `vite.config.ts`.
- **The README's reporting claims were false.** ">85% line coverage" and "All React components tested" were stated as coverage and never measured. The report was said to include "Coverage analysis" and "Screenshots (for UI tests)"; it produced neither (OBS-53).
- **The README and PROJECT.md described deleted tests.** Both pointed at `test_performance_load.py`, deleted in 3.1.4. PROJECT.md said the frontend had "no configured test runner" (OBS-54).
- `Backend/tests/test_reports/`, where the report wrote, was not gitignored.

---

## 4.1 Test Evaluation Summaries

### Form

A single Markdown file, `Backend/tests/test_reports/evaluation-summary.md`. It is readable in the repository viewer and the IDE, and can be pasted into a section's Execution Results. It is rendered by `tests/_report.py` from machine-readable artifacts the runs leave behind:

| Input | Written by |
|---|---|
| `Backend/tests/test_reports/junit-backend.xml` | backend coverage run (`--junitxml`, built into pytest) |
| `Backend/tests/test_reports/junit-performance.xml` | backend timing run |
| `Backend/tests/test_reports/coverage.json` | pytest-cov |
| `Frontend/test-reports/junit.xml` | Vitest's built-in junit reporter |
| `Frontend/coverage/coverage-summary.json` | `@vitest/coverage-v8` |

### Content

1. **Verdict.** `PASS`, `FAIL` or `INCOMPLETE`. FAIL means a failure or error in any run, an unreadable artifact, or coverage under a floor. INCOMPLETE means everything present passed but a tier was not run.
2. **Header.** Date, git branch and commit, Python and Node versions, platform, and the exact commands that produce the inputs.
3. **Runs.** Each run's status (`ok`, `not run`, or `unreadable` with the parser's reason), when its artifact was produced, and its case counts. The timestamp makes a stale frontend artifact visible.
4. **Verdict by test-plan section.** Cases, passed, failed, errors, skipped and time for every section in both tiers. Backend modules map to sections through `TEST_CATEGORIES`; `configuration.test.tsx` is 3.1.8 and every other client module is 3.1.3. A module in no category gets a visible **Unmapped** row instead of being dropped.
5. **Failures and errors.** Every failed or errored case, with its node id and the first line of its message.
6. **Coverage.** The 4.2 figures against their floors, the ten lowest-covered backend modules, and the paths of both HTML reports.
7. **Defects.** A pointer to the Findings tables of the section documents. Defect status is not parsed.

### Frequency

| When | Required |
|---|---|
| Every `python tests/run_tests.py report` run | Produced automatically |
| At the close of each test-plan section | **Yes** — the section's Execution Results cite its verdict and case counts |
| At the end of each test cycle (iteration) | **Yes** — the final summary is the cycle's evaluation record |
| Before a test branch is merged to `main` | **Yes** |

### How to produce it

```bash
cd Frontend && npm run test:coverage            # optional: include the client tier
cd Backend  && python tests/run_tests.py report # backend runs + summary of both tiers
python tests/run_tests.py summary               # re-render from existing artifacts, running nothing
```

`report` deletes the previous backend artifacts first, so a summary can never mix an old backend run with a new one. It runs the backend suite twice, each time as a child `python -m pytest` process with `cwd=Backend` and `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` in its environment:

1. **Coverage run:** `-m "not performance" --cov …`. pytest-cov applies `fail_under`, so this run is the coverage gate.
2. **Timing run:** `-m performance`, with no tracer (Special Consideration 1).

It exits non-zero if either run fails, and prints each run's PASS or FAIL.

---

## 4.2 Reporting on Test Coverage

### Form

| Report | Form | Location |
|---|---|---|
| Coverage section of the evaluation summary | Markdown tables | `Backend/tests/test_reports/evaluation-summary.md` |
| Backend coverage | HTML, per file, with missed lines and partial branches | `Backend/tests/test_reports/coverage-html/index.html` |
| Frontend coverage | HTML, per file | `Frontend/coverage/index.html` |
| Terminal summaries | Text | pytest-cov's gate message; Vitest's `text-summary` |
| Functional (requirements) coverage | Requirements Traceability Matrix in each section document | `Backend/tests/plans/3.1.x-*.md` |

### Content

- **Backend:** one combined statement-and-branch percentage over `app/` (coverage.py's total with `branch = True`), lines and branches separately, the ten lowest-covered modules, and the floor with its verdict.
- **Frontend:** lines, statements, branches and functions over every file under `src/`, including files no test imports. Each is shown against its threshold. Line coverage is split between `src/components/ui/` (vendored shadcn primitives) and the rest of the application, so the vendored code cannot hide or inflate the application's own figure.
- **Functional coverage** is not measured by a tool. It stays where it already is: each section's traceability matrix maps SRS requirements and SAD use cases to cases.

### Method and tools

| Tier | Tool | Configuration |
|---|---|---|
| Backend | coverage.py 7.16.0 through pytest-cov 7.1.0 | `Backend/.coveragerc`: `source = app`, `branch = True`, `parallel = True`, `patch = subprocess`; `fail_under` with `precision = 2` so the gate compares the unrounded figure; `exclude_also` for `TYPE_CHECKING`, `__main__` guards and `raise NotImplementedError` |
| Frontend | `@vitest/coverage-v8` 3.2.7 (V8's native coverage) | `coverage` block in `Frontend/vite.config.ts`; `npm run test:coverage` |
| Both | `tests/_report.py` (stdlib only) | Reads the artifacts above |

Coverage is **opt-in**. It is not in `pytest.ini`'s `addopts` (guarded by TD-16), so a plain `pytest` run carries no tracer.

### Frequency

The same as 4.1. Coverage is produced by the same `report` run, and the floors are checked on every run.

### Floor (ratchet) policy

- Each floor is `floor(measured baseline) − 1`. The one-point margin absorbs small shifts, such as a case moving between runs, without letting coverage fall materially.
- Floors are **raised** when a section lands and the measured figure rises.
- Floors are **never lowered** without a reason recorded in the document of the section that lowers them.
- The floors live only in the configuration: `fail_under` in `Backend/.coveragerc`, and `coverage.thresholds` in `Frontend/vite.config.ts`. No document restates them, so none can drift from the gate. The measured figures and the floors are reported in every evaluation summary. TD-14 and TD-15 fail if a floor is zero, and TD-14 also fails if the README states a coverage percentage.

---

## Special Considerations

1. **The tracer and the timing oracles.** The 3.1.4 and 3.1.5 cases, and FO-92 in 3.1.7, assert absolute wall-clock budgets taken from SRS PE-1 to PE-3. Under the coverage tracer those budgets no longer measure the system, and FO-92 failed in the first coverage run because of it (TEST-05). The timing cases therefore run in a second, untraced run, and their own verdict stays valid. Backend coverage is measured without them. The code they exercise is also driven by the functional cases, so the effect on the figure is small.
2. **Subprocess measurement.** Four cases run application code in fresh `sys.executable` interpreters: `test_config_roles.py`, `test_config_hardware.py` and `test_fr10_api_contract.py`. `patch = subprocess` (coverage ≥ 7.10) measures those children. This was checked with `COVERAGE_DEBUG=process`, which recorded the four child interpreters, and no `.coverage.*` fragments are left after a run.
3. **No CI.** The repository has no pipeline, so the reports are produced locally. The tests README shows the steps a pipeline would run and the directories it would upload.
4. **What the backend figure is made of.** Three large modules are almost entirely unexecuted (OBS-55). The suite runs offline with model stand-ins, so real model loading is outside it by design; the legacy routes are switched off by default.
5. **Report output and the lint gate.** The frontend HTML report writes three JavaScript files into `Frontend/coverage/`, and `eslint .` linted them: 116 problems against the 113 baseline. `eslint.config.js` now ignores `coverage` and `test-reports` (TD-17), and the count is back to 113.
6. **Timing runs need an idle machine.** The timing run's budgets are absolute. A timing run that overlaps other CPU-heavy work, such as a lint run or the frontend suite, can fail for reasons outside the system. Run `report` on its own.
7. **Install record.** `pytest-cov==7.1.0` (pulling `coverage==7.16.0`) went into `Backend/.venv` after a `pip --dry-run` confirmed that pytest 8.2.0 and every other pin stayed unchanged. It is pinned in `requirements-dev.txt`. `@vitest/coverage-v8@3.2.7` was added to `devDependencies`. Nothing else was installed.

---

## Test Case Matrix

`Backend/tests/test_deliverables_reporting.py`, `pytestmark = pytest.mark.important`, category `deliverables` (Section 4). No `conftest.py` changes: every fake is a per-case fixture, and no case touches the real `test_reports/` directory.

| ID | Case | Oracle | Guards | Pre-fix |
|---|---|---|---|---|
| **A. Category map** | | | | |
| TD-01 | Every `tests/test_*.py` is in exactly one category | Set comparison with the directory | TEST-04 | Fails: 36 modules in no category |
| TD-02 | Every listed module exists | Filesystem | — | Passes (ch) |
| TD-03 | Each category names its section; together they cover 3.1.1, 3.1.2, 3.1.4 to 3.1.8 | `section` key and description | TEST-04 | Fails: no `section` key |
| **B. Report command** (`subprocess.run` and `write_summary` faked) | | | | |
| TD-04 | Only installed plugins; `--junitxml` on both runs, `--cov` on the first; `cwd=Backend`; offline Hugging Face variables in the child environment | argv, cwd and env of each call | TEST-03 | Fails: argv carried `--html` |
| TD-05 | Coverage run deselects `performance`; timing run selects it and has no `--cov` | argv | TEST-05 | Fails: one run, no split |
| TD-06 | A failing child makes the command return failure and print `FAIL (exit n)` | Return value and stdout | TEST-03 | Fails: "generated" printed regardless |
| **C. Summary generator** (synthetic artifacts in `tmp_path`) | | | | |
| TD-07 | junit totals, node ids, module-level cases and collection errors | Hand-built pytest junit | — | New instrument |
| TD-08 | Section rollup in both tiers; an uncategorised module appears as **Unmapped** | Rows by tier and section | — | New instrument |
| TD-09 | Failures listed with node id and first message line, pipes escaped | Rendered Markdown | — | New instrument |
| TD-10 ×3 | Backend total against floors below, at and above it; lowest-module order; zero-statement files excluded | Rendered verdict | — | New instrument |
| TD-11 | Missing frontend artifacts read as "not run", verdict INCOMPLETE | Run status and verdict | — | New instrument |
| TD-12 ×2 | Frontend metrics against thresholds, met and missed; `ui/` and application split | Parsed json-summary | — | New instrument |
| TD-13 | Truncated junit and half-written coverage JSON are reported, not raised | Status `unreadable`, verdict FAIL | — | New instrument |
| TD-13b | `write_summary` writes the rendered file | File contents | — | New instrument |
| **D. Claims match gates** | | | | |
| TD-14 | `.coveragerc` `fail_under` is non-zero and compared at `precision` ≥ 2; the README states no coverage percentage | Config and README text | OBS-53 | Fails: no `.coveragerc`; README said >85% |
| TD-15 ×4 | Each `vite.config.ts` threshold (lines, statements, branches, functions) is non-zero | Config | OBS-53 | Fails: no thresholds |
| TD-16 | `pytest.ini` `addopts` carries no `--cov` | Config | Consideration 1 | Passes (ch) |
| TD-17 | Report outputs are gitignored in both tiers, every output path lies under an ignored directory, and eslint ignores the frontend outputs | `.gitignore` files, argv, config, `eslint.config.js` | — | Fails: `test_reports/` not ignored |

`ch` = characterisation: holds before and after, pinned so it cannot silently change. The B cases were not executed against the old `run_tests.py`, whose `report` called `pytest.main` in-process and would have started a full nested run. Their pre-fix column follows from the old argv and control flow, quoted in the Findings.

---

## Execution Results

Executed on 2026-09-12 on Windows 11 with Python 3.11.9 and Node 22.14.0. Every backend run used `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

| Module | File | Cases | Result |
|---|---|---|---|
| TD | `test_deliverables_reporting.py` | 24 (17 IDs; TD-10 ×3, TD-12 ×2, TD-15 ×4, TD-13b) | 24 passed (0.5 s) |

Before the configuration was finalised, TD-14 and TD-15 ×4 failed as designed while the floors were still 0 and the README still said ">85%". They passed once the floors were set and the README rewritten.

**Report runs.** Three backend `report` runs were made, in this order:

| Run | Coverage run | Timing run | Verdict | Notes |
|---|---|---|---|---|
| 1 — baseline (floor 0) | 924 passed, **1 failed**, 1 skipped (304.5 s) | 69 passed | FAIL | FO-92 failed under the tracer: TEST-05. The baseline the floors were set from was measured here |
| 2 — floors set | 947 passed, **1 failed**, 1 skipped | 68 passed, **2 failed** | FAIL | CF-51 failed because this section's README rewrite dropped the sample CI's `python-version: '3.11'` (restored; 3.1.8's guard worked as intended). The two LT failures in `test_load_background_workload.py` coincided with a lint run started during the timing run (Special Consideration 6) |
| 3 — final, machine idle | **948 passed, 1 skipped** (141.1 s) | **70 passed** (87.4 s) | **PASS** | 949 + 70 = 1,019 cases: the 995 pre-existing plus 24 TD |

The final summary, as rendered (header omitted):

| Tier | Section | Cases | Passed | Failed | Errors | Skipped |
|---|---|---|---|---|---|---|
| Backend | Data and Database Integrity Testing (3.1.1) | 84 | 84 | 0 | 0 | 0 |
| Backend | Function Testing (3.1.2) | 422 | 421 | 0 | 0 | 1 |
| Backend | Performance Profiling (3.1.4) | 48 | 48 | 0 | 0 | 0 |
| Backend | Load Testing (3.1.5) | 21 | 21 | 0 | 0 | 0 |
| Backend | Security and Access Control Testing (3.1.6) | 36 | 36 | 0 | 0 | 0 |
| Backend | Failover and Recovery Testing (3.1.7) | 191 | 191 | 0 | 0 | 0 |
| Backend | Configuration Testing (3.1.8) | 193 | 193 | 0 | 0 | 0 |
| Backend | Test Deliverables Tooling (4) | 24 | 24 | 0 | 0 | 0 |
| Frontend | User Interface Testing (3.1.3) | 129 | 129 | 0 | 0 | 0 |
| Frontend | Configuration Testing (3.1.8) | 17 | 17 | 0 | 0 | 0 |

The one skip is `test_fr7_acceptance.py`, a real-model acceptance test that runs only with `ECHO_MODEL_TESTS=1`.

**Coverage.** In run 3, both tiers met every floor: pytest-cov reported "Required test coverage … reached", and Vitest's thresholds passed with exit 0. The figures themselves are reported in the evaluation summary each run produces; they are not restated in this document.

**The gates fail when they should.** Each gate was checked against the measured totals:

| Check | Result |
|---|---|
| Backend floor set 0.01 above the measured total (`coverage report --fail-under`) | exit 2 |
| Backend floor set equal to the measured total | exit 0 |
| Coverage run over one small module (`pytest tests/test_results_cache.py --cov`), all tests passing | "FAIL Required test coverage … not reached", exit 1 |
| Frontend lines threshold set above the measured figure (`vitest run --coverage --coverage.thresholds.lines=…`) | "ERROR: Coverage for lines … does not meet global threshold", exit 1 |

The first attempt at this check found that a backend floor set just above the measured total **passed**. coverage.py compares the total after rounding it to `[report] precision`, which defaults to 0, so any total within half a point below the floor rounded up to it. `.coveragerc` now sets `precision = 2`, and TD-14 requires it. The frontend thresholds compare unrounded percentages.

**Subprocess measurement.** `COVERAGE_DEBUG=process` over `test_config_roles.py` recorded the pytest process and four child interpreters (`import app.main` ×2, the Celery loader, the worker import closure). No `.coverage.*` fragment remained after any run.

**Backend regression and order independence** (no coverage):

```
$ ./.venv/Scripts/python.exe -m pytest tests/ -q                           → 1017 passed, 1 failed, 1 skipped (210.86 s)   LT sustained peak
$ ./.venv/Scripts/python.exe -m pytest tests/ -q                           → 1017 passed, 1 failed, 1 skipped (256.38 s)   LT cold dataset scan
$ ./.venv/Scripts/python.exe -m pytest $(ls tests/test_*.py | sort -r) -q  → 1018 passed, 1 skipped (219.97 s)
$ ./.venv/Scripts/python.exe -m pytest tests/ -q                           → 1018 passed, 1 skipped (233.91 s)
$ ./.venv/Scripts/python.exe -m pytest tests/ -q -p no:cov                → 1018 passed, 1 skipped (196.87 s)
```

- 1018 = 994 pre-existing passes + the 24 TD cases. The skip is the same FR-7 acceptance case.
- **The two forward failures were investigated, not waved through (OBS-57).** Each run failed a *different* 3.1.5 case, and both are relative-latency oracles: late vs. early p50 under sustained peak, and loaded vs. baseline p50 during a background scan. The evidence that neither is a regression from this section:
  - both modules pass alone, twice (9 passed);
  - they pass twice when the new TD module runs first, the only ordering this section added (33 passed);
  - they pass in two further forward runs, one with pytest-cov loaded and one with it disabled;
  - the coverage install's `a1_coverage.pth` activates only when `COVERAGE_PROCESS_START` or `COVERAGE_PROCESS_CONFIG` is set, and a plain interpreter has no tracer;
  - no application or 3.1.5 code changed.
- No pre-existing test's assertions were modified. FO-92 gained a marker (TEST-05); nothing else in an existing test file changed.

**Client regression.**

```
$ npm run test:coverage                     → 11 files, 146 passed, thresholds met, exit 0
$ npm test                                  → 11 files, 146 passed, exit 0
$ npm run lint                              → 113 problems (baseline 113; 116 before eslint ignored coverage/)
$ npx tsc --noEmit -p tsconfig.app.json     → 6 errors (baseline 6; OBS-18)
```

**Isolation.** `find shared-storage uploads -newer <regression start> -type f` returned nothing (both tiers' paths checked). `git check-ignore` confirms that `tests/test_reports/`, `.coverage`, `Frontend/coverage/` and `Frontend/test-reports/` are ignored, and `git status` lists only the intended changes.

---

## Findings

### Test-run defects

| ID | Status | Location | Record |
|---|---|---|---|
| **TEST-03** | **Fixed** | `tests/run_tests.py` (formerly 186-213) | **The documented report command could not produce a report, and said it had.** `generate_test_report` passed `--html=…/test_report.html --self-contained-html`, options of pytest-html, which is not installed and in no requirements file. pytest stopped with a usage error, and line 212 then printed `Test report generated: …` whatever the result. Every README and CI example that relied on it depended on a file that was never written. **Resolution:** `report` runs two child pytest processes that use only built-in `--junitxml` and the approved pytest-cov, then renders the summary with `tests/_report.py`. It reports each run's result truthfully and exits non-zero on failure. `summary` re-renders without running anything. Guarded by TD-04 and TD-06. |
| **TEST-04** | **Fixed** | `tests/run_tests.py` `TEST_CATEGORIES` | **`all`, `critical` and `report` silently skipped 695 of 995 backend cases.** The map listed 15 of the 51 modules; nothing after the first version of each section had been added to it. It left out the whole of 3.1.8, fr7, fr10, the 3.1.1 store and artifact modules, `test_session_cookie.py` and 20 more. Anyone running the documented `python tests/run_tests.py` saw a PASS for under a third of the suite. **Resolution:** every module is in exactly one category, following its Test Case Matrix; `configuration` (3.1.8) and `deliverables` (4) are new categories, and each category carries its `section`. Guarded by TD-01 to TD-03. The unused `estimated_time` values (e.g. "30 minutes" for a section that runs in under three) were dropped. |
| **TEST-05** | **Fixed** | `test_failover_under_load.py`, FO-92 | **A wall-clock assertion outside the performance marker.** FO-92 asserts PE-1's 500 ms mean for submit and poll while half the workers die (line 151), but carried only the `failover` and `slow` markers. In the first coverage run it failed: mean submit latency was 523 ms under the tracer (`assert (29.305128000021796 / 56) < 0.5`). It is the only case outside the 3.1.4 and 3.1.5 modules that asserts a wall-clock budget (checked by searching every other module for timing assertions). **Resolution:** FO-92 is also marked `performance`, so it runs in the untraced timing run; its assertions are unchanged and it still belongs to 3.1.7. Found by the suite, not predicted. |

### Recorded

| ID | Status | Location | Record |
|---|---|---|---|
| **OBS-53** | **Resolved** | `Backend/tests/README.md` | **The coverage and reporting claims were stated, not measured.** ">85% line coverage", "All React components tested", and reports with "Coverage analysis" and "Screenshots (for UI tests)". Measured, both tiers are well below that claim. **Resolution:** the claims are removed. The README's Test Coverage section now says where the floors are enforced and where the measured figures are reported, and its reporting section lists what is actually produced. TD-14 fails if the README states a coverage percentage again. |
| **OBS-54** | **Fixed** | `Backend/tests/README.md`, `PROJECT.md` §10 | **The test documentation described deleted tests.** Both documents pointed at `test_performance_load.py` (deleted in 3.1.4). The README listed performance thresholds ("UI Response ≤100ms", "Memory Usage ≤2GB") that no case asserts, and told users to install Jest. PROJECT.md said the frontend had "no configured test runner" and pointed at the deleted `ui-components.test.tsx`. **Resolution:** corrected in both. The README points to the SRS budgets the performance cases actually assert. |
| **OBS-55** | Recorded | `api/routes/inferences.py`, `services/model_loader_service.py`, `services/jacobian_lens_service.py` | **Three large modules are almost entirely unexecuted:** `inferences.py`, `model_loader_service.py` and `jacobian_lens_service.py`. Together they hold over a fifth of the backend's statements. `inferences.py` is a legacy synchronous route mounted only when `ENABLE_LEGACY_SYNC_INFERENCE` is on, which the API image cannot do (OBS-47). The other two load and run real model weights, which an offline suite with stand-ins reaches only up to the adapter boundary. They set the ceiling of the backend figure; they are listed in every summary's lowest-modules table. |
| **OBS-56** | Recorded | `Frontend/vite.config.ts` coverage | **Most of the client's lines run under no test.** The figure counts every file under `src/`, including files no test imports. The vendored shadcn primitives in `src/components/ui/` are the least covered; the application's own code is covered better, but many panels are never rendered by a test. The summary reports the two separately. 3.1.3 targeted user-visible behaviour, not line counts. The floor now stops the figure falling; raising it is a matter for later UI work. |
| **OBS-57** | Recorded | `test_load_workload_profiles.py` (sustained peak), `test_load_background_workload.py` | **Two 3.1.5 relative-latency oracles fail intermittently in a long full run.** In 2 of 4 forward runs of the full suite, one of them failed. Sustained peak: late p50 239 ms against early 62 ms. Cold dataset scan: loaded against baseline. Each passes alone, after the TD module, and in later full runs (see Execution Results). They compare two windows of the same run, so a burst of unrelated activity on the host during one window fails them. Not changed here: the oracles belong to 3.1.5, and loosening them without a measured cause would hide real regressions. If they recur, the next step is to record the host load during the run. |

---

## Coverage Assessment and Residual Gaps

**Assessment: both deliverables the template names are specified, produced and verified.** Before this section the repository had one report command that could not run, a category map that hid 695 of the 995 backend cases, and coverage claims no tool had measured. Now one command produces a per-section, both-tier evaluation summary and measured coverage, and the coverage gate fails the run when a floor is not met.

The findings share one pattern, the same as 3.1.8's: **tooling that looked in place but was not in force.** The report command depended on an absent plugin and reported success regardless (TEST-03). The runner's map was frozen at the first version of each section (TEST-04). The README's coverage figures were goals written as facts (OBS-53). Every one is now either removed or guarded by a case that fails if it drifts back.

The coverage run itself found one defect that was not predicted: FO-92's latency budget had never been classified as a timing oracle (TEST-05).

Deliberately excluded, with justification:

| Excluded | Reason | Owner |
|---|---|---|
| A CI pipeline | The repository has none; the reports are produced locally (Consideration 3) | — |
| An HTML test-results report (pytest-html) | Not installed, by decision at planning time; the Markdown summary and coverage's own HTML replace it | — |
| Section 5 (risks, dependencies, assumptions, constraints) | Its own section | **5** |

---

## References

1. Rational Unified Process — *Test Plan* template, Section 4 Deliverables (4.1 Test Evaluation Summaries, 4.2 Reporting on Test Coverage). IBM Rational Software.
2. *AudioLens Software Requirements Specification*, Group 16, §3.4 PE-1 to PE-3 (the budgets behind Special Consideration 1). `docs/Group 16 - AudioLens SRS.pdf`.
3. coverage.py documentation — configuration reference (`[run] patch = subprocess`, `branch`, `parallel`, `[report] fail_under`, `exclude_also`), available at https://coverage.readthedocs.io/ (Accessed 2026-09-12).
4. pytest-cov documentation — `--cov`, `--cov-report`, `fail_under` handling, available at https://pytest-cov.readthedocs.io/ (Accessed 2026-09-12).
5. Vitest documentation — coverage (`provider: 'v8'`, `thresholds`, reporters) and the junit reporter, available at https://vitest.dev/guide/coverage (Accessed 2026-09-12).
6. `currentTest.md` (repository root) — the original gap analysis this section supersedes.
