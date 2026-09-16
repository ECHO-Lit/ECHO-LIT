# 5 Risks, Dependencies, Assumptions, and Constraints

**Project:** ECHO (Explainable Computation for Hearing Outputs) — Learning Interpretability Tool for audio models
**Test Plan section:** 5 Risks, Dependencies, Assumptions, and Constraints
**Iteration:** Master
**Version:** 1.0
**Date:** 2026-09-14
**Branch:** `test/fixes-and-deliverables`
**Status:** Implemented, executed and remediated. The four tables the template asks for are written below: 14 ranked risks, 11 dependencies, 10 assumptions and 10 constraints. Every mitigation the register calls in force, and every assumption it calls proven, is checked by a case in `test_risk_register.py` (RD-01 to RD-15). Writing the register found that three of the safeguards it needed were not in force; those test-run defects are fixed (TEST-06 to TEST-08), and three observations are recorded (OBS-58 to OBS-60). Nothing in the pre-existing suite regressed.

---

## Scope note

The RUP template asks Section 5 for four lists:

> **Risks** — "List any risks that may affect the successful execution of this Test Plan, and identify mitigation and contingency strategies for each risk. Also indicate a relative ranking for both the likelihood of occurrence and the impact if the risk is realized."
>
> **Dependencies** — "List any dependencies identified during the development of this Test Plan that may affect its successful execution if those dependencies are not honored."
>
> **Assumptions** — "List any assumptions made during the development of this Test Plan that may affect its successful execution if those assumptions are proven incorrect."
>
> **Constraints** — "List any constraints placed on the test effort that have had a negative effect on the way in which this Test Plan has been approached."

Section 5 is not a test technique, so this document does not follow the 3.1.x technique headings. It gives the four tables (5.1 to 5.4), records how the safeguards they name were checked, and records what that work found.

**Owners.** **Member 1** is the author of this test effort (Sections 3.1.1 to 3.1.5, 3.1.7, 3.1.8, 4 and 5). **Member 2** owns Section 3.1.6. A row that needs a joint or product decision lists both.

**Ranking.** Likelihood (L) and impact (I) are each High, Medium or Low (3, 2, 1). Likelihood is judged **before** mitigation, so a well-mitigated risk still shows how often it would bite without its mitigation. Risks are ranked by L × I, and ties are broken on impact.

**What "in force" means here.** A mitigation or assumption is in force only when a case fails if it stops holding. Each risk and assumption row names that case (RD-nn here, or a case from an earlier section), or says **Procedural** when the safeguard is a practice rather than a check. RD-14 fails if a row names neither.

**Decisions taken at planning time:**
- **Scope.** The register, plus guards for it, plus fixes for what the guards found.
- **Offline mode.** Enforced by `os.environ.setdefault` at the top of `tests/conftest.py`. This is the one approved change to the conftest. An explicit `HF_HUB_OFFLINE=0` still opts out, for the real-model cases.
- **Standing rules.** No downloads or installs; no git operations; no coverage figures in documents; `test_security.py` belongs to Member 2 and is not touched. Requirements not yet implemented get their own pass and are not named here.

---

## The state this section found

`currentTest.md` recorded Section 5 as missing: no Risk / Mitigation / Contingency register, and only the informal Troubleshooting part of `Backend/tests/README.md`. Writing the register showed it could not be written honestly yet, because several of the safeguards it would name were not in force:

- **A clean checkout failed instead of skipping (TEST-06).** `Backend/data/` holds 147 MB of corpora and none of it is tracked. Seven cases read the SAA or Common Voice corpus with no guard, and failed on a clean checkout. Two more guards were too narrow. One further case passed only by accident. `3.1.2-function-testing.md` said dataset-dependent modules "skip rather than error".
- **A wall-clock budget ran under the coverage tracer (TEST-07).** FO-31 asserts PE-1's 500 ms budget and had no `performance` marker, so it ran in the coverage run. Section 4 had said FO-92 was "the only case" of that kind.
- **Offline mode belonged to one command (TEST-08).** Only `run_tests.py report` set `HF_HUB_OFFLINE`. A plain `pytest tests/` was online, and so were `all`, `critical` and every category command. The Whisper agreement case checked only that the processor was cached, so it could download weights.
- **The README's Troubleshooting was wrong (OBS-58).** It advised a pytest flag that does not exist (`--ignore-missing-models`), described memory limits and containers the suite does not have, and gave per-category times of 10 to 30 minutes for a suite that runs in about five in total.
- **`currentTest.md` misquoted `test_security.py` (OBS-59).** It said the file declares system-level security out of scope. The file has no scope note.

---

## 5.1 Risks

Ranked by L × I, ties broken on impact.

| ID | Risk | L | I | Mitigation strategy | Contingency (risk is realized) | Owners |
|---|---|---|---|---|---|---|
| **R-01** | Model behaviour is checked only through stand-ins, so a model or transformers change that breaks real inference still passes (OBS-55) | H | H | Adapter-boundary cases; the real Whisper agreement case when the model is cached (OBS-60); the FR-7 acceptance case FT-104, opt-in; drift held to the recorded set by RD-10 | Run `ECHO_MODEL_TESTS=1 HF_HUB_OFFLINE=0` on a host that has the models; record the result in the owning section | Member 1, Member 2 |
| **R-02** | A run reaches the network or downloads weights (precedent: TEST-01) | M | H | Offline by default in the conftest (TEST-08); RD-01 to RD-03; child processes inherit the environment | Stop the run, delete the partial snapshot from the Hugging Face cache, rerun offline, record it | Member 1 |
| **R-03** | A tooling upgrade breaks the harness: httpx 0.28 drops `AsyncClient(app=)`, vitest 4 needs Vite 6, jsdom 30 needs Node 22.22 | M | H | Exact pins checked against the venv (RD-09); client ceilings checked against `package.json` and the lockfile (RD-11); lockfile committed | `pip install -r requirements-dev.txt`, `npm ci` | Member 1 |
| **R-04** | The corpora are absent on a clean clone | H | M | Every corpus-reading case carries `requires_corpora` (TEST-06); RD-04 runs them with the corpora hidden; RD-05 keeps the hiding complete | Obtain the corpora under licence into the `dataset_service.py` layout; otherwise report the skipped cases as unverified | Member 1 |
| **R-05** | The venv has drifted from the images (OBS-43). Already realized | H | M | RD-10 fails on any drift beyond the recorded set; S3 and the models are reached only through stand-ins | Rebuild the venv once downloads are approved, rerun, and shrink the recorded set | Member 1, Member 2 |
| **R-06** | There is no pipeline, so the suite runs only when someone runs it | H | M | `run_tests.py report` produces a truthful verdict (TD-04 to TD-06); the Section 4 rule that a report precedes every merge | Run the report before each merge; bisect a failure by category | Member 1, Member 2 |
| **R-07** | A timing oracle fails for a reason outside the system: the tracer, or host load (OBS-57) | M | M | `performance` marker on every clock oracle (TEST-05, TEST-07, RD-07); an untraced timing run (TD-05); an idle host | Rerun the module alone and record the host load; never loosen an oracle without a measured cause | Member 1 |
| **R-08** | The suite executes on Windows only, while the images are Linux | M | M | Static Dockerfile and compose cases and role probes (CF-49, CF-60 to CF-64); the documented Python version matches the images (CF-51) | Run the affected modules in a Linux container with the tests mounted | Member 1 |
| **R-09** | The 3.1.6 verdict rests on Member 2's modules, and most of `test_security.py` passes whatever the application does (OBS-07, OBS-61) | M | M | The security modules are in the `security` category and therefore in every run (TD-01); `3.1.6-security-testing.md` classifies every security case as evidence or not; `test_access_control.py` pairs each denial with the owner's success (SEC-01, SEC-09); the record is corrected (OBS-59) | Member 2 rewrites or removes the vacuous cases | Member 2 |
| **R-10** | Capacity and the academic timeframe (SAD §3) leave unowned passes unrun | M | M | Procedural: this ranked register; an owner on every residual gap; RD-14 keeps the register and its guards in step | Descope with a recorded reason | Member 1, Member 2 |
| **R-11** | fakeredis diverges from Redis 7 on something the app relies on | L | H | RD-08: the app uses no Lua, function or lock call, and the fake reports Redis 7; three logical databases in the conftest; WATCH verified in 3.1.1 and 3.1.5 (DI, LT cases) | Reproduce on the compose Redis 7 | Member 1 |
| **R-12** | The documents claim what the suite does not do | H | L | RD-12 to RD-14; TD-14; CF-50, CF-51 | Correct the document and add a guard | Member 1 |
| **R-13** | Test traffic reaches a developer's Redis or broker: an unpatched `send_task` goes to `localhost:6379/2` | L | M | Procedural: every job-submitting module patches `send_task` with a per-module `fake_broker` | Purge DB2 and the queue; fix the module | Member 1 |
| **R-14** | The suite writes into the repository tree | L | M | Procedural: per-module isolation fixtures; the `find … -newer` check after every regression | Delete the files; fix the module | Member 1 |

## 5.2 Dependencies

| Dependency between | Potential impact of dependency | Owners |
|---|---|---|
| **D-01** — the corpus cases ↔ the `Backend/data/` corpora (research licences) | Without them, 40 cases skip (see Execution Results) | Member 1 |
| **D-02** — the real-model cases ↔ the local Hugging Face cache | The Whisper agreement case skips; FR-7 acceptance needs downloads | Member 1 |
| **D-03** — the harness ↔ pytest 8.2.0, pytest-asyncio 0.23.7, pytest-cov 7.1.0, httpx 0.27.0, fakeredis 2.23.2, anyio 4.4.0 | Wholesale errors | Member 1 |
| **D-04** — the client tier ↔ the lockfile, vitest and @vitest/coverage-v8 3.2.7, jsdom 26.1.0, vite 5.4.20; Node 22.14 locally against `node:20-alpine` in the image | The client suite is blocked | Member 1 |
| **D-05** — the Section 4 summary ↔ `TEST_CATEGORIES` (TD-01) | Unmapped rows in the summary | Member 1 (Member 2 for the security modules) |
| **D-06** — the security verdict ↔ Member 2's 3.1.6 modules (OBS-07, OBS-61) | Vacuous cases counted as passes unless read with the classification in `3.1.6-security-testing.md` | Member 2 |
| **D-07** — the characterisation oracles ↔ the open decisions OBS-01, 12, 13, 20, 32, 37, 44, 46 | Oracles change when those are decided | Member 1, Member 2 |
| **D-08** — fixing OBS-43 ↔ permission to download | The drift persists | Member 1, Member 2 |
| **D-09** — the timing verdict ↔ an idle host | False FAIL | Member 1 |
| **D-10** — regression judgement ↔ the lint (113) and tsc (6) baselines (OBS-18) | Regressions hidden in the noise | Member 1 |
| **D-11** — real-browser, browser-matrix and screenshot passes ↔ an owner being assigned | They stay unrun | Member 1, Member 2 |

## 5.3 Assumptions

| Assumption to be proven | Impact of assumption being incorrect | Owners |
|---|---|---|
| **A-01** — No run needs the network (RD-01 to RD-03) | Downloads; nondeterministic results | Member 1 |
| **A-02** — fakeredis models every Redis feature the app uses (RD-08) | The integrity and failover verdicts are unsound | Member 1 |
| **A-03** — A clean checkout skips and never fails (RD-04, RD-05) | The suite reads as broken | Member 1 |
| **A-04** — Only `performance` cases assert timing budgets (RD-07) | The traced run's verdict depends on host speed | Member 1 |
| **A-05** — Drift is limited to {transformers, boto3}, and both are reached only through stand-ins (RD-10) | The suite passes a configuration no image builds | Member 1 |
| **A-06** — The caller's shell sets no `Settings` key (RD-15) | Silent reconfiguration of every case that reads `settings` | Member 1 |
| **A-07** — The suite is order-independent and leaves the tree clean (Procedural: forward twice, reverse, `find` after each section) | Verdicts depend on order | Member 1 |
| **A-08** — The stand-ins honour the real adapter contract. Partly shown: the Whisper agreement case (OBS-60) and FR-7 acceptance FT-104, opt-in | Real regressions are invisible | Member 1, Member 2 |
| **A-09** — Windows results transfer to Linux for platform-independent code. Shown statically for the images (CF-49, CF-51); otherwise unproven | POSIX-only defects go unseen | Member 1 |
| **A-10** — Relative-latency oracles are stable on an idle host (LT-03, LT-12; OBS-57). Evidence only | Flaky FAIL | Member 1 |

## 5.4 Constraints

| Constraint on | Impact constraint has on test effort | Owners |
|---|---|---|
| **C-01** — No downloads or installs | The drift cannot be fixed; real models only from the local cache | Member 1, Member 2 |
| **C-02** — Corpus licences; SAVEE is not redistributable (`scripts/prepare_savee_subset.py:3`) | The corpora are never committed (RD-06); fixtures are synthetic | Member 1 |
| **C-03** — A CPU-only host of about 16 GB (SRS §3.9.2) | GPU paths are reached only through device stand-ins | Member 1 |
| **C-04** — One OS and one interpreter (Windows 11, Python 3.11.9; DC-1) | POSIX-only branches are never executed | Member 1 |
| **C-05** — No pipeline | Local runs and local artifacts only (Section 4 SC-3) | Member 1, Member 2 |
| **C-06** — Absolute PE-1 to PE-3 budgets | A separate untraced run on an idle machine is needed | Member 1 |
| **C-07** — Tooling ceilings: vitest 3, jsdom 26, jest-dom 6, httpx below 0.28 (RD-09, RD-11) | Upgrades are blocked | Member 1 |
| **C-08** — In-process ASGI and jsdom, against the browser list of SRS §3.9.3 | Real-browser behaviour is outside the automated tiers | Member 1, Member 2 |
| **C-09** — The conftest is frozen, with one approved change | Isolation is per module | Member 1 |
| **C-10** — Three members and an academic timeframe (SAD §3); Member 2 owns 3.1.6 | Scope and ownership are split | Member 1, Member 2 |

---

## Special Considerations

1. **Offline mode has to be set before the first import.** huggingface_hub reads `HF_HUB_OFFLINE` once, when it is imported, as `HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE` over the raw strings. So the `setdefault`s sit above every import in `conftest.py` that can reach the hub (RD-03). Nothing imports the hub earlier: the only installed pytest plugins are anyio, pytest-asyncio and pytest-cov, and `tests/__init__.py` is empty. Because the check is on strings, `HF_HUB_OFFLINE=0` alone opts out even while `TRANSFORMERS_OFFLINE` defaults to 1, and `TRANSFORMERS_OFFLINE=0` alone does not. transformers 5 takes its offline flag from the hub, so the same opt-out covers it.
2. **Hiding corpora without touching them.** `tests/_corpora.py`, loaded with `-p tests._corpora`, repoints every corpus path the app holds at a temporary directory that is never created: the `dataset_service` registry and `DATA_DIR`, the L2-ARCTIC annotations path (and its `lru_cache`), and the legacy routes' paths when they are imported. `Backend/data/` is never read, moved or written. `ECHO_KEEP_CORPORA` keeps named corpora visible, which is how the partial-checkout runs below were made. The module imports nothing from `app` at load time, because `-p` loads it before the conftest sets offline mode.
3. **The RD-07 analysis is a heuristic, with controls.** It follows clock-derived values through assignment, `for`, `with` and `list.append` inside each test function, including nested functions. It does not follow values into or out of other functions. Two positive controls fix what it must see: FO-92, which derives its samples from `.ended - .started`, and a `test_perf_event_loop.py` case, which reads the `loop_lag_probe` helper. Equality checks on timestamps are not flagged; ordered comparisons and `pytest.approx` are.
4. **Two cases run a child pytest.** RD-02 runs RD-01 in a fresh interpreter with the offline variables stripped; RD-04 runs the corpus-reading modules with the corpora hidden. Both strip `COVERAGE_PROCESS_*` so that a coverage run's subprocess patch does not trace them, and `PYTEST_ADDOPTS`. RD-04 is marked `slow` and takes about a minute.
5. **Where the register comes from.** The risks are the ones the earlier sections met or recorded (TEST-01, OBS-43, OBS-55, OBS-57 and others), not a generic list. Likelihood is judged from this repository's own history: R-04 and R-05 are High because they have already happened.

---

## Test Case Matrix

`Backend/tests/test_risk_register.py`, `pytestmark = pytest.mark.important`, category `risks` (Section 5). No conftest fixture is requested, and each docstring gives the case's RD id and what it guards. The instrument is `tests/_corpora.py`.

| ID | Case | Oracle | Guards | Pre-fix |
|---|---|---|---|---|
| **A. Offline** | | | | |
| RD-01 | Offline mode is active in-process | Both variables truthy; `huggingface_hub.constants.HF_HUB_OFFLINE is True`; `is_offline_mode()`. Skips only for `ECHO_MODEL_TESTS` with `HF_HUB_OFFLINE=0` | TEST-08 | Passes with the shell prefix; fails without it (`'' in {'1', 'ON', 'TRUE', 'YES'}`) |
| RD-02 | Offline mode does not depend on the caller | A child pytest runs RD-01 with the offline, `ECHO_MODEL_TESTS`, `COVERAGE_PROCESS_*` and `PYTEST_ADDOPTS` variables stripped: exit 0, "1 passed" | TEST-08 | Fails: the child's RD-01 fails |
| RD-03 | The conftest sets offline mode before any hub-capable import | AST of `conftest.py`: both `setdefault`s precede the first import of `app`, `transformers`, `huggingface_hub` or `torch`; `tests/__init__.py` imports none of them | TEST-08 | Fails: "conftest.py does not default HF_HUB_OFFLINE" |
| **B. Corpora** | | | | |
| RD-04 (`slow`) | A clean checkout skips and never fails | A child run with `-p tests._corpora -m "not performance"` over every module a static pattern selects as a corpus reader (nine when this section ran; ten after the 3.1.6 merge added `test_access_control.py`, whose SEC-17 skips with its own `Backend/data` reason). From its junit XML: no failure or error; every TEST-06 case skipped; every skip reason names `Backend/data` | TEST-06 | Fails: 7 failed (`FileNotFoundError: Dataset metadata not found for: saa`) |
| RD-05 | The hiding plugin covers every corpus path in `app/` | Every `app/` file with `parents[N] / "data"` or `Path("data")` has a disposition in `_corpora.HIDDEN`, and no other file does; after `hide()` (through monkeypatch), every Path-valued global of the repointed modules is outside `Backend/data/` | TEST-06 | New instrument |
| RD-06 | The corpora are never committed | `Backend/.gitignore` has `/data`; `git ls-files -- data` is empty | C-02 | Passes (ch) |
| **C. Timing** | | | | |
| RD-07 | Every clock-derived assertion is a `performance` case | Data flow per test function (Special Consideration 3); markers read at module, class and function level. Positive controls: FO-92 and a `test_perf_event_loop.py` case are found as marked oracles. The allowlist is empty | TEST-07 | Fails: flags FO-31 (line 507) |
| **D. Redis** | | | | |
| RD-08 | The app uses nothing fakeredis cannot model | AST of `app/`: no `register_script`, `script_load`, `evalsha`, `fcall`, `fcall_ro` or `lock` call, and no `eval` with two or more arguments (torch's `.eval()` has none); `FakeServer().version >= (7,)` (SRS §3.10) | A-02 | Passes (ch) |
| **E. Environment** | | | | |
| RD-09 | The harness pins are exact and installed | `requirements-dev.txt` pins pytest, pytest-asyncio, pytest-cov, fakeredis, httpx and anyio with `==`, and the venv matches; httpx below 0.28; coverage 7.10 or later | R-03 | Passes (ch) |
| RD-10 | Drift is no worse than recorded | The `-r` chain from `requirements.txt`, parsed with `packaging`, markers evaluated: violations ⊆ {transformers, boto3}; at least 25 requirements parsed | OBS-43 | Passes (ch) |
| RD-11 | The frontend tooling ceilings hold | Range majors in `package.json` and versions in `package-lock.json`: vitest 3, @vitest/coverage-v8 3 and equal to vitest, jsdom 26, vite 5, jest-dom 6 | C-07 | Passes (ch) |
| RD-15 | The caller's shell sets no `Settings` key | No environment key matches `_config.SETTINGS_KEYS`, case-insensitively | A-06 | Passes (ch) |
| **F. Documents** | | | | |
| RD-12 | The README's pytest commands parse | Every pytest command in the README's bash fences parses with pytest's own parser with no unknown option; `-m` names only registered markers; test paths exist | OBS-58 | Fails: `--ignore-missing-models` |
| RD-13 | The README's runner and npm commands exist | Each `run_tests.py <x>` is a runner command or a category; each `npm run <s>` or `npm test` is a script in `package.json` | OBS-58 | Passes (ch) |
| RD-14 | The register and its guards agree | This matrix lists exactly the RD cases the module defines and cites no other; every R- and A- row names a case or says Procedural | — | Fails: no document |

`ch` = characterisation: holds before and after, pinned so that it cannot silently change. Cut at planning time: a static check that `send_task` is patched (it would give false positives and miss real cases; left as Procedural under R-13), and a child `pytest --help` for RD-12 (the in-process parser is exact).

---

## Execution Results

Executed on 2026-09-14 on Windows 11 with Python 3.11.9 and Node 22.14.0.

**Baseline** (before any change, idle machine, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`): backend **1023 passed, 1 skipped** (315.5 s). The count had moved from Section 4's 1018 because of later test commits (`62a2503` added the Whisper agreement module). Frontend **150 passed**, lint **113** problems, `tsc` **6** errors.

**Pre-fix evidence.** The instruments were written before any fix.

| Run | Result |
|---|---|
| RD module, with the shell prefix | 9 passed, 6 failed: RD-02, 03, 04, 07, 12 and 14, as predicted |
| RD-01 alone, without the prefix | Failed |
| Full suite with every corpus hidden (`-p tests._corpora`) | The 7 predicted TEST-06 failures: `test_fr10_orchestration.py` (partitions SAA), `test_fr10_api_contract.py` (groupable SAA), `test_fr10_cancellation_and_failures.py` ×2, FO-75, FO-26 ×2. The Common Voice case in `test_fr10_orchestration.py` and the Whisper agreement case **passed** where they should have skipped |
| `ECHO_KEEP_CORPORA=ravdess`, `test_dataset_serving_api.py` | FT-64 ×4 and FT-68 failed: the module's guard checked RAVDESS only |
| Every dataset kept, annotations hidden, `test_fr10_dataset_extensions.py` | 7 grounding cases failed: the guard did not name the annotations file |

**Post-fix: the register's guards.** All 15 RD cases pass, in every full run below. RD-01 also passes run alone **without** the shell prefix, which is the point of TEST-08. RD-04's child run takes 50 to 75 s.

**Post-fix: hidden corpora.** Every run 0 failed, 0 errors.

| Run | Result |
|---|---|
| Full suite with every corpus hidden, no shell prefix (`-p tests._corpora`) | **998 passed, 41 skipped** (180.2 s) |
| `ECHO_KEEP_CORPORA=ravdess`, `test_dataset_serving_api.py` | 18 passed, 5 skipped: FT-64 ×4 and FT-68, each naming its own corpus |
| `ECHO_KEEP_CORPORA=datasets` (annotations hidden), `test_fr10_dataset_extensions.py` | 14 skipped, reason `l2-arctic-annotations` |

The 41 skips are the FR-7 acceptance case (opt-in) and 40 corpus skips, each with a reason naming the absent corpus and `Backend/data/`:

| Reason (absent) | Cases |
|---|---|
| saa, l2-arctic, l2-arctic-annotations | 14 (`test_fr10_dataset_extensions.py`) |
| saa | 8 |
| saa, l2-arctic, common-voice | 7 (`test_fr10_partitioning.py`) |
| ravdess | 5 |
| common-voice | 3 |
| cv-valid-dev, l2-arctic | 1 each (FT-64) |
| the two Common Voice clips | 1 (the Whisper agreement case) |

**Post-fix: backend regression and order independence** (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, as the baseline):

```
$ ./.venv/Scripts/python.exe -m pytest tests/ -q                           → 1038 passed, 1 skipped (253.79 s)
$ ./.venv/Scripts/python.exe -m pytest tests/ -q                           → 1038 passed, 1 skipped (186.56 s)
$ ./.venv/Scripts/python.exe -m pytest $(ls tests/test_*.py | sort -r) -q  → 1038 passed, 1 skipped (189.28 s)
```

1038 = the 1023 baseline passes + the 15 RD cases. The skip is the FR-7 acceptance case. No OBS-57 recurrence: every timing case passed in all three runs. `test_deliverables_reporting.py` passes (24) with the new `risks` category, so TD-01 and TD-03 accept it.

No pre-existing assertion was changed. The changes to existing test files are skip guards (TEST-06), one marker with its comment (TEST-07), the offline defaults at the top of the conftest (TEST-08), and the docstrings and comments that described the old guards.

**Report run.** `python tests/run_tests.py report`, run on its own on an idle machine:

| Run | Result |
|---|---|
| Coverage run (`-m "not performance"`) | 967 passed, 1 skipped (163.4 s); pytest-cov reported the floor reached |
| Timing run (`-m performance`) | **71 passed** (90.0 s) — one more than Section 4's 70: FO-31 |
| Verdict | **PASS**, with a row "Risks, Dependencies, Assumptions and Constraints (Section 5)": 15 cases, 15 passed |

968 + 71 = 1,039 cases, the 1,024 of the baseline plus 15. Compared with the baseline's split, the coverage run has one case fewer from the existing suite and the timing run one more (FO-31).

**Client regression.** No client file changed in this section; RD-11 only reads `package.json` and the lockfile.

```
$ npm run test:coverage                     → 11 files, 150 passed, thresholds met, exit 0
$ npm run lint                              → 113 problems (baseline 113)
$ npx tsc --noEmit -p tsconfig.app.json     → 6 errors (baseline 6; OBS-18)
$ python tests/run_tests.py summary         → Evaluation summary [PASS], both tiers from this run
```

**Isolation.** `find shared-storage uploads -newer <start> -type f` returned nothing (both tiers' paths checked); no `.coverage.*` fragment remained after the report run; no hidden-corpora directory was created under the temporary directory; `git status` lists only the intended changes.

---

## Findings

### Test-run defects

| ID | Status | Location | Record |
|---|---|---|---|
| **TEST-06** | **Fixed** | `test_fr10_orchestration.py`, `test_fr10_api_contract.py`, `test_fr10_cancellation_and_failures.py`, `test_failover_corrupt_data.py` (FO-75), `test_failover_worker_recovery.py` (FO-26), `test_dataset_serving_api.py`, `test_fr10_dataset_extensions.py`, `test_whisper_transcript_consistency.py` | **A clean checkout failed instead of skipping.** `Backend/data/` is gitignored and holds none of its 147 MB in git. Seven cases read SAA or Common Voice with no guard and failed on a clean clone (`FileNotFoundError: Dataset metadata not found for: saa`). `test_dataset_serving_api.py` guarded every built-in case on RAVDESS alone, so FT-64 ×4 and FT-68 failed in a partial checkout; `test_fr10_dataset_extensions.py` did not name the annotations file its grounding cases read. Two cases passed by accident: the Common Voice orchestration case raised the error it expected for a different reason, and the Whisper case built its own path from `__file__`. `3.1.2-function-testing.md` said dataset-dependent modules skip. **Resolution:** `tests/_corpora.py` provides `requires_corpora(*names, files=())`, which checks a dataset's CSV and audio directory; every corpus-reading case carries it, per test or per parameter where only some cases read a corpus. The hiding plugin in the same file lets a machine with the corpora see what a clean clone sees. Guarded by RD-04 and RD-05. |
| **TEST-07** | **Fixed** | `test_failover_worker_recovery.py`, FO-31 | **A wall-clock budget outside the performance marker.** FO-31 asserts that submit and poll stay within PE-1's 500 ms mean while no worker runs (`assert sum(latencies) / len(latencies) < 0.5`), and carried only the `failover` and `critical` markers. It therefore ran under the coverage tracer: the same defect as TEST-05, and it makes Section 4's statement that FO-92 was "the only case" wrong. The text search Section 4 used had missed it. **Resolution:** FO-31 is also marked `performance`, at method level, with a comment in FO-92's style; its assertions are unchanged and it still belongs to 3.1.7. RD-07 replaces the text search with a data-flow check. Guarded by RD-07. |
| **TEST-08** | **Fixed** | `tests/conftest.py` | **Offline mode was a property of one command, not of the suite.** `run_tests.py report` set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` in its children; a plain `pytest tests/`, and the runner's `all`, `critical` and category commands, ran online. The Whisper agreement case (`test_whisper_transcript_consistency.py`) checks only that the processor is cached before loading the model, so online it could download the model weights. **Resolution:** the conftest sets both variables with `setdefault` above its first hub-capable import, so every run is offline unless the shell says otherwise; `ECHO_MODEL_TESTS=1 HF_HUB_OFFLINE=0` is the documented opt-out. Guarded by RD-01 to RD-03. |

### Recorded

| ID | Status | Location | Record |
|---|---|---|---|
| **OBS-58** | **Fixed** | `Backend/tests/README.md` | **The README's Troubleshooting, time estimates and mock claims were false.** It advised `pytest tests/ --ignore-missing-models`, an option no plugin defines, so the command stopped with a usage error. It described "memory limits" and tests that "run in containers", neither of which exist, told the reader to update mock outputs in the conftest, and gave per-category times of 10 to 30 minutes. **Resolution:** Troubleshooting now covers what actually goes wrong — absent corpora, offline mode and its opt-out, the cache-dependent Whisper case, Redis, timing failures, where the tests run — and ends with a pointer to this register. The time claims are gone, the environment and test-data notes describe the stand-ins and the corpora as they are, and Contributing says timing assertions belong only in `performance` cases. Guarded by RD-12 and RD-13. |
| **OBS-59** | Record corrected | `currentTest.md` (3.1.6 section and 3.1.6 gaps) | **The gap analysis quoted a scope note that does not exist.** It said `test_security.py` declares system-level security out of scope "per that file's own Special Considerations note". The file has no such note and no scope statement. The correction is recorded in `currentTest.md`; whether system-level security is in scope stays with Member 2. |
| **OBS-60** | Recorded | `test_whisper_transcript_consistency.py` | **The Whisper agreement case depends on host state.** It runs the real `openai/whisper-base` when the model is in the local Hugging Face cache and skips otherwise, so its verdict and its runtime differ between machines. It now also skips when its two Common Voice clips are absent, and it can no longer reach the network (TEST-08). It is the only default-run case that executes real weights; it partly shows A-08. |

**Found by the suite during this work.** CF-65 (3.1.8, guards BUG-74) failed when `test_risk_register.py` was first added: the new module imports `huggingface_hub` (RD-01) and `packaging` (RD-09, RD-10), and neither was declared in `requirements-dev.txt`. Both were already installed as dependencies of transformers and pytest. They are now declared, unpinned; nothing was installed. The first hidden-corpora run also found a fault in this section's own instrument: `hide()` could not repoint a module first imported after an earlier `hide()`, which RD-05 caught when run under the plugin. It was fixed before the post-fix runs. RD-14's first version then failed on this document: an assumption's id shares its cell with the assumption's text (the template's column is "Assumption to be proven"), so the row pattern found no A- rows. The pattern was corrected, not the table.

---

## Coverage Assessment and Residual Gaps

**Assessment: the four tables the template asks for are written, ranked and owned, and every safeguard they call in force is checked.** Before this section there was no register, and the informal Troubleshooting it would have replaced advised a flag that does not exist.

The findings share the pattern of Sections 3.1.8 and 4: **safeguards that looked in place but were not in force.** The corpus guards covered the modules written with them and none of the older ones (TEST-06). The timing rule depended on a text search (TEST-07). Offline mode depended on which command was typed (TEST-08). Each is now either enforced by the suite itself or checked by a case that fails if it drifts back, and RD-14 stops the register and its guards from drifting apart.

What the register cannot do is mitigate its highest-ranked risk. R-01 stays High × High: the default run checks models only through stand-ins, and the real-model evidence depends on the cache (OBS-60) or on an opt-in run (FT-104). That is recorded, with its contingency, rather than hidden.

Deliberately excluded, with justification:

| Excluded | Reason | Owner |
|---|---|---|
| A static check that job-submitting modules patch `send_task` | It would report false positives and miss indirect submissions; R-13 is Procedural | Member 1 |
| Rewriting `test_security.py`'s vacuous cases (OBS-07, OBS-61) | Section 3.1.6; classified case by case in `3.1.6-security-testing.md` | Member 2 |
| Requirements not yet implemented | They get their own pass after implementation | Member 1, Member 2 |

---

## References

1. Rational Unified Process — *Test Plan* template, Section 5 Risks, Dependencies, Assumptions, and Constraints. IBM Rational Software.
2. *AudioLens Software Requirements Specification*, Group 16: §3.4 PE-1 to PE-3 (C-06), §3.9.2 (C-03), §3.9.3 (C-08), §3.10 (RD-08). `docs/Group 16 - AudioLens SRS.pdf`.
3. *AudioLens Software Architecture Document*, Group 16, §3 (R-10, C-10). `docs/Group 16 - AudioLens SAD.pdf`.
4. huggingface_hub 1.23 — `constants.HF_HUB_OFFLINE` and `is_offline_mode()` (Special Consideration 1).
5. pytest 8.2 documentation — plugins loaded with `-p`, `pytest_configure`, and `skipif` marks, available at https://docs.pytest.org/ (Accessed 2026-09-14).
6. `Backend/tests/plans/4-deliverables.md` — TEST-05 and the Section 4 report command this section builds on.
7. `docs/risks.md` — the plan that produced this section.
8. `currentTest.md` (repository root) — the original gap analysis this section supersedes.
