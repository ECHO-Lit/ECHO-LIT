# Current Test Coverage — Analysis vs. Master Test Plan Template

Snapshot date: 2026-09-02. Branch: `Janith/fairness-merged`. Scanned `Backend/tests/` (28 files) and `Frontend/src/tests/` (1 file).

New since last pass: `Backend/tests/test_dataset_labels_service.py` (custom-dataset label ingestion / preview-probe parity).

---

## 3.1.1 Data and Database Integrity Testing — IMPLEMENTED

| File | Covers |
|---|---|
| `test_data_integrity.py` | Redis cache consistency (`TestRedisCacheIntegrity`), audio file checksum + metadata preservation (`TestAudioFileIntegrity`), session isolation (`TestSessionIntegrity`) |
| `test_results_cache.py` | Cache-key stability and parameter sensitivity |

Gap: no test drives the DBMS layer directly with valid/invalid seed data independent of the app (template's literal ask) — coverage is via Redis + file-integrity proxies only.

---

## 3.1.2 Function Testing — IMPLEMENTED (deepest section, exceeds template)

**Generic function/integration:**
- `test_function_testing.py` — model load/init (`TestMLModelIntegration`), audio pipeline (`TestAudioProcessingPipeline`), mock API endpoints (`TestAPIEndpointIntegration`), data flow (`TestDataFlowIntegration`)
- `test_async_jobs.py` — job-contract validation, model-adapter capability exposure, queue routing, model-registry reuse/eviction
- `test_device.py` — CUDA/ROCm/MPS/CPU device selection and fallback
- `test_queue.py` — task queue / background job management (mislabeled 3.1.3 in its own docstring — belongs here)

**Feature-specific, FR-7 (linguistic-vs-acoustic sweep):**
`test_fr7_dsp_invariants.py`, `test_fr7_metrics.py`, `test_fr7_orchestration.py`, `test_fr7_api_contract.py`

**Feature-specific, FR-10 (fairness analysis):**
`test_fr10_metrics.py`, `test_fr10_grounding.py`, `test_fr10_dataset_extensions.py`, `test_fr10_orchestration.py`, `test_fr10_api_contract.py`, `test_fr10_cancellation_and_failures.py`

**Feature-specific, layer representation / probing:**
`test_clustering.py` (HDBSCAN grouping, silhouette, degenerate inputs), `test_probing_service.py` (planted-signal recovery, baselines, label hygiene, reproducibility — "highest-value test file" per its own docstring), `test_layer_probe_job.py` (worker job wiring, cache-key identity), `test_layer_probe_schema.py` (request validation, alignment guards, cache identity), `test_jacobian_lens.py` (job contract, adapter exposure, legacy-lens rejection)

**Feature-specific, custom datasets:**
`test_custom_dataset_manifest.py` (manifest exposes only matched transcript pairs), `test_dataset_labels_service.py` **(new)** — CSV label parsing, filename-pattern label derivation, duration-band labeling, dataset-column merging, property summarisation, preview-vs-probe parity (`TestPreviewMatchesProbe`), end-to-end against the bundled SAVEE dataset (`TestSaveeEndToEnd`)

This bucket is well beyond the template's generic "exercise use-case flows" ask — it's real FR-level regression coverage plus numerical/statistical correctness (WER math, bootstrap CI width, Holm-Bonferroni, planted-signal recovery).

---

## 3.1.3 User Interface Testing — WEAK / PLACEHOLDER

`Frontend/src/tests/ui-components.test.tsx` — every test renders an inline `Mock*` component (`MockWaveformViewer`, `MockDropZone`, `MockModelSelector`, `MockPanelManager`, `MockAttentionVisualization`, etc.), not the real app components. Exercises the *idea* of waveform controls, upload, model config, panel nav, visualization zoom/pan, loading/error states, keyboard nav, ARIA — but zero assertions touch actual production component code.

Gap: no test imports a real component from the app source tree. This is scaffolding, not UI coverage.

---

## 3.1.4 & 3.1.5 Performance Profiling / Load Testing — SUPERSEDED

> **Superseded 2026-09-11** by `Backend/tests/plans/3.1.4-performance-profiling.md` and `Backend/tests/plans/3.1.5-load-testing.md`. `test_performance_load.py` described below was vacuous (it timed its own mocks' `sleep`) and has been deleted; 90 new cases and 15 fixed defects replace it. The analysis below is kept as the original record.

`test_performance_load.py`:
- `TestPerformanceProfiling` — Whisper/Wav2Vec2 inference timing, memory usage monitoring
- `TestLoadTesting` — concurrent-load class
- `TestResourceConstraints` — memory cleanup after ops, CPU usage monitoring

Plus `TestSessionCookiePerformance` in `test_session_cookie.py`.

Thresholds are documented in `run_tests.py` (10s inference / 30s clip, 5s upload / <10MB, 50ms cache op, 2GB peak memory, 10+ concurrent users) — matches template's call for defined success criteria.

---

## 3.1.6 Security and Access Control Testing — IMPLEMENTED

`test_security.py`: `TestAuthenticationSecurity`, `TestInputValidationSecurity`, `TestDataProtectionSecurity`, `TestAPISecurityMeasures`, `TestFileUploadSecurity`, `TestSessionSecurity`, `TestXSSPrevention`, `TestCSRFProtection`.

Plus `TestSessionCookieSecurity` in `test_session_cookie.py` (cookie flags, tampering resistance).

Template's "system-level security" (login/remote-access gateway testing) is explicitly out of scope per that file's own Special Considerations note — consistent with template guidance that this may be an ops/infra concern.

---

## 3.1.7 Failover and Recovery Testing — SUPERSEDED

> **Superseded 2026-09-12** by `Backend/tests/plans/3.1.7-failover-and-recovery.md`: 159 new cases in five `test_failover_*.py` modules, 11 fixed defects (BUG-50..60). The analysis below is kept as the original record.

No test simulates power interruption, network/DASD interruption, incomplete-cycle abort, or corrupted DB pointers/keys, and no recovery-procedure validation exists anywhere in the suite.

---

## 3.1.8 Configuration Testing — SUPERSEDED

> **Superseded 2026-09-12** by `Backend/tests/plans/3.1.8-configuration-testing.md`: 204 new cases (five `test_config_*.py` modules plus `Frontend/src/tests/configuration.test.tsx`), 18 fixed defects (BUG-61..78) and one fixed test-run defect (TEST-02). The analysis below is kept as the original record.

No test varies hardware/software configuration combinations, cross-browser behavior, or concurrent non-target-software resource contention. `test_device.py` (CUDA/ROCm/MPS/CPU selection) is adjacent but is functional device-selection logic, not the template's multi-configuration deployment testing.

---

## Section 4 — Deliverables — SUPERSEDED

> **Superseded 2026-09-12** by `Backend/tests/plans/4-deliverables.md`: `run_tests.py report` now writes a Markdown evaluation summary from junit XML and measures coverage in both tiers (pytest-cov, `@vitest/coverage-v8`) against ratchet floors; TEST-03..05 fixed. The analysis below is kept as the original record.

- **4.1 Test Evaluation Summaries**: `run_tests.py` prints console summaries and claims HTML report generation (`python tests/run_tests.py report`) — present as tooling, not verified against actual report output in this scan.
- **4.2 Reporting on Test Coverage**: no coverage tooling found in the repo (no `pytest-cov` / coverage config detected). README's ">85% line coverage" target is stated but unmeasured.

## Section 5 — Risks, Dependencies, Assumptions, Constraints

No formal risk register exists. `Backend/tests/README.md` has an informal Troubleshooting section (missing models, memory limits, Redis connection) that covers similar ground ad hoc, not in the template's Risk/Mitigation/Contingency table format.

---

## Summary

| Template Section | Status |
|---|---|
| 3.1.1 Data & DB Integrity | Implemented |
| 3.1.2 Function Testing | Implemented — exceeds template, FR7/FR10/probing/dataset-labels suites are the real depth |
| 3.1.3 UI Testing | Placeholder only (mock components, not real UI) |
| 3.1.4 Performance Profiling | Implemented |
| 3.1.5 Load Testing | Implemented |
| 3.1.6 Security & Access Control | Implemented |
| 3.1.7 Failover & Recovery | Implemented (superseded 2026-09-12 — see `Backend/tests/plans/3.1.7-failover-and-recovery.md`) |
| 3.1.8 Configuration Testing | Implemented (superseded 2026-09-12 — see `Backend/tests/plans/3.1.8-configuration-testing.md`) |
| 4. Deliverables | Implemented (superseded 2026-09-12 — see `Backend/tests/plans/4-deliverables.md`) |
| 5. Risks/Dependencies | Missing formal doc — informal equivalent exists |

---

# Gaps & Missing Tests — What Still Needs To Be Implemented

Grounded against actual repo shape, checked 2026-09-02:
- Store is **Redis only** (`Backend/app/core/redis.py`) — no SQL/Mongo DB found. No role/permission system found (`grep -rn "role|permission|is_admin"` returned nothing under `app/`) — sessions are per-user, not multi-role.
- `docker-compose.yml` + `Backend/Dockerfile` + `Frontend/Dockerfile` exist — configuration testing is feasible, currently unused for that purpose.
- Real frontend component tree exists at `Frontend/src/components/{analysis,audio,dataset,eda,fairness,layout,model,panels,predictions,probing,ui,visualization}` — none of it is imported by any test.

## 3.1.1 Data & Database Integrity — gaps
- No test drives Redis (or any store) as an independent subsystem bypassing the app/API layer per the template's literal technique (seed valid/invalid data directly, inspect store state). Current tests go through service/repository wrappers, not raw driver calls.
- No test for **dataset metadata corruption** (malformed manifest rows, partially-written cache entries) — only the happy-path manifest/label parsing is covered (`test_custom_dataset_manifest.py`, `test_dataset_labels_service.py`).
- No test for Redis **eviction / TTL expiry mid-job** (a job reads a key that expired between write and read) — `test_fr7_orchestration.py::test_status_transitions_and_ttl` covers TTL on the happy path only, not adversarial timing.
- **To implement:** direct-Redis integrity test (seed key, kill connection mid-write, verify no partial/corrupt value survives); TTL-expiry-during-active-job test; malformed-manifest / malformed-CSV robustness test beyond the two negative cases already in `test_custom_dataset_manifest.py`.

## 3.1.2 Function Testing — gaps
- `test_function_testing.py`'s `TestAPIEndpointIntegration` and `TestDataFlowIntegration` classes were seen using **mocked** endpoints/fixtures in the earlier scan, not the live FastAPI app via `httpx`/`TestClient` the way `test_fr7_api_contract.py` / `test_fr10_api_contract.py` do — those two FR suites are the only ones exercising real request/response contracts end-to-end.
- No test suite exists yet for **`analyses.py`, `results.py`, `saliency.py`, `inferences.py`, `session.py`** routes as a whole API-contract surface outside the FR7/FR10-scoped slices — i.e. no general "every route returns documented status codes for documented inputs" sweep.
- No test for **dataset upload → probe → clustering → results** as one continuous cross-service flow (each stage is tested in isolation: `test_custom_dataset_manifest`, `test_probing_service`, `test_clustering`, `test_results_cache` — but nothing chains them).
- **To implement:** end-to-end pipeline test (upload custom dataset → run probe job → cluster embeddings → fetch cached result) using real service wiring, not per-stage mocks; API-contract sweep for the four untested route modules.

## 3.1.3 User Interface Testing — gaps (largest gap in the suite)
- Zero tests import a real component from `Frontend/src/components/`. Every existing assertion is against a locally-defined `Mock*` stand-in.
- No test covers the **actual** waveform/audio player, upload dropzone, dataset picker (`fairness` custom reference picker added recently per git log), probing/attention panels, or the fairness confusion-matrix view that the recent commits (`add: custom reference picker for fairness analysis`, `add: phone confusion matrix`, `hesandi/layer-representation`) introduced.
- No visual/interaction test for **panel resize, layer/head selection in real attention view, or embedding zoom/pan** against production code — only mocked reproductions.
- No accessibility audit tool wired in (axe-core or equivalent) — current a11y tests hand-roll trivial contrast/ARIA checks against mock markup.
- **To implement (priority order):**
  1. Render tests for the real components under `src/components/fairness/` (custom reference picker, confusion matrix) and `src/components/probing/` (layer-representation UI) — these are the newest, least-covered, highest-churn areas per recent commits.
  2. Replace or supplement `ui-components.test.tsx` mocks with tests that import actual `WaveformViewer`, upload dropzone, and panel-manager components.
  3. Wire `jest-axe` (or similar) for real accessibility assertions instead of the hand-rolled contrast stub.

## 3.1.4 / 3.1.5 Performance & Load — gaps
- No test represents **sustained/peak workload variation** (Daily/Weekly/Monthly-style peak shapes the template asks for) — `TestLoadTesting` in `test_performance_load.py` was a single concurrent-load shape in the earlier scan, not multiple workload profiles.
- No test measures performance of the **FR10 fairness aggregation path** (bootstrap CI, Holm-Bonferroni over many groups) under realistic dataset sizes — FR10 tests validate correctness, not throughput/latency at scale.
- No background-workload-during-test scenario (template's "background workload on the server" consideration) — nothing drives concurrent Redis/API traffic while a benchmark runs.
- **To implement:** multi-shape load test (average vs peak vs sustained-peak) against the async job queue; latency/throughput benchmark for FR10 group-fairness aggregation at realistic group/row counts; concurrent-background-load variant of the existing inference benchmarks.

## 3.1.6 Security & Access Control — gaps
- Template's **application-level, per-user-type** security testing doesn't apply cleanly — there is no role system in this codebase (single-session-per-user model), so this is a legitimate scope reduction, not an oversight. Worth stating explicitly in the plan rather than leaving implicit.
- **System-level security** (login/remote-access gateway) is explicitly declared out of scope inside `test_security.py` itself — consistent with template's "may not be required, function of network/systems administration" escape hatch, but no one has confirmed who (if anyone) owns that testing.
- No test for **cross-session data leakage under concurrency** (two sessions racing on the same Redis keys/cache namespace) — isolation is tested serially in `TestSessionIntegrity`, not under concurrent load.
- No dependency/package vulnerability scan wired into the suite (e.g. `pip-audit`, `npm audit` as a test-suite gate).
- **To implement:** concurrent cross-session isolation test (race two sessions against shared cache namespace, assert no bleed); wire `pip-audit`/`npm audit` as a CI-gated test; explicit written decision in the plan on system-level security ownership.

## 3.1.7 Failover and Recovery — gaps (closed 2026-09-12, see `Backend/tests/plans/3.1.7-failover-and-recovery.md`)
- No power/communication-interruption simulation at any level (client, server, Redis).
- No **incomplete-cycle abort** test (kill a job mid-worker-execution, verify job status lands in a consistent terminal state rather than stuck/corrupted) — closest existing coverage is `test_fr10_cancellation_and_failures.py`, which tests *requested* cancellation, not an unplanned crash/kill.
- No test for **Redis connection loss mid-request** and recovery behavior (reconnect, retry, or fail cleanly).
- No test for **corrupted/invalid cache entries** being detected and handled rather than causing an unhandled exception downstream.
- **To implement:**
  1. Worker-crash-mid-job test: kill the worker process (or raise inside the shard executor) partway through a job and assert the job record converges to a terminal `failed` state, not stuck `running`.
  2. Redis-connection-drop test: simulate `ConnectionError` mid-request and assert the API returns a clean 5xx rather than an unhandled traceback.
  3. Corrupted-cache-entry test: write a malformed value under a known cache key and assert the read path detects and recomputes rather than crashing.
  (Full DASD/hardware-level failover from the literal template is not applicable to this architecture — no failover cluster exists — so this scope should be explicitly narrowed to worker/Redis/API resilience in the written plan.)

## 3.1.8 Configuration Testing — gaps (closed 2026-09-12, see `Backend/tests/plans/3.1.8-configuration-testing.md`)
- `docker-compose.yml` + separate `Backend/Dockerfile` / `Frontend/Dockerfile` exist but nothing in the test suite runs against the containerized configuration — all current tests run against the local dev environment directly.
- No test matrix across the device-backend permutations `test_device.py` already validates in isolation (CUDA/ROCm/MPS/CPU) combined with actual model inference — device selection logic is tested, but not "does inference actually run correctly on each detected backend."
- No browser-compatibility pass for the frontend (template asks Chrome/Firefox/Safari/Edge — README claims this as a goal, nothing implements it; no Playwright/Cypress cross-browser config found).
- No test for behavior under **constrained resources** (low memory container, capped CPU) — `TestResourceConstraints` in `test_performance_load.py` monitors usage but doesn't run under an artificially constrained environment.
- **To implement:**
  1. A smoke test that builds and runs `docker-compose.yml` and hits `health.py`'s endpoint — minimum viable configuration test.
  2. Cross-browser E2E smoke pass (Playwright, 2–3 browsers) covering the core upload → analyze → view-results flow.
  3. Resource-constrained CI job variant (cgroup/container memory+CPU cap) running the existing performance suite to confirm graceful degradation rather than crashes.

## Section 4 — Deliverables gaps (closed 2026-09-12, see `Backend/tests/plans/4-deliverables.md`)
- (The frontend runner is Vitest, not Jest — corrected here.)
- No coverage tool wired in (`pytest-cov` absent from `Backend/tests/`, no Jest coverage config confirmed for frontend) — the README's ">85% line coverage" and "component coverage" targets are unmeasured claims.
- No actual generated HTML report artifact found — `run_tests.py report` is a claimed command, not something this scan could verify produces output.
- **To implement:** add `pytest-cov` + `--cov-report=html` to the pytest config and wire it into `run_tests.py`; add Jest `--coverage` to the frontend test command; publish both as CI artifacts so the README's coverage claims become verifiable numbers instead of stated goals.

## Section 5 — Risks/Dependencies gaps
- No formal Risk / Mitigation / Contingency table exists per template Section 5. `Backend/tests/README.md`'s Troubleshooting section covers similar ground informally (missing models, memory limits, Redis connectivity) but isn't framed as risk-to-test-execution.
- **To implement:** write a short Section-5 risk table covering at minimum: (1) GPU/model unavailability blocking function/performance tests, (2) fakeredis/real-Redis divergence masking integration bugs, (3) test dataset (SAVEE, L2-ARCTIC) licensing/availability as an external dependency for `test_dataset_labels_service.py` and `test_fr10_dataset_extensions.py`.

---

## Consolidated To-Do List (highest → lowest priority)

1. **Real UI component tests** — replace/supplement mock-only `ui-components.test.tsx` with tests against actual `fairness`, `probing`, `audio`, `dataset` components (3.1.3 — biggest coverage hole given how much recent feature work landed there untested).
2. **Failover/recovery for the async job pipeline** — worker-crash-mid-job, Redis-connection-drop, corrupted-cache-entry tests (3.1.7 — fully missing, and this app's actual failure mode given it's queue/worker-based).
3. **Configuration/deployment smoke test** — docker-compose build+health-check test, since the infra already exists unused (3.1.8).
4. **Coverage tooling** — `pytest-cov` + Jest coverage wired into CI so existing README claims are verifiable (Section 4).
5. **Cross-session concurrency test** — race two sessions against shared Redis namespace (3.1.6 hardening).
6. **End-to-end pipeline test** — upload → probe → cluster → results as one chained test, not isolated per-service tests (3.1.2 hardening).
7. **Multi-shape load test** — average/peak/sustained-peak workload profiles against the job queue (3.1.5 hardening).
8. **Cross-browser E2E smoke** — Playwright pass over core user flow (3.1.8, lower priority — needs more setup investment).
9. **Formal Section 5 risk table** and **Section 4 report generation verification** — process/documentation items, cheap to close once someone sits down to write them.
