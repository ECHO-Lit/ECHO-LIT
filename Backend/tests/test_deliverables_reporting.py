"""Deliverables -- the evaluation summary and coverage reports are real and truthful.

Test Plan Section 4.  See tests/plans/4-deliverables.md.

Section 4 of the Master Test Plan names two deliverables: test evaluation
summaries (4.1) and coverage reports (4.2).  Both are produced by
`python tests/run_tests.py report`, so this module holds that tooling to the
same standard as the product: the category map cannot silently skip a module,
the report command depends only on installed plugins and says FAIL when a run
fails, the summary generator survives missing and truncated artifacts, and
every coverage number the README states is the number the gate enforces.
"""
from __future__ import annotations

import configparser
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import _report, run_tests

pytestmark = pytest.mark.important

TESTS_DIR = Path(__file__).resolve().parent
BACKEND = TESTS_DIR.parent
FRONTEND = BACKEND.parent / "Frontend"
README = TESTS_DIR / "README.md"

CATEGORIES = {
    "integrity": {"section": "3.1.1", "description": "Integrity (Section 3.1.1)", "files": ["test_alpha.py"]},
    "function": {"section": "3.1.2", "description": "Function (Section 3.1.2)", "files": ["test_beta.py"]},
}

META = {"branch": "b", "commit": "c", "python": "3", "node": "v0", "platform": "p"}


def _junit(cases: str) -> str:
    return f'<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">{cases}</testsuite></testsuites>'


PYTEST_JUNIT = _junit(
    '<testcase classname="tests.test_alpha.TestA" name="test_ok" time="0.5"/>'
    '<testcase classname="tests.test_alpha" name="test_module_level" time="0.25"/>'
    '<testcase classname="tests.test_beta.TestB" name="test_bad[1]" time="1.0">'
    '<failure message="assert 1 == 2 | pipe&#10;second line">trace</failure></testcase>'
    '<testcase classname="tests.test_beta.TestB" name="test_err" time="0">'
    '<error message="fixture blew up">trace</error></testcase>'
    '<testcase classname="tests.test_beta.TestB" name="test_skip" time="0">'
    '<skipped message="not here"/></testcase>'
    '<testcase classname="tests.test_gamma" name="test_orphan" time="0.1"/>'
)

VITEST_JUNIT = (
    '<?xml version="1.0" encoding="UTF-8" ?><testsuites>'
    '<testsuite name="src/tests/configuration.test.tsx">'
    '<testcase classname="src/tests/configuration.test.tsx" name="CW &gt; CF-01 resolves" time="0.1"/>'
    '</testsuite><testsuite name="src/tests/app-navigation.test.tsx">'
    '<testcase classname="src/tests/app-navigation.test.tsx" name="TestRouting &gt; UI-01" time="0.2"/>'
    '<testcase classname="src/tests/app-navigation.test.tsx" name="TestRouting &gt; UI-02" time="0.2"/>'
    '</testsuite></testsuites>'
)


def _file_summary(pct: float, statements: int) -> dict:
    return {"summary": {"percent_covered": pct, "num_statements": statements}}


BACKEND_COVERAGE = {
    "totals": {
        "percent_covered": 72.5, "num_statements": 200, "covered_lines": 150,
        "num_branches": 40, "covered_branches": 24,
    },
    "files": {
        "app\\services\\well.py": _file_summary(95.0, 50),
        "app/services/poor.py": _file_summary(10.0, 30),
        "app/__init__.py": _file_summary(100.0, 0),
        "app/api/mid.py": _file_summary(55.0, 20),
    },
}


def _metric(covered: int, total: int) -> dict:
    return {"total": total, "covered": covered, "skipped": 0, "pct": round(100 * covered / total, 2)}


FRONTEND_COVERAGE = {
    "total": {
        "lines": _metric(40, 100), "statements": _metric(40, 100),
        "branches": _metric(30, 50), "functions": _metric(10, 40),
    },
    "D:\\x\\Frontend\\src\\components\\ui\\button.tsx": {"lines": _metric(5, 60)},
    "D:\\x\\Frontend\\src\\App.tsx": {"lines": _metric(35, 40)},
}


def _artifacts(tmp_path: Path, *, floor: float = 70, thresholds: str = "lines: 39, statements: 39, branches: 59, functions: 24") -> _report.Artifacts:
    (tmp_path / ".coveragerc").write_text(f"[report]\nfail_under = {floor}\n", encoding="utf-8")
    (tmp_path / "vite.config.ts").write_text(
        f"export default {{ test: {{ coverage: {{ thresholds: {{ {thresholds} }} }} }} }};\n", encoding="utf-8"
    )
    return _report.Artifacts(
        junit_backend=tmp_path / "junit-backend.xml",
        junit_performance=tmp_path / "junit-performance.xml",
        coverage_json=tmp_path / "coverage.json",
        coverage_html=tmp_path / "coverage-html" / "index.html",
        coveragerc=tmp_path / ".coveragerc",
        frontend_junit=tmp_path / "fe-junit.xml",
        frontend_coverage=tmp_path / "coverage-summary.json",
        frontend_coverage_html=tmp_path / "fe-coverage" / "index.html",
        vite_config=tmp_path / "vite.config.ts",
        summary=tmp_path / "evaluation-summary.md",
    )


def _write_all(artifacts: _report.Artifacts, *, backend_junit: str = PYTEST_JUNIT) -> None:
    artifacts.junit_backend.write_text(backend_junit, encoding="utf-8")
    artifacts.junit_performance.write_text(_junit(""), encoding="utf-8")
    artifacts.coverage_json.write_text(json.dumps(BACKEND_COVERAGE), encoding="utf-8")
    artifacts.frontend_junit.write_text(VITEST_JUNIT, encoding="utf-8")
    artifacts.frontend_coverage.write_text(json.dumps(FRONTEND_COVERAGE), encoding="utf-8")


def _render(artifacts: _report.Artifacts, categories: dict = CATEGORIES) -> tuple[str, _report.Evaluation]:
    evaluation = _report.collect(categories, artifacts)
    return _report.render(evaluation, commands=["cmd"], meta=META, now=datetime(2026, 9, 12)), evaluation


# --------------------------------------------------------------------------- A. category map

class TestCategoryMap:
    def test_every_test_module_is_in_exactly_one_category(self):
        """TD-01 -- guards TEST-04: `all`/`critical`/`report` skipped unlisted modules."""
        on_disk = sorted(p.name for p in TESTS_DIR.glob("test_*.py"))
        listed = [f for cfg in run_tests.TEST_CATEGORIES.values() for f in cfg["files"]]
        duplicates = sorted({f for f in listed if listed.count(f) > 1})
        missing = sorted(set(on_disk) - set(listed))
        assert not duplicates, f"modules in more than one category: {duplicates}"
        assert not missing, f"modules in no category: {missing}"

    def test_every_listed_module_exists(self):
        """TD-02 -- a renamed or deleted module must leave the map too."""
        listed = [f for cfg in run_tests.TEST_CATEGORIES.values() for f in cfg["files"]]
        assert [f for f in listed if not (TESTS_DIR / f).exists()] == []

    def test_categories_name_their_section_and_cover_the_backend_sections(self):
        """TD-03 -- the summary's per-section rollup depends on each `section` key."""
        for name, cfg in run_tests.TEST_CATEGORIES.items():
            assert f"(Section {cfg['section']})" in cfg["description"], name
            assert cfg["priority"] in ("critical", "important"), name
        sections = {cfg["section"] for cfg in run_tests.TEST_CATEGORIES.values()}
        assert {"3.1.1", "3.1.2", "3.1.4", "3.1.5", "3.1.6", "3.1.7", "3.1.8"} <= sections


# --------------------------------------------------------------------------- B. report command

@pytest.fixture
def report_calls(monkeypatch, tmp_path):
    """Run `generate_test_report` against a fake subprocess; no real run, no real report dir."""
    calls: list[SimpleNamespace] = []
    codes: list[int] = []

    def fake_run(argv, *, cwd, env):
        calls.append(SimpleNamespace(argv=argv, cwd=cwd, env=env))
        return SimpleNamespace(returncode=codes.pop(0) if codes else 0)

    monkeypatch.setattr(run_tests, "REPORT_DIR", tmp_path / "test_reports")
    monkeypatch.setattr(run_tests.subprocess, "run", fake_run)
    monkeypatch.setattr(
        run_tests._report, "write_summary",
        lambda categories, commands: (tmp_path / "summary.md", SimpleNamespace(verdict="PASS")),
    )
    return SimpleNamespace(calls=calls, codes=codes)


class TestReportCommand:
    def test_depends_only_on_installed_plugins_and_runs_offline(self, report_calls):
        """TD-04 -- guards TEST-03: `report` passed pytest-html's --html, which is not installed."""
        assert run_tests.generate_test_report() is True
        assert len(report_calls.calls) == 2
        for call in report_calls.calls:
            assert call.argv[:3] == [sys.executable, "-m", "pytest"]
            assert not any(a.startswith(("--html", "--self-contained-html", "--benchmark")) for a in call.argv)
            assert any(a.startswith("--junitxml=tests/test_reports/") for a in call.argv)
            assert Path(call.cwd) == BACKEND
            assert call.env["HF_HUB_OFFLINE"] == "1" and call.env["TRANSFORMERS_OFFLINE"] == "1"
        coverage_run = report_calls.calls[0].argv
        assert "--cov" in coverage_run
        assert "--cov-report=json:tests/test_reports/coverage.json" in coverage_run

    def test_timing_cases_run_without_the_coverage_tracer(self, report_calls):
        """TD-05 -- `performance` cases assert wall-clock budgets the tracer would inflate."""
        run_tests.generate_test_report()
        coverage_run, timing_run = (c.argv[3:] for c in report_calls.calls)  # past `python -m pytest`
        assert coverage_run[coverage_run.index("-m") + 1] == "not performance"
        assert timing_run[timing_run.index("-m") + 1] == "performance"
        assert not any(a.startswith("--cov") for a in timing_run)

    def test_a_failing_run_fails_the_command_and_says_so(self, report_calls, capsys):
        """TD-06 -- guards TEST-03: the old command printed "generated" whatever happened."""
        report_calls.codes.extend([1, 0])
        assert run_tests.generate_test_report() is False
        out = capsys.readouterr().out
        assert "Coverage run: FAIL (exit 1)" in out
        assert "Timing run: PASS" in out


# --------------------------------------------------------------------------- C. summary generator

class TestSummaryGenerator:
    def test_junit_totals_and_node_ids(self, tmp_path):
        """TD-07"""
        path = tmp_path / "j.xml"
        path.write_text(PYTEST_JUNIT, encoding="utf-8")
        run = _report.read_junit(path, "Backend", "Backend")
        assert run.status == "ok"
        assert (run.count("passed"), run.count("failed"), run.count("error"), run.count("skipped")) == (3, 1, 1, 1)
        ids = [c.nodeid for c in run.cases]
        assert "tests/test_alpha.py::TestA::test_ok" in ids
        assert "tests/test_alpha.py::test_module_level" in ids
        assert {c.file for c in run.cases} == {"test_alpha.py", "test_beta.py", "test_gamma.py"}

    def test_section_rollup_keeps_unmapped_modules_visible(self, tmp_path):
        """TD-08 -- a module outside every category is reported, never dropped."""
        artifacts = _artifacts(tmp_path)
        _write_all(artifacts)
        _, evaluation = _render(artifacts)
        rows = {(r.tier, r.section): r for r in evaluation.rows}
        assert (rows["Backend", "3.1.1"].passed, rows["Backend", "3.1.1"].cases) == (2, 2)
        beta = rows["Backend", "3.1.2"]
        assert (beta.failed, beta.errors, beta.skipped) == (1, 1, 1)
        assert rows["Backend", "~"].label == "Unmapped" and rows["Backend", "~"].cases == 1
        assert rows["Frontend", "3.1.8"].cases == 1
        assert rows["Frontend", "3.1.3"].cases == 2

    def test_failures_are_listed_with_node_id_and_first_message_line(self, tmp_path):
        """TD-09"""
        artifacts = _artifacts(tmp_path)
        _write_all(artifacts)
        text, evaluation = _render(artifacts)
        assert "| Backend | failed | `tests/test_beta.py::TestB::test_bad[1]` | assert 1 == 2 \\| pipe |" in text
        assert "| Backend | error | `tests/test_beta.py::TestB::test_err` | fixture blew up |" in text
        assert "second line" not in text
        assert evaluation.verdict == "FAIL"

    @pytest.mark.parametrize("floor, meets", [(70, True), (72.5, True), (80, False)])
    def test_backend_coverage_against_floor_and_lowest_modules(self, tmp_path, floor, meets):
        """TD-10"""
        artifacts = _artifacts(tmp_path, floor=floor)
        _write_all(artifacts, backend_junit=_junit('<testcase classname="tests.test_alpha" name="t" time="0"/>'))
        text, evaluation = _render(artifacts)
        cov = evaluation.backend_coverage
        assert cov.percent == 72.5 and cov.floor == floor and cov.meets_floor is meets
        assert [name for name, _, _ in cov.lowest] == ["app/services/poor.py", "app/api/mid.py", "app/services/well.py"]
        assert ("meets floor" in text) is meets and ("BELOW FLOOR" in text) is not meets
        assert evaluation.verdict == ("PASS" if meets else "FAIL")

    def test_missing_frontend_artifacts_read_as_not_run(self, tmp_path):
        """TD-11 -- the backend `report` does not run the frontend tier."""
        artifacts = _artifacts(tmp_path)
        _write_all(artifacts, backend_junit=_junit('<testcase classname="tests.test_alpha" name="t" time="0"/>'))
        artifacts.frontend_junit.unlink()
        artifacts.frontend_coverage.unlink()
        text, evaluation = _render(artifacts)
        frontend = next(r for r in evaluation.runs if r.tier == "Frontend")
        assert frontend.status == "not run" and evaluation.frontend_coverage is None
        assert "npm run test:coverage" in text
        assert evaluation.verdict == "INCOMPLETE"

    @pytest.mark.parametrize(
        "thresholds, meets",
        [("lines: 39, statements: 39, branches: 59, functions: 24", True),
         ("lines: 41, statements: 39, branches: 59, functions: 24", False)],
    )
    def test_frontend_coverage_against_thresholds(self, tmp_path, thresholds, meets):
        """TD-12"""
        artifacts = _artifacts(tmp_path, thresholds=thresholds)
        _write_all(artifacts, backend_junit=_junit('<testcase classname="tests.test_alpha" name="t" time="0"/>'))
        text, evaluation = _render(artifacts)
        fcov = evaluation.frontend_coverage
        assert fcov.metrics == {"lines": 40.0, "statements": 40.0, "branches": 60.0, "functions": 25.0}
        assert fcov.meets_thresholds is meets
        assert fcov.ui_lines == (5, 60) and fcov.app_lines == (35, 40)
        assert ("BELOW" in text) is not meets
        assert evaluation.verdict == ("PASS" if meets else "FAIL")

    def test_truncated_artifacts_are_reported_not_raised(self, tmp_path):
        """TD-13 -- a killed run leaves truncated XML and half-written JSON."""
        artifacts = _artifacts(tmp_path)
        _write_all(artifacts)
        artifacts.junit_backend.write_text(PYTEST_JUNIT[: len(PYTEST_JUNIT) // 2], encoding="utf-8")
        artifacts.coverage_json.write_text('{"totals": {"percent_cov', encoding="utf-8")
        text, evaluation = _render(artifacts)
        backend = evaluation.runs[0]
        assert backend.status == "unreadable" and "junit-backend.xml" in backend.detail
        assert evaluation.backend_coverage is None and "coverage.json" in evaluation.backend_coverage_error
        assert "unreadable (junit-backend.xml" in text
        assert evaluation.verdict == "FAIL"

    def test_write_summary_writes_the_rendered_file(self, tmp_path, monkeypatch):
        """TD-13b"""
        artifacts = _artifacts(tmp_path)
        _write_all(artifacts)
        monkeypatch.setattr(_report, "environment", lambda: META)
        path, evaluation = _report.write_summary(CATEGORIES, ["cmd"], artifacts)
        text = path.read_text(encoding="utf-8")
        assert path == artifacts.summary
        assert text.startswith("# Test Evaluation Summary") and f"**Verdict: {evaluation.verdict}**" in text


# --------------------------------------------------------------------------- D. claims match gates

class TestClaimsMatchGates:
    def test_backend_floor_is_real_and_the_readme_states_no_figure(self):
        """TD-14 -- guards OBS-53: the README claimed >85% and nothing measured it.

        The floors live only in configuration; a percentage in the README is a
        claim that can drift from the gate, so none may appear there.
        """
        floor = _report.backend_floor(BACKEND / ".coveragerc")
        assert floor > 0, "a zero floor gates nothing"
        parser = configparser.ConfigParser()
        parser.read(BACKEND / ".coveragerc")
        # fail_under compares the total rounded to `precision`; at 0 a total half a point under the floor passes.
        assert parser.getint("report", "precision", fallback=0) >= 2
        assert re.findall(r"\d+(?:\.\d+)?\s?%", README.read_text(encoding="utf-8")) == []

    @pytest.mark.parametrize("metric", _report.FRONTEND_METRICS)
    def test_frontend_threshold_is_real(self, metric):
        """TD-15"""
        thresholds = _report.frontend_thresholds(FRONTEND / "vite.config.ts")
        assert thresholds.get(metric, 0) > 0, "a zero threshold gates nothing"

    def test_coverage_is_opt_in(self):
        """TD-16 -- `--cov` in addopts would put the tracer under every timing case."""
        parser = configparser.ConfigParser()
        parser.read(BACKEND / "pytest.ini")
        assert "--cov" not in parser.get("pytest", "addopts")

    def test_report_outputs_are_ignored(self):
        """TD-17 -- by git, and by eslint (the HTML report ships JS that `eslint .` would lint)."""
        backend_ignored = (BACKEND / ".gitignore").read_text(encoding="utf-8").splitlines()
        frontend_ignored = (FRONTEND / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert "tests/test_reports/" in backend_ignored
        assert {"coverage", "test-reports"} <= set(frontend_ignored)
        outputs = [a.split("=", 1)[1].split(":")[-1] for _, argv in run_tests.report_commands()
                   for a in argv if a.startswith(("--junitxml=", "--cov-report="))]
        assert outputs and all(o.startswith("tests/test_reports/") for o in outputs)
        vite = (FRONTEND / "vite.config.ts").read_text(encoding="utf-8")
        assert 'reportsDirectory: "./coverage"' in vite
        assert "--outputFile.junit=test-reports/" in (FRONTEND / "package.json").read_text(encoding="utf-8")
        eslint_ignores = re.search(r"ignores:\s*\[([^\]]*)\]", (FRONTEND / "eslint.config.js").read_text(encoding="utf-8"))
        assert eslint_ignores and {'"coverage"', '"test-reports"'} <= {s.strip() for s in eslint_ignores.group(1).split(",")}
