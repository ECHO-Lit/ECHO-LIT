"""Test evaluation summary for Section 4 -- Deliverables.

Test Plan Section 4.  See tests/plans/4-deliverables.md.

Reads the artifacts a `run_tests.py report` run leaves behind -- pytest and
vitest junit XML, coverage.py's JSON report, vitest's json-summary -- and
renders one Markdown evaluation summary.  Stdlib only: pytest-html is
deliberately not a dependency.

Every parser tolerates an absent artifact ("not run") and an unreadable one (a
killed run leaves truncated XML), so a partial run still yields a summary that
says what is missing instead of a traceback.
"""
from __future__ import annotations

import configparser
import json
import platform
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]
REPO = BACKEND.parent
FRONTEND = REPO / "Frontend"
REPORT_DIR = BACKEND / "tests" / "test_reports"

FRONTEND_METRICS = ("lines", "statements", "branches", "functions")
FRONTEND_SECTIONS = {"configuration.test.tsx": ("3.1.8", "Configuration Testing (Section 3.1.8)")}
FRONTEND_DEFAULT_SECTION = ("3.1.3", "User Interface Testing (Section 3.1.3)")
UNMAPPED = ("~", "Unmapped")
LOWEST_N = 10


@dataclass(frozen=True)
class Artifacts:
    junit_backend: Path
    junit_performance: Path
    coverage_json: Path
    coverage_html: Path
    coveragerc: Path
    frontend_junit: Path
    frontend_coverage: Path
    frontend_coverage_html: Path
    vite_config: Path
    summary: Path

    @classmethod
    def default(cls) -> "Artifacts":
        return cls(
            junit_backend=REPORT_DIR / "junit-backend.xml",
            junit_performance=REPORT_DIR / "junit-performance.xml",
            coverage_json=REPORT_DIR / "coverage.json",
            coverage_html=REPORT_DIR / "coverage-html" / "index.html",
            coveragerc=BACKEND / ".coveragerc",
            frontend_junit=FRONTEND / "test-reports" / "junit.xml",
            frontend_coverage=FRONTEND / "coverage" / "coverage-summary.json",
            frontend_coverage_html=FRONTEND / "coverage" / "index.html",
            vite_config=FRONTEND / "vite.config.ts",
            summary=REPORT_DIR / "evaluation-summary.md",
        )


@dataclass
class Case:
    file: str
    nodeid: str
    outcome: str  # passed | failed | error | skipped
    time: float
    message: str = ""


@dataclass
class Run:
    label: str
    tier: str
    status: str  # ok | not run | unreadable
    cases: list[Case] = field(default_factory=list)
    detail: str = ""
    produced: str = ""

    def count(self, outcome: str) -> int:
        return sum(1 for c in self.cases if c.outcome == outcome)

    @property
    def clean(self) -> bool:
        return self.status == "ok" and not (self.count("failed") or self.count("error"))


@dataclass
class Row:
    tier: str
    section: str
    label: str
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    time: float = 0.0

    @property
    def cases(self) -> int:
        return self.passed + self.failed + self.skipped + self.errors


@dataclass
class BackendCoverage:
    percent: float
    floor: float
    statements: int
    covered_lines: int
    branches: int
    covered_branches: int
    lowest: list[tuple[str, float, int]]

    @property
    def meets_floor(self) -> bool:
        return self.percent >= self.floor


@dataclass
class FrontendCoverage:
    metrics: dict[str, float]
    thresholds: dict[str, float]
    ui_lines: tuple[int, int]
    app_lines: tuple[int, int]

    @property
    def meets_thresholds(self) -> bool:
        return all(self.metrics.get(m, 0.0) >= self.thresholds.get(m, 0.0) for m in FRONTEND_METRICS)


@dataclass
class Evaluation:
    runs: list[Run]
    rows: list[Row]
    backend_coverage: BackendCoverage | None
    backend_coverage_error: str
    frontend_coverage: FrontendCoverage | None
    frontend_coverage_error: str

    @property
    def verdict(self) -> str:
        if any(r.status == "unreadable" or (r.status == "ok" and not r.clean) for r in self.runs):
            return "FAIL"
        if self.backend_coverage_error or self.frontend_coverage_error:
            return "FAIL"
        if self.backend_coverage and not self.backend_coverage.meets_floor:
            return "FAIL"
        if self.frontend_coverage and not self.frontend_coverage.meets_thresholds:
            return "FAIL"
        missing = [r.label for r in self.runs if r.status == "not run"]
        if missing or self.backend_coverage is None or self.frontend_coverage is None:
            return "INCOMPLETE"
        return "PASS"


# --------------------------------------------------------------------------- junit

def _first_line(text: str | None) -> str:
    for line in (text or "").splitlines():
        if line.strip():
            return line.strip()[:200]
    return ""


def _locate(classname: str, name: str) -> tuple[str, str]:
    """(file, nodeid) for a junit testcase from pytest or vitest."""
    if "/" in classname or classname.endswith((".ts", ".tsx")):
        file = Path(classname).name
        return file, f"{classname} > {name}"
    # pytest: classname "tests.test_x.TestY"; a collection error has an empty
    # classname and the dotted module path in `name`.
    dotted = classname or name
    parts = dotted.split(".")
    for i, part in enumerate(parts):
        if part.startswith("test_"):
            file = f"{part}.py"
            tail = parts[i + 1:]
            path = "/".join(parts[: i + 1]) + ".py"
            nodeid = "::".join([path, *tail, name] if classname else [path])
            return file, nodeid
    return dotted, f"{dotted}::{name}" if classname else dotted


def read_junit(path: Path, label: str, tier: str) -> Run:
    if not path.exists():
        return Run(label, tier, "not run")
    produced = f"{datetime.fromtimestamp(path.stat().st_mtime):%Y-%m-%d %H:%M}"
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        return Run(label, tier, "unreadable", detail=f"{path.name}: {exc}", produced=produced)
    cases: list[Case] = []
    for suite in root.iter("testsuite"):
        for tc in suite.findall("testcase"):
            file, nodeid = _locate(tc.get("classname", ""), tc.get("name", ""))
            outcome, message = "passed", ""
            for tag, result in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
                el = tc.find(tag)
                if el is not None:
                    outcome = result
                    message = _first_line(el.get("message")) or _first_line(el.text)
                    break
            cases.append(Case(file, nodeid, outcome, float(tc.get("time") or 0.0), message))
    return Run(label, tier, "ok", cases, produced=produced)


def file_sections(categories: dict) -> dict[str, tuple[str, str]]:
    return {f: (cfg["section"], cfg["description"]) for cfg in categories.values() for f in cfg["files"]}


def rollup(runs: list[Run], categories: dict) -> list[Row]:
    backend = file_sections(categories)
    rows: dict[tuple[str, str], Row] = {}
    for run in runs:
        for case in run.cases:
            if run.tier == "Frontend":
                section, label = FRONTEND_SECTIONS.get(case.file, FRONTEND_DEFAULT_SECTION)
            else:
                section, label = backend.get(case.file, UNMAPPED)
            row = rows.setdefault((run.tier, section), Row(run.tier, section, label))
            if case.outcome == "passed":
                row.passed += 1
            elif case.outcome == "failed":
                row.failed += 1
            elif case.outcome == "skipped":
                row.skipped += 1
            else:
                row.errors += 1
            row.time += case.time
    return sorted(rows.values(), key=lambda r: (r.tier, r.section))


# --------------------------------------------------------------------------- coverage

def backend_floor(coveragerc: Path) -> float:
    parser = configparser.ConfigParser()
    parser.read(coveragerc)
    return parser.getfloat("report", "fail_under", fallback=0.0)


def read_backend_coverage(path: Path, coveragerc: Path) -> BackendCoverage | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    totals = data["totals"]
    files = [
        (name.replace("\\", "/"), s["summary"]["percent_covered"], s["summary"]["num_statements"])
        for name, s in data.get("files", {}).items()
        if s["summary"]["num_statements"]
    ]
    files.sort(key=lambda f: (f[1], f[0]))
    return BackendCoverage(
        percent=totals["percent_covered"],
        floor=backend_floor(coveragerc),
        statements=totals["num_statements"],
        covered_lines=totals["covered_lines"],
        branches=totals.get("num_branches", 0),
        covered_branches=totals.get("covered_branches", 0),
        lowest=files[:LOWEST_N],
    )


def frontend_thresholds(vite_config: Path) -> dict[str, float]:
    match = re.search(r"thresholds:\s*\{([^}]*)\}", vite_config.read_text(encoding="utf-8"))
    if not match:
        return {}
    return {k: float(v) for k, v in re.findall(r"(\w+)\s*:\s*([\d.]+)", match.group(1))}


def read_frontend_coverage(path: Path, vite_config: Path) -> FrontendCoverage | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    ui, app = [0, 0], [0, 0]
    for name, summary in data.items():
        if name == "total":
            continue
        bucket = ui if "/src/components/ui/" in name.replace("\\", "/") else app
        bucket[0] += summary["lines"]["covered"]
        bucket[1] += summary["lines"]["total"]
    return FrontendCoverage(
        metrics={m: float(data["total"][m]["pct"]) for m in FRONTEND_METRICS},
        thresholds=frontend_thresholds(vite_config),
        ui_lines=(ui[0], ui[1]),
        app_lines=(app[0], app[1]),
    )


def _guarded(reader, *args):
    try:
        return reader(*args), ""
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f"{Path(args[0]).name}: {exc.__class__.__name__}: {exc}"


# --------------------------------------------------------------------------- summary

def collect(categories: dict, artifacts: Artifacts) -> Evaluation:
    runs = [
        read_junit(artifacts.junit_backend, "Backend (coverage run, -m \"not performance\")", "Backend"),
        read_junit(artifacts.junit_performance, "Backend (timing run, -m performance)", "Backend"),
        read_junit(artifacts.frontend_junit, "Frontend (vitest)", "Frontend"),
    ]
    backend_cov, backend_err = _guarded(read_backend_coverage, artifacts.coverage_json, artifacts.coveragerc)
    frontend_cov, frontend_err = _guarded(read_frontend_coverage, artifacts.frontend_coverage, artifacts.vite_config)
    return Evaluation(runs, rollup(runs, categories), backend_cov, backend_err, frontend_cov, frontend_err)


def environment() -> dict[str, str]:
    def probe(*cmd: str) -> str:
        try:
            out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=30, shell=False)
        except (OSError, subprocess.SubprocessError):
            return "unavailable"
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unavailable"

    return {
        "branch": probe("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": probe("git", "rev-parse", "--short", "HEAD"),
        "python": platform.python_version(),
        "node": probe("node", "--version"),
        "platform": platform.platform(),
    }


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _pct(covered: int, total: int) -> str:
    return f"{100.0 * covered / total:.2f}%" if total else "n/a"


def render(evaluation: Evaluation, *, commands: list[str], meta: dict[str, str], now: datetime | None = None) -> str:
    now = now or datetime.now()
    out: list[str] = [
        "# Test Evaluation Summary",
        "",
        f"**Verdict: {evaluation.verdict}**",
        "",
        f"- Generated: {now:%Y-%m-%d %H:%M}",
        f"- Branch / commit: `{meta.get('branch', 'unavailable')}` @ `{meta.get('commit', 'unavailable')}`",
        f"- Python {meta.get('python', '?')}, Node {meta.get('node', '?')}, {meta.get('platform', '?')}",
        "- Commands:",
        *[f"  - `{c}`" for c in commands],
        "",
        "## Runs",
        "",
        "| Run | Status | Produced | Cases | Passed | Failed | Errors | Skipped |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in evaluation.runs:
        status = run.status if run.status != "unreadable" else f"unreadable ({_cell(run.detail)})"
        out.append(
            f"| {run.label} | {status} | {run.produced or '-'} | {len(run.cases)} | {run.count('passed')} "
            f"| {run.count('failed')} | {run.count('error')} | {run.count('skipped')} |"
        )

    out += [
        "",
        "## Verdict by test-plan section",
        "",
        "| Tier | Section | Cases | Passed | Failed | Errors | Skipped | Time (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in evaluation.rows:
        out.append(
            f"| {row.tier} | {row.label} | {row.cases} | {row.passed} | {row.failed} | {row.errors} "
            f"| {row.skipped} | {row.time:.1f} |"
        )
    if not evaluation.rows:
        out.append("| - | no results | 0 | 0 | 0 | 0 | 0 | 0.0 |")

    broken = [(run.tier, c) for run in evaluation.runs for c in run.cases if c.outcome in ("failed", "error")]
    out += ["", "## Failures and errors", ""]
    if broken:
        out += ["| Tier | Outcome | Case | Message |", "|---|---|---|---|"]
        out += [f"| {tier} | {c.outcome} | `{_cell(c.nodeid)}` | {_cell(c.message)} |" for tier, c in broken]
    else:
        out.append("None.")

    out += ["", "## Coverage", "", "### Backend (coverage.py, statements + branches)", ""]
    cov = evaluation.backend_coverage
    if evaluation.backend_coverage_error:
        out.append(f"Unreadable: {evaluation.backend_coverage_error}")
    elif cov is None:
        out.append("Not run.")
    else:
        out += [
            f"- Total: **{cov.percent:.2f}%** against floor {cov.floor:g}% -- "
            f"{'meets floor' if cov.meets_floor else 'BELOW FLOOR'}",
            f"- Lines {cov.covered_lines}/{cov.statements} ({_pct(cov.covered_lines, cov.statements)}), "
            f"branches {cov.covered_branches}/{cov.branches} ({_pct(cov.covered_branches, cov.branches)})",
            "- HTML report: `Backend/tests/test_reports/coverage-html/index.html`",
            "",
            f"Lowest-covered modules (bottom {LOWEST_N}):",
            "",
            "| Module | Coverage | Statements |",
            "|---|---|---|",
            *[f"| `{name}` | {pct:.1f}% | {stmts} |" for name, pct, stmts in cov.lowest],
        ]

    out += ["", "### Frontend (vitest, v8)", ""]
    fcov = evaluation.frontend_coverage
    if evaluation.frontend_coverage_error:
        out.append(f"Unreadable: {evaluation.frontend_coverage_error}")
    elif fcov is None:
        out.append("Not run -- `cd Frontend && npm run test:coverage`, then `python tests/run_tests.py summary`.")
    else:
        out += ["| Metric | Coverage | Threshold | |", "|---|---|---|---|"]
        for m in FRONTEND_METRICS:
            pct, floor = fcov.metrics.get(m, 0.0), fcov.thresholds.get(m, 0.0)
            out.append(f"| {m} | {pct:.2f}% | {floor:g}% | {'ok' if pct >= floor else 'BELOW'} |")
        out += [
            "",
            f"- Line coverage, `src/components/ui/` (vendored shadcn primitives): {_pct(*fcov.ui_lines)} "
            f"({fcov.ui_lines[0]}/{fcov.ui_lines[1]})",
            f"- Line coverage, rest of the app: {_pct(*fcov.app_lines)} ({fcov.app_lines[0]}/{fcov.app_lines[1]})",
            "- HTML report: `Frontend/coverage/index.html`",
        ]

    out += [
        "",
        "## Defects",
        "",
        "Defect status is recorded per section in the Findings table of each `Backend/tests/plans/*.md` doc.",
        "",
    ]
    return "\n".join(out)


def write_summary(categories: dict, commands: list[str], artifacts: Artifacts | None = None) -> tuple[Path, Evaluation]:
    artifacts = artifacts or Artifacts.default()
    evaluation = collect(categories, artifacts)
    artifacts.summary.parent.mkdir(parents=True, exist_ok=True)
    artifacts.summary.write_text(render(evaluation, commands=commands, meta=environment()), encoding="utf-8")
    return artifacts.summary, evaluation
