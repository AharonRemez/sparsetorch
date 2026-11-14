"""Mutation testing of src/spartorch: break every statement, one at a time, and run the tests.

Each mutant is one change from a fixed set of operators, applied to a copy of the tree in a
temporary directory (the repository is only read, and the report is written once, at the end).
A mutant is killed only when a test fails on it and fails again when rerun alone; a failure that
does not repeat is flaky, and an error at collection or a timeout is reported as such, with the
test or file named for every one. A mutant that no test catches is a survivor; one listed in the
equivalents file is reported as equivalent. The exit code is 1 for any unlisted survivor, any
flaky mutant, or an entry of the list that matches no mutant or more than one.

    uv run python tools/mutate.py
"""

import argparse
import ast
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = [
    "tests/functional",
    "tests/modules",
    "tests/test_structure.py",
    "tests/test_spartorch_init.py",
    "tests/test_spartorch_types.py",
]
OPERATORS = {
    "delete": "a statement or a decorator removed (`pass` in its place); docstrings and imports are kept",
    "negate": "the test of an `if`/`elif`, a `while`, a conditional expression, an `assert`, a comprehension's "
    "`if` and a `case` guard wrapped in `not`; any `not` removed",
    "compare": "`<`/`<=`, `>`/`>=`, `==`/`!=`, `in`/`not in`, `is`/`is not` swapped, at each place of a chain",
    "boolop": "`and`/`or` swapped",
    "arith": "`+`/`-`, `*`/`/`, `//`/`%`, `|`/`&` swapped and `**` to `*`, in an expression and in an augmented "
    "assignment (`+=` to `-=`); any unary `-` removed",
    "constant": "an int or float `n` to `n + 1` (1 to 0), `True`/`False` swapped",
    "return": "`return x` to `return None` in a function not annotated `-> None`",
}
# Not mutated: annotations (types, not behaviour) and f-strings (the wording of a message).
COMPARE: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
    ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
}  # fmt: skip
ARITH: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult, ast.FloorDiv: ast.Mod,
    ast.Mod: ast.FloorDiv, ast.BitOr: ast.BitAnd, ast.BitAnd: ast.BitOr, ast.Pow: ast.Mult,
}  # fmt: skip
KEPT = ast.Import | ast.ImportFrom | ast.Pass | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef  # never deleted


@dataclass(frozen=True)
class Mutant:
    file: str  # relative to the root, posix
    line: int
    scope: str  # dotted qualname of the enclosing function or class, "<module>" at top level
    operator: str
    original: str  # the first original line of the mutated node, stripped
    mutant: str  # that line after the mutation, stripped
    source: str  # the whole mutated file

    def key(self) -> tuple[str, str, str, str, str]:
        return self.file, self.scope, self.operator, self.original, self.mutant


def generate(path: Path, rel: str) -> list[Mutant]:
    """Every mutant of one file, deduplicated by the mutated source, in source order."""
    text = path.read_text()
    data = text.encode()
    lines = text.splitlines()
    starts = [0]
    for line in data.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    tree = ast.parse(text)
    skip: set[int] = set()  # ids of nodes inside annotations and f-strings
    for node in ast.walk(tree):
        parts = [getattr(node, "annotation", None), getattr(node, "returns", None)]
        parts += [node] if isinstance(node, ast.JoinedStr) else []
        skip |= {id(n) for part in parts if part is not None for n in ast.walk(part)}

    found: dict[str, Mutant] = {}

    def emit(line: int, op: str, start: int, end: int, replacement: str, scope: str) -> None:
        new = (data[:start] + replacement.encode() + data[end:]).decode()
        if new == text or new in found:
            return
        compile(new, rel, "exec")  # a mutant that does not compile is a bug of this script
        new_line = new.splitlines()[line - 1] if replacement else ""  # "": the line itself was removed
        found[new] = Mutant(rel, line, scope, op, lines[line - 1].strip(), new_line.strip(), new)

    def replace(node: ast.expr | ast.stmt, op: str, replacement: str, scope: str) -> None:
        assert node.end_lineno is not None
        assert node.end_col_offset is not None
        start = starts[node.lineno - 1] + node.col_offset
        emit(node.lineno, op, start, starts[node.end_lineno - 1] + node.end_col_offset, replacement, scope)

    def expr(node: ast.expr, op: str, new: ast.expr, scope: str) -> None:
        replace(node, op, f"({ast.unparse(new)})", scope)

    def visit(node: ast.AST, scope: str, *, returns_none: bool) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            scope = node.name if scope == "<module>" else f"{scope}.{node.name}"
            if not isinstance(node, ast.ClassDef):
                returns_none = isinstance(node.returns, ast.Constant) and node.returns.value is None
            for dec in node.decorator_list:  # the whole line of the decorator, newline included
                assert dec.end_lineno is not None
                emit(dec.lineno, "delete", starts[dec.lineno - 1], starts[dec.end_lineno], "", scope)
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, [])
            for stmt in body if isinstance(body, list) else []:  # an IfExp's or a Lambda's body is one expression
                docstring = isinstance(stmt, ast.Expr) and isinstance(getattr(stmt.value, "value", None), str)
                if not docstring and not isinstance(stmt, KEPT):
                    replace(stmt, "delete", "pass", scope)
        if id(node) not in skip:
            mutate(node, scope, returns_none=returns_none)
        for child in ast.iter_child_nodes(node):
            visit(child, scope, returns_none=returns_none)

    def mutate(node: ast.AST, scope: str, *, returns_none: bool) -> None:
        if isinstance(node, ast.If | ast.While | ast.IfExp | ast.Assert):
            expr(node.test, "negate", ast.UnaryOp(ast.Not(), node.test), scope)
        elif isinstance(node, ast.comprehension | ast.match_case):
            for test in node.ifs if isinstance(node, ast.comprehension) else [node.guard] if node.guard else []:
                expr(test, "negate", ast.UnaryOp(ast.Not(), test), scope)
        elif isinstance(node, ast.AugAssign) and type(node.op) in ARITH:
            replace(node, "arith", ast.unparse(ast.AugAssign(node.target, ARITH[type(node.op)](), node.value)), scope)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not | ast.USub):
            expr(node, "negate" if isinstance(node.op, ast.Not) else "arith", node.operand, scope)
        elif isinstance(node, ast.Compare):
            for i, cmp in enumerate(node.ops):
                ops = [*node.ops[:i], COMPARE[type(cmp)](), *node.ops[i + 1 :]]
                expr(node, "compare", ast.Compare(node.left, ops, node.comparators), scope)
        elif isinstance(node, ast.BoolOp):
            expr(
                node, "boolop", ast.BoolOp(ast.Or() if isinstance(node.op, ast.And) else ast.And(), node.values), scope
            )
        elif isinstance(node, ast.BinOp) and type(node.op) in ARITH:
            expr(node, "arith", ast.BinOp(node.left, ARITH[type(node.op)](), node.right), scope)
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            replace(node, "constant", repr(not node.value), scope)
        elif isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            replace(node, "constant", repr(0 if node.value == 1 else node.value + 1), scope)
        elif isinstance(node, ast.Return) and node.value is not None and not returns_none:
            replace(node, "return", "return None", scope)

    visit(tree, "<module>", returns_none=False)
    return list(found.values())


def run_tests(tree: Path, tests: list[str], timeout: float | None) -> tuple[str, str, float]:
    """Run pytest in ``tree``, stopping at the first failure: the outcome, what caused it, the seconds.

    The outcome is "passed", "failed" (a test failed), "error" (an error at collection or in a
    fixture, or pytest itself failing) or "timeout"; the cause names the first failing test or file
    and its exception type, as in the report: the node id in backticks, then e.g. ``AssertionError``.
    """
    # No bytecode: a mutant of the same size written in the same second would load a stale .pyc.
    # One intra-op thread: the tensors are tiny, and many processes share the cores.
    env = os.environ | {"PYTHONPATH": str(tree / "src"), "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}
    command = [sys.executable, "-m", "pytest", "-x", "-q", "-rfE", "-p", "no:cacheprovider", *tests]
    start = time.perf_counter()
    try:
        done = subprocess.run(command, cwd=tree, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", "timeout", time.perf_counter() - start
    seconds = time.perf_counter() - start
    if done.returncode == 0:
        return "passed", "", seconds
    # pytest prints "FAILED <node> - Type: message", and for a plain assert "FAILED <node> - assert ...".
    found = re.search(r"^(FAILED|ERROR) (\S+)(?: - (?:(\w+)(?::|$)|(assert)))?", done.stdout, re.MULTILINE)
    if found is None:
        tail = (done.stdout + done.stderr).strip().splitlines()
        return "error", f"pytest exit {done.returncode}: {tail[-1] if tail else ''}", seconds
    outcome = "failed" if found[1] == "FAILED" and done.returncode == 1 else "error"
    kind = found[3] or ("AssertionError" if found[4] else "")
    return outcome, f"`{found[2]}` {kind}".strip(), seconds


def cell(text: str) -> str:
    return "`" + text.replace("|", "\\|") + "`" if text else "(removed)"


CATEGORIES = ("killed", "error", "timeout", "flaky", "equivalent", "survived")


def report(
    package: str,
    tests: list[str],
    mutants: list[Mutant],
    outcomes: dict[Mutant, tuple[str, str]],
    reasons: dict[tuple[str, ...], str],
    problems: list[str],
) -> str:
    """The Markdown report: per file the count of each category, then every mutant by category, with its cause."""
    out = [
        f"# Mutation report: `{package}`",
        "",
        "Generated by `uv run python tools/mutate.py`; do not edit. Tests: " + ", ".join(f"`{t}`" for t in tests) + ".",
        "",
        "| Operator | Mutation |",
        "|---|---|",
        *(f"| {name} | {text.replace('|', '\\|')} |" for name, text in OPERATORS.items()),
        "",
        (
            "Killed: a test failed, and failed again when rerun alone. Error: an error at collection or in a "
            "fixture, the same on a rerun. Timeout: the tests did not end, twice. Flaky: a failure that did not "
            "happen again."
        ),
        "",
        "| File | Mutants | " + " | ".join(c.capitalize() for c in CATEGORIES) + " |",
        "|---|---:|" + "---:|" * len(CATEGORIES),
    ]
    rows: dict[str, list[int]] = {}
    for m in mutants:
        row = rows.setdefault(m.file, [0] * (1 + len(CATEGORIES)))
        row[0] += 1
        row[1 + CATEGORIES.index(outcomes[m][0])] += 1
    total = [sum(col) for col in zip(*rows.values(), strict=True)] if rows else [0] * (1 + len(CATEGORIES))
    out += [f"| `{f}` | {' | '.join(map(str, r))} |" for f, r in sorted(rows.items())]
    out += [f"| **total** | {' | '.join(f'**{n}**' for n in total)} |"]
    head = "| File | Line | Scope | Operator | Original | Mutant |"

    def line(m: Mutant) -> str:
        return f"| `{m.file}` | {m.line} | `{m.scope}` | {m.operator} | {cell(m.original)} | {cell(m.mutant)} |"

    def section(title: str, chosen: list[Mutant], last: str, value: dict[Mutant, str]) -> None:
        nonlocal out
        out += ["", f"## {title}", ""]
        out += [f"{head} {last} |", "|---|---:|---|---|---|---|---|"] if chosen else ["None."]
        out += [f"{line(m)} {value[m]} |" for m in chosen]

    def having(*categories: str) -> list[Mutant]:
        return [m for m in mutants if outcomes[m][0] in categories]

    out += ["", "## Survivors", ""]
    out += [head, "|---|---:|---|---|---|---|"] if having("survived") else ["None."]
    out += [line(m) for m in having("survived")]
    causes = {m: outcomes[m][1] for m in mutants}
    section("Flaky", having("flaky"), "Cause", causes)
    section("Equivalent", having("equivalent"), "Reason", {m: reasons.get(m.key(), "") for m in mutants})
    section("Caught by an error or a timeout", having("error", "timeout"), "Cause", causes)
    out += ["", "## Killed", "", "| File | Line | Operator | Mutant | Test |", "|---|---:|---|---|---|"]
    out += [f"| `{m.file}` | {m.line} | {m.operator} | {cell(m.mutant)} | {causes[m]} |" for m in having("killed")]
    out += ["", "## Unusable entries of the equivalents list", ""]
    out += [f"- {p}" for p in problems] if problems else ["None."]
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Mutation testing: break every statement, run the tests.")
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="the tree to copy; package and tests are relative to it"
    )
    parser.add_argument("--package", default="src/spartorch")
    parser.add_argument("--tests", nargs="+", default=TESTS)
    parser.add_argument("--equivalents", type=Path, default=ROOT / "tools" / "equivalents.toml")
    parser.add_argument("--report", type=Path, default=ROOT / "evidence" / "mutation.md")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))  # each a CUDA context
    parser.add_argument("--timeout", type=float, help="seconds per test run; 10 times the slowest baseline, >= 60")
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):  # a kill runs the cleanup too, also from a background job
        signal.signal(sig, signal.default_int_handler)
    root: Path = args.root.resolve()
    started = time.perf_counter()

    files = sorted((root / args.package).rglob("*.py"))
    mutants = [m for f in files for m in generate(f, f.relative_to(root).as_posix())]
    entries = tomllib.loads(args.equivalents.read_text()).get("equivalent", []) if args.equivalents.exists() else []
    reasons = {(e["file"], e["scope"], e["operator"], e["original"], e["mutant"]): e["reason"] for e in entries}
    # An entry excuses exactly one mutant: one that matches none is left over from changed code,
    # one that matches several would wave through a mutant nobody reviewed.
    count = Counter(m.key() for m in mutants)
    problems = [
        f"{'stale' if count[k] == 0 else 'ambiguous'}: `{k[0]}` `{k[1]}` {k[2]} `{k[3]}` -> `{k[4]}` "
        + ("matches no mutant" if count[k] == 0 else f"matches {count[k]} mutants")
        for k in reasons
        if count[k] != 1
    ]
    reasons = {k: r for k, r in reasons.items() if count[k] == 1}
    tops = {Path(p).parts[0] for p in [args.package, *args.tests]} | {"pyproject.toml", "conftest.py"}

    stop = threading.Event()
    outcomes: dict[Mutant, tuple[str, str]] = {}
    with tempfile.TemporaryDirectory(prefix="mutate-") as tmp:

        def tree(name: str) -> Path:
            copy_root = Path(tmp) / name
            for top in tops:
                src = root / top
                if src.is_dir():
                    shutil.copytree(src, copy_root / top, ignore=shutil.ignore_patterns("__pycache__"))
                elif src.exists():
                    copy_root.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, copy_root / top)
            return copy_root

        def baseline(i: int) -> tuple[str, str, float]:
            return run_tests(tree(str(i)), args.tests, None)

        def work(i: int, timeout: float) -> None:
            here = Path(tmp) / str(i)
            for m in mutants[i :: args.workers]:
                if stop.is_set():
                    return
                target = here / m.file
                original = target.read_text()
                target.write_text(m.source)
                try:
                    outcome, cause, _ = run_tests(here, args.tests, timeout)
                    if outcome != "passed":  # the same cause again, alone, or it was not the mutant
                        where = re.match(r"`(\S+)`", cause)
                        again = run_tests(here, [where[1]] if where else args.tests, timeout)[0]
                        outcome, cause = (
                            ("flaky", f"{cause}, then passed on a rerun") if again == "passed" else (outcome, cause)
                        )
                    outcomes[m] = ("killed" if outcome == "failed" else outcome, cause)
                finally:
                    target.write_text(original)

        with ThreadPoolExecutor(args.workers) as pool:
            try:
                # The unmutated tests, in every copy at once: under the load of the run itself.
                baselines = list(pool.map(baseline, range(args.workers)))
                failing = [cause for outcome, cause, _ in baselines if outcome != "passed"]
                if failing:
                    print(f"the tests fail on the unmutated tree, under load: {failing[0]}", file=sys.stderr)
                    return 2
                timeout = args.timeout or max(60.0, 10 * max(seconds for *_, seconds in baselines))
                list(pool.map(work, range(args.workers), [timeout] * args.workers))
            finally:
                stop.set()

    for m in mutants:
        if outcomes[m][0] == "passed":
            outcomes[m] = ("equivalent" if m.key() in reasons else "survived", "")
    text = report(args.package, args.tests, mutants, outcomes, reasons, problems)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    partial = args.report.with_suffix(".tmp")
    partial.write_text(text)
    partial.replace(args.report)  # whole or not at all
    counts = Counter(outcome for outcome, _ in outcomes.values())
    for p in problems:
        print(p)
    for m in mutants:
        if outcomes[m][0] in ("flaky", "survived"):
            print(f"{outcomes[m][0]}: {m.file}:{m.line} {m.operator} {m.mutant!r} {outcomes[m][1]}".rstrip())
    summary = ", ".join(f"{counts[c]} {c}" for c in CATEGORIES)
    print(f"{len(mutants)} mutants: {summary}; {time.perf_counter() - started:.0f} s; report: {args.report}")
    return 1 if counts["survived"] or counts["flaky"] or problems else 0


if __name__ == "__main__":
    sys.exit(main())
