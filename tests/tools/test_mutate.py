"""tools/mutate.py is itself shown able to fail, on a tiny package in a temporary directory."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent.parent / "tools" / "mutate.py"
DOUBLE = '''def double(x: int) -> int:
    """2 x."""
    return x * 2
'''
UNTESTED = '''

def untested(x: int) -> None:
    """Writes nothing anyone reads."""
    y = x + 1
'''


TEST_DOUBLE = "from toy import double\n\n\ndef test_double():\n    assert double(3) == 6\n"


def run(
    root: Path, source: str, equivalents: str = "", test: str = TEST_DOUBLE, extra: tuple[str, ...] = ()
) -> tuple[int, str, str]:
    (root / "src" / "toy").mkdir(parents=True)
    (root / "src" / "toy" / "__init__.py").write_text(source)
    (root / "tests").mkdir()
    (root / "tests" / "test_toy.py").write_text(test)
    (root / "eq.toml").write_text(equivalents)
    args = ["--root", root, "--package", "src/toy", "--tests", "tests", "--workers", "2", *extra]
    args += ["--equivalents", root / "eq.toml", "--report", root / "report.md"]
    done = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True)
    report = root / "report.md"
    return done.returncode, report.read_text() if report.exists() else "", done.stdout + done.stderr


def test_every_mutant_of_a_tested_line_is_reported_killed(tmp_path):
    code, report, _ = run(tmp_path, DOUBLE)
    assert code == 0
    # return x * 2: deleted, return None, * to /, 2 to 3
    assert "| `src/toy/__init__.py` | 4 | 4 | 0 | 0 | 0 | 0 | 0 |" in report
    assert "## Survivors\n\nNone." in report


def test_a_line_no_test_covers_is_reported_as_a_survivor(tmp_path):
    code, report, _ = run(tmp_path, DOUBLE + UNTESTED)
    assert code == 1
    # y = x + 1: deleted, + to -, 1 to 0
    assert "| `src/toy/__init__.py` | 7 | 4 | 0 | 0 | 0 | 0 | 3 |" in report
    assert "| `src/toy/__init__.py` | 8 | `untested` | arith | `y = x + 1` | `y = (x - 1)` |" in report


def test_a_listed_survivor_is_reported_equivalent(tmp_path):
    entries = "".join(
        f'[[equivalent]]\nfile = "src/toy/__init__.py"\nscope = "untested"\noperator = "{op}"\n'
        f'original = "y = x + 1"\nmutant = "{mutant}"\nreason = "y is never read."\n'
        for op, mutant in [("delete", "pass"), ("arith", "y = (x - 1)"), ("constant", "y = x + 0")]
    )
    code, report, _ = run(tmp_path, DOUBLE + UNTESTED, entries)
    assert code == 0
    assert "| `src/toy/__init__.py` | 7 | 4 | 0 | 0 | 0 | 3 | 0 |" in report
    assert (
        "| `src/toy/__init__.py` | 8 | `untested` | arith | `y = x + 1` | `y = (x - 1)` | y is never read. |" in report
    )


def entry(op: str, original: str, mutant: str) -> str:
    return (
        f'[[equivalent]]\nfile = "src/toy/__init__.py"\nscope = "untested"\noperator = "{op}"\n'
        f'original = "{original}"\nmutant = "{mutant}"\nreason = "Planted."\n'
    )


def test_a_listed_entry_that_matches_no_mutant_is_reported(tmp_path):
    code, report, out = run(tmp_path, DOUBLE, entry("arith", "y = x + 1", "y = (x - 1)"))
    assert code == 1
    stale = "stale: `src/toy/__init__.py` `untested` arith `y = x + 1` -> `y = (x - 1)` matches no mutant"
    assert stale in report
    assert stale in out


TWINS = """

def untested(x: int) -> None:
    \"\"\"Two blocks that start alike.\"\"\"
    if x:
        y = 1
    if x:
        y = 2
"""


def test_an_entry_that_would_cover_two_mutants_is_refused(tmp_path):
    # Deleting either `if` block is the same locator: untested, delete, `if x:` -> `pass`.
    code, report, out = run(tmp_path, DOUBLE + TWINS, entry("delete", "if x:", "pass"))
    assert code == 1
    ambiguous = "ambiguous: `src/toy/__init__.py` `untested` delete `if x:` -> `pass` matches 2 mutants"
    assert ambiguous in report
    assert ambiguous in out
    assert "| `src/toy/__init__.py` | 8 | `untested` | delete | `if x:` | `pass` |" in report  # still a survivor


FORMS = """

def forms(x: int) -> list[int]:
    \"\"\"Every form an operator applies to, untested.\"\"\"
    x += 2
    assert x
    match x:
        case _ if x:
            pass
    return [y for y in range(x) if y]
"""


def test_every_operator_applies_to_every_form_it_names(tmp_path):
    _, report, _ = run(tmp_path, DOUBLE + FORMS)
    for original, mutant, operator in [
        ("x += 2", "x -= 2", "arith"),
        ("assert x", "assert (not x)", "negate"),
        ("case _ if x:", "case _ if (not x):", "negate"),
        ("return [y for y in range(x) if y]", "return [y for y in range(x) if (not y)]", "negate"),
    ]:
        assert f"| `forms` | {operator} | `{original}` | `{mutant}` |" in report


def test_a_killed_mutant_names_the_test_that_failed(tmp_path):
    _, report, _ = run(tmp_path, DOUBLE)
    assert (
        "| `src/toy/__init__.py` | 3 | arith | `return (x / 2)` | `tests/test_toy.py::test_double` AssertionError |"
        in report
    )


def test_tests_that_fail_unmutated_measure_nothing(tmp_path):
    code, _, out = run(tmp_path, DOUBLE, test="def test_fails():\n    assert False\n")
    assert code == 2
    assert "tests/test_toy.py::test_fails" in out


def test_a_mutant_that_breaks_the_import_is_an_error_not_a_kill(tmp_path):
    _, report, _ = run(tmp_path, "SCALE = 2\nHALF = SCALE // 2\n\n\n" + DOUBLE)
    # Deleting SCALE = 2 fails at import, so at collection: no test ran on it.
    assert (
        "| `src/toy/__init__.py` | 1 | `<module>` | delete | `SCALE = 2` | `pass` | `tests/test_toy.py` NameError |"
        in report
    )


COUNTDOWN = '''def double(x: int) -> int:
    """Counts x down to 0."""
    while x > 0:
        x -= 1
    return x
'''


def test_a_mutant_that_hangs_is_a_timeout_not_a_kill(tmp_path):
    test = "from toy import double\n\n\ndef test_countdown():\n    assert double(3) == 0\n"
    code, report, _ = run(tmp_path, COUNTDOWN, test=test, extra=("--timeout", "2"))
    assert code == 0
    # x -= 1 deleted, to x += 1, to x -= 0: the loop never ends.
    assert "| `src/toy/__init__.py` | 9 | 6 | 0 | 3 | 0 | 0 | 0 |" in report
    assert "| `src/toy/__init__.py` | 4 | `double` | arith | `x -= 1` | `x += 1` | timeout |" in report


def test_a_kill_that_does_not_happen_again_is_flaky(tmp_path):
    mark = tmp_path / "failed-once"
    test = (
        f"from pathlib import Path\n\nfrom toy import COIN\n\n\ndef test_coin():\n    mark = Path({str(mark)!r})\n"
        "    if COIN and not mark.exists():\n        mark.touch()\n        raise AssertionError\n"
    )
    code, report, out = run(tmp_path, "COIN = 0\n", test=test)
    assert code == 1
    assert "| `src/toy/__init__.py` | 2 | 0 | 1 | 0 | 1 | 0 | 0 |" in report
    flaky = "`tests/test_toy.py::test_coin` AssertionError, then passed on a rerun"
    assert f"| `src/toy/__init__.py` | 1 | `<module>` | constant | `COIN = 0` | `COIN = 1` | {flaky} |" in report
    assert "flaky" in out
