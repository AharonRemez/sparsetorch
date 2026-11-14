"""The API pages cover every public name of spartorch and name nothing it lacks; each check is shown able to fail."""

import ast
import dataclasses
import importlib
import inspect
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
from sphinx.pycode import ModuleAnalyzer

import spartorch

ROOT = Path(__file__).parent.parent
SRC = Path(spartorch.__path__[0])  # the package imported, so a test can point the checks at a copy
DOC = ROOT / "doc" / "source"


def resolve(dotted: str) -> object:
    """The module or attribute a dotted name stands for."""
    try:
        return importlib.import_module(dotted)
    except ModuleNotFoundError:
        parent, _, name = dotted.rpartition(".")
        return getattr(resolve(parent), name)


def public_names() -> list[str]:
    """Every public name of every public module: its ``__all__``, else what its top level defines."""
    names = []
    for path in sorted(SRC.rglob("*.py")):
        parts = path.relative_to(SRC.parent).with_suffix("").parts
        parts = parts[:-1] if parts[-1] == "__init__" else parts
        if any(part.startswith("_") for part in parts):
            continue
        module = ".".join(parts)
        exported = getattr(importlib.import_module(module), "__all__", None)
        if exported is None:
            body = ast.parse(path.read_text()).body
            exported = [n.name for n in body if isinstance(n, ast.FunctionDef | ast.ClassDef)]
            exported += [t.id for n in body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)]
            exported += [n.target.id for n in body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
        names += [f"{module}.{name}" for name in exported if not name.startswith("_")]
    return names


def documented(pages: Path) -> list[object]:
    """The objects the pages document: each ``automodule`` and each ``autosummary`` entry."""
    found = []
    for page in pages.glob("*.rst"):
        module, in_summary = "", False
        for line in page.read_text().splitlines():
            if m := re.match(r"\.\. (currentmodule|automodule):: (\S+)", line):
                module, in_summary = m[2], False
                if m[1] == "automodule":
                    found.append(resolve(module))
            elif line.startswith(".. autosummary::"):
                in_summary = True
            elif in_summary and line.startswith(" ") and not line.strip().startswith(":"):
                found.append(resolve(f"{module}.{line.strip()}"))
            elif line and not line.startswith(" "):
                in_summary = False
    return found


def missing(pages: Path) -> list[str]:
    """The public names whose object is on no page (a module counts once it has an ``automodule``)."""
    docs = documented(pages)
    return [name for name in public_names() if not any(resolve(name) is d for d in docs)]


def test_every_public_name_is_on_an_api_page():
    assert "spartorch.functional.backends.resolve" in public_names()  # a module without __all__ is read too
    assert missing(DOC) == []


def test_a_name_missing_from_the_pages_is_found(tmp_path):
    for page in DOC.glob("*.rst"):
        shutil.copy(page, tmp_path)
    page = tmp_path / "spartorch.functional.rst"
    page.write_text(page.read_text().replace("   sort_pattern\n", ""))
    assert missing(tmp_path) == ["spartorch.functional.sort_pattern"]


@pytest.mark.parametrize(
    ("plant", "message"),
    [
        ("   linear\n   nosuch\n", "failed to import nosuch"),
        (
            "   linear\n\nSee :func:`spartorch.functional.nosuch`.\n",
            "reference target not found: spartorch.functional.nosuch",
        ),
    ],
    ids=["autosummary-entry", "func-reference"],
)
def test_the_build_fails_on_a_name_spartorch_lacks(tmp_path, plant, message):
    source = tmp_path / "source"
    shutil.copytree(DOC, source, ignore=shutil.ignore_patterns("generated"))
    page = source / "spartorch.functional.rst"
    page.write_text(page.read_text().replace("   linear\n", plant, 1))
    build = [sys.executable, "-m", "sphinx", "-E", "-W", "-q", "-b", "html", str(source), str(tmp_path / "html")]
    result = subprocess.run(build, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert message in result.stderr


def public_objects() -> dict[str, object]:
    """Every public object as a user reaches it, by public path.

    The public names, the modules that export them, and the methods, properties and dataclass
    fields a public class defines itself (inherited ``nn.Module`` members and ``_`` names are not).
    """
    found: dict[str, object] = {}
    for name in public_names():
        found[name.rpartition(".")[0]] = resolve(name.rpartition(".")[0])
        found[name] = obj = resolve(name)
        if isinstance(obj, type):
            for member, value in vars(obj).items():
                if not member.startswith("_") and (inspect.isfunction(value) or isinstance(value, property)):
                    found[f"{name}.{member}"] = value
            for field in dataclasses.fields(obj) if dataclasses.is_dataclass(obj) else ():
                found[f"{name}.{field.name}"] = field
    return found


def section(doc: str, headers: tuple[str, ...]) -> set[str]:
    """The names a Google docstring lists in its sections called ``headers`` (``Args``, ...)."""
    names, inside = set(), False
    for line in inspect.cleandoc(doc).splitlines():
        if line and not line.startswith(" "):
            inside = line.endswith(":") and line[:-1] in headers
        elif inside and (m := re.match(r"    (\w+)( \(.*\))?:", line)):
            names.add(m[1])
    return names


def function(obj: object) -> types.FunctionType | None:
    """``obj`` if it is a function, the function a registered op (``torch.library.custom_op``) wraps, else None."""
    fn = getattr(obj, "__wrapped__", obj)
    return fn if inspect.isfunction(fn) else None


def undocumented() -> list[str]:
    """The public objects without a docstring, and the classes whose docstring misses a parameter.

    A class must name each parameter of its constructor in ``Args:`` or ``Attributes:``. A function,
    method or property with a parameter besides ``self``/``cls`` must have an ``Args:`` section;
    that it names every parameter is ruff's D417 (which, unlike D1, also runs in ``_x.py`` modules).
    A dataclass field is documented by its class's ``Attributes:``.
    """
    problems = []
    for path, obj in public_objects().items():
        if isinstance(obj, dataclasses.Field):
            continue
        module, _, name = path.rpartition(".")
        if inspect.ismodule(obj) or isinstance(obj, type) or function(obj) or isinstance(obj, property):
            doc = obj.__doc__ or ""
        else:  # data: its docstring is the string after the assignment
            doc = "\n".join(ModuleAnalyzer.for_module(module).find_attr_docs().get(("", name), []))
        if dataclasses.is_dataclass(obj) and doc.startswith(f"{name}("):
            doc = ""  # the signature dataclass writes when the class has no docstring
        if not doc.strip():
            problems.append(f"{path}: no docstring")
        elif isinstance(obj, type):
            named = section(doc, ("Args", "Attributes"))
            problems += [
                f"{path}: {p} is not in Args or Attributes" for p in inspect.signature(obj).parameters if p not in named
            ]
        elif fn := function(obj.fget if isinstance(obj, property) else obj):
            params = inspect.signature(fn).parameters
            if set(params) - {"self", "cls"} and not section(doc, ("Args",)):
                problems.append(f"{path}: has parameters and no Args section")
    return problems


def test_every_public_object_has_a_docstring():
    assert "spartorch.SparseLinear.forward" in public_objects()
    assert "spartorch.types.Format.shape" in public_objects()
    assert undocumented() == []


def drop_docstring(path: Path, name: str) -> None:
    """Delete the docstring of the function ``name`` in the file ``path``."""
    lines = path.read_text().splitlines(keepends=True)
    node = next(n for n in ast.walk(ast.parse("".join(lines))) if isinstance(n, ast.FunctionDef) and n.name == name)
    doc = node.body[0]
    del lines[doc.lineno - 1 : doc.end_lineno]
    path.write_text("".join(lines))


def test_the_public_objects_the_review_stripped_are_found(tmp_path):
    shutil.copytree(SRC, tmp_path / "spartorch")
    drop_docstring(tmp_path / "spartorch" / "modules" / "_weight.py", "format")
    drop_docstring(tmp_path / "spartorch" / "functional" / "_topology.py", "sort_pattern")
    drop_docstring(tmp_path / "spartorch" / "init.py", "fan_in_uniform_")
    script = "from tests.test_docs import SRC, undocumented; print(SRC); print(*undocumented(), sep='\\n')"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(ROOT)])}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env)
    src, *found = result.stdout.splitlines()
    assert src == str(tmp_path / "spartorch")
    assert found == [
        "spartorch.SparseWeight.format: no docstring",
        "spartorch.functional.sort_pattern: no docstring",
        "spartorch.init.fan_in_uniform_: no docstring",
        "spartorch.modules.SparseWeight.format: no docstring",
    ]


def test_the_functions_without_args_before_b03_are_found(tmp_path):
    shutil.copytree(SRC, tmp_path / "spartorch")
    old = {  # the two docstrings as they were before they had an Args section
        "resolve": '''"""The :class:`~spartorch.types.Backend` a preset name stands for, '''
        '''or ``backend`` itself if it is one.

    Raises:
        ValueError: for a name that is not in :data:`~spartorch.functional.backends.PRESETS`.
    """''',
        "forward": '''""":func:`~spartorch.functional.linear` on the layer's tensors.

        ``(..., in_features) -> (..., out_features)``.
        """''',
    }
    for path, name in [("functional/backends.py", "resolve"), ("modules/_linear.py", "forward")]:
        file = tmp_path / "spartorch" / path
        lines = file.read_text().splitlines(keepends=True)
        node = next(n for n in ast.walk(ast.parse("".join(lines))) if isinstance(n, ast.FunctionDef) and n.name == name)
        doc = node.body[0]
        lines[doc.lineno - 1 : doc.end_lineno] = [" " * doc.col_offset + old[name] + "\n"]
        file.write_text("".join(lines))
    script = "from tests.test_docs import undocumented; print(*undocumented(), sep='\\n')"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(ROOT)])}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env)
    assert result.stdout.splitlines() == [
        "spartorch.SparseLinear.forward: has parameters and no Args section",
        "spartorch.functional.backends.resolve: has parameters and no Args section",
        "spartorch.modules.SparseLinear.forward: has parameters and no Args section",
    ]


def test_a_registered_op_without_a_docstring_or_args_is_found(tmp_path):
    shutil.copytree(SRC, tmp_path / "spartorch")
    csr = tmp_path / "spartorch" / "functional" / "ops" / "_csr.py"
    drop_docstring(csr, "spmm")
    text = csr.read_text()
    csr.write_text(
        text[: text.index("    Args:\n        row_ptr:")] + text[text.index("    Returns:\n        ``(nnz,)``") :]
    )
    script = "from tests.test_docs import undocumented; print(*undocumented(), sep='\\n')"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(ROOT)])}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env)
    assert result.stdout.splitlines() == [
        "spartorch.functional.ops.sddmm: has parameters and no Args section",
        "spartorch.functional.ops.spmm: no docstring",
    ]


def test_a_class_without_a_parameter_in_its_docstring_is_found(tmp_path):
    shutil.copytree(SRC, tmp_path / "spartorch")
    linear = tmp_path / "spartorch" / "modules" / "_linear.py"
    linear.write_text(linear.read_text().replace("        nnz: exact number of connections.\n", ""))
    script = "from tests.test_docs import undocumented; print(*undocumented(), sep='\\n')"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(ROOT)])}
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, env=env)
    assert result.stdout.splitlines() == [
        "spartorch.SparseLinear: nnz is not in Args or Attributes",
        "spartorch.modules.SparseLinear: nnz is not in Args or Attributes",
    ]


LITERAL = re.compile(r"``(\w+(?:\.\w+)*)(?:\(\))?``")


def literal_mentions(src: Path) -> list[str]:
    """The ``literals`` in the docstrings under ``src`` that name a public spartorch object.

    A literal names one when it is a public path or a dotted suffix of one (``functional.linear``,
    ``SparseWeight.format``), or a bare last component (``sort_pattern``) that is not also a
    parameter name of a public function or class: a bare ``format`` or ``shape`` is read as the
    parameter. Code such as ``format.shape`` or ``F.linear`` names no public path and passes.
    """
    objects = public_objects()
    suffixes = {".".join(p.split(".")[i:]) for p in objects for i in range(len(p.split(".")))}
    params = {
        p
        for obj in objects.values()
        if inspect.isfunction(obj) or (isinstance(obj, type) and not dataclasses.is_dataclass(obj))
        for p in inspect.signature(obj).parameters
    }
    found = []
    for path in sorted(src.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                found += [
                    f"{path.relative_to(src.parent)}:{node.lineno}: {m[0]} names a spartorch object; use a role "
                    f"(:func:, :class:, :meth:, :attr:, :mod:, :data:) with its public path, so the build checks it"
                    for m in LITERAL.finditer(node.value.value)
                    if m[1] in suffixes and ("." in m[1] or m[1] not in params)
                ]
    return found


def test_no_docstring_names_a_spartorch_object_as_a_literal():
    assert literal_mentions(SRC) == []


@pytest.mark.parametrize(
    ("plant", "found"),
    [("``functional.linear``", 1), ("``SparseWeight``", 1), ("``SparseWeight.format()``", 1), ("``format``", 0)],
    ids=["dotted", "bare", "call", "parameter"],
)
def test_a_literal_naming_a_spartorch_object_is_found(tmp_path, plant, found):
    shutil.copytree(SRC, tmp_path / "spartorch")
    linear = tmp_path / "spartorch" / "modules" / "_linear.py"
    text = linear.read_text()
    linear.write_text(
        text.replace("    Args:\n        in_features:", f"    See {plant}.\n\n    Args:\n        in_features:", 1)
    )
    assert len(literal_mentions(tmp_path / "spartorch")) == len(literal_mentions(SRC)) + found
