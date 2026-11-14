"""The structure of the package, enforced on the AST of src/spartorch; each rule is also shown able to fail."""

import ast
from collections import defaultdict
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src" / "spartorch"

# The spartorch modules each module file may import, as dotted paths below spartorch (a row
# allows its submodules too). Every file has its own row, so a new file is reported until it is
# given one, which is the moment to ask whether it is needed. A package's __init__ may also
# import its own children; spartorch's own __init__ is the public face and imports anything.
ALLOWED: dict[str, set[str]] = {
    "types": set(),
    "init": set(),
    "functional": {"types"},
    "functional._topology": {"types"},
    "functional.storage": {"types"},
    "functional.storage._csr": set(),
    "functional.storage._dense": {"types"},
    "functional.ops": {"types"},
    "functional.ops._csr": {"types"},
    "functional.ops._dense": {"types"},
    "functional.backends": {"types", "functional._topology", "functional.storage", "functional.ops"},
    "functional._linear": {"types", "functional.backends"},
    "modules": {"functional", "types", "init"},
    "modules._weight": {"functional", "types", "init"},
    "modules._linear": {"functional", "types", "init", "modules._weight"},
}
# Parts that compute on plain tensors: of torch.nn they may use torch.nn.functional only, so no
# module class and no Parameter, and nothing of torch.optim.
TENSORS_ONLY = "functional"
# Every outside name src/spartorch may not use: an import by string hides an edge from this test.
DYNAMIC = ("importlib", "__import__", "builtins.__import__")


def inside(name: str, row: str) -> bool:
    return name == row or name.startswith(row + ".")


def forbidden_in_functional(name: str) -> bool:
    return (inside(name, "torch.nn") and not inside(name, "torch.nn.functional")) or inside(name, "torch.optim")


def dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and (base := dotted(node.value)):
        return f"{base}.{node.attr}"
    return None


def imported(tree: ast.Module, package: list[str]) -> list[str]:
    """Every spartorch module the tree imports, dotted below spartorch ("" is spartorch itself)."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = ".".join(["spartorch", *package[: len(package) - node.level + 1]])
                module = f"{base}.{node.module}" if node.module else base
            else:
                module = node.module or ""
            # `from a import b` imports the module a.b when there is one, else a name of a.
            found += [f"{module}.{a.name}" if f"{module}.{a.name}"[10:] in ALLOWED else module for a in node.names]
    return [name[10:] for name in found if name == "spartorch" or name.startswith("spartorch.")]


def used(tree: ast.Module) -> set[str]:
    """Every outside name the tree imports or refers to, resolved through the names its imports bind."""
    bound, found = {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                found.add(a.name)
                # `import a.b` binds a; `import a.b as c` binds c to a.b.
                bound[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for a in node.names:
                found.add(f"{node.module}.{a.name}")
                bound[a.asname or a.name] = f"{node.module}.{a.name}"
    inner = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name | ast.Attribute) and id(node) not in inner and (name := dotted(node)):
            head, dot, rest = name.partition(".")
            if head in bound:
                found.add(bound[head] + dot + rest)
            elif head == "__import__":
                found.add(head)
    return found


def violations(root: Path) -> list[str]:
    """Every breach of the structure rules in the package tree at ``root``."""
    found = []
    defined = defaultdict(list)
    for path in sorted(root.rglob("*.py")):
        parts = list(path.relative_to(root).with_suffix("").parts)
        is_init = parts[-1] == "__init__"
        name = ".".join(parts[:-1] if is_init else parts)
        tree = ast.parse(path.read_text())
        if name and name not in ALLOWED:
            found.append(f"{name}: not in the rule table")
            continue
        for target in imported(tree, parts[:-1]):
            if not name or (is_init and target.startswith(name + ".")):
                continue
            if not any(inside(target, row) for row in ALLOWED[name]):
                found.append(f"{name}: imports spartorch.{target}")
        outside = used(tree)
        bad = {n for n in outside if any(inside(n, d) for d in DYNAMIC)}
        if inside(name, TENSORS_ONLY):
            bad |= {n for n in outside if forbidden_in_functional(n)}
        found += [f"{name or 'spartorch'}: uses {n}" for n in sorted(bad)]
        for node in tree.body:
            targets = [node.target] if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [])
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                targets = [ast.Name(node.name)]
            for target in targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    defined[target.id].append(name)
    found += [f"{n} is defined in {', '.join(where)}" for n, where in defined.items() if len(where) > 1]
    return found


def unexported(root: Path) -> list[str]:
    """Every package ``__init__`` under ``root`` whose ``__all__`` is not exactly the names it imports."""
    found = []
    for path in sorted(root.rglob("__init__.py")):
        tree = ast.parse(path.read_text())
        names = sorted(
            a.asname or a.name for n in tree.body if isinstance(n, ast.Import | ast.ImportFrom) for a in n.names
        )
        listed = [
            ast.literal_eval(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets)
        ]
        if sorted(listed[0] if listed else []) != names:
            found.append(
                f"{path.relative_to(root)}: __all__ is {sorted(listed[0] if listed else [])}, it imports {names}"
            )
    return found


def test_spartorch_keeps_the_rules():
    assert (SRC / "__init__.py").is_file()
    assert violations(SRC) == []


def plant(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "spartorch"
    for name, text in {"__init__.py": "", "types.py": "", "functional/__init__.py": "", **files}.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


PLANTED = {
    "types": ({"types.py": "from spartorch import functional\n"}, "types: imports spartorch.functional"),
    "init": ({"init.py": "import spartorch.modules._linear\n"}, "init: imports spartorch.modules._linear"),
    "relative": (
        {"functional/ops/_dense.py": "from .._topology import x\n"},
        "functional.ops._dense: imports spartorch.functional._topology",
    ),
    "linear-ops": (
        {"functional/_linear.py": "from spartorch.functional.ops import f\n"},
        "functional._linear: imports spartorch.functional.ops",
    ),
    "functional-modules": (
        {"functional/_topology.py": "from spartorch.modules import SparseLinear\n"},
        "functional._topology: imports spartorch.modules",
    ),
    "unknown": ({"training.py": ""}, "training: not in the rule table"),
    "new-file-in-functional": (
        {
            "functional/_conv.py": "from spartorch.functional import ops\n"
            "from spartorch.functional._linear import linear\n"
        },
        "functional._conv: not in the rule table",
    ),
    "twice": ({"init.py": "def f(): ...\n", "types.py": "f = 1\n"}, "f is defined in init, types"),
    "module": (
        {"functional/_topology.py": "from torch import nn\nclass A(nn.Module): ...\n"},
        "functional._topology: uses torch.nn.Module",
    ),
    "module-attribute": (
        {"functional/_topology.py": "import torch\nclass A(torch.nn.Module): ...\n"},
        "functional._topology: uses torch.nn.Module",
    ),
    "module-alias": (
        {"functional/_topology.py": "import torch.nn as tnn\nclass A(tnn.Module): ...\n"},
        "functional._topology: uses torch.nn.Module",
    ),
    "module-subclass": (
        {"functional/_topology.py": "from torch import nn\nclass A(nn.Linear): ...\n"},
        "functional._topology: uses torch.nn.Linear",
    ),
    "parameter": (
        {"functional/ops/_dense.py": "from torch.nn import Parameter\n"},
        "functional.ops._dense: uses torch.nn.Parameter",
    ),
    "parameter-submodule": (
        {"functional/ops/_dense.py": "from torch.nn.parameter import Parameter\n"},
        "functional.ops._dense: uses torch.nn.parameter.Parameter",
    ),
    "optim": ({"functional/_linear.py": "import torch.optim\n"}, "functional._linear: uses torch.optim"),
    "import-module": (
        {"types.py": "import importlib\nimportlib.import_module('spartorch.modules')\n"},
        "types: uses importlib.import_module",
    ),
    "import-module-from": (
        {"init.py": "from importlib import import_module as im\nim('spartorch.modules')\n"},
        "init: uses importlib.import_module",
    ),
    "dunder-import": ({"init.py": "__import__('spartorch.modules')\n"}, "init: uses __import__"),
}


@pytest.mark.parametrize(("files", "report"), PLANTED.values(), ids=PLANTED.keys())
def test_a_planted_violation_is_reported(tmp_path, files, report):
    assert report in violations(plant(tmp_path, files))


def test_every_package_exports_exactly_what_it_imports():
    assert unexported(SRC) == []


def test_an_unexported_import_is_reported(tmp_path):
    root = plant(tmp_path, {"functional/__init__.py": "from ._linear import linear\n"})
    assert unexported(root) == ["functional/__init__.py: __all__ is [], it imports ['linear']"]


def test_allowed_imports_are_not_reported(tmp_path):
    root = plant(
        tmp_path,
        {
            "functional/__init__.py": "from ._linear import linear\nfrom . import backends\n",
            "functional/backends.py": "from spartorch.functional import ops, storage\nfrom spartorch.types import B\n",
            "functional/ops/__init__.py": "from ._dense import f\n",
            "functional/ops/_dense.py": "def f(): ...\n",
            "modules/_linear.py": "from spartorch import functional, init\nfrom spartorch.functional import backends\n",
            "functional/_linear.py": "import torch.nn.functional as F\nimport torch\nF.relu(torch.nn.functional.elu)\n",
            "functional/_topology.py": "from torch.nn import functional\nfrom torch.nn.functional import linear\n",
            "init.py": "",
        },
    )
    assert violations(root) == []
