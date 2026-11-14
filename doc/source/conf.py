import sys
from pathlib import Path

# Configuration file for the Sphinx documentation builder, after TorchCurves' doc/source/conf.py.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # `import spartorch` from the source tree, installed or not

project = "SparTorch"
copyright = "2026, the SparTorch authors"
author = "the SparTorch authors"

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinxext.opengraph",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_typehints = "description"  # rely on PEP-484 annotations

# A reference to a name spartorch does not have is a warning, so -W fails on it. The outside
# names its annotations use are listed instead of fetched with intersphinx: the build stays offline.
nitpicky = True
nitpick_ignore = [
    ("py:class", "torch.Tensor"),
    ("py:class", "collections.abc.Callable"),
    ("py:class", "collections.abc.Mapping"),
    ("py:data", "Ellipsis"),
]

templates_path = ["_templates"]
exclude_patterns = []
pygments_style = "sphinx"
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

language = "en"

html_theme = "pydata_sphinx_theme"
