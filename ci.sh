#!/usr/bin/env bash
# The CI. It runs here, where the GPU is, before anything goes to GitHub.
set -euo pipefail
cd "$(dirname "$0")"

uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -q
rm -rf doc/source/generated  # autosummary keeps stubs of names that were removed
uv run sphinx-build -E -W -q -b html doc/source doc/build/html
