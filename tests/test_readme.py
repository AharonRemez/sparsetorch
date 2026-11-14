import re
from pathlib import Path

README = Path(__file__).parent.parent / "README.md"


def test_the_code_of_the_readme_runs(tmp_path, monkeypatch):
    # The README is the first code a new user copies: it must match the real API.
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), re.DOTALL)
    assert blocks
    monkeypatch.chdir(tmp_path)  # the snippets save model.pt
    namespace = {}
    for block in blocks:
        exec(block, namespace)
    assert (tmp_path / "model.pt").exists()
