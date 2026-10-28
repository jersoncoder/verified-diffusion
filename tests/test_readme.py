"""The README's quick-start output is the real output of examples/quickstart.py."""

import contextlib
import io
import re
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_quickstart_output_matches_the_readme() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(
        r"<!-- quickstart-output:start -->\n```text\n(.*?)```\n<!-- quickstart-output:end -->",
        readme,
        re.DOTALL,
    )
    assert match is not None
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        runpy.run_path(str(ROOT / "examples" / "quickstart.py"), run_name="__main__")
    assert buffer.getvalue() == match.group(1)


def test_readme_code_block_is_the_example() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    example = (ROOT / "examples" / "quickstart.py").read_text(encoding="utf-8")
    body = example.split('"""', 2)[2].lstrip("\n")
    assert f"```python\n{body}```" in readme
