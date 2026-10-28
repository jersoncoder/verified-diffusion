"""The report script: table rendering, the README block, and (slow) an end-to-end run."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from verified_diffusion.verification import ConvergenceStudy

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verification_report.py"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verification_report", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def study() -> ConvergenceStudy:
    return ConvergenceStudy(
        name="demo",
        description="demo study",
        parameter="h",
        levels=(10, 20, 40),
        sizes=(0.1, 0.05, 0.025),
        errors_l2=(4e-2, 1e-2, 2.5e-3),
        errors_max=(8e-2, 2e-2, 5e-3),
        expected_order=2.0,
    )


def test_study_table_lists_every_level_with_its_order() -> None:
    report = load_script()
    lines = report.study_table(study()).splitlines()
    assert lines[0].startswith("| cells per direction | h |")
    assert len(lines) == 2 + 3
    assert lines[3] == "| 20 | 0.05 | 1.000e-02 | 2.00 | 2.000e-02 | 2.00 |"


def test_readme_block_is_replaced_between_the_markers(tmp_path: Path) -> None:
    report = load_script()
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{report.README_START}\nold\n{report.README_END}\noutro\n")
    results = report.Results(
        space=[study()], time=[], analytic=[], composite=None, gci=[], balance=[], seconds=0.0
    )
    assert report.update_readme(readme, results)
    text = readme.read_text()
    assert "old" not in text
    assert "| demo study | 2 | 2.00 | 2.00 |  |" in text
    assert text.startswith("intro\n")
    assert text.endswith("outro\n")
    (tmp_path / "plain.md").write_text("no markers\n")
    assert not report.update_readme(tmp_path / "plain.md", results)


@pytest.mark.slow
def test_full_report_runs(tmp_path: Path) -> None:
    report = load_script()
    output = tmp_path / "verification.md"
    assert report.main(["--output", str(output), "--readme", str(tmp_path / "none.md")]) == 0
    text = output.read_text()
    for heading in ("## Summary", "## Order in time", "## Grid Convergence Index"):
        assert heading in text
