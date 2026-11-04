"""outputs/MANIFEST.md names exactly the files under outputs/results/, and its scripts exist."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs/results"
NAMED = set(re.findall(r"`([\w./-]+\.(?:csv|png|json|txt|py))`", (ROOT / "outputs/MANIFEST.md").read_text()))


def test_every_named_file_exists() -> None:
    assert not [name for name in NAMED if not ((RESULTS / name).exists() or (ROOT / name).exists())]


def test_every_output_file_is_named() -> None:
    files = {str(path.relative_to(RESULTS)) for path in RESULTS.rglob("*") if path.is_file()}
    assert not sorted(files - NAMED)
