import os, subprocess, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


@pytest.fixture(scope="session", autouse=True)
def trained():
    """Make sure the saved models exist before any test runs."""
    if not (ROOT / "artifacts" / "curves.json").exists():
        subprocess.check_call([sys.executable, "-m", "src.train"], cwd=ROOT)
