"""Central paths so every module finds the data and saved models regardless of the working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "bank-full.csv"
ARTIFACTS = ROOT / "artifacts"
